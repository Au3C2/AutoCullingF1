"""
sharpness.py — FFT high-frequency ratio sharpness scoring.
Refined for robustness against detection jitter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import cv2
import numpy as np

if TYPE_CHECKING:
    from .detector import Detection

log = logging.getLogger(__name__)

_HF_LO: float = 0.0005
_HF_HI: float = 0.012
_LAP_REJECT: float = 3.0
_MIN_CROP_PX: int = 32
_ROI_BUFFER: float = 0.10  # 10% expansion to handle bbox jitter

def _hf_ratio(gray: np.ndarray) -> float:
    """Compute high-frequency energy ratio in frequency domain.

    Optimized using C++ cv2.dft on float32 and unshifted radial distance
    broadcasting (eliminating np.fftshift memory copy and np.mgrid/sqrt overhead).
    Mathematically identical to np.fft.fft2 + fftshift + mgrid (diff < 1e-6).
    """
    g_f32 = gray.astype(np.float32)
    h, w = gray.shape
    dft = cv2.dft(g_f32, flags=cv2.DFT_COMPLEX_OUTPUT)

    # In unshifted DFT, distance to DC component (0,0) in periodic domain is:
    # dy = min(y, h - y), dx = min(x, w - x) -> r_sq = dy^2 + dx^2
    dy = np.minimum(np.arange(h), h - np.arange(h)) ** 2
    dx = np.minimum(np.arange(w), w - np.arange(w)) ** 2
    r_sq = dy[:, None] + dx[None, :]

    max_r = min(h // 2, w // 2)
    threshold_r_sq = (max_r * 0.5) ** 2
    mask = r_sq > threshold_r_sq

    # Squared magnitude directly from complex components (real^2 + imag^2)
    mag_sq = dft[..., 0] ** 2 + dft[..., 1] ** 2
    total = mag_sq.sum()
    return float(mag_sq[mask].sum() / total) if total > 1e-9 else 0.0

def laplacian_variance(gray: np.ndarray) -> float:
    p = np.pad(gray.astype(np.float64), 1, mode='edge')
    lap = p[1:-1, 0:-2] + p[1:-1, 2:] + p[0:-2, 1:-1] + p[2:, 1:-1] - 4 * p[1:-1, 1:-1]
    return float(np.var(lap))

def score_sharpness(
    img_rgb: np.ndarray,
    detection: Detection | None,
    hf_lo: float = _HF_LO,
    hf_hi: float = _HF_HI,
    lap_reject: float = _LAP_REJECT,
) -> float:
    h, w = img_rgb.shape[:2]
    if detection:
        # Expand ROI slightly to be robust to bbox jitter
        bw, bh = detection.x2 - detection.x1, detection.y2 - detection.y1
        pad_w, pad_h = bw * _ROI_BUFFER, bh * _ROI_BUFFER

        x1 = max(0, int(detection.x1 - pad_w))
        y1 = max(0, int(detection.y1 - pad_h))
        x2 = min(w, int(detection.x2 + pad_w))
        y2 = min(h, int(detection.y2 + pad_h))

        region = img_rgb[y1:y2, x1:x2] if (x2-x1) >= _MIN_CROP_PX and (y2-y1) >= _MIN_CROP_PX else img_rgb
    else:
        region = img_rgb

    # Grayscale via cv2 (fixed-point BT.601): ~12x faster than the float dot
    # product; differs by <=1 LSB on ~64% of pixels. The Laplacian gate runs
    # on cv2.Laplacian(CV_32F) (~2.4x faster, variance within ~1% of the f64
    # reference). Both changes are gate-verified (see performance_baseline.md).
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    if lap_reject > 0.0:
        lvar = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        if lvar <= 0: return 0.0
        lv = float(np.log(lvar + 1e-9))
        if lv < lap_reject: return 0.0

    hf = _hf_ratio(gray)
    return float(np.clip((hf - hf_lo) / (hf_hi - hf_lo), 0.0, 1.0))
