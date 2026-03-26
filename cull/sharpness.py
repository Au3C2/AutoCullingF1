"""
sharpness.py — FFT high-frequency ratio sharpness scoring.
Refined for robustness against detection jitter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
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
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag_sq = np.abs(fshift) ** 2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = min(cx, cy)
    mask = r > max_r * 0.5
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

    # Grayscale: BT.601 exactly to match OpenCV color conversion
    gray = (0.299 * region[..., 0] + 0.587 * region[..., 1] + 0.114 * region[..., 2]).astype(np.uint8)

    if lap_reject > 0.0:
        lvar = laplacian_variance(gray)
        if lvar <= 0: return 0.0
        lv = float(np.log(lvar + 1e-9))
        if lv < lap_reject: return 0.0

    hf = _hf_ratio(gray)
    return float(np.clip((hf - hf_lo) / (hf_hi - hf_lo), 0.0, 1.0))
