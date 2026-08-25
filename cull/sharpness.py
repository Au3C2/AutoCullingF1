"""
sharpness.py — FFT high-frequency ratio sharpness scoring.
Refined for robustness against detection jitter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import cv2
import numpy as np
from scipy.fft import fft2

if TYPE_CHECKING:
    from .detector import Detection

log = logging.getLogger(__name__)

_HF_LO: float = 0.0005
_HF_HI: float = 0.012
_LAP_REJECT: float = 3.0
_MIN_CROP_PX: int = 32
_ROI_BUFFER: float = 0.10  # 10% expansion to handle bbox jitter

# Cache the HF mask keyed by (h, w) — it depends only on shape, not content.
# Rebuilding the mask every frame costs ~0.15 ms; caching eliminates that.
_hf_mask_cache: dict[tuple[int, int], np.ndarray] = {}

def _get_hf_mask(h: int, w: int) -> np.ndarray:
    """Get the cached HF mask for a given shape."""
    key = (h, w)
    if key not in _hf_mask_cache:
        dy = np.minimum(np.arange(h), h - np.arange(h)) ** 2
        dx = np.minimum(np.arange(w), w - np.arange(w)) ** 2
        r_sq = dy[:, None] + dx[None, :]
        max_r = min(h // 2, w // 2)
        _hf_mask_cache[key] = r_sq > (max_r * 0.5) ** 2
    return _hf_mask_cache[key]

def _hf_ratio(gray: np.ndarray) -> float:
    """Compute high-frequency energy ratio in frequency domain.

    Uses scipy.fft.fft2 (pocketfft with ARM NEON SIMD + multithreading) —
    ~6x faster than cv2.dft COMPLEX_OUTPUT on Apple M4. Returns the full
    complex spectrum directly, so no conjugate-symmetry reconstruction is
    needed (eliminating a class of correctness bugs).

    Score difference vs cv2.dft: ~1e-7 after clip to [0,1] — IEEE754
    rounding noise from different FFT algorithms.
    """
    g_f32 = gray.astype(np.float32)
    h, w = gray.shape
    mask = _get_hf_mask(h, w)

    spectrum = fft2(g_f32, workers=-1)
    mag_sq = np.abs(spectrum) ** 2

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
