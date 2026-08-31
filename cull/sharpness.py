"""
sharpness.py — FFT high-frequency ratio sharpness scoring.
Refined for robustness against detection jitter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import cv2
import numpy as np
from scipy.fft import rfft2

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
_hf_mask_rfft_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

def _get_hf_mask_rfft(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Row mask for the rfft2 half-spectrum plus per-column symmetry weights.

    The full-spectrum HF condition ``r_sq > (max_r*0.5)**2`` is symmetric
    under both axis flips, so for half-spectrum column j <= w//2 the row mask
    equals the full-spectrum mask at that column (dx = min(j, w-j) == j).
    Columns 1..w//2 pair with their mirrored columns and count twice;
    column 0 — and column w/2 when w is even — are self-paired and count
    once. Summing half the spectrum with these weights yields the same HF
    energy ratio as summing the full fft2 spectrum."""
    key = (h, w)
    cached = _hf_mask_rfft_cache.get(key)
    if cached is None:
        dy = np.minimum(np.arange(h), h - np.arange(h)) ** 2
        cols = w // 2 + 1
        dx_sq = np.arange(cols, dtype=np.float64) ** 2
        r_sq = dy[:, None] + dx_sq[None, :]
        max_r = min(h // 2, w // 2)
        row_mask = r_sq > (max_r * 0.5) ** 2
        weights = np.full(cols, 2.0)
        weights[0] = 1.0
        if w % 2 == 0:
            weights[-1] = 1.0
        _hf_mask_rfft_cache[key] = cached = (row_mask, weights)
    return cached

def _hf_ratio(gray: np.ndarray) -> float:
    """Compute high-frequency energy ratio in frequency domain.

    Uses scipy.fft.rfft2 (pocketfft with ARM NEON SIMD + multithreading) on
    the half spectrum — real input makes the negative-frequency half redundant,
    so this does the same O(n log n) work on ~half the output grid vs fft2
    (~1.9x faster on Apple M4). The conjugate-symmetric columns are folded
    back in via the cached weights; result matches the full-spectrum fft2
    ratio to IEEE754 summation-order noise (~1e-7 relative), same magnitude
    as the earlier cv2.dft -> scipy.fft2 swap (gate-verified precedent)."""
    g_f32 = gray.astype(np.float32)
    h, w = gray.shape
    row_mask, col_weights = _get_hf_mask_rfft(h, w)

    try:
        from cull.deterministic import is_deterministic as _is_det_s
        _workers = 1 if _is_det_s() else -1
    except Exception:
        _workers = -1
    spectrum = rfft2(g_f32, workers=_workers)  # (h, w//2+1) complex64
    mag_sq = np.abs(spectrum) ** 2

    total = float(mag_sq.sum(axis=0) @ col_weights)
    if total <= 1e-9:
        return 0.0
    masked_col = (mag_sq * row_mask).sum(axis=0)
    return float((masked_col @ col_weights) / total)

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
