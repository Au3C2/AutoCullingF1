"""
sharpness.py — FFT high-frequency ratio sharpness scoring.
Refined for robustness against detection jitter.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING
import cv2
import numpy as np
from numpy.fft import rfft2  # darwin path; kept for cross-check tooling

if TYPE_CHECKING:
    from .detector import Detection

log = logging.getLogger(__name__)

_HF_LO: float = 0.0005
_HF_HI: float = 0.012
_LAP_REJECT: float = 3.0
_MIN_CROP_PX: int = 32
_ROI_BUFFER: float = 0.10  # 10% expansion to handle bbox jitter

# FFT backend selection (platform-adaptive, same HF condition on every path):
# - darwin: numpy.fft.rfft2 on the half spectrum. On Apple M4 numpy's pocketfft
#   is the fastest option (~1.9x vs full-spectrum fft2) and keeps scipy out of
#   the macOS bundle.
# - elsewhere: the pocketfft_fast wheel (same pocketfft algorithm family as
#   numpy/scipy vendor; SIMD-compiled, bit-identical output to scipy.fft,
#   zero heavy deps) when installed, falling back to cv2.dft on the full
#   spectrum otherwise. numpy 2.2's pocketfft has no SIMD dispatch on Windows
#   (19 ms/frame vs scipy's 5 ms — measured 2026-10-31), and scipy.fft drags
#   the full linalg/sparse/openblas closure into the bundle (~80 MB), so
#   neither is a usable Windows backend.
try:
    from pocketfft_fast import rfft2 as _pf_rfft2
    _HF_BACKEND = "pocketfft_fast"
except ImportError:
    _pf_rfft2 = None
    _HF_BACKEND = "numpy" if sys.platform == "darwin" else "cv2"

# Cache the HF masks keyed by (h, w) — they depend only on shape, not content.
# Rebuilding a mask every frame costs ~0.15 ms; caching eliminates that.
_hf_mask_rfft_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_hf_mask_full_cache: dict[tuple[int, int], np.ndarray] = {}

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

def _get_hf_mask_full(h: int, w: int) -> np.ndarray:
    """Boolean mask over the full fft2 spectrum marking HF bins
    (``r_sq > (max_r*0.5)**2`` with r from the mirrored frequency grid)."""
    key = (h, w)
    cached = _hf_mask_full_cache.get(key)
    if cached is None:
        dy = np.minimum(np.arange(h), h - np.arange(h)) ** 2
        dx = np.minimum(np.arange(w), w - np.arange(w)) ** 2
        r_sq = dy[:, None] + dx[None, :]
        max_r = min(h // 2, w // 2)
        _hf_mask_full_cache[key] = cached = r_sq > (max_r * 0.5) ** 2
    return cached

def _hf_half_spectrum(gray: np.ndarray, rfft2_fn) -> float:
    """Half-spectrum HF ratio via a numpy-compatible rfft2."""
    g_f32 = gray.astype(np.float32)
    h, w = gray.shape
    row_mask, col_weights = _get_hf_mask_rfft(h, w)

    spectrum = rfft2_fn(g_f32)  # (h, w//2+1) complex
    mag_sq = np.abs(spectrum) ** 2

    total = float(mag_sq.sum(axis=0) @ col_weights)
    if total <= 1e-9:
        return 0.0
    masked_col = (mag_sq * row_mask).sum(axis=0)
    return float((masked_col @ col_weights) / total)

def _hf_ratio_pocketfft(gray: np.ndarray) -> float:
    """Half-spectrum HF ratio via the pocketfft_fast wheel (non-darwin path).

    Bit-identical to scipy.fft (same pocketfft algorithm), ~4 ms/frame."""
    return _hf_half_spectrum(gray, _pf_rfft2)

def _hf_ratio_rfft2(gray: np.ndarray) -> float:
    """Half-spectrum HF ratio via numpy.fft.rfft2 (darwin path)."""
    return _hf_half_spectrum(gray, rfft2)

def _hf_ratio_cv2(gray: np.ndarray) -> float:
    """Full-spectrum HF ratio via cv2.dft (non-darwin path).

    Mathematically identical to the half-spectrum weighted sum (symmetry of
    the HF condition under both axis flips); differs only in float rounding
    (~1e-4 relative on the energy sums, far inside the 0.005 raw gate)."""
    g_f32 = gray.astype(np.float32)
    spectrum = cv2.dft(g_f32, flags=cv2.DFT_COMPLEX_OUTPUT)  # (h, w, 2) f32
    re, im = spectrum[:, :, 0], spectrum[:, :, 1]
    mag_sq = re * re + im * im

    total = float(mag_sq.sum())
    if total <= 1e-9:
        return 0.0
    return float(mag_sq[_get_hf_mask_full(*mag_sq.shape)].sum() / total)

def _hf_ratio(gray: np.ndarray) -> float:
    """Compute high-frequency energy ratio in frequency domain.

    Backend is platform-adaptive (see _HF_BACKEND): pocketfft_fast (or a
    cv2.dft fallback) off darwin, numpy.fft.rfft2 on darwin. All backends
    implement the same HF condition; results agree to float rounding noise
    (~1e-4 relative worst case), same magnitude as the earlier cv2.dft <->
    fft library swaps (optimization #B precedent)."""
    if _HF_BACKEND == "pocketfft_fast":
        return _hf_ratio_pocketfft(gray)
    if _HF_BACKEND == "numpy":
        return _hf_ratio_rfft2(gray)
    return _hf_ratio_cv2(gray)

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
