"""
sharpness.py — FFT high-frequency ratio sharpness scoring.

Evaluates sharpness via the ratio of high-frequency energy to total energy
in the 2D FFT of the detection ROI.  Motion blur kills high-frequency detail,
making this metric an effective discriminator (d' = 0.83 on corrected
experiment, validated on 76 images with controlled sampling).

A fast Laplacian pre-check is used to quickly reject extremely blurry images
without the overhead of FFT computation.

Score normalisation
-------------------
``hf_ratio`` is clipped to [HF_LO, HF_HI] and linearly mapped to [0, 1].
Calibration was performed on the full 2018-image test set at 1664×1088
preview resolution (Sony A7C2 HIF stream #6, ROI from cascade detection):

    p1  = 0.000406    → near-zero high-freq (extremely blurry)
    p50 = 0.008985    → median
    p90 = 0.026290    → very sharp
    p99 = 0.061478

Using HF_LO=0.001 (near-p1) and HF_HI=0.025 (near-p90) to give good
dynamic range: most images map between 0.0-1.0 with natural spread.

Previous Laplacian-only approach had a severe ceiling effect (58.5% of images
scored >= 0.85, 46.7% scored >= 0.95), making burst-group ranking nearly
impossible on sharpness alone.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .detector import Detection

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calibration constants — FFT high-frequency energy ratio
# Calibrated on 2018-image test set at 1664×1088 preview resolution.
# ---------------------------------------------------------------------------
_HF_LO: float = 0.001    # hf_ratio corresponding to score ≈ 0 (very blurry)
_HF_HI: float = 0.025    # hf_ratio corresponding to score ≈ 1 (very sharp)

# ---------------------------------------------------------------------------
# Laplacian pre-check — fast rejection of extreme blur
# If log(Laplacian variance) < this threshold, skip the FFT and return 0.
# This catches completely out-of-focus / black frames quickly.
# At 1664×1088 preview resolution, p1 of log_lap ≈ 3.0.
# ---------------------------------------------------------------------------
_LAP_REJECT: float = 3.0

# Minimum bbox size (pixels) to bother evaluating; smaller crops are unreliable
_MIN_CROP_PX: int = 32


def _hf_ratio(gray: np.ndarray) -> float:
    """Compute high-frequency energy ratio via 2D FFT.

    Returns the fraction of total spectral energy residing in the outer 50%
    of the frequency space (i.e., frequencies above half the Nyquist limit).

    Motion blur suppresses high-frequency content, reducing this ratio.
    Sharp images with detailed textures/edges have higher ratios.

    Parameters
    ----------
    gray:
        Grayscale image (uint8 or float).

    Returns
    -------
    float
        High-frequency energy ratio, typically in [0.0001, 0.06].
    """
    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    mag_sq = np.abs(fshift) ** 2

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    y_grid, x_grid = np.mgrid[0:h, 0:w]
    radii = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    max_r = min(cx, cy)

    high_mask = radii > max_r * 0.5  # outer 50% of frequency space
    total_energy = mag_sq.sum()
    if total_energy < 1e-9:
        return 0.0
    return float(mag_sq[high_mask].sum() / total_energy)


def score_sharpness(
    img_bgr: np.ndarray,
    detection: Detection | None,
    hf_lo: float = _HF_LO,
    hf_hi: float = _HF_HI,
    lap_reject: float = _LAP_REJECT,
) -> float:
    """Compute a sharpness score for the primary subject region.

    Uses a two-stage approach:
      1. Fast Laplacian pre-check — reject extremely blurry images (log_lap < 3.0)
      2. FFT high-frequency ratio — compute the main sharpness score

    Parameters
    ----------
    img_bgr:
        Full-resolution image as a BGR uint8 numpy array (as returned by
        ``cv2.imread``).  Colour order does not matter for sharpness.
    detection:
        The primary (highest-scoring) Detection.  If ``None``, the whole
        image is evaluated (less reliable but better than nothing).
    hf_lo:
        hf_ratio that maps to score 0.0.
    hf_hi:
        hf_ratio that maps to score 1.0.
    lap_reject:
        Log-Laplacian-variance below which we skip FFT and return 0.0
        immediately.  Set to 0.0 to disable the fast rejection.

    Returns
    -------
    float
        Sharpness score in [0.0, 1.0].
    """
    h, w = img_bgr.shape[:2]

    if detection is not None:
        x1 = max(0, int(detection.x1))
        y1 = max(0, int(detection.y1))
        x2 = min(w, int(detection.x2))
        y2 = min(h, int(detection.y2))

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w >= _MIN_CROP_PX and crop_h >= _MIN_CROP_PX:
            region = img_bgr[y1:y2, x1:x2]
        else:
            log.debug(
                "Bbox too small (%dx%d); using full image for sharpness",
                crop_w, crop_h,
            )
            region = img_bgr
    else:
        region = img_bgr

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # --- Stage 1: Fast Laplacian pre-check -----------------------------------
    if lap_reject > 0.0:
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var <= 0:
            return 0.0
        log_var = float(np.log(lap_var + 1e-9))
        if log_var < lap_reject:
            log.debug("Laplacian pre-reject: log_lap=%.2f < %.2f", log_var, lap_reject)
            return 0.0

    # --- Stage 2: FFT high-frequency ratio -----------------------------------
    hf = _hf_ratio(gray)
    score = (hf - hf_lo) / (hf_hi - hf_lo)
    return float(np.clip(score, 0.0, 1.0))
