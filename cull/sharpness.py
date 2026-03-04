"""
sharpness.py — Laplacian-variance sharpness scoring.

Only the region inside the primary detection bounding box is evaluated.
Background blur (e.g. bokeh) is intentionally ignored, which better matches
a photographer's judgement of whether the *subject* is in focus.

Score normalisation
-------------------
Raw Laplacian variance is log-transformed and mapped to [0, 1] via empirical
calibration constants.  The defaults are calibrated for ~1664×1088 preview
images extracted from Sony A7C2 HIF files (embedded HEVC stream #6).

If working at a different resolution, adjust ``LAP_LO`` / ``LAP_HI`` via
the function parameters or re-run ``_calibrate_sharpness.py``.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .detector import Detection

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calibration constants for normalisation
# log(variance) is clipped to [LAP_LO, LAP_HI] then mapped to [0, 1]
#
# Calibrated on 94 sample Sony A7C2 HIF files decoded at 1664×1088
# (embedded preview stream #6):
#   p5 = 3.58,  p50 = 5.03,  p95 = 6.00
# Using slightly wider range for safety margin.
# ---------------------------------------------------------------------------
_LAP_LO: float = 3.0   # log-variance corresponding to score ≈ 0 (very blurry)
_LAP_HI: float = 6.5   # log-variance corresponding to score ≈ 1 (very sharp)

# Minimum bbox size (pixels) to bother evaluating; smaller crops are unreliable
_MIN_CROP_PX: int = 32


def score_sharpness(
    img_bgr: np.ndarray,
    detection: Detection | None,
    lap_lo: float = _LAP_LO,
    lap_hi: float = _LAP_HI,
) -> float:
    """Compute a sharpness score for the primary subject region.

    Parameters
    ----------
    img_bgr:
        Full-resolution image as a BGR uint8 numpy array (as returned by
        ``cv2.imread``).  Colour order does not matter for sharpness.
    detection:
        The primary (highest-scoring) Detection.  If ``None``, the whole
        image is evaluated (less reliable but better than nothing).
    lap_lo:
        Log-variance that maps to score 0.0.
    lap_hi:
        Log-variance that maps to score 1.0.

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
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if lap_var <= 0:
        return 0.0

    log_var = float(np.log(lap_var + 1e-9))
    score = (log_var - lap_lo) / (lap_hi - lap_lo)
    return float(np.clip(score, 0.0, 1.0))
