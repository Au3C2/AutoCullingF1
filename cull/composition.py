"""
composition.py — Composition scoring for culling.

Three sub-scores are computed and combined:

  S_fill    (weight 0.35) — How well the primary subject fills the frame.
  S_thirds  (weight 0.35) — Alignment with rule-of-thirds lines or image centre.
  S_lead    (weight 0.30) — Lead-room: space in the direction of motion / gaze.
                            The first frame of a burst group receives a neutral
                            score of 0.6 (direction unknown for single frames).

Final composition score:
  S_comp = 0.35 * S_fill + 0.35 * S_thirds + 0.30 * S_lead
"""

from __future__ import annotations

import logging

import numpy as np

from .detector import Detection

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-score weights
# ---------------------------------------------------------------------------
_W_FILL   = 0.35
_W_THIRDS = 0.35
_W_LEAD   = 0.30

# Neutral lead-room score when direction cannot be determined (first frame)
_LEAD_NEUTRAL = 0.6

# Optimal fill range: subject should cover between FILL_LO and FILL_HI of
# the frame area.  Outside this range the score tapers off.
_FILL_LO = 0.04   # 4% — subject too small (distant car)
_FILL_HI = 0.50   # 50% — subject fills half the frame (close-up)


# ---------------------------------------------------------------------------
# S_fill
# ---------------------------------------------------------------------------


def _score_fill(detection: Detection, img_w: int, img_h: int) -> float:
    """Score how well the subject fills the frame.

    Score is 1.0 when area_ratio is in [FILL_LO, FILL_HI].
    Tapers linearly to 0 outside that range.
    """
    ratio = detection.area_ratio(img_w, img_h)

    if _FILL_LO <= ratio <= _FILL_HI:
        return 1.0

    if ratio < _FILL_LO:
        # Too small
        return ratio / _FILL_LO

    # ratio > _FILL_HI: too large (cropped / out-of-frame car)
    over = (ratio - _FILL_HI) / (1.0 - _FILL_HI)
    return max(0.0, 1.0 - over)


# ---------------------------------------------------------------------------
# S_thirds
# ---------------------------------------------------------------------------

# Third-line positions as fractions of image dimension
_THIRD_POSITIONS = (1 / 3, 2 / 3)


def _score_thirds(detection: Detection, img_w: int, img_h: int) -> float:
    """Score alignment of the subject centre with rule-of-thirds or centre.

    The score is the maximum of:
      - proximity to any of the four thirds-line intersections, and
      - proximity to the image centre.

    Proximity is defined as 1 − (normalised distance).
    """
    cx_norm = detection.cx / img_w   # [0, 1]
    cy_norm = detection.cy / img_h   # [0, 1]

    # Thirds intersections (x, y) in normalised coordinates
    interest_points = [
        (tx, ty)
        for tx in _THIRD_POSITIONS
        for ty in _THIRD_POSITIONS
    ]
    # Also include image centre
    interest_points.append((0.5, 0.5))

    max_score = 0.0
    for px, py in interest_points:
        dist = ((cx_norm - px) ** 2 + (cy_norm - py) ** 2) ** 0.5
        # Max possible normalised distance is √2 / 2 ≈ 0.707
        score = max(0.0, 1.0 - dist / (2 ** 0.5 * 0.5))
        max_score = max(max_score, score)

    return max_score


# ---------------------------------------------------------------------------
# S_lead
# ---------------------------------------------------------------------------


def _score_lead(
    detection: Detection,
    img_w: int,
    img_h: int,
    prev_detection: Detection | None,
    is_first_frame: bool,
) -> float:
    """Score lead-room: space in the direction the subject is moving.

    Parameters
    ----------
    detection:
        Current frame's primary detection.
    img_w, img_h:
        Image dimensions.
    prev_detection:
        Primary detection from the *previous* frame (same burst group).
        ``None`` when unavailable.
    is_first_frame:
        If ``True`` (no motion vector available), return ``_LEAD_NEUTRAL``.

    Returns
    -------
    float ∈ [0, 1]
    """
    if is_first_frame or prev_detection is None:
        return _LEAD_NEUTRAL

    # Motion direction vector (horizontal dominates for racing)
    dx = detection.cx - prev_detection.cx
    dy = detection.cy - prev_detection.cy

    if abs(dx) < 1.0 and abs(dy) < 1.0:
        # No significant movement detected → neutral
        return _LEAD_NEUTRAL

    # Determine lead room: space ahead of the subject in the direction of motion
    cx_norm = detection.cx / img_w

    if dx > 0:
        # Moving right → lead room is space to the right
        lead_norm = 1.0 - cx_norm
    else:
        # Moving left → lead room is space to the left
        lead_norm = cx_norm

    # Apply a soft preference: optimal lead ≈ 30-60% of frame width ahead
    # Score peaks when lead_norm ≈ 0.35, drops for very tight or very loose framing
    optimal = 0.35
    deviation = abs(lead_norm - optimal)
    score = max(0.0, 1.0 - deviation / optimal)

    return float(score)


# ---------------------------------------------------------------------------
# Public API: score_composition
# ---------------------------------------------------------------------------


def score_composition(
    detections: list[Detection],
    img_w: int,
    img_h: int,
    prev_detections: list[Detection] | None = None,
    is_first_frame: bool = True,
) -> float:
    """Compute the overall composition score for a single image.

    Parameters
    ----------
    detections:
        Ordered list of detections (primary subject is index 0).
        If empty, returns 0.0.
    img_w, img_h:
        Image dimensions in pixels.
    prev_detections:
        Detections from the previous frame (for lead-room calculation).
        ``None`` or empty → ``is_first_frame`` is forced to ``True``.
    is_first_frame:
        Whether this is the first frame of a burst group (no prior motion).

    Returns
    -------
    float ∈ [0.0, 1.0]
    """
    if not detections:
        return 0.0

    primary = detections[0]
    prev_primary = (prev_detections[0] if prev_detections else None)

    # Force neutral lead when there is no previous frame reference
    if prev_primary is None:
        is_first_frame = True

    s_fill   = _score_fill(primary, img_w, img_h)
    s_thirds = _score_thirds(primary, img_w, img_h)
    s_lead   = _score_lead(primary, img_w, img_h, prev_primary, is_first_frame)

    s_comp = _W_FILL * s_fill + _W_THIRDS * s_thirds + _W_LEAD * s_lead

    log.debug(
        "Composition  fill=%.3f  thirds=%.3f  lead=%.3f  → S_comp=%.3f",
        s_fill, s_thirds, s_lead, s_comp,
    )

    return float(np.clip(s_comp, 0.0, 1.0))
