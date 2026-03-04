"""
scorer.py — Aggregate scores, apply veto rules, map to Lightroom Rating.

Scoring pipeline per image
--------------------------
1. Veto check:
   - No detections           → Rating = -1 (Lightroom "Rejected" flag)
   - S_sharp < SHARP_THRESH  → Rating = -1

2. Raw score:
   raw = 3.5 * S_sharp + 1.5 * S_comp          (sharpness 70%, comp 30%)
   raw is in approximately [0, 5]

3. Rating mapping:
   raw   → Rating
   < 1   →  1
   1-2   →  2
   2-3   →  3
   3-4   →  4
   ≥ 4   →  5

Burst group TopN selection
--------------------------
After scoring all frames in a burst group, keep the top-N by raw score.
The kept frames receive their computed Rating (1-5).
All others are downgraded to Rating = -1 (rejected).

N is configurable (default 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARP_THRESH: float = 0.15   # veto threshold — below this → Rating -1

# Rating breakpoints for raw score → 1-5 stars
_RATING_BREAKS = [1.0, 2.0, 3.0, 4.0]   # boundaries between ratings 1/2/3/4/5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ImageScore:
    """Scoring result for a single image."""

    path: Path
    s_sharp: float           # sharpness score ∈ [0, 1]
    s_comp: float            # composition score ∈ [0, 1]
    raw_score: float         # combined raw score ∈ [0, 5]
    rating: int              # Lightroom Rating: -1 (reject) or 1-5 (stars)
    vetoed: bool = False     # True if veto rule triggered
    veto_reason: str = ""    # human-readable reason for veto


# ---------------------------------------------------------------------------
# Rating mapping
# ---------------------------------------------------------------------------


def _raw_to_rating(raw: float) -> int:
    """Map a raw score (≈ 0-5) to a Lightroom Rating (1-5)."""
    for stars, threshold in enumerate(_RATING_BREAKS, start=1):
        if raw < threshold:
            return stars
    return 5


# ---------------------------------------------------------------------------
# Public API: score_image
# ---------------------------------------------------------------------------


def score_image(
    path: Path,
    detections: list,           # list[Detection] — type-hinted loosely to avoid circular
    s_sharp: float,
    s_comp: float,
    sharp_thresh: float = SHARP_THRESH,
) -> ImageScore:
    """Compute the final score and Rating for a single image.

    Parameters
    ----------
    path:
        Image file path (stored in the result for traceability).
    detections:
        Detection list from ``detector.detect``.  Empty list → veto.
    s_sharp:
        Sharpness score from ``sharpness.score_sharpness``.
    s_comp:
        Composition score from ``composition.score_composition``.
    sharp_thresh:
        Minimum sharpness to avoid veto.

    Returns
    -------
    ImageScore
    """
    # --- Veto checks ---------------------------------------------------------
    if not detections:
        raw = 3.5 * s_sharp + 1.5 * s_comp
        return ImageScore(
            path=path,
            s_sharp=s_sharp,
            s_comp=s_comp,
            raw_score=raw,
            rating=-1,
            vetoed=True,
            veto_reason="no_detection",
        )

    if s_sharp < sharp_thresh:
        raw = 3.5 * s_sharp + 1.5 * s_comp
        return ImageScore(
            path=path,
            s_sharp=s_sharp,
            s_comp=s_comp,
            raw_score=raw,
            rating=-1,
            vetoed=True,
            veto_reason=f"sharpness={s_sharp:.3f} < threshold={sharp_thresh:.3f}",
        )

    # --- Normal scoring ------------------------------------------------------
    raw = 3.5 * s_sharp + 1.5 * s_comp
    rating = _raw_to_rating(raw)

    log.debug(
        "%s  sharp=%.3f  comp=%.3f  raw=%.3f  → Rating %d",
        path.name, s_sharp, s_comp, raw, rating,
    )

    return ImageScore(
        path=path,
        s_sharp=s_sharp,
        s_comp=s_comp,
        raw_score=raw,
        rating=rating,
        vetoed=False,
    )


# ---------------------------------------------------------------------------
# Public API: select_best_n
# ---------------------------------------------------------------------------


def select_best_n(
    scores: list[ImageScore],
    top_n: int = 3,
) -> list[ImageScore]:
    """Apply burst-group TopN selection.

    Within a group, only the ``top_n`` frames by raw_score keep their Rating.
    All other frames are downgraded to Rating = -1.

    Already-vetoed frames (rating == -1) do not count toward TopN.

    Parameters
    ----------
    scores:
        All ImageScore objects for frames in a single burst group, in any order.
    top_n:
        Maximum number of frames to keep (default 3).

    Returns
    -------
    list[ImageScore]
        Same list, mutated in-place (rating fields updated), returned for
        convenience.
    """
    if top_n <= 0:
        for s in scores:
            s.rating = -1
        return scores

    # Only non-vetoed frames are candidates
    candidates = [s for s in scores if not s.vetoed]
    candidates.sort(key=lambda s: s.raw_score, reverse=True)

    keep_set = {id(s) for s in candidates[:top_n]}

    for s in scores:
        if not s.vetoed and id(s) not in keep_set:
            s.rating = -1
            s.veto_reason = "burst_group_topn"

    return scores
