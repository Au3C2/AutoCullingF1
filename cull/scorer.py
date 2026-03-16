"""
scorer.py — Aggregate scores, apply veto rules, map to Lightroom Rating.

Scoring pipeline per image
--------------------------
1. Veto check:
   - Wire fence detected      → Rating = -1 (new: P3 fence detection)
   - No detections           → Rating = -1 (Lightroom "Rejected" flag)
   - S_sharp < SHARP_THRESH  → Rating = -1
   - raw < MIN_RAW            → Rating = -1

2. Raw score:
   raw = W_SHARP * S_sharp + W_COMP * S_comp    (sharpness 50%, comp 50%)
   raw is in approximately [0, 6.0]

3. Rating mapping (tuned for non-vetoed raw range ~2.9-6.0):
   raw       → Rating
   < 3.40    →  1★
   3.40-3.80 →  2★
   3.80-4.20 →  3★
   4.20-4.60 →  4★
   ≥ 4.60    →  5★

Burst group TopN selection
--------------------------
After scoring all frames in a burst group, keep the top-N by raw score.
The kept frames receive their computed Rating (1-5).
All others are downgraded to Rating = -1 (rejected).

N is configurable (default 11).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants  (v5 — tuned via offline grid search with user feedback fixed labels)
# ---------------------------------------------------------------------------

SHARP_THRESH: float = 0.05   # veto threshold — below this → Rating -1
W_SHARP: float = 1.5          # weight for sharpness in raw score formula
W_COMP: float = 2.5           # weight for composition in raw score formula
MIN_RAW: float = 3.1          # minimum raw score — below this → Rating -1

# Fence detection (P3)
ENABLE_FENCE_VETO: bool = False  # enable fence veto (set to False to disable)

# Orientation & Integrity detection (P4)
ENABLE_P4 = True                 # enable P4 evaluation
P4_ORIENT_VETO = ['rear']        # completely veto these orientations
P4_CUT_PENALTY = 0.6             # reduce raw_score by this much if cut/occluded

# Rating breakpoints for raw score → 1-5 stars.
# Tuned for the range [MIN_RAW, ~6.0] to give a natural star gradient.
# With hf_ratio sharpness, s_sharp has much better spread (no ceiling effect)
# so raw scores span a wider range than the old Laplacian-based version.
_RATING_BREAKS = [3.40, 3.80, 4.20, 4.60]   # boundaries between ratings 1/2/3/4/5
# ---------------------------------------------------------------------------
# Global fence classifier instance (lazy-loaded on first use)
# ---------------------------------------------------------------------------

_FENCE_CLASSIFIER = None  # Lazy-loaded FenceClassifier instance


def _get_fence_classifier():
    """Get or initialize the global fence classifier (lazy-loaded)."""
    global _FENCE_CLASSIFIER
    if _FENCE_CLASSIFIER is None:
        try:
            from cull.fence_classifier import FenceClassifier
            _FENCE_CLASSIFIER = FenceClassifier(arch="mobilenetv2")
            log.info("Fence classifier loaded (MobileNetV2, F1=0.9796)")
        except Exception as e:
            log.error(f"Failed to load fence classifier: {e}. Fence veto disabled.")
            _FENCE_CLASSIFIER = False  # Mark as failed
    return _FENCE_CLASSIFIER if _FENCE_CLASSIFIER is not False else None


_P4_CLASSIFIER = None  # Lazy-loaded P4Classifier instance

def _get_p4_classifier():
    """Get or initialize the global P4 classifier (lazy-loaded)."""
    global _P4_CLASSIFIER
    if _P4_CLASSIFIER is None:
        try:
            from cull.p4_classifier import P4Classifier
            _P4_CLASSIFIER = P4Classifier()
        except Exception as e:
            log.error(f"Failed to load P4 classifier: {e}. P4 disabled.")
            _P4_CLASSIFIER = False
    return _P4_CLASSIFIER if _P4_CLASSIFIER is not False else None


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
    n_detections: int = 0    # number of detections (for offline replay)
    burst_group: int = -1    # burst group index (set by run())
    fence_pred: int = 0      # fence classifier: 0 (no fence) or 1 (fence) [new: P3]
    fence_confidence: float = 0.0  # fence classifier confidence [0, 1] [new: P3]
    p4_orient: str = "unknown"     # orientation label [new: P4]
    p4_orient_conf: float = 0.0    # orientation confidence [new: P4]
    p4_integ: int = 1              # integrity: 1 (full), 0 (cut) [new: P4]
    p4_integ_prob: float = 1.0     # integrity confidence [new: P4]
    is_manual: bool = False        # True if rating was loaded from existing XMP sidecar


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
    w_sharp: float = W_SHARP,
    w_comp: float = W_COMP,
    min_raw: float = MIN_RAW,
    check_fence: bool = ENABLE_FENCE_VETO,
    check_p4: bool = ENABLE_P4,
    img_rgb: np.ndarray | None = None,
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
    w_sharp:
        Weight for sharpness in raw score formula.
    w_comp:
        Weight for composition in raw score formula.
    min_raw:
        Minimum raw score to avoid veto (0.0 to disable).
    check_fence:
        Whether to check for wire fence occlusion (P3).
    img_rgb:
        Preloaded RGB numpy array. If provided, ROI cropping is used for fence classification.

    Returns
    -------
    ImageScore
    """
    n_det = len(detections)
    raw = w_sharp * s_sharp + w_comp * s_comp
    
    fence_pred = 0
    fence_confidence = 0.0

    # --- Wire fence veto check (P3) -----------------------------------------------
    if check_fence and n_det > 0:
        try:
            classifier = _get_fence_classifier()
            if classifier is not None:
                if img_rgb is not None:
                    # Pass the primary subject's ROI for more robust fence detection
                    primary = detections[0]
                    bbox = (primary.x1, primary.y1, primary.x2, primary.y2)
                    fence_pred, fence_confidence = classifier.predict_roi(img_rgb, bbox)
                else:
                    # Fallback to full image inference
                    fence_pred, fence_confidence = classifier.predict_image(path)
                    
                if fence_pred == 1 and fence_confidence > 0.7 and raw < 4.0:
                    return ImageScore(
                        path=path,
                        s_sharp=s_sharp,
                        s_comp=s_comp,
                        raw_score=raw,
                        rating=-1,
                        vetoed=True,
                        veto_reason=f"fence_detected (confidence={fence_confidence:.3f})",
                        n_detections=n_det,
                        fence_pred=fence_pred,
                        fence_confidence=fence_confidence,
                    )
        except Exception as e:
            log.warning(f"Fence detection error for {path}: {e}")

    if not detections:
        return ImageScore(
            path=path,
            s_sharp=s_sharp,
            s_comp=s_comp,
            raw_score=raw,
            rating=-1,
            vetoed=True,
            veto_reason="no_detection",
            n_detections=0,
            fence_pred=fence_pred,
            fence_confidence=fence_confidence,
        )
        
    p4_orient = "unknown"
    p4_orient_conf = 0.0
    p4_integ = 1
    p4_integ_prob = 1.0
    
    # --- P4 Evaluation (Orientation + Integrity) ---
    if check_p4 and img_rgb is not None:
        p4_c = _get_p4_classifier()
        if p4_c is not None:
            primary = detections[0]
            bbox = (primary.x1, primary.y1, primary.x2, primary.y2)
            p4_orient, p4_orient_conf, p4_integ, p4_integ_prob = p4_c.predict_roi(img_rgb, bbox)
            
            # Apply P4 Penalties
            if p4_integ == 0:
                # Apply penalty for Cut/Occluded (reduces raw score, likely causing min_raw veto!)
                raw -= P4_CUT_PENALTY

    if s_sharp < sharp_thresh:
        return ImageScore(
            path=path,
            s_sharp=s_sharp,
            s_comp=s_comp,
            raw_score=raw,
            rating=-1,
            vetoed=True,
            veto_reason=f"sharpness={s_sharp:.3f} < threshold={sharp_thresh:.3f}",
            n_detections=n_det,
            fence_pred=fence_pred,
            fence_confidence=fence_confidence,
            p4_orient=p4_orient,
            p4_orient_conf=p4_orient_conf,
            p4_integ=p4_integ,
            p4_integ_prob=p4_integ_prob,
        )

    if min_raw > 0.0 and raw < min_raw:
        return ImageScore(
            path=path,
            s_sharp=s_sharp,
            s_comp=s_comp,
            raw_score=raw,
            rating=-1,
            vetoed=True,
            veto_reason=f"raw={raw:.3f} < min_raw={min_raw:.3f} (cut penalty applied? {p4_integ == 0})",
            n_detections=n_det,
            fence_pred=fence_pred,
            fence_confidence=fence_confidence,
            p4_orient=p4_orient,
            p4_orient_conf=p4_orient_conf,
            p4_integ=p4_integ,
            p4_integ_prob=p4_integ_prob,
        )
        
    if p4_orient in P4_ORIENT_VETO:
        return ImageScore(
            path=path,
            s_sharp=s_sharp,
            s_comp=s_comp,
            raw_score=raw,
            rating=-1,
            vetoed=True,
            veto_reason=f"p4_orient={p4_orient} (confidence={p4_orient_conf:.2f})",
            n_detections=n_det,
            fence_pred=fence_pred,
            fence_confidence=fence_confidence,
            p4_orient=p4_orient,
            p4_orient_conf=p4_orient_conf,
            p4_integ=p4_integ,
            p4_integ_prob=p4_integ_prob,
        )

    # --- Normal scoring ------------------------------------------------------
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
        n_detections=n_det,
        fence_pred=fence_pred,
        fence_confidence=fence_confidence,
        p4_orient=p4_orient,
        p4_orient_conf=p4_orient_conf,
        p4_integ=p4_integ,
        p4_integ_prob=p4_integ_prob,
    )


# ---------------------------------------------------------------------------
# Public API: select_best_n
# ---------------------------------------------------------------------------


def select_best_n(
    scores: list[ImageScore],
    top_n: int = 12,
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
        Maximum number of frames to keep (default 12).

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

    # Only non-vetoed and non-manual frames are candidates for auto-culling
    candidates = [s for s in scores if not s.vetoed and not s.is_manual]
    candidates.sort(key=lambda s: s.raw_score, reverse=True)

    keep_set = {id(s) for s in candidates[:top_n]}

    for s in scores:
        if not s.vetoed and not s.is_manual and id(s) not in keep_set:
            s.rating = -1
            s.veto_reason = "burst_group_topn"

    return scores
