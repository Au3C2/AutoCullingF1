"""preview.py — thumbnail rendering with detection/crop overlays for the GUI.

Decoding and overlay drawing happen in a background thread (``render_pil``,
no Tkinter involved); the main thread wraps the result in a ``CTkImage``
(which scales correctly on HighDPI displays).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cull.loader import load_image_rgb
from cull.scorer import ImageScore

log = logging.getLogger(__name__)

MAX_PREVIEW = 640  # longest side of the preview thumbnail, in pixels
_CACHE_SIZE = 16   # decoded images kept in memory

_DETECTION_COLOR = (46, 204, 113)   # green
_CROP_COLOR = (243, 156, 18)        # orange


def _fit(img: np.ndarray, max_size: int) -> np.ndarray:
    """Downscale *img* (H,W,3 uint8) so its longest side is <= max_size."""
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return img
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.asarray(Image.fromarray(img).resize(new_size, Image.BILINEAR))


@lru_cache(maxsize=_CACHE_SIZE)
def _load_cached(path: Path, max_size: int) -> np.ndarray | None:
    """Decode and downscale an image (cacheable by path, no Tk involved)."""
    try:
        img = load_image_rgb(path, scale_width=max_size)
    except Exception:
        log.warning("Preview decode failed for %s", path)
        return None
    if img is None:
        return None
    return _fit(img, max_size)


def render_pil(score: ImageScore, max_size: int = MAX_PREVIEW) -> Image.Image | None:
    """Render *score.path* with detection/crop overlays.

    Safe to call from any thread (pure Pillow/numpy). Decoded frames are
    cached by path. Returns None when the image cannot be decoded.
    """
    img = _load_cached(score.path, max_size)
    if img is None:
        return None

    h, w = img.shape[:2]
    pil = Image.fromarray(img).convert("RGB")

    if score.detections or score.crop:
        draw = ImageDraw.Draw(pil)
        fx = w / max(1, score.img_w)
        fy = h / max(1, score.img_h)
        for det in score.detections:
            draw.rectangle(
                [det.x1 * fx, det.y1 * fy, det.x2 * fx, det.y2 * fy],
                outline=_DETECTION_COLOR, width=2,
            )
            draw.text((det.x1 * fx + 2, det.y1 * fy + 2), det.label, fill=_DETECTION_COLOR)
        if score.crop is not None:
            top, left, bottom, right = score.crop
            draw.rectangle(
                [left * w, top * h, right * w, bottom * h],
                outline=_CROP_COLOR, width=3,
            )

    return pil