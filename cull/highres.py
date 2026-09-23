"""High-resolution data provider for intelligent on-demand zoom.

Implements a 3-tier passthrough and extraction pipeline:
- Tier 1: Zero-transcode, zero-write passthrough for native JPG, JPEG, HIF, HEIF, HEIC.
- Tier 2: Zero-recode binary stream Range I/O extraction for RAW files with embedded JPEGs (ARW/NEF/CR3).
- Tier 3: Hardware-accelerated decode & HEIF fallback (when no full-res preview is embedded).
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from cull.loader import (
    COOKED_EXTS,
    HEIF_EXTS,
    RAW_EXTS,
    _extract_raw_tiff_direct,
    load_image_rgb,
)

log = logging.getLogger(__name__)


@dataclass
class HighResRequest:
    file_path: Path
    gen_id: int
    roi: Optional[list[float]] = None  # [nx1, ny1, nx2, ny2]


@dataclass
class HighResResponse:
    gen_id: int
    resolved_path: Path
    format: str
    width: int
    height: int
    tier: str


class HighResProvider:
    """Manages high-resolution image resolution, caching, and generation cancellation."""

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "auto_culling" / "highres"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._active_generation = 0

    def update_active_generation(self, gen_id: int) -> None:
        """Update active generation token. Requests with gen_id < active are discarded."""
        if gen_id > self._active_generation:
            self._active_generation = gen_id

    def resolve(self, req: HighResRequest) -> Optional[HighResResponse]:
        """Resolve high-res asset for the request, honoring generation token."""
        if req.gen_id < self._active_generation:
            log.debug("Discarding obsolete highres request gen_id=%d < active=%d",
                      req.gen_id, self._active_generation)
            return None

        p = req.file_path.resolve()
        if not p.exists():
            log.warning("HighRes target file not found: %s", p)
            return None

        ext = p.suffix.lower()

        # Tier 1: Native Cooked / HEIF passthrough (0 ms, 0 transcode, 0 disk write)
        if ext in COOKED_EXTS or ext in HEIF_EXTS:
            w, h = self._probe_dimensions(p)
            fmt = "heif" if ext in HEIF_EXTS else ext.lstrip(".")
            return HighResResponse(
                gen_id=req.gen_id,
                resolved_path=p,
                format=fmt,
                width=w,
                height=h,
                tier="tier1_passthrough",
            )

        # Tier 2: RAW files with embedded full-resolution preview streams
        if ext in RAW_EXTS:
            res = self._resolve_raw_embedded(p, req.gen_id)
            if res is not None:
                return res

        # Tier 3: Sensor Demosaic & fallback
        return self._resolve_tier3_fallback(p, req.gen_id)

    def _resolve_raw_embedded(self, raw_path: Path, gen_id: int) -> Optional[HighResResponse]:
        """Tier 2: Range I/O direct binary extraction without decompressing or recoding."""
        file_stat = raw_path.stat()
        # Cache key based on file path, size, and mtime
        key_str = f"{raw_path.resolve()}_{file_stat.st_size}_{file_stat.st_mtime_ns}"
        cache_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]
        cached_file = self.cache_dir / f"{raw_path.stem}_{cache_hash}.jpg"

        if cached_file.exists() and cached_file.stat().st_size > 10000:
            w, h = self._probe_dimensions(cached_file)
            return HighResResponse(
                gen_id=gen_id,
                resolved_path=cached_file,
                format="jpg",
                width=w,
                height=h,
                tier="tier2_embedded_raw",
            )

        # Extract direct binary chunk using range I/O
        jpeg_bytes = _extract_raw_tiff_direct(raw_path)
        if jpeg_bytes is None or len(jpeg_bytes) < 10000:
            return None

        # Check generation cancellation before write
        if gen_id < self._active_generation:
            return None

        tmp_cached = self.cache_dir / f"{cached_file.name}.tmp.{os.getpid()}"
        try:
            with open(tmp_cached, "wb") as f:
                f.write(jpeg_bytes)
            tmp_cached.replace(cached_file)
        except Exception as e:
            log.warning("Failed writing highres cache for %s: %s", raw_path, e)
            if tmp_cached.exists():
                tmp_cached.unlink(missing_ok=True)
            return None

        w, h = self._probe_dimensions(cached_file)
        return HighResResponse(
            gen_id=gen_id,
            resolved_path=cached_file,
            format="jpg",
            width=w,
            height=h,
            tier="tier2_embedded_raw",
        )

    def _resolve_tier3_fallback(self, raw_path: Path, gen_id: int) -> Optional[HighResResponse]:
        """Tier 3: Fallback when no high-res preview is embedded."""
        img = load_image_rgb(raw_path, scale_width=0)
        if img is None:
            return None

        h, w = img.shape[:2]
        cached_file = self.cache_dir / f"{raw_path.stem}_fallback.jpg"
        pil_img = Image.fromarray(img)
        pil_img.save(cached_file, format="JPEG", quality=92)

        return HighResResponse(
            gen_id=gen_id,
            resolved_path=cached_file,
            format="jpg",
            width=w,
            height=h,
            tier="tier3_fallback",
        )

    def _probe_dimensions(self, path: Path) -> Tuple[int, int]:
        """Probe width and height without decoding all pixels."""
        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg", ".png"):
            try:
                with Image.open(path) as img:
                    return img.size
            except Exception:
                pass
        elif ext in HEIF_EXTS:
            try:
                import av
                with av.open(str(path)) as container:
                    for s in container.streams.video:
                        if s.codec_context.name == "hevc":
                            return s.codec_context.width, s.codec_context.height
            except Exception:
                pass

        try:
            with Image.open(path) as img:
                return img.size
        except Exception:
            return (0, 0)
