"""High-resolution data provider for intelligent on-demand zoom.

Implements a 3-tier passthrough and extraction pipeline:
- Tier 1: Zero-transcode, zero-write passthrough for native JPG, JPEG, HIF, HEIC.
  HEIF is deliberately EXCLUDED: a camera HEIF's primary image is the full-res
  sensor tile grid (e.g. 7008x4672) — decoding it inside the WebKit webview
  during a culling run starves the engine. HEIF instead uses the embedded
  1664x1088 preview stream, software-decoded once and cached as JPEG.
- Tier 2: Zero-recode binary stream Range I/O extraction for RAW files with embedded JPEGs (ARW/NEF/CR3).
- Tier 3: Software decode fallback (when no full-res preview is embedded).

The cache directory MUST NOT live under a dot-directory (e.g. ~/.cache): the
Tauri asset-protocol scope on unix defaults to require_literal_leading_dot,
so "**" never matches dotfile paths and every asset:// request would 403
(WebKit surfaces this as EncodingError: Loading error).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from cull.loader import (
    COOKED_EXTS,
    HEIF_EXTS,
    RAW_EXTS,
    _extract_raw_tiff_direct,
    load_image_ffmpeg,
    load_image_rgb,
)

log = logging.getLogger(__name__)

# Serializes cache writes for the SAME target file (unique tmp names make
# cross-file writes safe; same-file concurrent extraction must not interleave).
_CACHE_WRITE_LOCK = threading.Lock()


def _default_cache_dir() -> Path:
    """OS-appropriate, NON-dotfile cache directory (asset-protocol friendly)."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "AutoCulling" / "highres"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "AutoCulling" / "highres"
    return Path.home() / ".cache" / "auto_culling" / "highres"


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
            self.cache_dir = _default_cache_dir()
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

        # Tier 1: Native cooked-image passthrough (0 ms, 0 transcode, 0 disk
        # write). HEIF is handled further down — its primary image is the
        # full-res sensor tile grid and must not be decoded inside the webview.
        if ext in COOKED_EXTS and ext not in HEIF_EXTS:
            w, h = self._probe_dimensions(p)
            return HighResResponse(
                gen_id=req.gen_id,
                resolved_path=p,
                format=ext.lstrip("."),
                width=w,
                height=h,
                tier="tier1_passthrough",
            )

        # Tier 2: RAW files with embedded full-resolution preview streams
        if ext in RAW_EXTS:
            res = self._resolve_raw_embedded(p, req.gen_id)
            if res is not None:
                return res

        # HEIF: decode the embedded preview stream (software HEVC, ~1664x1088)
        # once and cache it as JPEG. Zooming to 250% on a 640 px preview needs
        # ~1600 px, so the preview stream is exactly the right resolution for
        # focus verification — without dragging the 32 MP primary image
        # (100+ MB texture) through the webview mid-culling.
        if ext in HEIF_EXTS:
            res = self._resolve_heif_preview(p, req.gen_id)
            if res is not None:
                return res

        # Tier 3: Software decode fallback
        return self._resolve_tier3_fallback(p, req.gen_id)

    def _cache_target(self, src: Path, tag: str) -> Tuple[Path, str]:
        """Deterministic cache path + hash for a source file (path+size+mtime)."""
        file_stat = src.stat()
        key_str = f"{src.resolve()}_{file_stat.st_size}_{file_stat.st_mtime_ns}"
        cache_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{src.stem}_{tag}_{cache_hash}.jpg", cache_hash

    def _resolve_raw_embedded(self, raw_path: Path, gen_id: int) -> Optional[HighResResponse]:
        """Tier 2: Range I/O direct binary extraction without decompressing or recoding."""
        cached_file, _ = self._cache_target(raw_path, "hr")

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

        if not self._atomic_write(cached_file, jpeg_bytes):
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

    def _resolve_heif_preview(self, heif_path: Path, gen_id: int) -> Optional[HighResResponse]:
        """HEIF: software-decode the embedded preview stream, cache as JPEG.

        Software HEVC only (hwaccel=False): VideoToolbox sessions from preview
        generation must never contend with the culling engine's decode pool.
        """
        cached_file, _ = self._cache_target(heif_path, "heif")

        if cached_file.exists() and cached_file.stat().st_size > 10000:
            w, h = self._probe_dimensions(cached_file)
            return HighResResponse(
                gen_id=gen_id,
                resolved_path=cached_file,
                format="jpg",
                width=w,
                height=h,
                tier="tier2_heif_preview",
            )

        # Decode the 1664x1088 preview stream at full stream resolution.
        img = load_image_ffmpeg(heif_path, scale_width=0, hwaccel=False)
        if img is None:
            return None

        # Check generation cancellation before write
        if gen_id < self._active_generation:
            return None

        import io as _io
        buf = _io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=92)
        if not self._atomic_write(cached_file, buf.getvalue()):
            return None

        h, w = img.shape[:2]
        return HighResResponse(
            gen_id=gen_id,
            resolved_path=cached_file,
            format="jpg",
            width=w,
            height=h,
            tier="tier2_heif_preview",
        )

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> bool:
        """Write bytes via a UNIQUE tmp file + atomic rename.

        The tmp name must be unique per call (PID alone collides across
        concurrent worker threads writing the same target, corrupting the
        JPEG). The write lock additionally serializes same-target races.
        """
        with _CACHE_WRITE_LOCK:
            tmp_cached = target.with_name(f"{target.name}.tmp.{uuid.uuid4().hex}")
            try:
                with open(tmp_cached, "wb") as f:
                    f.write(data)
                tmp_cached.replace(target)
                return True
            except Exception as e:
                log.warning("Failed writing highres cache %s: %s", target, e)
                try:
                    tmp_cached.unlink(missing_ok=True)
                except Exception:
                    pass
                return False

    def _resolve_tier3_fallback(self, raw_path: Path, gen_id: int) -> Optional[HighResResponse]:
        """Tier 3: Fallback when no high-res preview is embedded (software decode)."""
        cached_file, _ = self._cache_target(raw_path, "fb")

        if cached_file.exists() and cached_file.stat().st_size > 10000:
            w, h = self._probe_dimensions(cached_file)
            return HighResResponse(
                gen_id=gen_id,
                resolved_path=cached_file,
                format="jpg",
                width=w,
                height=h,
                tier="tier3_fallback",
            )

        img = load_image_rgb(raw_path, scale_width=0, hwaccel=False)
        if img is None:
            return None

        if gen_id < self._active_generation:
            return None

        h, w = img.shape[:2]
        import io as _io
        buf = _io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=92)
        if not self._atomic_write(cached_file, buf.getvalue()):
            return None

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
