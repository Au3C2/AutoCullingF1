"""
loader.py — Image loading utilities for F1 photo culling.
Supports HIF (via FFmpeg), RAW (via ExifTool), and standard formats.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
EXTENSIONS = {".hif", ".heif", ".heic", ".nef", ".arw", ".cr2", ".cr3",
              ".orf", ".rw2", ".raf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}

RAW_EXTS = {".arw", ".nef", ".cr2", ".cr3", ".orf", ".rw2", ".raf", ".dng"}
COOKED_EXTS = {".jpg", ".jpeg", ".hif", ".heif", ".heic", ".png", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# HIF Preview Stream Probing
# ---------------------------------------------------------------------------

def probe_embedded_preview(path: Path, min_width: int = 800) -> Tuple[int, int, int] | None:
    """Find an embedded preview stream suitable for fast decode.
    Returns (stream_index, width, height).
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v",
                "-show_entries", "stream=index,width,height,codec_name:stream_disposition=dependent",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None

        import json
        data = json.loads(proc.stdout)
        best: Tuple[int, int, int] | None = None
        best_w = 0
        for s in data.get("streams", []):
            idx = s.get("index", -1)
            w = s.get("width", 0)
            h = s.get("height", 0)
            codec = s.get("codec_name", "")
            dep = s.get("disposition", {}).get("dependent", 0)

            if dep == 1 or codec != "hevc":
                continue
            if w < min_width:
                continue
            if w > best_w and w < 5000:
                best = (idx, w, h)
                best_w = w

        return best
    except Exception:
        return None


def probe_full_dimensions(path: Path) -> Tuple[int, int] | None:
    """Return (width, height) of the primary (full-res) image via ffprobe."""
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0:s=x",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_line = proc.stdout.strip().split("\n")[0].strip()
        parts = first_line.split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None


# Cache the probed preview stream index for the first file in a directory.
_preview_stream_cache: dict[Path, Tuple[int, int, int] | None] = {}


def get_preview_stream(path: Path) -> Tuple[int, int, int] | None:
    """Return (stream_index, width, height) for the preview stream, with caching."""
    cache_key = path.parent
    if cache_key not in _preview_stream_cache:
        _preview_stream_cache[cache_key] = probe_embedded_preview(path)
        info = _preview_stream_cache[cache_key]
        if info:
            log.info("HIF preview stream: #%d (%dx%d) in %s",
                     info[0], info[1], info[2], cache_key.name)
        else:
            log.info("No suitable HIF preview stream found in %s", cache_key.name)
    return _preview_stream_cache[cache_key]


# ---------------------------------------------------------------------------
# Image Decoders
# ---------------------------------------------------------------------------

def load_image_ffmpeg(
    path: Path,
    scale_width: int = 1280,
) -> np.ndarray | None:
    """Decode an HIF image via ffmpeg, using the fastest available strategy."""
    preview = get_preview_stream(path)
    if preview is not None:
        s_idx, s_w, s_h = preview
        expected_bytes = s_w * s_h * 3
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-v", "error",
                    "-hwaccel", "auto",
                    "-i", str(path),
                    "-map", f"0:{s_idx}",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-frames:v", "1",
                    "-y", "pipe:1",
                ],
                capture_output=True,
                timeout=30,
            )
            if proc.returncode == 0 and len(proc.stdout) == expected_bytes:
                img = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(s_h, s_w, 3)
                if scale_width > 0 and s_w > scale_width * 1.2:
                    new_h = int(round(s_h * scale_width / s_w))
                    img = cv2.resize(img, (scale_width, new_h),
                                     interpolation=cv2.INTER_AREA)
                return img
        except Exception:
            pass

    # Generic decode fallback
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-v", "error",
                "-hwaccel", "auto",
                "-i", str(path),
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-frames:v", "1",
                "-y", "pipe:1",
            ],
            capture_output=True,
            timeout=60,
        )
    except Exception as exc:
        log.warning("ffmpeg full decode failed for %s: %s", path.name, exc)
        return None

    if proc.returncode != 0 or len(proc.stdout) == 0:
        return None

    dims = probe_full_dimensions(path)
    if dims is None:
        return None
    full_w, full_h = dims
    expected_bytes = full_w * full_h * 3
    if len(proc.stdout) != expected_bytes:
        log.warning("ffmpeg full-res size mismatch for %s: expected %d, got %d",
                     path.name, expected_bytes, len(proc.stdout))
        return None

    img = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(full_h, full_w, 3)
    if scale_width > 0:
        new_h = int(round(full_h * scale_width / full_w))
        img = cv2.resize(img, (scale_width, new_h), interpolation=cv2.INTER_AREA)
    return img


def load_image_rgb(
    path: Path,
    scale_width: int = 0,
) -> np.ndarray | None:
    """Load an image as an RGB numpy array.
    Supports HIF, RAW, and standard formats.
    """
    suffix = path.suffix.lower()

    if suffix in (".hif", ".heif", ".heic"):
        if scale_width > 0:
            img = load_image_ffmpeg(path, scale_width=scale_width)
            if img is not None:
                return img
        
        # Fallback: pillow-heif
        try:
            import pillow_heif
            from PIL import Image
            pillow_heif.register_heif_opener()
            pil_img = Image.open(path).convert("RGB")
            arr = np.array(pil_img, dtype=np.uint8)
            if scale_width > 0:
                h, w = arr.shape[:2]
                new_w = scale_width
                new_h = int(round(h * new_w / w))
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return arr
        except Exception as exc:
            log.warning("pillow-heif failed for %s: %s", path.name, exc)

    # Standard / RAW Fallback
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    
    if img_bgr is None and suffix in RAW_EXTS:
        for tag in ["-JpgFromRaw", "-PreviewImage"]:
            try:
                proc = subprocess.run(
                    ["exiftool", "-b", tag, str(path)],
                    capture_output=True, timeout=10
                )
                if proc.returncode == 0 and len(proc.stdout) > 0:
                    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
                    temp_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if temp_bgr is not None:
                        img_rgb = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2RGB)
                        if scale_width > 0:
                            h, w = img_rgb.shape[:2]
                            new_h = int(round(h * scale_width / w))
                            img_rgb = cv2.resize(img_rgb, (scale_width, new_h), interpolation=cv2.INTER_AREA)
                        return img_rgb
            except Exception:
                continue

    if img_bgr is None:
        return None
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if scale_width > 0:
        h, w = img_rgb.shape[:2]
        new_w = scale_width
        new_h = int(round(h * new_w / w))
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_rgb


def update_image_metadata(img_path: Path, rating: int, crop: tuple[float, float, float, float] | None = None) -> tuple[bool, str]:
    """Update a single image's Rating, Pick flag, and optional Crop using ExifTool."""
    if not img_path.exists():
        return False, f"File not found: {img_path.name}"

    # Lightroom Metadata logic:
    # Rating: 0-5
    # Pick flag (XMP-xmpDM:Pick): 1=Picked, 0=None, -1=Rejected
    et_rating = max(0, rating)
    pick_flag = 1 if rating > 0 else -1
    
    cmd = [
        "exiftool",
        "-overwrite_original",
        f"-XMP-xmp:Rating={et_rating}",
        f"-XMP-xmpDM:Pick={pick_flag}",
    ]

    # Add Crop info if present (Lightroom-compatible crs namespace)
    if crop:
        t, l, b, r = crop
        cmd.extend([
            "-XMP-crs:HasCrop=True",
            "-XMP-crs:AlreadyApplied=False",
            f"-XMP-crs:CropTop={t:.6f}",
            f"-XMP-crs:CropLeft={l:.6f}",
            f"-XMP-crs:CropBottom={b:.6f}",
            f"-XMP-crs:CropRight={r:.6f}",
            "-XMP-crs:CropAngle=0",
            "-XMP-crs:CropConstrainToWarp=0",
            "-XMP-crs:CropConstrainToUnitSquare=1",
        ])

    cmd.append(str(img_path))
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True, img_path.name
    except subprocess.CalledProcessError as e:
        return False, f"Error updating {img_path.name}: {e.stderr.decode().strip()}"

