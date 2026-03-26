"""
loader.py — Image loading utilities for F1 photo culling.
Pillow/FFmpeg VERSION — NO OpenCV.
Supports HIF (via FFmpeg), RAW (via ExifTool), and standard formats.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

EXTENSIONS = {".hif", ".heif", ".heic", ".nef", ".arw", ".cr2", ".cr3",
              ".orf", ".rw2", ".raf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}

RAW_EXTS = {".arw", ".nef", ".cr2", ".cr3", ".orf", ".rw2", ".raf", ".dng"}
COOKED_EXTS = {".jpg", ".jpeg", ".hif", ".heif", ".heic", ".png", ".tiff", ".tif"}

def probe_embedded_preview(path: Path, min_width: int = 800) -> Tuple[int, int, int] | None:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", 
             "stream=index,width,height,codec_name:stream_disposition=dependent", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=10
        )
        if proc.returncode != 0: return None
        import json
        data = json.loads(proc.stdout)
        best: Tuple[int, int, int] | None = None
        best_w = 0
        for s in data.get("streams", []):
            idx, w, h, codec = s.get("index", -1), s.get("width", 0), s.get("height", 0), s.get("codec_name", "")
            if s.get("disposition", {}).get("dependent", 0) == 1 or codec != "hevc": continue
            if w < min_width: continue
            if w > best_w and w < 5000:
                best = (idx, int(w), int(h))
                best_w = int(w)
        return best
    except Exception: return None

def probe_full_dimensions(path: Path) -> Tuple[int, int] | None:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", 
             "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
            capture_output=True, text=True, timeout=10
        )
        parts = proc.stdout.strip().split("\n")[0].split("x")
        if len(parts) == 2: return int(parts[0]), int(parts[1])
    except Exception: pass
    return None

_preview_stream_cache: dict[Path, Tuple[int, int, int] | None] = {}

def get_preview_stream(path: Path) -> Tuple[int, int, int] | None:
    cache_key = path.parent
    if cache_key not in _preview_stream_cache:
        _preview_stream_cache[cache_key] = probe_embedded_preview(path)
    return _preview_stream_cache[cache_key]

def load_image_ffmpeg(path: Path, scale_width: int = 1280) -> np.ndarray | None:
    preview = get_preview_stream(path)
    if preview is not None:
        idx, w, h = preview
        try:
            cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-hwaccel", "auto", "-i", str(path), "-map", f"0:{idx}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-frames:v", "1", "-y", "pipe:1"]
            proc = subprocess.run(cmd, capture_output=True, timeout=30)
            if proc.returncode == 0 and len(proc.stdout) == w * h * 3:
                img = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(h, w, 3)
                if scale_width > 0 and w > scale_width * 1.2:
                    new_h = int(round(h * scale_width / w))
                    # Use Pillow for resizing
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((scale_width, new_h), Image.BILINEAR)
                    return np.array(pil_img)
                return img
        except Exception: pass
    return None

def load_image_rgb(path: Path, scale_width: int = 0) -> np.ndarray | None:
    suffix = path.suffix.lower()
    if suffix in (".hif", ".heif", ".heic"):
        img = load_image_ffmpeg(path, scale_width=scale_width)
        if img is not None: return img
        # Pillow Fallback
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            pil_img = Image.open(path).convert("RGB")
            if scale_width > 0:
                w, h = pil_img.size
                new_h = int(round(h * scale_width / w))
                pil_img = pil_img.resize((scale_width, new_h), Image.BILINEAR)
            return np.array(pil_img)
        except Exception as e:
            log.warning(f"pillow-heif failed for {path.name}: {e}")

    # JPEG / PNG / TIFF via Pillow
    try:
        pil_img = Image.open(path).convert("RGB")
        if scale_width > 0:
            w, h = pil_img.size
            new_h = int(round(h * scale_width / w))
            pil_img = pil_img.resize((scale_width, new_h), Image.BILINEAR)
        return np.array(pil_img)
    except Exception:
        pass

    # RAW Fallback via ExifTool
    if suffix in RAW_EXTS:
        for tag in ["-JpgFromRaw", "-PreviewImage"]:
            try:
                proc = subprocess.run(["exiftool", "-b", tag, str(path)], capture_output=True, timeout=10)
                if proc.returncode == 0 and len(proc.stdout) > 0:
                    import io
                    pil_img = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
                    if scale_width > 0:
                        w, h = pil_img.size
                        new_h = int(round(h * scale_width / w))
                        pil_img = pil_img.resize((scale_width, new_h), Image.BILINEAR)
                    return np.array(pil_img)
            except Exception: continue
    return None

def update_image_metadata(img_path: Path, rating: int, crop: tuple[float, float, float, float] | None = None) -> tuple[bool, str]:
    et_rating, pick_flag = max(0, rating), (1 if rating > 0 else -1)
    cmd = ["exiftool", "-overwrite_original", f"-XMP-xmp:Rating={et_rating}", f"-XMP-xmpDM:Pick={pick_flag}"]
    if crop:
        t, l, b, r = crop
        cmd.extend(["-XMP-crs:HasCrop=True", "-XMP-crs:AlreadyApplied=False", 
                    f"-XMP-crs:CropTop={t:.6f}", f"-XMP-crs:CropLeft={l:.6f}", 
                    f"-XMP-crs:CropBottom={b:.6f}", f"-XMP-crs:CropRight={r:.6f}",
                    "-XMP-crs:CropAngle=0", "-XMP-crs:CropConstrainToWarp=0", "-XMP-crs:CropConstrainToUnitSquare=1"])
    cmd.append(str(img_path))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True, img_path.name
    except subprocess.CalledProcessError as e:
        return False, f"Error updating {img_path.name}: {e.stderr.decode().strip()}"
