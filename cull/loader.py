"""
loader.py — Image loading utilities for F1 photo culling.
Pillow/FFmpeg VERSION — NO OpenCV.
Supports HIF (via FFmpeg), RAW (via ExifTool), and standard formats.
"""

from __future__ import annotations

import logging
import io
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
from PIL import Image

# Decode workers (spawned by the engine's ProcessPoolExecutor) run CPU-heavy
# JPEG/RAW decode; demote them below the main consumer thread so scoring
# latency isn't starved on the shared 8 physical cores.
try:
    import multiprocessing as _mp
    if _mp.current_process().name != "MainProcess":
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

log = logging.getLogger(__name__)

EXTENSIONS = {".hif", ".heif", ".heic", ".nef", ".arw", ".cr2", ".cr3",
              ".orf", ".rw2", ".raf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}

RAW_EXTS = {".arw", ".nef", ".cr2", ".cr3", ".orf", ".rw2", ".raf", ".dng"}
COOKED_EXTS = {".jpg", ".jpeg", ".hif", ".heif", ".heic", ".png", ".tiff", ".tif"}

def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).parent.parent.resolve()
    return base_path / relative_path

def _find_exiftool_path() -> list[str]:
    """Return command list for exiftool (bundled or system-wide)."""
    # 1. Check for bundled Perl script + Bundled Perl Interpreter (Self-contained)
    ext = ".exe" if sys.platform == "win32" else ""
    bundled_perl = get_resource_path(f"external/exiftool/perl{ext}")
    bundled_pl = get_resource_path("external/exiftool/exiftool.pl")
    lib_path = get_resource_path("external/exiftool/lib")

    if bundled_perl.exists() and bundled_pl.exists() and lib_path.exists():
        return [str(bundled_perl), "-I", str(lib_path), str(bundled_pl)]

    # 2. Check for bundled binary/launcher
    bundled_bin = get_resource_path(f"external/exiftool/exiftool{ext}")
    if bundled_bin.exists():
        if lib_path.exists():
            return ["perl", "-I", str(lib_path), str(bundled_bin)]
        return [str(bundled_bin)]

    # 3. Fallback to system-wide
    return ["exiftool"]

def _find_ffmpeg_path() -> str:
    """Return path to bundled ffmpeg if exists, otherwise assume system-wide."""
    ext = ".exe" if sys.platform == "win32" else ""
    # 1. Bundled (if we decided to bundle, but we won't for size)
    bundled = get_resource_path(f"external/ffmpeg/ffmpeg{ext}")
    if bundled.exists(): return str(bundled)
    # 2. Known local path on this machine
    for cand in [Path(r"D:\ProgramData\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe")]:
        if cand.exists(): return str(cand)
    # 3. System-wide
    return "ffmpeg"

def _find_ffprobe_path() -> str:
    """Return path to bundled ffprobe if exists, otherwise assume system-wide."""
    ext = ".exe" if sys.platform == "win32" else ""
    # 1. Bundled
    bundled = get_resource_path(f"external/ffmpeg/ffprobe{ext}")
    if bundled.exists(): return str(bundled)
    # 2. Known local path
    for cand in [Path(r"D:\ProgramData\ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe")]:
        if cand.exists(): return str(cand)
    # 3. System-wide
    return "ffprobe"

def probe_embedded_preview(path: Path, min_width: int = 800) -> Tuple[int, int, int] | None:
    try:
        ffprobe_bin = _find_ffprobe_path()
        proc = subprocess.run(
            [ffprobe_bin, "-v", "error", "-select_streams", "v", "-show_entries", 
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
        ffprobe_bin = _find_ffprobe_path()
        proc = subprocess.run(
            [ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-show_entries", 
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
            ffmpeg_bin = _find_ffmpeg_path()
            # Use hardware acceleration if available (cuda for NVIDIA, d3d11va for generic Windows)
            hw_accel = "cuda" if sys.platform == "win32" else "auto"
            cmd = [
                ffmpeg_bin, "-hide_banner", "-v", "error", 
                "-hwaccel", hw_accel, 
                "-i", str(path), "-map", f"0:{idx}", 
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-frames:v", "1", "-y", "pipe:1"
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=30)
            if proc.returncode == 0 and len(proc.stdout) == w * h * 3:
                img = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(h, w, 3)
                if scale_width > 0 and w > scale_width * 1.2:
                    new_h = int(round(h * scale_width / w))
                    return cv2.resize(img, (scale_width, new_h), interpolation=cv2.INTER_AREA)
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
            img_arr = np.array(pil_img)
            if scale_width > 0:
                h, w = img_arr.shape[:2]
                new_h = int(round(h * scale_width / w))
                return cv2.resize(img_arr, (scale_width, new_h), interpolation=cv2.INTER_AREA)
            return img_arr
        except Exception as e:
            log.warning(f"pillow-heif failed for {path.name}: {e}")

    # JPEG / PNG / TIFF via Pillow + cv2.INTER_AREA.
    # NOTE: libjpeg DCT draft decode was A/B'd (#13) and REVERTED: draft
    # pixels flip the P4 integrity classifier on 2/6 real JPGs (raw drops
    # 0.5 via cut penalty -> min_raw veto -> keep->reject). Pixel changes
    # are therefore off-limits; speed comes from zero-copy + parallelism.
    try:
        pil_img = Image.open(path).convert("RGB")
        # Zero-copy view of the contiguous RGB buffer (np.array would copy
        # ~72 MB per 24MP frame; asarray is pixel-identical).
        img_arr = np.asarray(pil_img)
        if scale_width > 0:
            h, w = img_arr.shape[:2]
            new_h = int(round(h * scale_width / w))
            return cv2.resize(img_arr, (scale_width, new_h), interpolation=cv2.INTER_AREA)
        return img_arr
    except Exception:
        pass

    # RAW Fallback via ExifTool (embedded preview, persistent session)
    if suffix in RAW_EXTS:
        try:
            data = _extract_embedded_raw(path, ["-JpgFromRaw", "-PreviewImage"])
            if data:
                pil_img = Image.open(io.BytesIO(data)).convert("RGB")
                img_arr = np.array(pil_img)
                if scale_width > 0:
                    h, w = img_arr.shape[:2]
                    new_h = int(round(h * scale_width / w))
                    return cv2.resize(img_arr, (scale_width, new_h), interpolation=cv2.INTER_AREA)
                return img_arr
        except Exception:
            pass
    return None

# ---------------------------------------------------------------------------
# RAW embedded-preview extraction via a persistent per-process exiftool
# session (-stay_open). A per-file exiftool spawn costs ~460 ms (Perl
# interpreter startup); the persistent session amortizes it to ~60-100 ms.
# The extraction bytes are identical to the per-file spawn (same
# ``-b -JpgFromRaw``), so decoded pixels and downstream scores are unchanged.
# Each -execute handles exactly ONE file, giving clean binary framing via the
# stderr "{ready}" marker (a batch of N files per -execute would concatenate
# payloads without separators — that path was A/B'd and dropped in #4).
# ---------------------------------------------------------------------------

class _PersistentExiftool:
    """One persistent exiftool ``-stay_open`` session for RAW embedded-preview
    extraction (``-b -w <tmp>/%f.jpg``).

    A per-file ``exiftool -b <tag>`` spawn costs ~460 ms (Perl interpreter
    startup); the persistent session amortizes that. Each -execute handles
    exactly ONE file and exiftool writes the binary payload to a temp file
    (``-w``), so framing is filesystem-based — the extraction bytes are
    identical to the per-file spawn (verified — the precision gates lock the
    decoded pixels). ``{ready}`` arrives as a text line on STDOUT for this
    launcher, which the caller reads line-by-line. Order-preserving and
    single-threaded per process."""

    def __init__(self) -> None:
        import tempfile
        import threading
        exiftool_cmd = _find_exiftool_path()
        kwargs = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.proc = subprocess.Popen([*exiftool_cmd, "-stay_open", "True", "-@", "-"], **kwargs)
        self._outdir = Path(tempfile.mkdtemp(prefix="raw_extract_"))
        self._ready = threading.Event()
        self._dead = False
        threading.Thread(target=self._read_stdout, daemon=True).start()

    def _read_stdout(self) -> None:
        try:
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    self._dead = True
                    self._ready.set()
                    return
                if line.strip() == b"{ready}":
                    self._ready.set()
        except Exception:
            self._dead = True
            self._ready.set()

    def extract(self, path: Path, tag: str) -> bytes | None:
        if self._dead:
            return None
        out = self._outdir / f"{path.stem}.jpg__"
        cmd = ["-b", "-w", f"{self._outdir.as_posix()}/%f.jpg__", tag, str(path)]
        try:
            self.proc.stdin.write(("\n".join(cmd) + "\n-execute\n").encode("utf-8", "replace"))
            self.proc.stdin.flush()
        except Exception:
            self._dead = True
            return None
        self._ready.clear()
        import time
        self._ready.wait(timeout=60.0)
        if self._dead:
            return None
        try:
            return out.read_bytes() if out.exists() else None
        finally:
            try: out.unlink(missing_ok=True)
            except Exception: pass

    def close(self) -> None:
        import shutil
        if not self._dead:
            try:
                self.proc.stdin.write(b"-stay_open\nFalse\n-execute\n")
                self.proc.stdin.flush()
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
        shutil.rmtree(self._outdir, ignore_errors=True)

_raw_session: _PersistentExiftool | None = None

def _get_raw_session() -> _PersistentExiftool | None:
    global _raw_session
    if _raw_session is None:
        try:
            _raw_session = _PersistentExiftool()
            import atexit
            atexit.register(_raw_session.close)
        except Exception:
            _raw_session = None  # fall back to per-file spawns
    return _raw_session


def _extract_embedded_raw(path: Path, tags: list[str]) -> bytes | None:
    """Return embedded preview bytes (e.g. JpgFromRaw/PreviewImage), using the
    persistent session when available, else the per-file spawn fallback."""
    session = _get_raw_session()
    if session is not None:
        for tag in tags:
            data = session.extract(path, tag)
            if data:
                return data
        return None
    exiftool_cmd = _find_exiftool_path()
    for tag in tags:
        try:
            proc = subprocess.run([*exiftool_cmd, "-b", tag, str(path)],
                                  capture_output=True, timeout=10)
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout
        except Exception:
            continue
    return None


def update_image_metadata(img_path: Path, rating: int, crop: tuple[float, float, float, float] | None = None) -> tuple[bool, str]:
    et_rating, pick_flag = max(0, rating), (1 if rating > 0 else -1)
    exiftool_cmd = _find_exiftool_path()
    cmd = [*exiftool_cmd, "-overwrite_original", f"-XMP-xmp:Rating={et_rating}", f"-XMP-xmpDM:Pick={pick_flag}"]
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


def update_image_metadata_batch(
    items: list[tuple[Path, int, tuple[float, float, float, float] | None]],
) -> list[tuple[bool, str]]:
    """Write rating/pick/crop to many files through ONE persistent exiftool
    ``-stay_open`` session (text protocol), replacing per-file subprocess
    spawns (~399 ms/file → single process, ~10 ms/file). Order-preserving.
    """
    results: list[tuple[bool, str]] = []
    if not items:
        return results
    import subprocess as _s

    exiftool_cmd = _find_exiftool_path()
    kwargs = {"stdin": _s.PIPE, "stdout": _s.PIPE, "stderr": _s.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = _s.CREATE_NO_WINDOW
    proc = _s.Popen([*exiftool_cmd, "-stay_open", "True", "-@", "-"],
                    universal_newlines=False, **kwargs)
    try:
        for path, rating, crop in items:
            args = [
                "-overwrite_original",
                f"-XMP-xmp:Rating={max(0, rating)}",
                f"-XMP-xmpDM:Pick={1 if rating > 0 else -1}",
            ]
            if crop:
                t, l, b, r = crop
                args += [
                    "-XMP-crs:HasCrop=True", "-XMP-crs:AlreadyApplied=False",
                    f"-XMP-crs:CropTop={t:.6f}", f"-XMP-crs:CropLeft={l:.6f}",
                    f"-XMP-crs:CropBottom={b:.6f}", f"-XMP-crs:CropRight={r:.6f}",
                    "-XMP-crs:CropAngle=0", "-XMP-crs:CropConstrainToWarp=0",
                    "-XMP-crs:CropConstrainToUnitSquare=1",
                ]
            args.append(str(path))
            proc.stdin.write(("\n".join(args) + "\n-execute\n").encode("utf-8", "replace"))
            proc.stdin.flush()
            ok, msg = _wait_stay_open(proc, path.name)
            results.append((ok, msg))
        proc.stdin.write(b"-stay_open\nFalse\n-execute\n")
        proc.stdin.flush()
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
    return results


def _wait_stay_open(proc, name: str) -> tuple[bool, str]:
    """Read exiftool stdout until the current -execute completes."""
    while True:
        line = proc.stdout.readline()
        if not line:
            return False, f"exiftool closed early for {name}"
        text = line.decode("utf-8", "replace").strip()
        if "image files updated" in text or "files updated" in text:
            return True, name
        if text.lower().startswith("error"):
            return False, f"{name}: {text}"
