"""
cull_photos.py — Rule-based F1 photo culling pipeline.

For each image in the input directory:
  1. Read EXIF metadata and group into burst sequences.
  2. Detect subjects via cascade: F1 YOLO model → COCO YOLOv8n fallback.
  3. Score sharpness (Laplacian variance inside the primary detection bbox).
  4. Score composition (fill, rule-of-thirds, lead-room).
  5. Apply veto rules (no detection / too blurry → Rating -1).
  6. Select the top-N frames per burst group; reject the rest.
  7. Write XMP sidecar files (same directory as the originals).

Usage
-----
    # Basic — process all HIF/NEF/JPG in a directory
    python cull_photos.py --input-dir /path/to/photos

    # F1 photos: scan HIF/ subdirectory, compare with ARW ground truth
    python cull_photos.py \\
        --input-dir "/path/to/2025-03-21 上午 练习赛/HIF" \\
        --label-check \\
        --rf-api-key NQI4igULp4JqFsyY2L2t \\
        --dry-run

    # Recursive scan — process all images in subdirectories too
    python cull_photos.py --input-dir /path/to/session --recursive

    # With F1 model (download first with models/download_f1_model.py)
    python cull_photos.py \\
        --input-dir /path/to/photos \\
        --f1-model models/f1_yolov8n.onnx

    # Cloud-API fallback for F1 detection (no local ONNX needed)
    python cull_photos.py \\
        --input-dir /path/to/photos \\
        --rf-api-key NQI4igULp4JqFsyY2L2t

    # Dry run — show what would be written without creating any .xmp files
    python cull_photos.py --input-dir /path/to/photos --dry-run

    # Tune parameters
    python cull_photos.py \\
        --input-dir /path/to/photos \\
        --top-n 5 \\
        --sharp-thresh 0.10 \\
        --conf 0.30
"""

from __future__ import annotations

import argparse
import csv as _csv
import json as _json
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from cull.renamer import rename_images

import cv2
import numpy as np

from cull.exif_reader import ExifData, BurstGroup, read_exif, group_bursts
from cull.detector import Detection, load_f1_model, load_coco_model, detect
from cull.sharpness import score_sharpness
from cull.composition import score_composition
from cull.scorer import ImageScore, score_image, select_best_n, SHARP_THRESH, W_SHARP, W_COMP, MIN_RAW
from cull.xmp_writer import write_xmp_batch
from cull.xmp_reader import read_xmp_rating

log = logging.getLogger(__name__)

def setup_logging(base_dir: Path):
    """Setup logging to both console and a file in the base_dir/logs folder."""
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cull_{timestamp}.log"
    
    # Reset existing handlers if any
    root_log = logging.getLogger()
    for handler in root_log.handlers[:]:
        root_log.removeHandler(handler)
        
    root_log.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    root_log.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_log.addHandler(console_handler)
    
    log.info("Logging to %s", log_file)
    return log_file

# ---------------------------------------------------------------------------
# Supported image extensions (case-insensitive comparison used at runtime)
# ---------------------------------------------------------------------------
_EXTENSIONS = {".hif", ".heif", ".heic", ".nef", ".arw", ".cr2", ".cr3",
               ".orf", ".rw2", ".raf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}

_RAW_EXTS = {".arw", ".nef", ".cr2", ".cr3", ".orf", ".rw2", ".raf", ".dng"}
_COOKED_EXTS = {".jpg", ".jpeg", ".hif", ".heif", ".heic", ".png", ".tiff", ".tif"}


def _collect_images(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Scan *input_dir* for supported image files, sorted by name.
    Uses os.scandir for high performance on Windows (prevents redundant stat() calls).
    """
    import os
    found: list[Path] = []
    
    log.info("Collecting files from %s...", input_dir)
    
    def _scan(target: str):
        try:
            with os.scandir(target) as it:
                for entry in it:
                    if entry.is_file():
                        # Filter out macOS metadata files (e.g., ._IMG_1234.JPG)
                        if entry.name.startswith("._"):
                            continue
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in _EXTENSIONS:
                            found.append(Path(entry.path))
                    elif recursive and entry.is_dir():
                        # Skip hidden directories (e.g., .git, .thumbnails)
                        if entry.name.startswith("."):
                            continue
                        _scan(entry.path)
        except PermissionError:
            pass

    _scan(str(input_dir))
    return sorted(found)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _probe_embedded_preview(path: Path, min_width: int = 800) -> tuple[int, int, int] | None:
    """Find an embedded preview stream suitable for fast decode.

    Sony HIF files contain a Tile Grid (streams 0-5 that are stitched into
    the full 7008x4672 image) plus several standalone streams at lower
    resolutions.  The Tile Grid uses an internal complex filtergraph so we
    cannot apply ``-vf scale`` on top of it.

    This function probes the file and returns ``(stream_index, width, height)``
    for the best standalone HEVC preview stream whose width >= *min_width*
    and whose ``disposition.dependent`` flag is 0 (i.e. not a tile member).

    Returns ``None`` if no suitable stream is found.
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
        best: tuple[int, int, int] | None = None
        best_w = 0
        for s in data.get("streams", []):
            idx = s.get("index", -1)
            w = s.get("width", 0)
            h = s.get("height", 0)
            codec = s.get("codec_name", "")
            dep = s.get("disposition", {}).get("dependent", 0)

            # Skip tile members (dependent=1) and non-HEVC streams
            if dep == 1 or codec != "hevc":
                continue
            # Must be at least min_width
            if w < min_width:
                continue
            # Pick the largest that is still smaller than the full grid
            # (full grid is typically 7008 wide for A7C2)
            if w > best_w and w < 5000:
                best = (idx, w, h)
                best_w = w

        return best
    except Exception:
        return None


def _probe_full_dimensions(path: Path) -> tuple[int, int] | None:
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
        # May return multiple lines for Tile Grid files; take the first
        first_line = proc.stdout.strip().split("\n")[0].strip()
        parts = first_line.split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None


# Cache the probed preview stream index for the first file in a directory.
# All HIF files from the same camera/session share the same structure.
_preview_stream_cache: dict[Path, tuple[int, int, int] | None] = {}


def _get_preview_stream(path: Path) -> tuple[int, int, int] | None:
    """Return (stream_index, width, height) for the preview stream, with caching."""
    cache_key = path.parent
    if cache_key not in _preview_stream_cache:
        _preview_stream_cache[cache_key] = _probe_embedded_preview(path)
        info = _preview_stream_cache[cache_key]
        if info:
            log.info("HIF preview stream: #%d (%dx%d) in %s",
                     info[0], info[1], info[2], cache_key.name)
        else:
            log.info("No suitable HIF preview stream found in %s", cache_key.name)
    return _preview_stream_cache[cache_key]


def _load_image_ffmpeg(
    path: Path,
    scale_width: int = 1280,
) -> np.ndarray | None:
    """Decode an HIF image via ffmpeg, using the fastest available strategy.

    Strategy priority:
      1. Extract embedded preview stream (e.g. stream #6 at 1664x1088) — ~0.11s
      2. Full-resolution Tile Grid decode + cv2.resize               — ~0.60s

    Returns an RGB numpy array, or ``None`` on failure.
    """
    # --- Strategy 1: embedded preview stream (fastest) -----------------------
    preview = _get_preview_stream(path)
    if preview is not None:
        s_idx, s_w, s_h = preview
        expected_bytes = s_w * s_h * 3
        try:
            proc = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-v", "error",
                    "-hwaccel", "auto",  # Use available hardware acceleration
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
                # Optionally resize further if needed
                if scale_width > 0 and s_w > scale_width * 1.2:
                    new_h = int(round(s_h * scale_width / s_w))
                    img = cv2.resize(img, (scale_width, new_h),
                                     interpolation=cv2.INTER_AREA)
                return img
        except Exception:
            pass
        log.debug("Preview stream extraction failed for %s, trying full decode", path.name)

    # --- Strategy 2: full-resolution decode + resize -------------------------
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

    # Determine dimensions from the Tile Grid (default stream)
    dims = _probe_full_dimensions(path)
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


def _load_image_rgb(
    path: Path,
    scale_width: int = 0,
) -> np.ndarray | None:
    """Load an image as an RGB numpy array.

    For HIF/HEIF files, uses ffmpeg for fast decode + scale.  Falls back to
    pillow-heif, then OpenCV.

    When *scale_width* > 0, the image is scaled to that width (aspect-ratio
    preserved) **during decode** for HIF files (via ffmpeg).  For non-HIF
    files the image is loaded at full resolution and then resized.

    Returns ``None`` on failure.
    """
    suffix = path.suffix.lower()

    if suffix in (".hif", ".heif", ".heic"):
        # --- Fast path: ffmpeg decode + scale --------------------------------
        if scale_width > 0:
            img = _load_image_ffmpeg(path, scale_width=scale_width)
            if img is not None:
                return img
            log.warning(
                "ffmpeg decode failed for %s — falling back to pillow-heif",
                path.name,
            )

        # --- Fallback: pillow-heif -------------------------------------------
        try:
            import pillow_heif  # type: ignore
            from PIL import Image  # type: ignore
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
            log.warning(
                "pillow-heif failed for %s: %s — trying OpenCV",
                path.name, exc,
            )

    # --- Non-HIF: Try OpenCV -------------------------------------------------
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    
    # --- RAW Fallback: Extract embedded preview via ExifTool -----------------
    if img_bgr is None and suffix in _RAW_EXTS:
        log.debug("OpenCV failed for RAW %s — trying ExifTool preview extraction", path.name)
        # Try -JpgFromRaw then -PreviewImage
        for tag in ["-JpgFromRaw", "-PreviewImage"]:
            try:
                proc = subprocess.run(
                    ["exiftool", "-b", tag, str(path)],
                    capture_output=True, timeout=10
                )
                if proc.returncode == 0 and len(proc.stdout) > 0:
                    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
                    img_rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img_rgb is not None:
                        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                        log.debug("Successfully extracted %s from %s", tag, path.name)
                        img_bgr = img_rgb # used for return logic below
                        break
            except Exception:
                continue

    if img_bgr is None:
        log.warning("Could not load image: %s (tried FFmpeg/Pillow/OpenCV/ExifTool)", path.name)
        return None
        
    # If we got here, img_bgr is actually RGB if we used the ExifTool path, 
    # or BGR if we used imread. Let's unify.
    if suffix in _RAW_EXTS and 'img_rgb' in locals() and img_rgb is not None:
        # already RGB
        pass
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
    if scale_width > 0:
        h, w = img_rgb.shape[:2]
        new_w = scale_width
        new_h = int(round(h * new_w / w))
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_rgb


# ---------------------------------------------------------------------------
# Cloud-API F1 detection fallback (inference_sdk)
# ---------------------------------------------------------------------------


class _CloudF1Detector:
    """Thin wrapper around inference_sdk for F1 detection via Roboflow cloud."""

    _MODEL_ID = "formula-one-car-detection/1"
    _API_URL  = "https://serverless.roboflow.com"

    def __init__(self, api_key: str) -> None:
        try:
            from inference_sdk import InferenceHTTPClient  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "inference-sdk is required for cloud F1 detection.\n"
                "  Install: pip install inference-sdk"
            ) from exc
        self._client = InferenceHTTPClient(
            api_url=self._API_URL,
            api_key=api_key,
        )
        log.info("Cloud F1 detector initialised (model: %s)", self._MODEL_ID)

    def detect(self, img_rgb: np.ndarray, conf: float = 0.25) -> list[Detection]:
        """Run cloud inference and return Detection list."""
        from PIL import Image  # type: ignore
        pil_img = Image.fromarray(img_rgb)

        try:
            result = self._client.infer(pil_img, model_id=self._MODEL_ID)
        except Exception as exc:
            log.warning("Cloud F1 inference failed: %s", exc)
            return []

        detections: list[Detection] = []
        for pred in result.get("predictions", []):
            if pred.get("confidence", 0) < conf:
                continue
            x   = pred["x"]
            y   = pred["y"]
            w   = pred["width"]
            h   = pred["height"]
            detections.append(Detection(
                label="f1_car",
                weight=1.0,
                conf=float(pred["confidence"]),
                x1=x - w / 2,
                y1=y - h / 2,
                x2=x + w / 2,
                y2=y + h / 2,
            ))

        h_img, w_img = img_rgb.shape[:2]
        detections.sort(
            key=lambda d: d.subject_score(w_img, h_img), reverse=True
        )
        return detections


# ---------------------------------------------------------------------------
# Per-burst-group processing
# ---------------------------------------------------------------------------


def _process_group(
    group: BurstGroup,
    exif_map: dict[Path, ExifData],
    f1_model,
    coco_model,
    cloud_f1: _CloudF1Detector | None,
    top_n: int,
    sharp_thresh: float,
    w_sharp: float,
    w_comp: float,
    min_raw: float,
    conf: float,
    dry_run: bool,
    force: bool = False,
    p4_policy: str = "always",
    scale_width: int = 0,
    prefetch_executor: ThreadPoolExecutor | None = None,
) -> list[ImageScore]:
    """Score all frames in one burst group and apply TopN selection.

    When *prefetch_executor* is provided, the next frame's image decode is
    submitted to the thread pool while the current frame is being processed
    (detection + sharpness + composition).  This overlaps I/O-bound HIF
    decoding with CPU-bound scoring.
    """
    scores: list[ImageScore] = []
    prev_detections: list[Detection] | None = None
    frames = group.frames

    # --- Prefetch setup -------------------------------------------------------
    # Submit decode of the first frame immediately
    pending_future = None
    if prefetch_executor is not None and len(frames) > 0:
        pending_future = prefetch_executor.submit(
            _load_image_rgb, frames[0], scale_width,
        )

    # Determine P4 check based on policy
    check_p4 = True
    if p4_policy == "never":
        check_p4 = False
    elif p4_policy == "auto":
        # Auto mode: Only enable P4 if we see "F1" or "Grand Prix" or "GP" in the immediate directory name
        dir_name = frames[0].parent.name.lower()
        keywords = ["f1", "gp", "grand prix", "race", "quali", "practice"]
        check_p4 = any(k in dir_name for k in keywords)
        
        # Exception: sprint_quali (F1 session where P4 showed negative impact in benchmarks)
        if "sprint_quali" in dir_name:
            check_p4 = False

    for frame_idx, frame_path in enumerate(frames):
        is_first = frame_idx == 0

        # --- Consume prefetch ASAP to keep sync -------------------------------
        img_rgb_prefetched = None
        if pending_future is not None:
            img_rgb_prefetched = pending_future.result()
            pending_future = None

        # --- Check for existing culling status (XMP sidecar or internal metadata)
        xmp_rating, xmp_pick = read_xmp_rating(frame_path)
        exif = exif_map.get(frame_path)
        
        # Priority: 1. XMP sidecar, 2. Internal metadata
        final_rating = xmp_rating if xmp_rating is not None else (exif.rating if exif else None)
        final_pick = xmp_pick if xmp_pick is not None else (exif.pick if exif else None)

        # Logic: 0 stars or 0 pick flag means "unrated/no decision" in LR.
        # We only skip if the user has explicitly set a non-zero value.
        is_rating_set = final_rating is not None and final_rating != 0
        is_pick_set = final_pick is not None and final_pick != 0

        # Skip if EITHER non-zero rating or pick status is present (unless --force)
        if not force and (is_rating_set or is_pick_set):
            # Infer missing value for consistent internal scoring
            if not is_rating_set:
                final_rating = 1 if final_pick == 1 else (-1 if final_pick == -1 else 0)
            if not is_pick_set:
                final_pick = 1 if final_rating > 0 else (-1 if final_rating == -1 else 0)

            source = "XMP sidecar" if xmp_rating is not None or xmp_pick is not None else "Internal Metadata"
            log.info("  [%s]  Existing decision found in %s (Rating=%d, Pick=%d) - skipping analysis", 
                     frame_path.name, source, final_rating, final_pick)
            
            img_score = ImageScore(
                path=frame_path,
                s_sharp=1.0,   # Placeholder
                s_comp=1.0,    # Placeholder
                raw_score=10.0 if final_rating > 0 else 0.0, 
                rating=final_rating,
                vetoed=(final_rating == -1),
                veto_reason="manual_metadata" if final_rating == -1 else "",
                is_manual=True
            )
            scores.append(img_score)
            continue

        # --- Load image (from prefetch or directly) ---------------------------
        if img_rgb_prefetched is not None:
            img_rgb = img_rgb_prefetched
        else:
            img_rgb = _load_image_rgb(frame_path, scale_width=scale_width)

        # Submit next frame for prefetch (overlap decode with current processing)
        if prefetch_executor is not None and frame_idx + 1 < len(frames):
            pending_future = prefetch_executor.submit(
                _load_image_rgb, frames[frame_idx + 1], scale_width,
            )

        if img_rgb is None:
            log.warning("Skipping unreadable frame: %s", frame_path.name)
            continue

        # Derive BGR from the already-loaded RGB to avoid double-decoding
        # (especially important for large HIF/NEF RAW files)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w = img_rgb.shape[:2]

        # --- Detection --------------------------------------------------------
        if cloud_f1 is not None:
            # Use cloud F1; if no result, fall through to COCO
            detections = cloud_f1.detect(img_rgb, conf=conf)
            if not detections:
                detections = detect(img_rgb, None, coco_model, conf=conf)
        else:
            detections = detect(img_rgb, f1_model, coco_model, conf=conf)

        primary = detections[0] if detections else None

        # --- Sharpness --------------------------------------------------------
        s_sharp = score_sharpness(img_bgr, primary)

        # --- Composition ------------------------------------------------------
        s_comp = score_composition(
            detections=detections,
            img_w=w,
            img_h=h,
            prev_detections=prev_detections,
            is_first_frame=is_first,
        )

        # --- Score ------------------------------------------------------------
        img_score = score_image(
            path=frame_path,
            detections=detections,
            s_sharp=s_sharp,
            s_comp=s_comp,
            sharp_thresh=sharp_thresh,
            w_sharp=w_sharp,
            w_comp=w_comp,
            min_raw=min_raw,
            check_p4=check_p4,
            img_rgb=img_rgb,
        )
        scores.append(img_score)

        log.info(
            "  [%s]  sharp=%.3f  comp=%.3f  raw=%.2f  Rating=%+d%s",
            frame_path.name,
            s_sharp,
            s_comp,
            img_score.raw_score,
            img_score.rating,
            f"  ({img_score.veto_reason})" if img_score.vetoed else "",
        )

        prev_detections = detections if detections else None

    # --- Burst TopN selection ------------------------------------------------
    select_best_n(scores, top_n=top_n)

    return scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        return 1
        
    log_file = setup_logging(input_dir)

    # --- Collect image files -------------------------------------------------
    image_paths = _collect_images(input_dir, recursive=args.recursive)
    if not image_paths:
        log.error("No supported image files found in %s", input_dir)
        return 1

    log.info("Found %d image files in %s%s",
             len(image_paths), input_dir,
             " (recursive)" if args.recursive else "")

    # --- Rename --------------------------------------------------------------
    if args.rename:
        log.info("Renaming images by EXIF timestamp...")
        new_map = rename_images(image_paths, dry_run=args.dry_run)
        image_paths = sorted(list(new_map.values()))

    # --- Prioritize JPG/HIF over RAW -----------------------------------------
    stems: dict[str, Path] = {}
    has_raw: dict[str, bool] = {}
    
    for p in image_paths:
        stem = p.stem.lower()
        ext = p.suffix.lower()
        if ext in _RAW_EXTS:
            has_raw[stem] = True
        
        if stem not in stems:
            stems[stem] = p
        else:
            prev = stems[stem]
            if ext in _COOKED_EXTS and prev.suffix.lower() in _RAW_EXTS:
                stems[stem] = p
    
    # These are the files we will actually process (decode & score)
    image_paths = sorted(stems.values())
    
    # Track which cooked files are "standalone" (no RAW partner)
    standalone_cooked: set[Path] = set()
    for stem, p in stems.items():
        if p.suffix.lower() in _COOKED_EXTS and not has_raw.get(stem):
            standalone_cooked.add(p)

    log.info("Processing %d unique shots (%d standalone cooked)",
             len(image_paths), len(standalone_cooked))

    # --- Read EXIF & group bursts --------------------------------------------
    log.info("Reading EXIF metadata …")
    exif_list = read_exif(image_paths)
    exif_map  = {e.path: e for e in exif_list}

    log.info("Grouping burst sequences …")
    groups = group_bursts(exif_list)
    log.info(
        "  %d groups  (%d burst, %d single)",
        len(groups),
        sum(1 for g in groups if g.is_burst),
        sum(1 for g in groups if not g.is_burst),
    )

    # --- Load detection models -----------------------------------------------
    f1_model    = None
    cloud_f1    = None
    coco_model  = load_coco_model()

    f1_onnx = Path(args.f1_model)
    if args.rf_api_key:
        log.info("Cloud F1 detector enabled (api_key provided)")
        cloud_f1 = _CloudF1Detector(args.rf_api_key)
    elif f1_onnx.exists():
        f1_model = load_f1_model(f1_onnx)
    else:
        log.warning(
            "No F1 model available (--f1-model not found and --rf-api-key not set). "
            "Only COCO detection will be used."
        )

    # --- P4 Policy check ---
    if args.p4_policy == "auto":
        dir_name = input_dir.name.lower()
        keywords = ["f1", "gp", "grand prix", "race", "quali", "practice"]
        is_f1 = any(k in dir_name for k in keywords)
        is_sprint = "sprint_quali" in dir_name
        if is_f1 and not is_sprint:
            log.info("P4 Multi-task model: ENABLED (auto-policy match)")
        else:
            reason = "sprint_quali excluded" if is_sprint else "no keywords found"
            log.info("P4 Multi-task model: DISABLED (auto-policy: %s)", reason)
    elif args.p4_policy == "always":
        log.info("P4 Multi-task model: ENABLED (policy: always)")
    else:
        log.info("P4 Multi-task model: DISABLED (policy: never)")

    # --- Decode scale and prefetch config ------------------------------------
    scale_width = args.scale_width
    n_workers   = args.workers
    log.info("Decode scale: %s,  prefetch workers: %d",
             f"{scale_width}px" if scale_width > 0 else "full-res",
             n_workers)

    # --- Process each group (Parallel) ---------------------------------------
    all_scores: list[ImageScore] = []
    t_start = time.perf_counter()

    # Create a wrapper for parallel execution
    def process_one_group(g_info):
        idx, group = g_info
        label = "burst" if group.is_burst else "single"
        log.info("Processing Group %d/%d  [%s, %d frame(s)]", idx, len(groups), label, len(group.frames))
        
        try:
            group_scores = _process_group(
                group=group,
                exif_map=exif_map,
                f1_model=f1_model,
                coco_model=coco_model,
                cloud_f1=cloud_f1,
                top_n=args.top_n,
                sharp_thresh=args.sharp_thresh,
                w_sharp=args.w_sharp,
                w_comp=args.w_comp,
                min_raw=args.min_raw,
                conf=args.conf,
                dry_run=args.dry_run,
                force=args.force,
                p4_policy=args.p4_policy,
                scale_width=scale_width,
                prefetch_executor=None,
            )
            for s in group_scores:
                s.burst_group = idx
            return group_scores
        except Exception as e:
            log.error("Error processing group %d: %s", idx, e)
            return []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        group_results = list(executor.map(process_one_group, enumerate(groups, start=1)))
    
    for res in group_results:
        all_scores.extend(res)

    elapsed = time.perf_counter() - t_start

    # --- Summary statistics --------------------------------------------------
    n_total   = len(all_scores)
    n_reject  = sum(1 for s in all_scores if s.rating == -1)
    n_keep    = n_total - n_reject
    ips       = n_total / elapsed if elapsed > 0 else float("inf")

    log.info(
        "\nDone in %.1fs  (%.1f img/s)  total=%d  keep=%d  reject=%d",
        elapsed, ips, n_total, n_keep, n_reject,
    )

    rating_dist: dict[int, int] = {}
    for s in all_scores:
        rating_dist[s.rating] = rating_dist.get(s.rating, 0) + 1
    for r in sorted(rating_dist):
        label = "Rejected" if r == -1 else f"{r}★"
        log.info("  %8s : %d", label, rating_dist[r])

    # --- Write XMP sidecars --------------------------------------------------
    xmp_pairs = [(s.path, s.rating) for s in all_scores if not s.is_manual]
    written = write_xmp_batch(xmp_pairs, overwrite=True, dry_run=args.dry_run)

    action = "Would write" if args.dry_run else "Wrote"
    log.info("%s %d XMP sidecar(s) alongside original files.", action, len(written))

    # --- Sync to standalone cooked files -------------------------------------
    to_sync = [s for s in all_scores if s.path in standalone_cooked and not s.is_manual]
    if to_sync and not args.dry_run:
        log.info("Syncing metadata to %d standalone cooked files via ExifTool...", len(to_sync))
        from apply_exif_ratings import update_single_image
        # Using a few workers to avoid overwhelming the system
        with ThreadPoolExecutor(max_workers=min(8, n_workers)) as sync_executor:
            for s in to_sync:
                sync_executor.submit(update_single_image, s.path, s.rating)

    # --- Dump per-image scores to CSV (for offline parameter tuning) ----------
    if args.dump_scores:
        _dump_scores_csv(all_scores, Path(args.dump_scores), input_dir,
                         args.label_check or args.label_check_dir is not None)

    # --- Label check (compare with ARW ground truth) -------------------------
    if args.label_check:
        _run_label_check(all_scores, input_dir, args.label_check_dir)

    return 0


# ---------------------------------------------------------------------------
# Dump per-image scores to CSV (for offline parameter tuning)
# ---------------------------------------------------------------------------


def _dump_scores_csv(
    all_scores: list[ImageScore],
    out_path: Path,
    input_dir: Path,
    include_gt: bool,
) -> None:
    """Write per-image scores to a CSV file for offline parameter tuning.

    Columns: filename, s_sharp, s_comp, raw_score, rating, vetoed,
    veto_reason, n_detections, burst_group, [has_arw].

    When *include_gt* is True, also writes a ``has_arw`` column by scanning
    for same-stem ARW files (ground truth).
    """
    # Build ARW ground truth set if needed
    arw_stems: set[str] = set()
    if include_gt:
        for search_dir in [input_dir, input_dir.parent]:
            if search_dir.is_dir():
                for p in search_dir.iterdir():
                    if p.suffix.lower() == ".arw":
                        arw_stems.add(p.stem.lower())
        log.info("Ground truth: %d ARW stems found for CSV export", len(arw_stems))

    fieldnames = [
        "filename", "s_sharp", "s_comp", "raw_score", "rating",
        "vetoed", "veto_reason", "n_detections", "burst_group",
    ]
    if include_gt:
        fieldnames.append("has_arw")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_scores:
            row: dict = {
                "filename": s.path.name,
                "s_sharp": f"{s.s_sharp:.6f}",
                "s_comp": f"{s.s_comp:.6f}",
                "raw_score": f"{s.raw_score:.6f}",
                "rating": s.rating,
                "vetoed": int(s.vetoed),
                "veto_reason": s.veto_reason,
                "n_detections": s.n_detections,
                "burst_group": s.burst_group,
            }
            if include_gt:
                row["has_arw"] = int(s.path.stem.lower() in arw_stems)
            writer.writerow(row)

    log.info("Scores dumped to %s  (%d rows)", out_path, len(all_scores))


# ---------------------------------------------------------------------------
# Label-check evaluation (--label-check)
# ---------------------------------------------------------------------------


def _run_label_check(
    all_scores: list[ImageScore],
    input_dir: Path,
    arw_dir: Path | None,
) -> None:
    """Compare system ratings against ARW ground truth and report metrics.

    Ground truth rule: if a same-stem .ARW file exists, the image is a "keep"
    (label=1).  Otherwise it is a "discard" (label=0).

    For the F1 photo directory layout::

        session/
        ├── *.ARW           ← ground truth (keep)
        └── HIF/*.HIF       ← all shots

    When *arw_dir* is ``None``, the function searches both:
      1. The HIF file's own directory (same dir)
      2. The HIF file's parent directory (one level up, i.e. session root)

    When *arw_dir* is provided, it is used as the sole directory to search.
    """
    log.info("\n" + "=" * 60)
    log.info("LABEL CHECK — comparing with ARW ground truth")
    log.info("=" * 60)

    # Build set of ARW stems (case-insensitive)
    arw_stems: set[str] = set()

    if arw_dir is not None:
        # User-specified directory
        if arw_dir.is_dir():
            arw_stems = {p.stem.lower() for p in arw_dir.iterdir()
                         if p.suffix.lower() == ".arw"}
            log.info("ARW directory: %s  (%d ARW files)", arw_dir, len(arw_stems))
        else:
            log.error("ARW check directory not found: %s", arw_dir)
            return
    else:
        # Auto-detect: check input_dir itself + parent
        for search_dir in [input_dir, input_dir.parent]:
            if search_dir.is_dir():
                for p in search_dir.iterdir():
                    if p.suffix.lower() == ".arw":
                        arw_stems.add(p.stem.lower())
        log.info("Auto-detected %d ARW files (searched %s and %s)",
                 len(arw_stems), input_dir, input_dir.parent)

    if not arw_stems:
        log.warning("No ARW files found — cannot run label check.")
        return

    # Classify each scored image
    tp = fp = fn = tn = 0
    mismatches: list[tuple[str, str, str]] = []  # (filename, predicted, actual)

    for score in all_scores:
        stem = score.path.stem.lower()
        actual_keep = stem in arw_stems
        predicted_keep = score.rating > 0  # Rating 1-5 = keep, -1 = reject

        if predicted_keep and actual_keep:
            tp += 1
        elif predicted_keep and not actual_keep:
            fp += 1
            mismatches.append((score.path.name, "KEEP", "discard"))
        elif not predicted_keep and actual_keep:
            fn += 1
            mismatches.append((score.path.name, "REJECT", "keep"))
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    log.info("")
    log.info("Confusion matrix:")
    log.info("                 Predicted KEEP    Predicted REJECT")
    log.info("  Actual keep:     TP = %-5d        FN = %-5d", tp, fn)
    log.info("  Actual discard:  FP = %-5d        TN = %-5d", fp, tn)
    log.info("")
    log.info("  Total images:  %d", total)
    log.info("  ARW (keep):    %d  (%.1f%%)", tp + fn, 100 * (tp + fn) / total if total else 0)
    log.info("  No-ARW (disc): %d  (%.1f%%)", fp + tn, 100 * (fp + tn) / total if total else 0)
    log.info("")
    log.info("  Accuracy:   %.4f  (%d / %d correct)", accuracy, tp + tn, total)
    log.info("  Precision:  %.4f  (of predicted keeps, %.1f%% are real keeps)",
             precision, 100 * precision)
    log.info("  Recall:     %.4f  (of real keeps, %.1f%% are recovered)",
             recall, 100 * recall)
    log.info("  F1 score:   %.4f", f1)

    # Show rating distribution for actual-keep vs actual-discard
    keep_ratings: dict[int, int] = {}
    disc_ratings: dict[int, int] = {}
    for score in all_scores:
        stem = score.path.stem.lower()
        bucket = keep_ratings if stem in arw_stems else disc_ratings
        bucket[score.rating] = bucket.get(score.rating, 0) + 1

    log.info("")
    log.info("Rating distribution by ground truth:")
    log.info("  Rating   Keep(ARW)  Discard(no-ARW)")
    for r in sorted(set(list(keep_ratings.keys()) + list(disc_ratings.keys()))):
        label = "Reject" if r == -1 else f"  {r}★  "
        log.info("  %6s    %-5d      %-5d", label, keep_ratings.get(r, 0), disc_ratings.get(r, 0))

    # Show some mismatches for debugging
    if mismatches:
        n_show = min(20, len(mismatches))
        log.info("")
        log.info("Sample mismatches (showing %d of %d):", n_show, len(mismatches))
        for name, pred, actual in mismatches[:n_show]:
            log.info("  %-25s  predicted=%-7s  actual=%s", name, pred, actual)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cull_photos",
        description=(
            "Rule-based F1 photo culling pipeline.\n"
            "Generates Lightroom XMP sidecar files with star ratings."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Input --------------------------------------------------------------
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing RAW / JPEG / HIF files to process.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively scan subdirectories (e.g. HIF/ under a session dir).",
    )

    # ---- Models -------------------------------------------------------------
    parser.add_argument(
        "--f1-model",
        type=Path,
        default=Path("models/f1_yolov8n.onnx"),
        help=(
            "Path to the local F1 YOLO ONNX model.  "
            "Run models/download_f1_model.py to generate it."
        ),
    )
    parser.add_argument(
        "--rf-api-key",
        default=None,
        metavar="KEY",
        help=(
            "Roboflow API key for cloud-based F1 detection.  "
            "Used instead of --f1-model when provided."
        ),
    )

    # ---- Scoring parameters -------------------------------------------------
    parser.add_argument(
        "--top-n",
        type=int,
        default=11,
        help="Maximum frames to keep per burst group.",
    )
    parser.add_argument(
        "--sharp-thresh",
        type=float,
        default=SHARP_THRESH,
        help="Minimum sharpness score; below this → Rating -1 (vetoed).",
    )
    parser.add_argument(
        "--w-sharp",
        type=float,
        default=W_SHARP,
        help="Weight for sharpness in raw score formula.",
    )
    parser.add_argument(
        "--w-comp",
        type=float,
        default=W_COMP,
        help="Weight for composition in raw score formula.",
    )
    parser.add_argument(
        "--min-raw",
        type=float,
        default=MIN_RAW,
        help="Minimum raw score to keep; below this → Rating -1 (vetoed). 0 to disable.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence for object detections.",
    )
    parser.add_argument(
        "--p4-policy",
        choices=["always", "never", "auto"],
        default="always",
        help=(
            "P4 evaluation policy: 'always' (run for all), 'never' (disable), "
            "'auto' (enable only for F1 sessions based on path/keywords, excluding sprint_quali)."
        ),
    )

    # ---- Output -------------------------------------------------------------
    parser.add_argument(
        "--rename",
        action="store_true",
        help="Rename images to IMG_YYYYMMDD_HHMMSS_MS format before processing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log what would be written without creating any .xmp files.",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Ignore existing ratings/picks and re-analyze all images.",
    )
    parser.add_argument(
        "--dump-scores",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help=(
            "Dump per-image scores (s_sharp, s_comp, raw_score, detections, "
            "burst_group, etc.) to a CSV file for offline parameter tuning.  "
            "When --label-check is active, includes a 'has_arw' ground truth column."
        ),
    )

    # ---- Label check (ground truth comparison) ------------------------------
    parser.add_argument(
        "--label-check",
        action="store_true",
        default=False,
        help=(
            "After scoring, compare predictions against ARW ground truth.  "
            "An image is 'keep' if a same-stem .ARW exists in the parent dir.  "
            "Reports accuracy, precision, recall, and F1."
        ),
    )
    parser.add_argument(
        "--label-check-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing ARW files for label check.  "
            "If not specified, searches input-dir and its parent automatically."
        ),
    )

    # ---- Performance --------------------------------------------------------
    parser.add_argument(
        "--scale-width",
        type=int,
        default=1280,
        help=(
            "Decode images at this width (aspect-ratio preserved).  "
            "Set to 0 for full-resolution decode (slow for HIF)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Number of prefetch threads for image decoding.  "
            "Overlaps decode I/O with scoring computation."
        ),
    )

    # ---- Misc ---------------------------------------------------------------
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
