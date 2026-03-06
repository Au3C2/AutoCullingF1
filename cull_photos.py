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

import cv2
import numpy as np

from cull.exif_reader import ExifData, BurstGroup, read_exif, group_bursts
from cull.detector import Detection, load_f1_model, load_coco_model, detect
from cull.sharpness import score_sharpness
from cull.composition import score_composition
from cull.scorer import ImageScore, score_image, select_best_n, SHARP_THRESH, W_SHARP, W_COMP, MIN_RAW
from cull.xmp_writer import write_xmp_batch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported image extensions (case-insensitive comparison used at runtime)
# ---------------------------------------------------------------------------
_EXTENSIONS = {".hif", ".heif", ".heic", ".nef", ".arw", ".cr2", ".cr3",
               ".orf", ".rw2", ".raf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}


def _collect_images(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Scan *input_dir* for supported image files, sorted by name.

    When *recursive* is ``True``, descend into subdirectories (e.g. ``HIF/``).
    Extension matching is case-insensitive.
    """
    if recursive:
        candidates = (p for p in input_dir.rglob("*") if p.is_file())
    else:
        candidates = (p for p in input_dir.iterdir() if p.is_file())

    return sorted(
        p for p in candidates
        if p.suffix.lower() in _EXTENSIONS
    )


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

    # --- Non-HIF: OpenCV (returns BGR; convert to RGB) -----------------------
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        log.warning("Could not load image: %s", path)
        return None
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

    for frame_idx, frame_path in enumerate(frames):
        is_first = frame_idx == 0

        # --- Load image (from prefetch or directly) ---------------------------
        if pending_future is not None:
            img_rgb = pending_future.result()
            pending_future = None
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

    # --- Collect image files -------------------------------------------------
    image_paths = _collect_images(input_dir, recursive=args.recursive)
    if not image_paths:
        log.error("No supported image files found in %s", input_dir)
        return 1

    log.info("Found %d images in %s%s",
             len(image_paths), input_dir,
             " (recursive)" if args.recursive else "")

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

    # --- Decode scale and prefetch config ------------------------------------
    scale_width = args.scale_width
    n_workers   = args.workers
    log.info("Decode scale: %s,  prefetch workers: %d",
             f"{scale_width}px" if scale_width > 0 else "full-res",
             n_workers)

    # --- Process each group --------------------------------------------------
    all_scores: list[ImageScore] = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_workers) as prefetch_pool:
        for g_idx, group in enumerate(groups, start=1):
            label = "burst" if group.is_burst else "single"
            log.info(
                "Group %d/%d  [%s, %d frame(s)]  %s",
                g_idx, len(groups), label, len(group.frames),
                group.frames[0].name if group.frames else "?",
            )
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
                scale_width=scale_width,
                prefetch_executor=prefetch_pool,
            )
            # Tag each score with its burst group index
            for s in group_scores:
                s.burst_group = g_idx
            all_scores.extend(group_scores)

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
    xmp_pairs = [(s.path, s.rating) for s in all_scores]
    written = write_xmp_batch(xmp_pairs, overwrite=True, dry_run=args.dry_run)

    action = "Would write" if args.dry_run else "Wrote"
    log.info("%s %d XMP sidecar(s) alongside original files.", action, len(written))

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

    # ---- Output -------------------------------------------------------------
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Log what would be written without creating any .xmp files.",
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
