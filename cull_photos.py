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

    # With F1 model (download first with utils/download_f1_model.py)
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
from cull.detector import Detection, CloudF1Detector, load_f1_model, load_coco_model, detect
from cull.loader import load_image_rgb, update_image_metadata, RAW_EXTS, COOKED_EXTS, EXTENSIONS
from cull.sharpness import score_sharpness
from cull.composition import score_composition
from cull.scorer import ImageScore, score_image, select_best_n, SHARP_THRESH, W_SHARP, W_COMP, MIN_RAW
from cull.xmp_writer import write_xmp_batch
from cull.xmp_reader import read_xmp_rating
from cull.cropper import calculate_crop, has_crop_info

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
                        if ext in EXTENSIONS:
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
# Per-burst-group processing
# ---------------------------------------------------------------------------


def _process_group(
    group: BurstGroup,
    exif_map: dict[Path, ExifData],
    f1_model,
    coco_model,
    cloud_f1: CloudF1Detector | None,
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
    autocrop: bool = False,
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
            load_image_rgb, frames[0], scale_width,
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
                is_manual=True,
                crop=None # Don't re-calculate if skipping
            )
            scores.append(img_score)
            continue

        # --- Load image (from prefetch or directly) ---------------------------
        if img_rgb_prefetched is not None:
            img_rgb = img_rgb_prefetched
        else:
            img_rgb = load_image_rgb(frame_path, scale_width=scale_width)

        # Submit next frame for prefetch (overlap decode with current processing)
        if prefetch_executor is not None and frame_idx + 1 < len(frames):
            pending_future = prefetch_executor.submit(
                load_image_rgb, frames[frame_idx + 1], scale_width,
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
            img_w=w,
            img_h=h,
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

    # After Top-N selection, calculate crops for keepers if requested
    if autocrop:
        for s in scores:
            if s.rating >= 1 and not s.is_manual and s.detections:
                f1_cars = [d for d in s.detections if d.label == "f1_car"]
                if len(f1_cars) == 1:
                    det = f1_cars[0]
                    # Ensure coordinates are normalized (0-1) for calculate_crop
                    if s.img_w and s.img_h:
                        nx1, ny1 = det.x1 / s.img_w, det.y1 / s.img_h
                        nx2, ny2 = det.x2 / s.img_w, det.y2 / s.img_h
                        img_ar = s.img_w / s.img_h
                        s.crop = calculate_crop(nx1, ny1, nx2, ny2, img_ar=img_ar)
                    else:
                        # Fallback if dims missing
                        s.crop = calculate_crop(det.x1, det.y1, det.x2, det.y2, img_ar=1.5)

    return scores


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


from cull.engine import CullingEngine, EngineConfig

log = logging.getLogger(__name__)

def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        return 1
        
    log_file = setup_logging(input_dir)
    
    # Map argparse Namespace to EngineConfig
    config = EngineConfig(
        input_dir=input_dir,
        recursive=args.recursive,
        f1_model_path=Path(args.f1_model),
        rf_api_key=args.rf_api_key,
        top_n=args.top_n,
        sharp_thresh=args.sharp_thresh,
        w_sharp=args.w_sharp,
        w_comp=args.w_comp,
        min_raw=args.min_raw,
        conf=args.conf,
        dry_run=args.dry_run,
        force=args.force,
        p4_policy=args.p4_policy,
        scale_width=args.scale_width,
        autocrop=not args.crop_off,
        rename=args.rename,
        workers=args.workers,
        dump_scores=Path(args.dump_scores) if args.dump_scores else None,
        label_check=args.label_check,
        label_check_dir=Path(args.label_check_dir) if args.label_check_dir else None
    )

    engine = CullingEngine(config)
    
    # Progress callback for CLI
    def progress(msg, p):
        log.info("[%d%%] %s", int(p * 100), msg)

    try:
        all_scores, elapsed = engine.run(progress_callback=progress)
    except Exception as e:
        log.exception("Engine failed: %s", e)
        return 1

    # Summary statistics
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

    # Dump scores CSV if requested
    if config.dump_scores:
        engine.export_scores_csv(config.dump_scores)

    # Label check if requested
    if config.label_check:
        engine.run_label_check(config.label_check_dir)

    return 0

# (Keep _collect_images and other helper functions if they are still used by LabelCheck, 
# although they should ideally be in the Engine as well. For now, I'll move them to Engine)


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
    parser.add_argument(
        "--crop-off",
        action="store_true",
        help="Disable auto-cropping for keepers (enabled by default)",
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
