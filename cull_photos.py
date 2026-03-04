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
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cull.exif_reader import ExifData, BurstGroup, read_exif, group_bursts
from cull.detector import Detection, load_f1_model, load_coco_model, detect
from cull.sharpness import score_sharpness
from cull.composition import score_composition
from cull.scorer import ImageScore, score_image, select_best_n, SHARP_THRESH
from cull.xmp_writer import write_xmp_batch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------
_EXTENSIONS = {
    ".hif", ".HIF",
    ".nef", ".NEF",
    ".arw", ".ARW",
    ".jpg", ".jpeg", ".JPG", ".JPEG",
    ".png", ".PNG",
    ".tiff", ".tif", ".TIFF", ".TIF",
}


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_image_rgb(path: Path) -> np.ndarray | None:
    """Load an image as an RGB numpy array.

    Tries pillow-heif for HIF files first, then falls back to OpenCV for
    all other formats.

    Returns ``None`` on failure.
    """
    suffix = path.suffix.lower()

    if suffix in (".hif", ".heif", ".heic"):
        try:
            import pillow_heif  # type: ignore
            from PIL import Image  # type: ignore
            pillow_heif.register_heif_opener()
            pil_img = Image.open(path).convert("RGB")
            return np.array(pil_img, dtype=np.uint8)
        except Exception as exc:
            log.warning("pillow-heif failed for %s: %s — trying OpenCV", path.name, exc)

    # OpenCV: returns BGR; convert to RGB
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        log.warning("Could not load image: %s", path)
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


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
    conf: float,
    dry_run: bool,
) -> list[ImageScore]:
    """Score all frames in one burst group and apply TopN selection."""
    scores: list[ImageScore] = []
    prev_detections: list[Detection] | None = None

    for frame_idx, frame_path in enumerate(group.frames):
        is_first = frame_idx == 0

        # --- Load image -------------------------------------------------------
        img_rgb = _load_image_rgb(frame_path)

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
    image_paths: list[Path] = sorted(
        p for p in input_dir.iterdir()
        if p.suffix in _EXTENSIONS
    )
    if not image_paths:
        log.error("No supported image files found in %s", input_dir)
        return 1

    log.info("Found %d images in %s", len(image_paths), input_dir)

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

    # --- Process each group --------------------------------------------------
    all_scores: list[ImageScore] = []
    t_start = time.perf_counter()

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
            conf=args.conf,
            dry_run=args.dry_run,
        )
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

    return 0


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
        help="Directory containing RAW / JPEG files to process.",
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
        default=3,
        help="Maximum frames to keep per burst group.",
    )
    parser.add_argument(
        "--sharp-thresh",
        type=float,
        default=SHARP_THRESH,
        help="Minimum sharpness score; below this → Rating -1 (vetoed).",
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
