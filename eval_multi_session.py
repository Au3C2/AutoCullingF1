"""
eval_multi_session.py — Run pipeline on a sampled test set from multiple sessions.

Reads ``test_set.csv`` (produced by sample_test_set.py), runs the detection +
scoring pipeline on the sampled HIF files, and outputs a merged scores CSV
with ground truth.  This CSV can then be fed to ``tune_params.py`` for
offline parameter tuning.

The script also outputs per-session and overall evaluation metrics.

Usage
-----
    python eval_multi_session.py --test-set test_set.csv --output scores_multi.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from cull.exif_reader import ExifData, read_exif, group_bursts
from cull.detector import Detection, load_f1_model, load_coco_model, detect
from cull.sharpness import score_sharpness
from cull.composition import score_composition
from cull.scorer import (
    ImageScore, score_image, select_best_n,
    SHARP_THRESH, W_SHARP, W_COMP, MIN_RAW,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image loading (mirrors cull_photos.py logic)
# ---------------------------------------------------------------------------

def _load_image_rgb(path: Path, scale_width: int = 1280):
    """Load HIF image via ffmpeg preview stream extraction."""
    import subprocess
    import numpy as np

    try:
        # Try ffmpeg preview stream extraction first (fastest for HIF)
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-map", "0:6",          # preview stream
                "-frames:v", "1",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-vcodec", "rawvideo",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and len(result.stdout) > 0:
            # Preview stream is 1664x1088
            w, h = 1664, 1088
            expected = w * h * 3
            if len(result.stdout) == expected:
                img = np.frombuffer(result.stdout, dtype=np.uint8).reshape(h, w, 3)
                return img

        # Fallback: ffmpeg generic decode
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-frames:v", "1",
                "-f", "image2pipe",
                "-vcodec", "png",
                "pipe:1",
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0:
            buf = np.frombuffer(result.stdout, dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    except Exception as e:
        log.warning("Failed to decode %s: %s", path.name, e)

    return None


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def _process_session(
    session_id: str,
    hif_dir: Path,
    filenames: list[str],
    arw_set: set[str],
    f1_model,
    coco_model,
    *,
    top_n: int,
    sharp_thresh: float,
    w_sharp: float,
    w_comp: float,
    min_raw: float,
    conf: float,
    n_workers: int = 2,
) -> list[dict]:
    """Process a subset of files from one session.

    Returns a list of score dicts ready for CSV output.
    """
    # Build full paths for the subset
    file_paths = [hif_dir / f for f in filenames]

    # Read EXIF for burst grouping
    log.info("[%s] Reading EXIF for %d files...", session_id, len(file_paths))
    exif_list = read_exif(file_paths)
    exif_map = {e.path: e for e in exif_list}

    # Group bursts
    groups = group_bursts(exif_list)
    n_burst = sum(1 for g in groups if g.is_burst)
    log.info("[%s] %d groups (%d burst, %d single)",
             session_id, len(groups), n_burst, len(groups) - n_burst)

    # Process each group
    all_rows: list[dict] = []
    group_offset = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for g_idx, group in enumerate(groups, start=1):
            group_offset += 1
            scores: list[ImageScore] = []
            prev_detections: list[Detection] | None = None
            det_confs: list[float] = []  # max detection confidence per frame

            # Prefetch first frame
            pending_future = None
            if len(group.frames) > 0:
                pending_future = pool.submit(_load_image_rgb, group.frames[0])

            for frame_idx, frame_path in enumerate(group.frames):
                is_first = frame_idx == 0

                # Load image
                if pending_future is not None:
                    img_rgb = pending_future.result()
                    pending_future = None
                else:
                    img_rgb = _load_image_rgb(frame_path)

                # Prefetch next
                if frame_idx + 1 < len(group.frames):
                    pending_future = pool.submit(
                        _load_image_rgb, group.frames[frame_idx + 1]
                    )

                if img_rgb is None:
                    log.warning("  Skipping unreadable: %s", frame_path.name)
                    continue

                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                h, w = img_rgb.shape[:2]

                # Detection
                detections = detect(img_rgb, f1_model, coco_model, conf=conf)
                primary = detections[0] if detections else None

                # Max detection confidence
                max_conf = max((d.conf for d in detections), default=0.0)
                # Max F1 detection confidence
                f1_max_conf = max(
                    (d.conf for d in detections if d.label == "f1_car"),
                    default=0.0,
                )

                # Sharpness
                s_sharp = score_sharpness(img_bgr, primary)

                # Composition
                s_comp = score_composition(
                    detections=detections,
                    img_w=w,
                    img_h=h,
                    prev_detections=prev_detections,
                    is_first_frame=is_first,
                )

                # Score
                img_score = score_image(
                    path=frame_path,
                    detections=detections,
                    s_sharp=s_sharp,
                    s_comp=s_comp,
                    sharp_thresh=sharp_thresh,
                    w_sharp=w_sharp,
                    w_comp=w_comp,
                    min_raw=min_raw,
                    img_rgb=img_rgb,
                )
                img_score.burst_group = group_offset
                scores.append(img_score)
                det_confs.append((max_conf, f1_max_conf))

                prev_detections = detections if detections else None

            # Burst TopN selection
            select_best_n(scores, top_n=top_n)

            # Collect rows
            for s, (mc, f1c) in zip(scores, det_confs):
                has_arw = 1 if s.path.stem.lower() in arw_set else 0
                all_rows.append({
                    "session": session_id,
                    "filename": s.path.name,
                    "s_sharp": f"{s.s_sharp:.6f}",
                    "s_comp": f"{s.s_comp:.6f}",
                    "raw_score": f"{s.raw_score:.6f}",
                    "rating": s.rating,
                    "vetoed": int(s.vetoed),
                    "veto_reason": s.veto_reason,
                    "n_detections": s.n_detections,
                    "burst_group": s.burst_group,
                    "max_det_conf": f"{mc:.4f}",
                    "f1_max_conf": f"{f1c:.4f}",
                    "has_arw": has_arw,
                    "fence_pred": s.fence_pred,
                    "fence_confidence": f"{s.fence_confidence:.6f}",
                    "p4_orient": s.p4_orient,
                    "p4_orient_conf": f"{s.p4_orient_conf:.4f}",
                    "p4_integ": s.p4_integ,
                    "p4_integ_prob": f"{s.p4_integ_prob:.4f}",
                })

    return all_rows


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def _print_metrics(rows: list[dict], label: str = "Overall") -> dict:
    """Compute and print confusion matrix metrics. Returns dict of metrics."""
    tp = fp = fn = tn = 0
    for r in rows:
        predicted_keep = int(r["rating"]) > 0
        actual_keep = int(r["has_arw"]) == 1
        if predicted_keep and actual_keep:
            tp += 1
        elif predicted_keep and not actual_keep:
            fp += 1
        elif not predicted_keep and actual_keep:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    n_keep = tp + fp
    n_reject = fn + tn
    n_arw = tp + fn

    log.info("--- %s ---", label)
    log.info("  Total=%d  Keep=%d (%.1f%%)  Reject=%d  ARW=%d (%.1f%%)",
             total, n_keep, 100 * n_keep / total, n_reject, n_arw, 100 * n_arw / total)
    log.info("  TP=%d  FP=%d  FN=%d  TN=%d", tp, fp, fn, tn)
    log.info("  Precision=%.3f  Recall=%.3f  F1=%.4f  Accuracy=%.3f",
             precision, recall, f1, accuracy)

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate pipeline on multi-session sampled test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-set", type=Path, default=Path("test_set.csv"),
                        help="Sampled test set CSV from sample_test_set.py.")
    parser.add_argument("--output", type=Path, default=Path("scores_multi.csv"),
                        help="Output scores CSV path.")
    parser.add_argument("--f1-model", type=Path, default=Path("models/f1_yolov8n.onnx"),
                        help="Path to F1 YOLO ONNX model.")
    parser.add_argument("--top-n", type=int, default=11)
    parser.add_argument("--sharp-thresh", type=float, default=SHARP_THRESH)
    parser.add_argument("--w-sharp", type=float, default=W_SHARP)
    parser.add_argument("--w-comp", type=float, default=W_COMP)
    parser.add_argument("--min-raw", type=float, default=MIN_RAW)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    # Load test set CSV
    if not args.test_set.exists():
        log.error("Test set CSV not found: %s", args.test_set)
        return 1

    # Parse test set: group by session
    sessions: dict[str, dict] = {}  # session_id -> {hif_dir, filenames, arw_set}
    with open(args.test_set, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sid = r["session"]
            if sid not in sessions:
                sessions[sid] = {
                    "hif_dir": Path(r["hif_dir"]),
                    "filenames": [],
                    "arw_set": set(),
                }
            sessions[sid]["filenames"].append(r["filename"])
            if int(r["has_arw"]) == 1:
                sessions[sid]["arw_set"].add(Path(r["filename"]).stem.lower())

    total_files = sum(len(s["filenames"]) for s in sessions.values())
    log.info("Test set: %d files across %d sessions", total_files, len(sessions))

    # Load models (once)
    log.info("Loading detection models...")
    coco_model = load_coco_model()
    f1_model = None
    if args.f1_model.exists():
        f1_model = load_f1_model(args.f1_model)
    else:
        log.warning("F1 model not found: %s", args.f1_model)

    # Process each session
    all_rows: list[dict] = []
    t_start = time.perf_counter()

    for sid, sdata in sessions.items():
        t_session = time.perf_counter()
        rows = _process_session(
            session_id=sid,
            hif_dir=sdata["hif_dir"],
            filenames=sdata["filenames"],
            arw_set=sdata["arw_set"],
            f1_model=f1_model,
            coco_model=coco_model,
            top_n=args.top_n,
            sharp_thresh=args.sharp_thresh,
            w_sharp=args.w_sharp,
            w_comp=args.w_comp,
            min_raw=args.min_raw,
            conf=args.conf,
            n_workers=args.workers,
        )
        elapsed_s = time.perf_counter() - t_session
        log.info("[%s] %d images in %.1fs (%.1f img/s)",
                 sid, len(rows), elapsed_s,
                 len(rows) / elapsed_s if elapsed_s > 0 else 0)
        _print_metrics(rows, label=sid)
        all_rows.extend(rows)

    elapsed_total = time.perf_counter() - t_start
    log.info("\nTotal: %d images in %.1fs (%.1f img/s)",
             len(all_rows), elapsed_total,
             len(all_rows) / elapsed_total if elapsed_total > 0 else 0)

    # Overall metrics
    _print_metrics(all_rows, label="OVERALL")

    # Write output CSV
    fieldnames = [
        "session", "filename", "s_sharp", "s_comp", "raw_score", "rating",
        "vetoed", "veto_reason", "n_detections", "burst_group",
        "max_det_conf", "f1_max_conf", "has_arw", "fence_pred", "fence_confidence",
        "p4_orient", "p4_orient_conf", "p4_integ", "p4_integ_prob",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    log.info("Scores written to %s (%d rows)", args.output, len(all_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
