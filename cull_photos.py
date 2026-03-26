"""
cull_photos.py — Rule-based F1 photo culling pipeline CLI.
(LITE VERSION — NO OpenCV)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cull.engine import CullingEngine, EngineConfig
from cull.scorer import SHARP_THRESH, W_SHARP, W_COMP, MIN_RAW
import time

log = logging.getLogger(__name__)

def setup_logging(base_dir: Path):
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cull_{timestamp}.log"
    root_log = logging.getLogger()
    for h in root_log.handlers[:]: root_log.removeHandler(h)
    root_log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    root_log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    root_log.addHandler(ch)
    log.info("Logging to %s", log_file)
    return log_file

def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        return 1
        
    setup_logging(input_dir)
        
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

    if config.dump_scores:
        engine.export_scores_csv(config.dump_scores)

    if config.label_check:
        engine.run_label_check(config.label_check_dir)

    return 0

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cull_photos",
        description="Rule-based F1 photo culling pipeline (LITE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-dir", type=Path, required=True, help="Directory to process.")
    parser.add_argument("--recursive", action="store_true", help="Recursive scan.")
    parser.add_argument("--f1-model", type=Path, default=Path("models/f1_yolov8n.onnx"), help="Path to F1 ONNX model.")
    parser.add_argument("--rf-api-key", default=None, help="Roboflow API key.")
    parser.add_argument("--crop-off", action="store_true", help="Disable auto-cropping.")
    parser.add_argument("--top-n", type=int, default=11, help="Max frames per burst.")
    parser.add_argument("--sharp-thresh", type=float, default=SHARP_THRESH)
    parser.add_argument("--w-sharp", type=float, default=W_SHARP)
    parser.add_argument("--w-comp", type=float, default=W_COMP)
    parser.add_argument("--min-raw", type=float, default=MIN_RAW)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--p4-policy", choices=["always", "never", "auto"], default="always")
    parser.add_argument("--rename", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--dump-scores", type=str, default=None)
    parser.add_argument("--label-check", action="store_true")
    parser.add_argument("--label-check-dir", type=Path, default=None)
    parser.add_argument("--scale-width", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("-v", "--verbose", action="store_true")

    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return run(args)

if __name__ == "__main__":
    sys.exit(main())
