#!/usr/bin/env python3
"""benchmarks/profile_pipeline.py — per-stage timing of the culling pipeline.

Reuses the real engine (CLI-equivalent) on a copied subset of files and logs
cumulative per-stage wall time for the CONSUMER thread and the decode-wait
(``Future.result``) as seen by the consumer. Spawn-safe.

Note: decode pool *worker* CPU time is not captured here — consumer-side
decode_wait shows how much of wall time is decode-supply bound.

Usage:
    python benchmarks/profile_pipeline.py --dir tests/test_img --pattern *.jpg
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cull.engine import CullingEngine, EngineConfig  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")

TIMINGS: dict[str, float] = defaultdict(float)
COUNTS: dict[str, int] = defaultdict(int)


def _wrap(name: str, fn):
    def inner(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            TIMINGS[name] += time.perf_counter() - t0
            COUNTS[name] += 1
    return inner


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir", default="tests/test_img", help="source dataset dir")
    parser.add_argument("--pattern", default="*.jpg")
    parser.add_argument("--copies", type=int, default=1, help="replicate files xN to load CPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scale-width", type=int, default=1280)
    args = parser.parse_args()

    src = ROOT / args.dir
    files = sorted(src.glob(args.pattern))
    if not files:
        print(f"no files in {src}")
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="prof_"))
    try:
        for i in range(args.copies):
            for p in files:
                shutil.copy(p, tmp / f"{p.stem}_{i:02d}{p.suffix}")
        paths = sorted(tmp.iterdir())
        print(f"profiling {len(paths)} files, workers={args.workers}, scale={args.scale_width}")

        import cull.engine as engine_mod

        # Patch names bound in the engine module (imports are by-name there).
        import concurrent.futures as cf
        _orig_result = cf.Future.result
        def _wait_wrap(self, timeout=None):
            t0 = time.perf_counter()
            try:
                return _orig_result(self, timeout)
            finally:
                TIMINGS["decode_wait"] += time.perf_counter() - t0
                COUNTS["decode_wait"] += 1
        cf.Future.result = _wait_wrap

        engine_mod.detect = _wrap("detect", engine_mod.detect)
        engine_mod.score_sharpness = _wrap("sharpness", engine_mod.score_sharpness)
        engine_mod.score_composition = _wrap("composition", engine_mod.score_composition)
        engine_mod.score_image = _wrap("score_image", engine_mod.score_image)
        engine_mod.read_xmp_rating = _wrap("xmp_read", engine_mod.read_xmp_rating)

        # Per-model detail: patch the class methods (f1 and coco instances).
        import cull.detector as detector_mod
        _orig_yolo = detector_mod.LiteYOLO.detect
        def _yolo_wrap(self, *a, **k):
            t0 = time.perf_counter()
            try:
                return _orig_yolo(self, *a, **k)
            finally:
                key = "yolo[%s]" % self.model_path.name
                TIMINGS[key] += time.perf_counter() - t0
                COUNTS[key] += 1
        detector_mod.LiteYOLO.detect = _yolo_wrap
        _orig_lb = detector_mod.LiteYOLO.letterbox_pil
        def _lb_wrap(self, *a, **k):
            t0 = time.perf_counter()
            try:
                return _orig_lb(self, *a, **k)
            finally:
                TIMINGS["letterbox"] += time.perf_counter() - t0
                COUNTS["letterbox"] += 1
        detector_mod.LiteYOLO.letterbox_pil = _lb_wrap
        import onnxruntime as _ort
        _orig_run = _ort.InferenceSession.run
        def _run_wrap(self, *a, **k):
            t0 = time.perf_counter()
            try:
                return _orig_run(self, *a, **k)
            finally:
                TIMINGS["session.run"] += time.perf_counter() - t0
                COUNTS["session.run"] += 1
        _ort.InferenceSession.run = _run_wrap
        import cull.p4_classifier as p4_mod
        _orig_p4 = p4_mod.P4Classifier.predict_roi
        def _p4_wrap(self, *a, **k):
            t0 = time.perf_counter()
            try:
                return _orig_p4(self, *a, **k)
            finally:
                TIMINGS["p4_roi"] += time.perf_counter() - t0
                COUNTS["p4_roi"] += 1
        p4_mod.P4Classifier.predict_roi = _p4_wrap

        cfg = EngineConfig(
            input_dir=tmp, scale_width=args.scale_width, workers=args.workers,
            dry_run=True, force=True, p4_policy="always", top_n=11)
        eng = CullingEngine(cfg)
        t0 = time.perf_counter()
        eng.run()
        elapsed = time.perf_counter() - t0

        n = len(eng.all_scores)
        print(f"wall {elapsed:.2f}s  ->  {n/elapsed:.2f} img/s")
        rows = sorted(TIMINGS.items(), key=lambda kv: -kv[1])
        for name, total in rows:
            cnt = COUNTS[name]
            print(f"  {name:<12} total {total*1000:8.1f} ms  ({cnt} calls, {total/max(1,cnt)*1000:6.1f} ms/call)")
        serial = sum(v for k, v in TIMINGS.items() if k != "decode_wait")
        print(f"consumer serial work {serial*1000:.1f} ms total, {serial/max(1,n)*1000:.1f} ms/frame")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())