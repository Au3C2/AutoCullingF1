#!/usr/bin/env python3
"""benchmarks/bench_consumer_scaling.py — scoring-chain (consumer) stress test.

Feeds PRE-DECODED frames through the scoring chain only (detect -> sharpness ->
P4 -> score_image), bypassing decode, and measures:

  - serial baseline fps + GPU utilization (is the GPU saturated?)
  - multi-thread fps with per-thread ONNX sessions (+ determinism check)
  - raw batched session.run throughput (batch 1/2/4/8)

Usage:
    python benchmarks/bench_consumer_scaling.py --dir tests/test_img --pattern *.jpg
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cull.loader import load_image_rgb  # noqa: E402
from cull.detector import load_f1_model, detect  # noqa: E402
from cull.p4_classifier import P4Classifier  # noqa: E402
from cull.sharpness import score_sharpness  # noqa: E402
from cull.scorer import score_image  # noqa: E402


class GpuSampler:
    """Sample nvidia-smi utilization.gpu in a background thread."""

    def __init__(self, interval: float = 0.25):
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(interval,), daemon=True)
        self._t.start()

    def _run(self, interval: float):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                self.samples.append(int(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            self._stop.wait(interval)

    def stop(self) -> tuple[float, int]:
        self._stop.set()
        self._t.join(timeout=2)
        if not self.samples:
            return 0.0, 0
        return float(np.mean(self.samples)), int(np.max(self.samples))


def make_chain():
    f1 = load_f1_model(Path("models/f1_yolov8n.onnx"))
    p4 = P4Classifier()
    return f1, p4


def score_frame(f1, p4, img, conf=0.30):
    dets = detect(img, f1, None, conf=conf)
    s_sharp = score_sharpness(img, dets[0] if dets else None)
    sc = score_image(path=Path("x"), detections=dets, s_sharp=s_sharp, s_comp=0.5,
                     check_p4=True, img_rgb=img,
                     img_w=img.shape[1], img_h=img.shape[0])
    return dets, sc.raw_score


def fingerprint(dets) -> str:
    return "|".join(f"{d.label}:{d.conf:.6f}:{d.x1:.4f},{d.y1:.4f},{d.x2:.4f},{d.y2:.4f}"
                    for d in dets)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir", default="tests/test_img")
    parser.add_argument("--pattern", default="*.jpg")
    parser.add_argument("--scale-width", type=int, default=1280)
    parser.add_argument("--n", type=int, default=120, help="frames per mode")
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()

    files = sorted((ROOT / args.dir).glob(args.pattern))
    frames = []
    for p in files:
        im = load_image_rgb(p, scale_width=args.scale_width)
        if im is not None:
            frames.append(im)
    print(f"cached {len(frames)} frames @ {args.scale_width}px")

    # ---- serial baseline -------------------------------------------------
    f1, p4 = make_chain()
    for i in range(8):
        score_frame(f1, p4, frames[i % len(frames)])  # warmup
    ref_fp = [fingerprint(score_frame(f1, p4, im)[0]) for im in frames]

    gpu = GpuSampler()
    t0 = time.perf_counter()
    for i in range(args.n):
        score_frame(f1, p4, frames[i % len(frames)])
    dt = time.perf_counter() - t0
    avg, peak = gpu.stop()
    print(f"\nserial x1 : {args.n/dt:6.1f} fps  ({dt*1000/args.n:.1f} ms/frame)"
          f"  GPU util avg {avg:.0f}% peak {peak}%")

    # ---- multi-thread, per-thread sessions -------------------------------
    for nt in args.threads:
        gpu = GpuSampler()
        barrier = threading.Barrier(nt)
        results: list[list[str]] = [[] for _ in range(nt)]
        errors: list[str] = []

        def worker(wid: int):
            try:
                f1w, p4w = make_chain()
                barrier.wait()
                for i in range(args.n // nt):
                    dets, _ = score_frame(f1w, p4w, frames[(wid * 7 + i * nt) % len(frames)])
                    results[wid].append(fingerprint(dets))
            except Exception as e:  # pragma: no cover
                errors.append(repr(e))

        t0 = time.perf_counter()
        ts = [threading.Thread(target=worker, args=(w,)) for w in range(nt)]
        for t in ts: t.start()
        for t in ts: t.join()
        dt = time.perf_counter() - t0
        avg, peak = gpu.stop()
        done = sum(len(r) for r in results)
        all_fp = [fp for r in results for fp in r]
        mism = sum(1 for fp in all_fp if fp not in set(ref_fp))
        print(f"threads x{nt}: {done/dt:6.1f} fps  ({dt*1000/max(1,done):.1f} ms/frame)"
              f"  GPU util avg {avg:.0f}% peak {peak}%  det-mismatch {mism}/{len(all_fp)}"
              + (f"  ERRORS: {errors[:1]}" if errors else ""))

    # ---- raw batched session.run ----------------------------------------
    import onnxruntime as ort  # noqa: F401
    from cull.detector import LiteYOLO

    def letterbox_batch(imgs, new_shape=(640, 640)):
        canvases = []
        for img in imgs:
            canvas, _, _ = f1.letterbox_numpy(img, new_shape=new_shape)
            canvases.append(canvas)
        return np.stack(canvases)

    print()
    for b in args.batches:
        if b > len(frames):
            continue
        xb = letterbox_batch(frames[:b]).astype(np.float32) / 255.0
        xb = np.transpose(xb, (0, 3, 1, 2))  # BHWC -> BCHW
        f1.session.run(None, {f1.input_name: xb[:1]})  # warmup
        reps = max(1, 60 // b)
        t0 = time.perf_counter()
        for _ in range(reps):
            f1.session.run(None, {f1.input_name: xb})
        dt = time.perf_counter() - t0
        n = reps * b
        print(f"batch x{b}: session.run {dt*1000/n:6.1f} ms/img  ({n/dt:6.1f} img/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())