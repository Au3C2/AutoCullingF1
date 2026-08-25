#!/usr/bin/env python3
"""benchmarks/run_benchmarks.py — performance gate (regression gate for optimizations).

Runs the 4-dataset protocol (JPG ×60 / HEIF ×24 / ARW ×20 / NEF ×20) at
``--workers 4 --dry-run`` and asserts end-to-end throughput stays above the
locked baselines. Exit code 0 = pass, 1 = regression.

Baselines locked 2026-08-22 (master@CUDA, RTX 4070 Ti): JPG 5.9, HEIF 6.4,
ARW 4.4, NEF 3.3 img/s. Thresholds leave noise headroom (~25% below).

Usage:
    python benchmarks/run_benchmarks.py            # all available datasets
    python benchmarks/run_benchmarks.py --verbose  # print per-dataset details
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYS = sys.executable

# (dataset dir, glob, source files to take, workers, min img/s)
# Gate thresholds are locked against THIS script's sample sizes (60 JPG via
# 10x replication / 24 HEIF / 20 ARW / 20 NEF), which run slower than the
# 100-image baselines in results/performance_baseline.md because fixed
# per-job overhead (model load, EXIF, process spawn) is amortized over fewer
# frames. Locked 2026-08-22 after the initial run.
DATASETS = [
    ("tests/test_img", "*.jpg", 6, 4, 4.2),    # measured 5.26 (60 imgs via 10x)
    ("test_import",    "*.heif", 24, 4, 3.0),  # measured 3.76
    ("test_arw",       "*.ARW", 20, 4, 2.3),   # measured 2.86
    ("test_nef",       "*.nef", 20, 4, 1.9),   # measured 2.39
]


def _available(subdir: str, glob: str, count: int) -> list[Path] | None:
    base = ROOT / subdir
    if not base.is_dir():
        return None
    files = sorted(base.glob(glob))[:count]
    if len(files) < count:
        return None
    return files


def bench_jpg(files: list[Path], workers: int) -> float:
    """160/120-file JPG bench via 10x copy (matches precision-gate throughput)."""
    tmp = Path(tempfile.mkdtemp(prefix="bench_jpg_"))
    try:
        for i in range(10):
            for p in files:
                shutil.copy(p, tmp / f"{p.stem}_{i:02d}{p.suffix}")
        n = len(list(tmp.iterdir()))
        t0 = time.perf_counter()
        proc = subprocess.run(
            [SYS, str(ROOT / "cull_photos.py"), "--input-dir", str(tmp),
             "--workers", str(workers), "--force", "--dry-run"],
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)},
            timeout=600,
        )
        dt = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "")[-400:])
        return n / dt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bench_copies(files: list[Path], workers: int, extra_flags: list[str] | None = None) -> float:
    """Direct-dir bench for RAW/HEIF sets (copy subset, no replication)."""
    tmp = Path(tempfile.mkdtemp(prefix="bench_raw_"))
    try:
        for p in files:
            shutil.copy(p, tmp / p.name)
        n = len(list(tmp.iterdir()))
        t0 = time.perf_counter()
        cmd = [
            SYS, str(ROOT / "cull_photos.py"), "--input-dir", str(tmp),
            "--workers", str(workers), "--force", "--dry-run"
        ]
        if extra_flags:
            cmd.extend(extra_flags)
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)},
            timeout=600,
        )
        dt = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "")[-400:])
        return n / dt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--enable-apple-hwdecoder", action="store_true",
                        help="pass --enable-apple-hwdecoder to cull_photos.py")
    args = parser.parse_args()

    extra = ["--enable-apple-hwdecoder"] if args.enable_apple_hwdecoder else None

    failures = []
    for subdir, glob, count, workers, threshold in DATASETS:
        files = _available(subdir, glob, count)
        if files is None:
            print(f"[skip] {subdir}: dataset not available")
            continue
        if subdir == "tests/test_img":
            ips = bench_jpg(files, workers)
        else:
            ips = bench_copies(files, workers, extra_flags=extra)
        flag = "OK " if ips >= threshold else "FAIL"
        print(f"[{flag}] {subdir}: {ips:.2f} img/s (min {threshold})")
        if ips < threshold:
            failures.append((subdir, ips, threshold))

    if failures:
        print("\nREGRESSION — " + ", ".join(f"{d} {i:.2f}<{t}" for d, i, t in failures))
        return 1
    print("\nAll available datasets within baseline. Pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())