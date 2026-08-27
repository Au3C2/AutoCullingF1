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

# (dataset dir, glob, source files to take, default workers, default threshold)
# Thresholds are locked per platform.
# Windows (legacy CUDA 4070 Ti baseline locked 2026-08-22): JPG 4.2 / HEIF 3.0 / ARW 2.3 / NEF 1.9 img/s
# macOS (Apple Silicon M4 dedicated baseline locked 2026-08-27):
#   - JPG  : 14.0 img/s (measured 17.6 - 18.3, ~20% safety margin against system load)
#   - HEIF :  6.0 img/s (measured  7.6 -  7.9)
#   - ARW  :  5.0 img/s (measured  6.7 -  6.9)
#   - NEF  :  5.5 img/s (measured  7.1 -  7.3)
DARWIN_THRESHOLDS = {
    "tests/test_img": 14.0,
    "test_import":     6.0,
    "test_arw":        5.0,
    "test_nef":        5.5,
}

WIN_THRESHOLDS = {
    "tests/test_img": 4.2,
    "test_import":    3.0,
    "test_arw":       2.3,
    "test_nef":       1.9,
}

DATASETS = [
    ("tests/test_img", "*.jpg", 6,  4),
    ("test_import",    "*.heif", 24, 4),
    ("test_arw",       "*.ARW", 20, 4),
    ("test_nef",       "*.nef", 20, 4),
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


def bench_copies(files: list[Path], workers: int) -> float:
    """Direct-dir bench for RAW/HEIF sets (copy subset, no replication)."""
    tmp = Path(tempfile.mkdtemp(prefix="bench_raw_"))
    try:
        for p in files:
            shutil.copy(p, tmp / p.name)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    thresholds_map = DARWIN_THRESHOLDS if sys.platform == "darwin" else WIN_THRESHOLDS
    platform_name = "macOS Apple Silicon (Dedicated High-Precision Baseline)" if sys.platform == "darwin" else "Windows / Linux (Generic Baseline)"
    print(f"Running Performance Gate on {platform_name} (workers={args.workers})...\n")

    failures = []
    for subdir, glob, count, default_w in DATASETS:
        files = _available(subdir, glob, count)
        if files is None:
            print(f"[skip] {subdir}: dataset not available")
            continue
        threshold = thresholds_map.get(subdir, 2.0)
        workers = args.workers
        if subdir == "tests/test_img":
            ips = bench_jpg(files, workers)
        else:
            ips = bench_copies(files, workers)
        flag = "OK " if ips >= threshold else "FAIL"
        margin = (ips - threshold) / threshold * 100
        print(f"[{flag}] {subdir:<15}: {ips:5.2f} img/s (threshold: {threshold:4.1f}, margin: {margin:+5.1f}%)")
        if ips < threshold:
            failures.append((subdir, ips, threshold))

    if failures:
        print("\nREGRESSION — " + ", ".join(f"{d} {i:.2f}<{t}" for d, i, t in failures))
        return 1
    print("\nAll available datasets within performance baseline. Pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())