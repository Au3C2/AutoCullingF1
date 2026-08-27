#!/usr/bin/env python3
"""packaging/ci_guard.py — CI guards entrypoint (GitHub Actions, macOS).

Runs the three guarded concerns on the seed protocol (ci_sample/):
   1. precision  — ci_seed_precision.py --compare: packaged-vs-source
                   per-file raw_score equality + rating-multiset equality.
                   Deterministic, needs NO runner calibration.
   2. packaging  — build.py --onedir (artifact check).
   3. performance— run_benchmarks.py --seed-dir (source + packaged) against
                   the runner-calibrated baselines from ci_config.json
                   (tolerance 0.85 — GitHub-hosted runners are noisier than
                   the locked local M4). SKIPPED until the perf-calibrate
                   workflow has produced the baselines.

Usage:
    python packaging/ci_guard.py [--tolerance 0.85] [--samples 3]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python" if (ROOT / ".venv").exists() else "python"
ONEDIR = ROOT / "dist" / "auto_cull_v0.1_macos_arm64" / "auto_cull_v0.1_macos_arm64"
CONFIG = ROOT / "ci_config.json"


def run(cmd: list[str], label: str, env: dict | None = None) -> int:
    t0 = time.perf_counter()
    print(f"\n=== {label} ===")
    print(f"    {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
    print(f"--- {label}: rc={proc.returncode} ({time.perf_counter()-t0:.0f}s)")
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tolerance", type=float, default=0.85,
                    help="steady-state floor fraction for the runner")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4,
                    help="decoder workers (CULL_WORKERS env overrides; CI VM "
                         "uses 2 — workers=4 was observed to stall there)")
    args = ap.parse_args()

    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = str(ROOT)

    config: dict = {}
    if CONFIG.exists():
        config = json.loads(CONFIG.read_text())
    baselines = config.get("baselines", {"source": {}, "onedir": {}})
    calibrated = bool(baselines.get("source"))

    rcs = []
    # 1. packaging flow FIRST — the precision gate compares against the
    #    packaged binary, so the onedir artifact must exist.
    rcs.append(run([str(PY), "packaging/build.py", "--onedir"],
                   "1. packaging flow (build onedir)", env_base))
    if not ONEDIR.exists():
        print(f"\nERROR: onedir artifact missing: {ONEDIR}", file=sys.stderr)
        return 1

    # 2. precision — packaged == source (calibration-free)
    rcs.append(run([str(PY), "benchmarks/ci_seed_precision.py", "--compare"],
                   "2. precision (seed consistency, packaged vs source)",
                   env_base))

    # 3. performance — seed steady-state (needs calibrated baselines)
    if calibrated:
        workers = os.environ.get("CULL_WORKERS", str(args.workers))
        env_pkg = {**env_base, "CULL_EXE": str(ONEDIR)}
        rcs.append(run([str(PY), "benchmarks/run_benchmarks.py",
                        "--workers", workers, "--seed-dir", str(ROOT / "ci_sample"),
                        "--count", "200", "--samples", str(args.samples),
                        "--tolerance", str(args.tolerance),
                        "--baseline-file", str(CONFIG), "--no-prewarm",
                        "--json", str(ROOT / "build" / "ci_source.json")],
                       "3a. performance (seed steady, source)", env_base))
        rcs.append(run([str(PY), "benchmarks/run_benchmarks.py",
                        "--workers", workers, "--seed-dir", str(ROOT / "ci_sample"),
                        "--count", "200", "--samples", str(args.samples),
                        "--tolerance", str(args.tolerance),
                        "--baseline-file", str(CONFIG),
                        "--json", str(ROOT / "build" / "ci_onedir.json")],
                       "3b. performance (seed steady, packaged)", env_pkg))
    else:
        print("\n[skip] ci_config.json baselines not calibrated for this runner"
              " — performance gate skipped.")
        print("       Run the perf-calibrate workflow and commit the result "
              "to ci_config.json to enable it.")

    if any(rc != 0 for rc in rcs):
        print("\nCI GUARDS FAILED")
        return 1
    print("\nCI GUARDS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())