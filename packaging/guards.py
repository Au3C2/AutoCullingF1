#!/usr/bin/env python3
"""packaging/guards.py — unified local regression guards (macOS-first).

Runs the five guards in order and exits non-zero on the first failure:

  1. source precision  — pytest tests/test_cull.py + test_precision_heif +
                         test_precision_raw (9 gates, workers 1/4/6)
  2. source performance— benchmarks/run_benchmarks.py, workers=4
                         (steady-state floors + setup-tax ceilings, ±10%)
  3. packaging flow    — packaging/build.py --onedir (rebuild + artifact check)
  4. packaged precision— same pytest suite with CULL_EXE=<onedir>
  5. packaged perf     — benchmarks/run_benchmarks.py with CULL_EXE=<onedir>

Steps 2 and 5 use the ~500-file protocol and run in ~3 min each on Apple M4
(full suite ~12-15 min). Pass --skip-build to reuse an existing onedir
artifact for steps 4-5 (CI-style reruns).

Usage:
    python packaging/guards.py             # all five guards
    python packaging/guards.py --workers 6 # run perf gates at workers=6
    python packaging/guards.py --skip-build
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if sys.platform == "win32":
    PY = ROOT / ".venv" / "Scripts" / "python.exe"
    ONEDIR = ROOT / "dist" / "auto_cull_v0.1_win_x64" / "auto_cull_v0.1_win_x64.exe"
else:
    PY = ROOT / ".venv" / "bin" / "python"
    ONEDIR = ROOT / "dist" / "auto_cull_v0.1_macos_arm64" / "auto_cull_v0.1_macos_arm64"

PRECISION_TESTS = [
    "tests/test_cull.py",
    "tests/test_precision_heif.py",
    "tests/test_precision_raw.py",
]
# Packaged binary precision also covers the JPG golden baseline.
PACKAGED_PRECISION_TESTS = [
    "tests/test_package.py",
    "tests/test_precision_heif.py",
    "tests/test_precision_raw.py",
]


def run(cmd: list[str], label: str, env: dict | None = None) -> int:
    t0 = time.perf_counter()
    print(f"\n=== {label} ===")
    print(f"    {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
    dt = time.perf_counter() - t0
    print(f"--- {label}: rc={proc.returncode} ({dt:.0f}s)")
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-build", action="store_true",
                    help="reuse the existing onedir artifact")
    ap.add_argument("--scope", choices=["all", "precision", "performance",
                                        "packaging"], default="all",
                    help="which of the three guarded concerns to run: "
                         "precision (1+4), performance (2+5), packaging (3), "
                         "or all five steps")
    args = ap.parse_args()

    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = str(ROOT)
    env_pkg = {**env_base, "CULL_EXE": str(ONEDIR)}

    rcs = []
    do_precision = args.scope in ("all", "precision")
    do_perf = args.scope in ("all", "performance")
    do_packaging = args.scope in ("all", "packaging")

    if do_precision:
        rcs.append(run([str(PY), "-m", "pytest", "-q", *PRECISION_TESTS],
                       "1. source precision gates", env_base))
        rcs.append(run([str(PY), "-m", "pytest", "-q", *PACKAGED_PRECISION_TESTS],
                       "4. packaged precision gates", env_pkg))
    if do_perf:
        # Local guards use the full multi-source 500-file protocol against the
        # locked baselines. The CI workflow (separate) uses the seed protocol
        # with runner-calibrated baselines + wider tolerance.
        rcs.append(run([str(PY), "benchmarks/run_benchmarks.py",
                        "--workers", str(args.workers), "--json",
                        str(ROOT / "build" / "gate_source.json")],
                       "2. source performance gate", env_base))
        rcs.append(run([str(PY), "benchmarks/run_benchmarks.py",
                        "--workers", str(args.workers), "--json",
                        str(ROOT / "build" / "gate_onedir.json")],
                       "5. packaged performance gate", env_pkg))
    if do_packaging:
        rcs.append(run([str(PY), "packaging/build.py", "--onedir"],
                       "3. packaging flow (build onedir)", env_base))

    if not ONEDIR.exists():
        print(f"\nERROR: onedir artifact missing: {ONEDIR}", file=sys.stderr)
        return 1

    fails = [i for i, rc in enumerate(rcs, 1) if rc != 0]
    if fails:
        print(f"\nGUARDS FAILED: steps {fails}")
        return 1
    print("\nALL GUARDS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())