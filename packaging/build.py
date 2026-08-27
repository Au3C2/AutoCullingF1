#!/usr/bin/env python3
"""packaging/build.py — one-shot packaging build for cull_photos.py.

Builds the single-file PyInstaller executable from ``cull_photos.spec``,
copies it to the project root under the versioned platform name
(``auto_cull_v0.1_macos_arm64`` / ``auto_cull_v0.1_win_x64.exe``) and
prints the archive composition so size regressions are visible.

The spec is cross-platform (macOS + Windows). macOS ships the frozen ONNX
graphs, Windows the dynamic exports; see the spec header.

Usage:
    python packaging/build.py              # build onefile + copy to root + size report
    python packaging/build.py --onedir     # directory form (stable inodes, see below)
    python packaging/build.py --no-copy    # keep the artifact in dist/ only
    python packaging/build.py --keep-old   # keep previous root artifact as *.prev

macOS note: the kernel verifies the adhoc code signature of every bundled
Mach-O the first time it is loaded (inode-keyed cache). The onefile form
re-extracts to a fresh temp dir on every run, so it re-pays that tax per
launch (~15-25 s on Apple M4); the onedir form pays it once per boot, after
which throughput matches the source pipeline. Run the performance gate
against the onedir artifact (CULL_EXE=dist/<name>/<name>) and treat the
onefile cold-start tax as a platform property.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYINSTALLER = ROOT / ".venv" / "bin" / "pyinstaller"

EXE_NAMES = {
    "Darwin": "auto_cull_v0.1_macos_arm64",
    "Windows": "auto_cull_v0.1_win_x64.exe",
    "Linux": "auto_cull",
}


def _archive_summary(path: Path) -> str:
    """zlib-compressed CArchive sizes grouped by top-level name."""
    viewer = ROOT / ".venv" / "bin" / "pyi-archive_viewer"
    proc = subprocess.run([str(viewer), "-l", str(path)],
                          capture_output=True, text=True, check=True)
    groups: dict[str, tuple[int, int, int]] = defaultdict(lambda: [0, 0, 0])
    for line in proc.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        try:
            length, raw = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        name = line.split(",", 5)[-1].strip()
        top = name.split("/")[0].strip("'")
        g = groups[top]
        g[0] += length
        g[1] += raw
        g[2] += 1
    rows = sorted(groups.items(), key=lambda kv: -kv[1][0])
    out = [f"{'top-level':<24}{'compressed':>12}{'raw':>14}{'files':>8}",
           "-" * 60]
    for name, (comp, raw, cnt) in rows:
        out.append(f"{name:<24}{comp:>12,}{raw:>14,}{cnt:>8}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-copy", action="store_true",
                        help="do not copy the artifact to the project root")
    parser.add_argument("--keep-old", action="store_true",
                        help="rename an existing root artifact to *.prev instead of replacing it")
    parser.add_argument("--onedir", action="store_true",
                        help="build the directory form (stable inodes, source-identical "
                             "throughput; run the perf gate against it)")
    args = parser.parse_args()

    if not PYINSTALLER.exists():
        print(f"ERROR: PyInstaller not found at {PYINSTALLER}. "
              "Install it: uv pip install pyinstaller", file=sys.stderr)
        return 1

    print("== Building executable "
          "(PyInstaller {}) ==".format("onedir" if args.onedir else "onefile"))
    env = os.environ.copy()
    if args.onedir:
        env["CULL_ONEDIR"] = "1"
    proc = subprocess.run(
        [str(PYINSTALLER), "--noconfirm", "--clean", "cull_photos.spec"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-6000:])
        print(proc.stderr[-6000:], file=sys.stderr)
        return proc.returncode

    exe_name = EXE_NAMES.get(platform.system(), "auto_cull")
    if args.onedir:
        artifact_dir = ROOT / "dist" / exe_name
        artifact = artifact_dir / exe_name
        if not artifact.exists():
            print(f"ERROR: expected artifact {artifact} missing", file=sys.stderr)
            return 1
        total = sum(p.stat().st_size for p in artifact_dir.rglob("*")
                    if p.is_file())
        exe_mb = artifact.stat().st_size / 1_048_576
        print(f"\n== Build OK (onedir): {artifact_dir.name}/ ==")
        print(f"Executable: {artifact.stat().st_size:,} bytes ({exe_mb:.1f} MiB)")
        print(f"Bundle total: {total:,} bytes ({total / 1_048_576:.1f} MiB)")
        print("\nRun the perf gate against it, e.g.:")
        print(f"  CULL_EXE={artifact} .venv/bin/python benchmarks/run_benchmarks.py")
        return 0

    artifact = ROOT / "dist" / exe_name
    if not artifact.exists():
        print(f"ERROR: expected artifact {artifact} missing", file=sys.stderr)
        return 1

    size_mb = artifact.stat().st_size / 1_048_576
    print(f"\n== Build OK: {artifact.name} ==")
    print(f"Size: {artifact.stat().st_size:,} bytes ({size_mb:.1f} MiB)")
    print()
    print(_archive_summary(artifact))

    if args.no_copy:
        return 0
    target = ROOT / exe_name
    if target.exists() and args.keep_old:
        shutil.move(target, target.with_suffix(target.suffix + ".prev"))
    shutil.copy2(artifact, target)
    print(f"\nCopied to {target} ({target.stat().st_size / 1_048_576:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())