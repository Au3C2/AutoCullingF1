#!/usr/bin/env python3
"""packaging/build.py — unified PyInstaller build for AutoCulling engine & CLI.

Compiles ``engine.spec`` into a onedir distribution in ``dist/engine/``,
containing:
  - auto_culling_cli (.exe on Windows)   — user-facing console CLI
  - auto_culling_engine (.exe on Windows)— windowed GUI sidecar
  - lib/                                 — shared dependencies & models

Used by local regression checks (``packaging/test.py``) and CI
(``.github/workflows/engine-test.yml``, ``.github/workflows/release.yml``).

Usage:
    python packaging/build.py            # builds dist/engine/
    python packaging/build.py --onedir   # alias (for backward compatibility)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_PYI = "pyinstaller.exe" if sys.platform == "win32" else "pyinstaller"
PYINSTALLER = ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / _PYI


def cli_binary_path() -> Path:
    """Return the expected path to the compiled CLI binary."""
    ext = ".exe" if sys.platform == "win32" else ""
    return ROOT / "dist" / "engine" / f"auto_culling_cli{ext}"


def engine_binary_path() -> Path:
    """Return the expected path to the compiled windowed engine binary."""
    ext = ".exe" if sys.platform == "win32" else ""
    return ROOT / "dist" / "engine" / f"auto_culling_engine{ext}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--onedir",
        action="store_true",
        default=True,
        help="kept for backward compatibility (onedir is now the only mode)",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="clean PyInstaller cache before building (use --no-clean to preserve)",
    )
    args = parser.parse_args()

    if not PYINSTALLER.exists():
        pyi_cmd = Path(sys.executable).parent / _PYI
        if not pyi_cmd.exists():
            print(
                f"ERROR: PyInstaller not found at {PYINSTALLER} or {pyi_cmd}. "
                "Install it: uv pip install pyinstaller",
                file=sys.stderr,
            )
            return 1
        pyinstaller_bin = pyi_cmd
    else:
        pyinstaller_bin = PYINSTALLER

    spec_path = ROOT / "engine.spec"
    if not spec_path.exists():
        print(f"ERROR: spec file {spec_path} missing", file=sys.stderr)
        return 1

    print("== Building AutoCulling engine (PyInstaller engine.spec) ==")
    env = os.environ.copy()
    env["CULL_ONEDIR"] = "1"

    cmd = [str(pyinstaller_bin), "--noconfirm"]
    if args.clean:
        cmd.append("--clean")
    cmd.append(str(spec_path))

    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    if proc.returncode != 0:
        return proc.returncode

    cli = cli_binary_path()
    engine = engine_binary_path()

    if not cli.exists():
        print(f"ERROR: expected CLI binary missing at {cli}", file=sys.stderr)
        return 1
    if not engine.exists():
        print(f"ERROR: expected engine binary missing at {engine}", file=sys.stderr)
        return 1

    artifact_dir = ROOT / "dist" / "engine"
    total_bytes = sum(p.stat().st_size for p in artifact_dir.rglob("*") if p.is_file())
    cli_mb = cli.stat().st_size / 1_048_576
    total_mb = total_bytes / 1_048_576

    print(f"\n== Build OK (onedir): {artifact_dir} ==")
    print(f"CLI:    {cli.name} ({cli_mb:.1f} MiB)")
    print(f"Engine: {engine.name}")
    print(f"Bundle: {total_mb:.1f} MiB ({artifact_dir / 'lib'}/)")
    print(f"\nUse in gates:\n  CULL_EXE={cli}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
