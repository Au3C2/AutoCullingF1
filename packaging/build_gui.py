#!/usr/bin/env python3
"""packaging/build_gui.py — automated packaging builder for AutoCulling Tauri GUI.

Produces:
- macOS: AutoCulling_v{ver}_macos_arm64.dmg (and AutoCulling.app)
- Windows: AutoCulling_v{ver}_win_x64_setup.exe (NSIS) and AutoCulling_v{ver}_win_x64_portable.zip
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_TAURI = ROOT / "src-tauri"


def _version() -> str:
    env_ver = os.environ.get("CULL_VERSION", "").strip().lstrip("v")
    if env_ver:
        return env_ver
    toml = ROOT / "pyproject.toml"
    if toml.exists():
        for line in toml.read_text().splitlines():
            if line.strip().startswith("version"):
                m = re.search(r'"([^"]+)"', line)
                if m:
                    return m.group(1).strip()
    return "0.1"


def get_rust_target_triple() -> str:
    """Detect current rust host target triple."""
    try:
        res = subprocess.run(["rustc", "-vV"], capture_output=True, text=True, check=True)
        for line in res.stdout.splitlines():
            if line.startswith("host:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    # Fallbacks
    sys_plat = sys.platform
    if sys_plat == "darwin":
        return "aarch64-apple-darwin" if platform.machine() == "arm64" else "x86_64-apple-darwin"
    elif sys_plat == "win32":
        return "x86_64-pc-windows-msvc"
    return "x86_64-unknown-linux-gnu"


def build_sidecar_binary() -> Path:
    """Compile cull_photos into a windowed sidecar binary using PyInstaller."""
    print("=== Step 1: Building Python Sidecar Binary ===")
    spec_path = ROOT / "cull_sidecar.spec"
    if not spec_path.exists():
        raise FileNotFoundError("cull_sidecar.spec not found in project root")

    _pyi = "pyinstaller.exe" if sys.platform == "win32" else "pyinstaller"
    pyinstaller = ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin") / _pyi
    if not pyinstaller.exists():
        pyinstaller = Path(shutil.which(_pyi) or _pyi)

    cmd = [str(pyinstaller), str(spec_path), "--noconfirm", "--clean"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)

    sidecar_out_name = "cull_sidecar.exe" if sys.platform == "win32" else "cull_sidecar"
    compiled_sidecar = ROOT / "dist" / sidecar_out_name
    if not compiled_sidecar.exists():
        raise FileNotFoundError(f"Expected compiled sidecar at {compiled_sidecar}")

    print(f"PASS: Compiled sidecar binary: {compiled_sidecar} ({compiled_sidecar.stat().st_size / 1024 / 1024:.1f} MB)")

    # Copy to src-tauri/binaries/cull-sidecar-<triple>[.exe]
    binaries_dir = SRC_TAURI / "binaries"
    binaries_dir.mkdir(parents=True, exist_ok=True)

    triple = get_rust_target_triple()
    ext = ".exe" if sys.platform == "win32" else ""
    target_bin_name = f"cull-sidecar-{triple}{ext}"
    target_bin_path = binaries_dir / target_bin_name

    shutil.copy2(compiled_sidecar, target_bin_path)
    # Ensure executable permissions on POSIX
    if sys.platform != "win32":
        os.chmod(target_bin_path, 0o755)

    print(f"PASS: Staged sidecar to {target_bin_path}")
    return target_bin_path


def build_tauri_gui() -> None:
    """Run tauri build to produce desktop bundle packages."""
    print("\n=== Step 2: Building Tauri Desktop GUI Packages ===")
    
    # Try npx tauri build or cargo tauri build
    tauri_cli = None
    if shutil.which("cargo-tauri"):
        tauri_cli = ["cargo", "tauri", "build"]
    elif shutil.which("npx"):
        tauri_cli = ["npx", "@tauri-apps/cli", "build"]
    else:
        tauri_cli = ["cargo", "tauri", "build"]

    print(f"Running Tauri build: {' '.join(tauri_cli)}")
    try:
        subprocess.run(tauri_cli, cwd=str(ROOT), check=True)
    except subprocess.CalledProcessError:
        # Known Tauri 2 + macOS Sequoia issue: create-dmg's Finder AppleScript
        # prettify step times out (AppleEvent -1712) even though .app + dmg
        # script are fine. Fall back to a plain DMG with --skip-jenkins so the
        # Applications drag link still works without Finder cosmetics.
        bundle_dmg = SRC_TAURI / "target/release/bundle/dmg/bundle_dmg.sh"
        app_dir = SRC_TAURI / "target/release/bundle/macos"
        dmg_out = SRC_TAURI / "target/release/bundle/dmg/AutoCulling_0.1.0_aarch64.dmg"
        if bundle_dmg.exists() and app_dir.exists():
            print("Tauri DMG prettify failed; retrying with --skip-jenkins fallback...")
            subprocess.run(
                ["bash", str(bundle_dmg), "--skip-jenkins", "--volname", "AutoCulling",
                 "--window-size", "660", "400", "--icon-size", "128",
                 "--app-drop-link", "480", "170",
                 str(dmg_out), str(app_dir)],
                cwd=str(ROOT), check=True)
        else:
            raise


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def organize_dist_artifacts() -> list[Path]:
    """Copy and standardize packaged artifacts into dist/ directory."""
    print("\n=== Step 3: Organizing Release Distribution Artifacts ===")
    dist_dir = ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    ver = _version()

    collected_artifacts: list[Path] = []
    bundle_dir = SRC_TAURI / "target/release/bundle"

    if sys.platform == "darwin":
        # Create standard DMG using hdiutil on macOS
        app_path = bundle_dir / "macos/AutoCulling.app"
        if not app_path.exists():
            # Fallback to direct release target
            app_path = SRC_TAURI / "target/release/bundle/macos/AutoCulling.app"
        
        if app_path.exists():
            dst_dmg = dist_dir / f"AutoCulling_v{ver}_macos_arm64.dmg"
            if dst_dmg.exists():
                dst_dmg.unlink()
            cmd = ["hdiutil", "create", "-volname", "AutoCulling", "-srcfolder", str(app_path), "-ov", "-format", "UDZO", str(dst_dmg)]
            print(f"Creating DMG via hdiutil: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print(f"Output DMG: {dst_dmg} ({dst_dmg.stat().st_size / 1024 / 1024:.1f} MB)")
            collected_artifacts.append(dst_dmg)
        else:
            print("WARNING: AutoCulling.app not found to create DMG.")

    elif sys.platform == "win32":
        # 1. NSIS Setup Exe
        nsis_candidates = list((bundle_dir / "nsis").glob("*.exe"))
        if nsis_candidates:
            src_exe = nsis_candidates[0]
            dst_exe = dist_dir / f"AutoCulling_v{ver}_win_x64_setup.exe"
            shutil.copy2(src_exe, dst_exe)
            print(f"Output Setup EXE: {dst_exe} ({dst_exe.stat().st_size / 1024 / 1024:.1f} MB)")
            collected_artifacts.append(dst_exe)

        # 2. Portable ZIP
        release_exe = SRC_TAURI / "target/release/AutoCulling.exe"
        if release_exe.exists():
            portable_zip = dist_dir / f"AutoCulling_v{ver}_win_x64_portable.zip"
            with zipfile.ZipFile(portable_zip, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(release_exe, "AutoCulling.exe")
                # Include sidecar if exists in release dir
                for f in (SRC_TAURI / "target/release").glob("cull-sidecar*.exe"):
                    z.write(f, f.name)
            print(f"Output Portable ZIP: {portable_zip} ({portable_zip.stat().st_size / 1024 / 1024:.1f} MB)")
            collected_artifacts.append(portable_zip)

    # Generate SHA256 checksum files
    for artifact in collected_artifacts:
        sha = sha256_file(artifact)
        sha_file = dist_dir / f"{artifact.name}.sha256"
        sha_file.write_text(f"{sha}  {artifact.name}\n", encoding="utf-8")
        print(f"SHA256: {sha_file.name} -> {sha}")

    return collected_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-sidecar", action="store_true", help="Skip compiling sidecar binary")
    args = parser.parse_args()

    try:
        if not args.skip_sidecar:
            build_sidecar_binary()
        build_tauri_gui()
        organize_dist_artifacts()
        print("\nAll GUI Packaging steps completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Build command failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
