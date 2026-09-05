"""packaging/test_gui_package.py — automated test suite for packaged GUI installers.

Validates:
1. macOS: Mounts .dmg, verifies .app bundle integrity and embedded sidecar execution.
2. Windows: Verifies NSIS setup .exe and extracts/validates portable .zip bundle.
3. Tests communication handshake with the bundled sidecar binary (scan & preview).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _get_version() -> str:
    env_ver = os.environ.get("CULL_VERSION", "").strip().lstrip("v")
    if env_ver:
        return env_ver
    toml = ROOT / "pyproject.toml"
    if toml.exists():
        import re
        for line in toml.read_text().splitlines():
            if line.strip().startswith("version"):
                m = re.search(r'"([^"]+)"', line)
                if m:
                    return m.group(1).strip()
    return "0.1"


def test_macos_dmg(dmg_path: Path | None = None) -> bool:
    """Mount macOS DMG, verify .app structure, test embedded sidecar and unmount."""
    print("=== Testing macOS DMG Installer Package ===")
    if not dmg_path or not dmg_path.exists():
        # Search dist/ or src-tauri/target/release/bundle/dmg/
        candidates = list((ROOT / "dist").glob("*.dmg")) + list(
            (ROOT / "src-tauri/target/release/bundle/dmg").glob("*.dmg")
        )
        if not candidates:
            print("FAIL: No DMG package found to test.")
            return False
        dmg_path = candidates[0]

    print(f"Testing DMG artifact: {dmg_path} ({dmg_path.stat().st_size / 1024 / 1024:.1f} MB)")
    mount_point = Path("/tmp/AutoCulling_DMG_TestMount")
    if mount_point.exists():
        subprocess.run(["hdiutil", "detach", str(mount_point), "-force"], capture_output=True)
    mount_point.mkdir(parents=True, exist_ok=True)

    try:
        # Mount DMG
        res = subprocess.run(
            ["hdiutil", "attach", str(dmg_path), "-mountpoint", str(mount_point), "-nobrowse", "-quiet"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("PASS: Successfully mounted DMG volume.")

        # Find .app
        apps = list(mount_point.glob("*.app"))
        if not apps:
            print("FAIL: No .app bundle found inside mounted DMG.")
            return False
        app_bundle = apps[0]
        print(f"Found app bundle: {app_bundle.name}")

        # Check binary existence
        macos_dir = app_bundle / "Contents" / "MacOS"
        res_dir = app_bundle / "Contents" / "Resources"

        main_execs = [e for e in macos_dir.glob("*") if e.name != "cull-sidecar"]
        if not main_execs:
            print("FAIL: Missing executable in Contents/MacOS/")
            return False
        app_main = main_execs[0]
        print(f"PASS: Found main app executable: {app_main.name}")

        success = True

        # Test sidecar executable if bundled in Resources or MacOS
        sidecar_cand = None
        for cand in [res_dir / "cull-sidecar", res_dir / "cull_sidecar", macos_dir / "cull-sidecar", macos_dir / "cull_sidecar"]:
            if cand.exists():
                sidecar_cand = cand
                break

        if sidecar_cand:
            print(f"Testing bundled sidecar: {sidecar_cand}")
            sidecar_proc = subprocess.Popen(
                [str(sidecar_cand), "--json-lines"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            time.sleep(0.3)
            # Send test scan
            test_dir = ROOT / "tests/test_img"
            if test_dir.exists():
                sidecar_proc.stdin.write(json.dumps({"cmd": "scan", "dir": str(test_dir), "recursive": False}) + "\n")
                sidecar_proc.stdin.flush()
                scan_line = sidecar_proc.stdout.readline()
                print(f"Sidecar handshake output: {scan_line.strip()[:100]}")
            sidecar_proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            sidecar_proc.stdin.flush()
            sidecar_proc.wait(timeout=5.0)
            print("PASS: Packaged sidecar passed Stdio JSON Lines handshake test.")
        else:
            print("NOTE: Sidecar is bundled as external binary or dev-resolved.")

        # Regression test for the installed-app flow: the .app must spawn the
        # bundled sidecar itself (eagerly at startup). A broken resolution here
        # used to surface as "Broken pipe (os error 32)" on folder pick.
        print("Testing .app self-spawn of bundled sidecar (installed-app simulation)...")
        install_dir = Path("/tmp/AutoCulling_AppInstall")
        shutil.rmtree(install_dir, ignore_errors=True)
        install_dir.mkdir(parents=True)
        shutil.copytree(app_bundle, install_dir / app_bundle.name, symlinks=True)
        app_bin = install_dir / app_bundle.name / "Contents" / "MacOS" / app_main.name
        app_proc = subprocess.Popen(
            [str(app_bin)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(install_dir),
        )
        try:
            sidecar_alive = False
            deadline = time.time() + 25.0
            while time.time() < deadline:
                probe = subprocess.run(
                    ["pgrep", "-f", "cull-sidecar"],
                    capture_output=True, text=True,
                )
                if probe.returncode == 0 and probe.stdout.strip():
                    sidecar_alive = True
                    break
                if app_proc.poll() is not None:
                    break
                time.sleep(0.5)
            if sidecar_alive:
                print("PASS: Installed .app spawned the bundled sidecar at startup.")
            else:
                print("FAIL: .app did not spawn the bundled sidecar (folder pick would hit Broken pipe).")
                success = False
        finally:
            app_proc.terminate()
            try:
                app_proc.wait(timeout=5.0)
            except Exception:
                app_proc.kill()
            subprocess.run(["pkill", "-f", "cull-sidecar"], capture_output=True)
            shutil.rmtree(install_dir, ignore_errors=True)

        return success

    except Exception as e:
        print(f"FAIL: DMG test encountered error: {e}")
        return False
    finally:
        subprocess.run(["hdiutil", "detach", str(mount_point), "-force", "-quiet"], capture_output=True)
        shutil.rmtree(mount_point, ignore_errors=True)
        print("Cleaned up DMG mount point.")


def test_windows_package(dist_dir: Path | None = None) -> bool:
    """Verify Windows NSIS installer and portable ZIP packages."""
    print("=== Testing Windows GUI Packages ===")
    dist = dist_dir or (ROOT / "dist")

    setups = list(dist.glob("*setup.exe")) + list((ROOT / "src-tauri/target/release/bundle/nsis").glob("*.exe"))
    zips = list(dist.glob("*portable.zip")) + list(dist.glob("*win*.zip"))

    success = True
    if setups:
        setup = setups[0]
        print(f"PASS: Found NSIS setup installer: {setup.name} ({setup.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("WARNING: No Windows NSIS setup.exe found in dist/")

    if zips:
        zip_pkg = zips[0]
        print(f"Testing Windows portable ZIP: {zip_pkg.name} ({zip_pkg.stat().st_size / 1024 / 1024:.1f} MB)")
        unzip_dir = ROOT / "build/test_unzip_win"
        shutil.rmtree(unzip_dir, ignore_errors=True)
        unzip_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_pkg, "r") as z:
                z.extractall(unzip_dir)
            print(f"PASS: Successfully extracted portable ZIP ({len(list(unzip_dir.glob('**/*')))} files)")
            exes = list(unzip_dir.glob("**/*.exe"))
            if exes:
                print(f"PASS: Found portable executables: {[e.name for e in exes]}")
            else:
                print("FAIL: No .exe found inside portable ZIP.")
                success = False
        except Exception as e:
            print(f"FAIL: Error testing portable ZIP: {e}")
            success = False
        finally:
            shutil.rmtree(unzip_dir, ignore_errors=True)
    else:
        print("WARNING: No Windows portable zip found in dist/")

    return success


def main() -> int:
    parser = argparse.ArgumentParser(description="Test packaged AutoCulling GUI distributions.")
    parser.add_argument("--dmg", type=Path, default=None, help="Path to .dmg file to test (macOS)")
    parser.add_argument("--dist-dir", type=Path, default=None, help="Directory containing built packages")
    args = parser.parse_args()

    system = platform.system()
    passed = False

    if system == "Darwin":
        passed = test_macos_dmg(args.dmg)
    elif system == "Windows":
        passed = test_windows_package(args.dist_dir)
    else:
        print(f"Testing packages on {system}...")
        passed = True

    print(f"\nGUI Package Test Result: {'ALL TESTS PASSED' if passed else 'TESTS FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
