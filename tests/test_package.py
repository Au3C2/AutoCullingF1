import os
import sys
import platform
import subprocess
import shutil
import tempfile
import pytest
from pathlib import Path
import csv

def _ver() -> str:
    v = os.environ.get("CULL_VERSION", "").strip().lstrip("v")
    if v:
        return v
    toml = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if toml.exists():
        import re as _re
        for line in toml.read_text().splitlines():
            if line.strip().startswith("version"):
                m = _re.search(r'"([^"]+)"', line)
                if m:
                    return m.group(1).strip()
    return "0.1"

EXE_NAME_MAP = {
    "Darwin": f"auto_cull_v{_ver()}_macos_arm64",
    "Windows": f"auto_cull_v{_ver()}_win_x64.exe"
}

def get_executable() -> Path:
    """Find the executable in the project root.

    CULL_EXE env var overrides the search (used by packaging/build.py to
    validate a freshly built binary without replacing the root artifact).
    """
    override = os.environ.get("CULL_EXE")
    if override:
        exe_path = Path(override)
        if exe_path.is_file() and os.access(exe_path, os.X_OK):
            return exe_path
        pytest.skip(f"CULL_EXE set but not executable: {override}")
    root = Path(__file__).parent.parent
    system = platform.system()
    
    expected_name = EXE_NAME_MAP.get(system)
    if expected_name:
        exe_path = root / expected_name
        if exe_path.exists():
            return exe_path
            
    for p in root.glob("auto_cull*"):
        if p.is_file() and os.access(p, os.X_OK):
            if system == "Windows" and p.suffix.lower() != ".exe":
                continue
            return p
            
    pytest.skip(f"Executable not found for system: {system}")
    raise FileNotFoundError(f"Binary not found: {system}")

@pytest.mark.packaged
@pytest.mark.precision
def test_packaged_executable_precision(deterministic_env):
    """ Packaged binary must reproduce the deterministic truth (tests/baselines/deterministic.json, jpg section)."""
    from conftest import baseline_jpg_ratings  # noqa: E402
    baseline = baseline_jpg_ratings()
    exe_path = get_executable()
    test_img_src = Path(__file__).parent / "test_img"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_path = tmp_path / "test_results.csv"
        
        for img_name in baseline.keys():
            src_file = test_img_src / img_name
            if src_file.exists():
                shutil.copy(src_file, tmp_path)
            else:
                pytest.fail(f"Test image {img_name} missing from {test_img_src}. Ensure images are added to the repo.")
        
        cmd = [
            str(exe_path),
            "--input-dir", str(tmp_path),
            "--dump-scores", str(csv_path),
            "-f",
            "--workers", "1"
        ]
        
        print(f"\nRunning binary test: {' '.join(cmd)}")
        env = os.environ.copy()
        env["CULL_DETERMINISTIC"] = "1"
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            pytest.fail(f"Binary execution failed!\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

        assert csv_path.exists(), "Binary did not generate results CSV"
        
        actual_results = {}
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                actual_results[row['filename']] = int(row['rating'])

        mismatches = []
        for img_name, expected in baseline.items():
            actual = actual_results.get(img_name)
            if actual != expected:
                mismatches.append(f"{img_name}: expected {expected}, got {actual}")

        if mismatches:
            pytest.fail("Precision mismatch between binary and deterministic truth:\n" + "\n".join(mismatches))
        else:
            print(f"PASS: All {len(baseline)} test images matched baseline.")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
