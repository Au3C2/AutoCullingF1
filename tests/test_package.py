import os
import sys
import platform
import subprocess
import shutil
import tempfile
import pytest
from pathlib import Path
import csv

# Standard executable names based on OS
EXE_NAME_MAP = {
    "Darwin": "auto_cull_v0.1_macos_arm64",
    "Windows": "auto_cull_v0.1_win_x64.exe"
}

def get_executable() -> Path:
    """Find the executable in the project root."""
    root = Path(__file__).parent.parent
    system = platform.system()
    
    # Try the specific name first (as renamed in previous steps)
    expected_name = EXE_NAME_MAP.get(system)
    if expected_name:
        exe_path = root / expected_name
        if exe_path.exists():
            return exe_path
            
    # Fallback: search for anything starting with auto_cull and having execute permission
    for p in root.glob("auto_cull*"):
        if p.is_file() and os.access(p, os.X_OK):
            if system == "Windows" and p.suffix.lower() != ".exe":
                continue
            return p
            
    pytest.skip(f"Executable not found for system: {system}")
    raise FileNotFoundError(f"Binary not found: {system}")

# Golden Baseline generated from tests/test_img (v0.1 logic)
BASELINE = {
    "IMG_20260314_151744_020.jpg": 3,
    "IMG_20260314_160317_680.jpg": 2,
    "IMG_20260314_160318_240.jpg": -1,
    "IMG_20260314_160343_870.jpg": 3,
    "IMG_20260314_160344_380.jpg": 3,
    "IMG_20260315_150404_550.jpg": -1
}

def test_packaged_executable_precision():
    """ Verify the packaged binary produces same results as the source code baseline. """
    exe_path = get_executable()
    test_img_src = Path(__file__).parent / "test_img"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_path = tmp_path / "test_results.csv"
        
        # 1. Prepare Test Data
        for img_name in BASELINE.keys():
            src_file = test_img_src / img_name
            if src_file.exists():
                shutil.copy(src_file, tmp_path)
            else:
                pytest.fail(f"Test image {img_name} missing from {test_img_src}. Ensure images are added to the repo.")
        
        # 2. Run Packaged Binary
        # We use --dump-scores to verify ratings since internal metadata 
        # is harder to read without external tools in this test.
        # We force (-f) to ignore any existing metadata.
        cmd = [
            str(exe_path),
            "--input-dir", str(tmp_path),
            "--dump-scores", str(csv_path),
            "-f",
            "--workers", "1"
        ]
        
        print(f"\nRunning binary test: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            pytest.fail(f"Binary execution failed!\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")

        # 3. Verify Baseline
        assert csv_path.exists(), "Binary did not generate results CSV"
        
        actual_results = {}
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                actual_results[row['filename']] = int(row['rating'])
        
        # Compare
        mismatches = []
        for img_name, expected in BASELINE.items():
            actual = actual_results.get(img_name)
            if actual != expected:
                mismatches.append(f"{img_name}: expected {expected}, got {actual}")
        
        if mismatches:
            pytest.fail("Precision mismatch between binary and baseline:\n" + "\n".join(mismatches))
        else:
            print(f"PASS: All {len(BASELINE)} test images matched baseline.")

if __name__ == "__main__":
    # Allow running this script directly
    sys.exit(pytest.main([__file__]))
