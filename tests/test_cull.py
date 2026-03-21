"""
tests/test_cull.py — pytest suite for F1 photo culling pipeline.
"""

import os
import subprocess
import shutil
import tempfile
import sys
import re
from pathlib import Path
import pytest

# Add project root to sys.path for any direct module imports in tests
sys.path.append(os.getcwd())

def get_xmp_data(xmp_path: Path):
    """Parse XMP and return rating, pick, and crop info using regex (namespace independent)."""
    if not xmp_path.exists():
        return {}
    
    content = xmp_path.read_text(encoding="utf-8")
    data = {}
    
    # Rating/Pick
    rating = re.search(r'xmp:Rating="([^"]+)"', content)
    pick = re.search(r'xmpDM:pick="([^"]+)"', content)
    
    if rating: data['rating'] = int(rating.group(1))
    if pick: data['pick'] = int(pick.group(1))
    
    # Crop
    has_crop = re.search(r'crs:HasCrop="([^"]+)"', content)
    if has_crop:
        data['has_crop'] = has_crop.group(1).lower() == "true"
        for attr in ['CropTop', 'CropLeft', 'CropBottom', 'CropRight']:
            match = re.search(rf'crs:{attr}="([^"]+)"', content)
            if match:
                data[attr.lower()] = float(match.group(1))
                
    return data

@pytest.fixture
def test_env(tmp_path):
    """Fixture to set up a clean test environment with sample images."""
    src_dir = Path("tests/test_img")
    for f in src_dir.glob("*.jpg"):
        shutil.copy(f, tmp_path)
    return tmp_path

def run_cull(input_dir: Path, backend: str):
    """Helper to run the cull_photos script."""
    env = os.environ.copy()
    env["CULL_BACKEND"] = backend
    env["PYTHONPATH"] = os.getcwd()
    
    cmd = [
        sys.executable, "cull_photos.py",
        "--input-dir", str(input_dir),
        "--workers", "4",
        "--force"
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)

@pytest.mark.parametrize("backend", ["onnx"] + (["coreml"] if sys.platform == "darwin" else []))
def test_cull_execution(test_env, backend):
    """Test that the script executes successfully and produces XMP files."""
    proc = run_cull(test_env, backend)
    assert proc.returncode == 0, f"Script failed with stderr: {proc.stderr}"
    
    # Verify XMP creation
    jpgs = list(test_env.glob("*.jpg"))
    xmps = list(test_env.glob("*.xmp"))
    assert len(xmps) == len(jpgs), f"Expected {len(jpgs)} XMPs, found {len(xmps)}"

def test_labels_correctness(test_env):
    """Verify that images from 14th are kept (Rating > 0) and 15th is rejected (Rating -1)."""
    # Use ONNX for label check as it's cross-platform
    run_cull(test_env, "onnx")
    
    jpgs = list(test_env.glob("*.jpg"))
    for img in jpgs:
        xmp_path = img.with_suffix(".xmp")
        data = get_xmp_data(xmp_path)
        
        if "20260314" in img.name:
            assert data.get('rating', -1) > 0, f"Image {img.name} should be KEPT (Rating > 0)"
        elif "20260315" in img.name:
            assert data.get('rating', -1) == -1, f"Image {img.name} should be REJECTED (Rating -1)"

def test_golden_xmp_comparison(test_env):
    """Compare generated XMP for a specific image against a golden version."""
    run_cull(test_env, "onnx")
    
    target_stem = "IMG_20260314_151744_020"
    generated_xmp = test_env / f"{target_stem}.xmp"
    golden_xmp = Path("tests/test_img") / f"{target_stem}_golden.xmp"
    
    assert generated_xmp.exists()
    assert golden_xmp.exists()
    
    gen = get_xmp_data(generated_xmp)
    gold = get_xmp_data(golden_xmp)
    
    # Core fields
    assert gen.get('rating') == gold.get('rating')
    assert gen.get('has_crop') == gold.get('has_crop')
    
    # Crop coordinates (within 5% tolerance for logic evolution)
    if gen.get('has_crop'):
        for coord in ['croptop', 'cropleft', 'cropbottom', 'cropright']:
            assert abs(gen.get(coord, 0) - gold.get(coord, 0)) < 0.05
