#!/usr/bin/env python3
"""
tests/test_cull_integration.py — Integration test for cull_photos.py.
Tests both ONNX and CoreML backends, label generation, and XMP accuracy.
"""

import os
import subprocess
import shutil
import tempfile
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

# Add project root for utilities if needed
sys.path.append(os.getcwd())

def get_xmp_data(xmp_path: Path):
    """Parse XMP and return rating, pick, and crop info."""
    if not xmp_path.exists():
        return None
    
    content = xmp_path.read_text()
    
    # Simple string search as XMP namespaces can be tricky for ET
    data = {}
    
    import re
    rating_match = re.search(r'xmp:Rating="([^"]+)"', content)
    pick_match = re.search(r'xmpDM:pick="([^"]+)"', content)
    has_crop_match = re.search(r'crs:HasCrop="([^"]+)"', content)
    
    if rating_match:
        data['rating'] = int(rating_match.group(1))
    if pick_match:
        data['pick'] = int(pick_match.group(1))
    if has_crop_match:
        data['has_crop'] = has_crop_match.group(1).lower() == "true"
        
    # Crop coordinates
    for attr in ['CropTop', 'CropLeft', 'CropBottom', 'CropRight']:
        match = re.search(rf'crs:{attr}="([^"]+)"', content)
        if match:
            data[attr.lower()] = float(match.group(1))
            
    return data

def run_test():
    print("=" * 70)
    print("CULL PHOTOS INTEGRATION TEST")
    print("=" * 70)
    
    test_img_src = Path("tests/test_img")
    if not test_img_src.exists():
        print(f"ERROR: {test_img_src} not found.")
        return 1
        
    results = []
    
    # Test Backends: ONNX (all) and CoreML (Mac only)
    import platform
    backends = ["onnx"]
    if platform.system() == "Darwin":
        backends.append("coreml")
        
    for backend in backends:
        print(f"\n>>> Running Test Case: Backend={backend}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Copy JPGs
            jpgs = list(test_img_src.glob("*.jpg"))
            if not jpgs:
                print("ERROR: No test images found.")
                return 1
                
            for f in jpgs:
                shutil.copy(f, tmp_path)
            
            # Run cull_photos.py
            env = os.environ.copy()
            env["CULL_BACKEND"] = backend
            env["PYTHONPATH"] = os.getcwd() # Project root
            
            cmd = [
                sys.executable, "cull_photos.py",
                "--input-dir", str(tmp_path),
                "--workers", "4"
            ]
            
            print(f"Executing: {' '.join(cmd)}")
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if proc.returncode != 0:
                print(f"FAIL: cull_photos.py exited with {proc.returncode}")
                print("STDOUT:", proc.stdout)
                print("STDERR:", proc.stderr)
                results.append((f"{backend}_execution", False))
                continue
            
            results.append((f"{backend}_execution", True))
            
            # 1. Verify Label Stats (5 keep from 14th, 1 reject from 15th)
            # 14th: IMG_20260314_* (5 images)
            # 15th: IMG_20260315_* (1 image)
            
            keeps = []
            rejects = []
            
            for f in jpgs:
                xmp_path = tmp_path / f.with_suffix(".xmp").name
                data = get_xmp_data(xmp_path)
                
                if data and data.get('rating', -1) > 0:
                    keeps.append(f.name)
                elif data and data.get('rating', -1) == -1:
                    rejects.append(f.name)
            
            print(f"  Result Stats: {len(keeps)} keep, {len(rejects)} reject")
            
            # Verify specific date logic
            ok_14th = all("20260314" in name for name in keeps) and len(keeps) == 5
            ok_15th = len(rejects) == 1 and "20260315" in rejects[0]
            
            results.append((f"{backend}_labels_14th", ok_14th))
            results.append((f"{backend}_labels_15th", ok_15th))
            
            if not ok_14th or not ok_15th:
                print(f"  FAIL: Expected 5 keeps from 14th and 1 reject from 15th.")
                print(f"  Actual keeps: {keeps}")
                print(f"  Actual rejects: {rejects}")
            else:
                print(f"  PASS: Labels correctly assigned by date.")

            # 2. Golden Validation (Compare generated XMP with golden)
            # Only for IMG_20260314_151744_020.jpg
            target_name = "IMG_20260314_151744_020"
            generated_xmp = tmp_path / f"{target_name}.xmp"
            golden_xmp = test_img_src / f"{target_name}_golden.xmp"
            
            if generated_xmp.exists() and golden_xmp.exists():
                gen_data = get_xmp_data(generated_xmp)
                gold_data = get_xmp_data(golden_xmp)
                
                # Compare Rating
                rating_ok = gen_data.get('rating') == gold_data.get('rating')
                results.append((f"{backend}_golden_rating", rating_ok))
                
                # Compare Crop existence
                crop_exist_ok = gen_data.get('has_crop') == gold_data.get('has_crop')
                results.append((f"{backend}_golden_crop_exists", crop_exist_ok))
                
                # Compare Crop coords (loosely, as we might have minor floating point diffs)
                if rating_ok and crop_exist_ok and gen_data.get('has_crop'):
                    coords_ok = True
                    for coord in ['croptop', 'cropleft', 'cropbottom', 'cropright']:
                        diff = abs(gen_data.get(coord, 0) - gold_data.get(coord, 0))
                        if diff > 0.05: # 5% tolerance for logic improvements
                            coords_ok = False
                            print(f"  FAIL: Crop coord {coord} diff too large: gen={gen_data.get(coord)}, gold={gold_data.get(coord)}")
                    results.append((f"{backend}_golden_crop_coords", coords_ok))
                
                print(f"  Golden Comparison for {target_name}: Rating={gen_data.get('rating')}, Crop={gen_data.get('has_crop')}")

    # Final Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"{name:40s}: {status}")
        
    print(f"\nScore: {passed}/{total} passed")
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(run_test())
