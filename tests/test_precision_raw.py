"""
tests/test_precision_raw.py — RAW score-precision gates (ARW / NEF).

Locks per-file rating/raw_score for 20 Sony ARW + 20 Nikon NEF stills.
Guards optimization #4 (RAW batch extraction via exiftool `-stay_open`): any
decode-path change must keep scoring identical. Skipped when datasets absent.
"""
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent))

from score_gate import run_cull_on_copies, assert_scores_match  # noqa: E402

ARW_DIR = Path("test_arw")
NEF_DIR = Path("test_nef")

# Baseline locked 2026-08-22 on the CURRENT pipeline (cv2.INTER_AREA decode
# resize + optimized cv2.dft sharpness).
ARW_BASELINE = {
    "DSC00827.ARW": (-1, 2.431),
    "DSC00845.ARW": (2, 3.295),
    "DSC00849.ARW": (3, 3.513),
    "DSC00851.ARW": (3, 3.727),
    "DSC00879.ARW": (-1, 2.05),
    "DSC00880.ARW": (-1, 1.19),
    "DSC00886.ARW": (-1, 2.263),
    "DSC00887.ARW": (-1, 1.858),
    "DSC00888.ARW": (-1, 2.502),
    "DSC00890.ARW": (-1, 1.339),
    "DSC00892.ARW": (-1, 2.162),
    "DSC00893.ARW": (-1, 3.027),
    "DSC00894.ARW": (-1, 0.665),
    "DSC00895.ARW": (-1, 1.841),
    "DSC00896.ARW": (-1, 0.86),
    "DSC00897.ARW": (-1, 1.277),
    "DSC00942.ARW": (-1, 2.984),
    "DSC00951.ARW": (-1, 1.367),
    "DSC00952.ARW": (-1, 2.626),
    "DSC00958.ARW": (-1, 1.539),
}

NEF_BASELINE = {
    "IMG_20260315_164102_480.nef": (-1, 2.188),
    "IMG_20260315_164102_540.nef": (-1, 1.711),
    "IMG_20260315_164102_600.nef": (-1, 1.882),
    "IMG_20260315_164102_660.nef": (-1, 1.524),
    "IMG_20260315_164102_730.nef": (-1, 1.516),
    "IMG_20260315_164102_790.nef": (-1, 1.287),
    "IMG_20260315_164133_610.nef": (-1, 2.556),
    "IMG_20260315_164133_680.nef": (-1, 1.883),
    "IMG_20260315_164133_750.nef": (-1, 1.921),
    "IMG_20260315_164133_810.nef": (-1, 1.838),
    "IMG_20260315_164133_870.nef": (-1, 2.409),
    "IMG_20260315_164133_930.nef": (-1, 1.511),
    "IMG_20260315_164133_990.nef": (-1, 1.459),
    "IMG_20260315_164134_050.nef": (-1, 1.481),
    "IMG_20260315_164134_110.nef": (-1, 0.932),
    "IMG_20260315_164136_090.nef": (-1, 2.438),
    "IMG_20260315_164136_160.nef": (-1, 2.943),
    "IMG_20260315_164136_220.nef": (-1, 2.501),
    "IMG_20260315_164136_280.nef": (3, 3.501),
    "IMG_20260315_164136_340.nef": (2, 3.191),
}


def test_arw_precision_matches_baseline():
    if not ARW_DIR.is_dir():
        pytest.skip("test_arw dataset not present")
    src = [ARW_DIR / n for n in ARW_BASELINE]
    if any(not p.exists() for p in src):
        pytest.skip("test_arw incomplete")
    actual = run_cull_on_copies(src)
    assert_scores_match(actual, ARW_BASELINE, "arw")


def test_nef_precision_matches_baseline():
    if not NEF_DIR.is_dir():
        pytest.skip("test_nef dataset not present")
    src = [NEF_DIR / n for n in NEF_BASELINE]
    if any(not p.exists() for p in src):
        pytest.skip("test_nef incomplete")
    actual = run_cull_on_copies(src)
    assert_scores_match(actual, NEF_BASELINE, "nef")