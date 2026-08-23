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

# Baseline locked 2026-08-23 on the CURRENT pipeline (libjpeg draft decode +
# cv2.INTER_AREA resize + cv2 letterbox/cv2 P4-ROI + cv2.dft sharpness with
# cv2 gray/Laplacian internals + P4 v2 model). All keep/reject decisions
# unchanged; raw records drift <=0.016 (gray-conversion LSB rounding).
ARW_BASELINE = {
    "DSC00827.ARW": (-1, 2.397),
    "DSC00845.ARW": (2, 3.178),
    "DSC00849.ARW": (3, 3.424),
    "DSC00851.ARW": (3, 3.726),
    "DSC00879.ARW": (-1, 2.02),
    "DSC00880.ARW": (-1, 1.186),
    "DSC00886.ARW": (-1, 2.217),
    "DSC00887.ARW": (-1, 1.891),
    "DSC00888.ARW": (-1, 2.508),
    "DSC00890.ARW": (-1, 1.341),
    "DSC00892.ARW": (-1, 2.178),
    "DSC00893.ARW": (-1, 3.025),
    "DSC00894.ARW": (-1, 0.652),
    "DSC00895.ARW": (-1, 1.807),
    "DSC00896.ARW": (-1, 2.131),
    "DSC00897.ARW": (-1, 1.255),
    "DSC00942.ARW": (-1, 2.985),
    "DSC00951.ARW": (-1, 0.463),
    "DSC00952.ARW": (-1, 2.673),
    "DSC00958.ARW": (-1, 2.495),
}

NEF_BASELINE = {
    "IMG_20260315_164102_480.nef": (-1, 2.167),
    "IMG_20260315_164102_540.nef": (-1, 2.305),
    "IMG_20260315_164102_600.nef": (-1, 2.48),
    "IMG_20260315_164102_660.nef": (-1, 2.226),
    "IMG_20260315_164102_730.nef": (-1, 1.276),
    "IMG_20260315_164102_790.nef": (-1, 0.926),
    "IMG_20260315_164133_610.nef": (-1, 2.55),
    "IMG_20260315_164133_680.nef": (-1, 2.463),
    "IMG_20260315_164133_750.nef": (-1, 2.505),
    "IMG_20260315_164133_810.nef": (-1, 2.426),
    "IMG_20260315_164133_870.nef": (-1, 2.398),
    "IMG_20260315_164133_930.nef": (-1, 1.491),
    "IMG_20260315_164133_990.nef": (-1, 1.439),
    "IMG_20260315_164134_050.nef": (-1, 1.475),
    "IMG_20260315_164134_110.nef": (-1, 0.922),
    "IMG_20260315_164136_090.nef": (-1, 2.404),
    "IMG_20260315_164136_160.nef": (-1, 2.89),
    "IMG_20260315_164136_220.nef": (-1, 3.072),
    "IMG_20260315_164136_280.nef": (3, 3.434),
    "IMG_20260315_164136_340.nef": (2, 3.143),
}

NEF_BASELINE = {
    "IMG_20260315_164102_480.nef": (-1, 2.167),
    "IMG_20260315_164102_540.nef": (-1, 2.305),
    "IMG_20260315_164102_600.nef": (-1, 2.48),
    "IMG_20260315_164102_660.nef": (-1, 2.226),
    "IMG_20260315_164102_730.nef": (-1, 1.276),
    "IMG_20260315_164102_790.nef": (-1, 0.926),
    "IMG_20260315_164133_610.nef": (-1, 2.55),
    "IMG_20260315_164133_680.nef": (-1, 2.463),
    "IMG_20260315_164133_750.nef": (-1, 2.505),
    "IMG_20260315_164133_810.nef": (-1, 2.426),
    "IMG_20260315_164133_870.nef": (-1, 2.398),
    "IMG_20260315_164133_930.nef": (-1, 1.491),
    "IMG_20260315_164133_990.nef": (-1, 1.439),
    "IMG_20260315_164134_050.nef": (-1, 1.475),
    "IMG_20260315_164134_110.nef": (-1, 0.922),
    "IMG_20260315_164136_090.nef": (-1, 2.404),
    "IMG_20260315_164136_160.nef": (-1, 2.89),
    "IMG_20260315_164136_220.nef": (-1, 3.072),
    "IMG_20260315_164136_280.nef": (3, 3.434),
    "IMG_20260315_164136_340.nef": (2, 3.143),
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