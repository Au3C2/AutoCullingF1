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

# Baseline locked 2026-08-22 (master@CUDA, exiftool -JpgFromRaw / -PreviewImage).
ARW_BASELINE = {
    "DSC00827.ARW": (-1, 2.436),
    "DSC00845.ARW": (-1, 3.037),
    "DSC00849.ARW": (2, 3.238),
    "DSC00851.ARW": (3, 3.47),
    "DSC00879.ARW": (-1, 1.941),
    "DSC00880.ARW": (-1, 1.175),
    "DSC00886.ARW": (-1, 1.985),
    "DSC00887.ARW": (-1, 1.749),
    "DSC00888.ARW": (-1, 2.501),
    "DSC00890.ARW": (-1, 1.262),
    "DSC00892.ARW": (-1, 2.676),
    "DSC00893.ARW": (-1, 2.893),
    "DSC00894.ARW": (-1, 0.588),
    "DSC00895.ARW": (-1, 1.745),
    "DSC00896.ARW": (-1, 0.77),
    "DSC00897.ARW": (-1, 1.179),
    "DSC00942.ARW": (-1, 2.971),
    "DSC00951.ARW": (-1, 1.232),
    "DSC00952.ARW": (-1, 2.373),
    "DSC00958.ARW": (-1, 1.142),
}

NEF_BASELINE = {
    "IMG_20260315_164102_480.nef": (-1, 2.165),
    "IMG_20260315_164102_540.nef": (-1, 1.697),
    "IMG_20260315_164102_600.nef": (-1, 1.87),
    "IMG_20260315_164102_660.nef": (-1, 1.491),
    "IMG_20260315_164102_730.nef": (-1, 1.235),
    "IMG_20260315_164102_790.nef": (-1, 0.899),
    "IMG_20260315_164133_610.nef": (-1, 2.482),
    "IMG_20260315_164133_680.nef": (-1, 1.817),
    "IMG_20260315_164133_750.nef": (-1, 1.849),
    "IMG_20260315_164133_810.nef": (-1, 1.79),
    "IMG_20260315_164133_870.nef": (-1, 2.372),
    "IMG_20260315_164133_930.nef": (-1, 1.455),
    "IMG_20260315_164133_990.nef": (-1, 1.405),
    "IMG_20260315_164134_050.nef": (-1, 1.419),
    "IMG_20260315_164134_110.nef": (-1, 0.899),
    "IMG_20260315_164136_090.nef": (-1, 2.342),
    "IMG_20260315_164136_160.nef": (-1, 2.786),
    "IMG_20260315_164136_220.nef": (-1, 2.951),
    "IMG_20260315_164136_280.nef": (2, 3.32),
    "IMG_20260315_164136_340.nef": (-1, 2.989),
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