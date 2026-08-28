"""
tests/test_precision_raw.py — RAW score-precision gates (ARW / NEF).

Expectations are read from the deterministic truth
(tests/baselines/deterministic.json, arw/nef sections) so the old
hard-coded ARW/NEF dictionaries cannot diverge from it.  Kept as focused
wrappers for `pytest -k arw/nef` selectivity.  Guards optimization #4
(RAW batch extraction via exiftool -stay_open). Skipped when datasets absent.
"""
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent))

from score_gate import run_cull_on_copies, assert_scores_match  # noqa: E402
from conftest import baseline_section, SOURCE_DIRS  # noqa: E402

ARW_DIR = SOURCE_DIRS["arw"]
NEF_DIR = SOURCE_DIRS["nef"]


@pytest.mark.precision
@pytest.mark.deterministic
def test_arw_precision_matches_baseline(deterministic_env):
    if not ARW_DIR.is_dir():
        pytest.skip("test_arw dataset not present")
    baseline = baseline_section("arw")
    src = [ARW_DIR / n for n in baseline]
    if any(not p.exists() for p in src):
        pytest.skip("test_arw incomplete")
    actual = run_cull_on_copies(src)
    assert_scores_match(actual, baseline, "arw")


@pytest.mark.precision
@pytest.mark.deterministic
def test_nef_precision_matches_baseline(deterministic_env):
    if not NEF_DIR.is_dir():
        pytest.skip("test_nef dataset not present")
    src = [NEF_DIR / n for n in baseline_section("nef")]
    if any(not p.exists() for p in src):
        pytest.skip("test_nef incomplete")
    actual = run_cull_on_copies(src)
    assert_scores_match(actual, baseline_section("nef"), "nef")
