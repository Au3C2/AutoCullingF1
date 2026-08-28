"""
tests/test_precision_heif.py — HEIF score-precision gate.

Now reads its expectation (24 HEIF stills) from the deterministic truth
(tests/baselines/deterministic.json, heif section) so it cannot diverge
from the cross-platform gate.  Kept as a focused wrapper around the
shared (jpg/heif/arw/nef) deterministic gate for `pytest -k heif`
selectivity.  Guards optimization #1 (ffmpeg -vf scale / decode-path
changes). Skipped when the source dataset is absent.
"""
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent))

from score_gate import run_cull_on_copies, assert_scores_match  # noqa: E402
from conftest import DET_RAW_TOL, PLATFORM, baseline_section, SOURCE_DIRS  # noqa: E402

HEIF_DIR = SOURCE_DIRS["heif"]


def _src_files() -> list[Path]:
    if not HEIF_DIR.is_dir():
        return []
    # Dict insertion order of deterministic.json heif section matches gate order
    from conftest import load_deterministic_baseline  # noqa: E402
    bl = load_deterministic_baseline()
    return [HEIF_DIR / name for name in bl.get("heif", {}) if (HEIF_DIR / name).exists()]


def _verify_srcs(files: list[Path]) -> list[Path]:
    if not files:
        pytest.skip("test_import dataset not present")
    from conftest import load_deterministic_baseline  # noqa: E402
    bl = load_deterministic_baseline()
    if len(files) < len(bl.get("heif", {})):
        pytest.skip(f"test_import incomplete: {len(files)}/{len(bl.get('heif', {}))} files")
    return files


@pytest.mark.precision
@pytest.mark.deterministic
def test_heif_rating_precision_matches_baseline(deterministic_env):
    """24 HEIF stills must match deterministic truth (rating exact, raw ±0.005)."""
    files = _verify_srcs(_src_files())
    run_files = files  # already in deterministic order
    actual = run_cull_on_copies(run_files)
    baseline = baseline_section("heif")
    assert_scores_match(actual, baseline, "heif", raw_tol=DET_RAW_TOL[PLATFORM])
