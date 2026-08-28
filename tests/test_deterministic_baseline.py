"""
tests/test_deterministic_baseline.py — Deterministic truth + backend alignment.

The deterministic artifact (tests/baselines/deterministic.json) is the
cross-platform truth.  Four per-format sections (jpg/heif/arw/nef) cover
70 files.  Two complementary gates sit on the same truth:

  test_deterministic_matches_baseline — CULL_DETERMINISTIC=1 run must equal
      the truth exactly (rating strict + raw <=0.005)
  test_backends_align_to_deterministic — default backends (CUDA, CoreML, …)
      align to the truth (rating strict + raw <=0.03 envelope from measured
      CPU-vs-CUDA / decode LSB drift).

The focused HEIF/RAW/jpg wrappers (test_precision_heif.py etc.) are kept
for -k selectivity but now also read through conftest helpers so they
cannot diverge from this file.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent))

from score_gate import assert_scores_match, run_cull_on_copies  # noqa: E402
from conftest import baseline_section, SOURCE_DIRS, ALIGN_RAW_TOL  # noqa: E402


def _src_files(key: str) -> list[Path]:
    section = baseline_section(key)
    d = SOURCE_DIRS[key]
    if not d.is_dir():
        pytest.skip(f"source dir missing: {d}")
    files = [d / name for name in section]
    missing = [p for p in files if not p.exists()]
    if missing:
        pytest.skip(f"{key}: {len(missing)} files missing (e.g. {missing[0].name})")
    return files


def _assert_align(actual: dict, baseline: dict, what: str, raw_tol: float) -> None:
    assert set(actual) == set(baseline), \
        f"{what}: file sets differ — missing={set(baseline)-set(actual)} extra={set(actual)-set(baseline)}"
    for name, (exp_rating, exp_raw) in baseline.items():
        act_rating, act_raw, veto, ndet = actual[name]
        assert act_rating == exp_rating, \
            f"{what}::{name}: rating {act_rating} != deterministic truth {exp_rating} (raw {act_raw} vs {exp_raw}, veto={veto!r})"
        assert abs(act_raw - exp_raw) <= raw_tol, \
            f"{what}::{name}: raw_score {act_raw} drifted from deterministic truth {exp_raw} by {abs(act_raw-exp_raw):.4f} > {raw_tol}"


@pytest.mark.precision
@pytest.mark.deterministic
@pytest.mark.parametrize("key", ["jpg", "heif", "arw", "nef"])
def test_deterministic_matches_baseline(key, deterministic_env):
    """Deterministic run (CULL_DETERMINISTIC=1) must equal the committed truth."""
    src_files = _src_files(key)
    baseline = baseline_section(key)
    actual = run_cull_on_copies(src_files, workers=4)
    assert_scores_match(actual, baseline, f"deterministic/{key}")


@pytest.mark.precision
@pytest.mark.alignment
@pytest.mark.parametrize("key", ["jpg", "heif", "arw", "nef"])
def test_backends_align_to_deterministic(key, nondeterministic_env):
    """Default backends must align to deterministic truth (rating strict + raw envelope)."""
    src_files = _src_files(key)
    baseline = baseline_section(key)
    actual = run_cull_on_copies(src_files, workers=4)
    _assert_align(actual, baseline, f"align/{key}", ALIGN_RAW_TOL)
