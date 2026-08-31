"""
tests/test_deterministic_baseline.py — Deterministic truth + backend alignment.

The deterministic artifact (tests/baselines/deterministic.json) carries
assertions with two DIFFERENT strictness levels (measured mac-vs-win
2026-08-28):

  RATING — the product-level contract: identical star/reject outcome on
      any platform/backend. The deterministic backend achieves 70/70
      ratings (macOS run vs the Windows-generated truth), so rating gates
      are STRICT everywhere.
  RAW — decode SIMD (NEON vs AVX) makes bit-identical raw_score across
      platforms impossible (JPG aligns; HEIF/ARW/NEF drift 0.013–0.039).
      Raw is asserted only as a platform-internal regression window
      (mac ±0.05, win ±0.005), never as cross-platform equality.

Doors (one shared run per format via _det_run cache):

  test_deterministic_ratings_cross_platform — CULL_DETERMINISTIC=1
      ratings == truth, strict. THE cross-platform gate.
  test_deterministic_raw_regression — CULL_DETERMINISTIC=1 raw within the
      platform window around the truth.
  test_backends_align_to_deterministic — default backends (CUDA, CoreML,
      …): rating strict minus the documented ANE knife-edge files
      (KNOWN_RATING_DIVERGENCE), raw within the platform alignment window
      (mac ±0.06, win ±0.03) with P4 cut-boundary files excluded
      (KNOWN_CUT_BOUNDARY — rating stays strict for them).

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
from conftest import (  # noqa: E402
    ALIGN_RAW_TOL,
    DET_RAW_TOL,
    KNOWN_CUT_BOUNDARY,
    KNOWN_RATING_DIVERGENCE,
    PLATFORM,
    baseline_section,
    SOURCE_DIRS,
)

# One deterministic run per format, shared by the rating and raw gates
# (the run itself is bit-reproducible on a given platform, so caching is
# safe and halves the gate cost).
_DET_RUN_CACHE: dict[str, dict] = {}


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


def _det_run(key: str) -> dict:
    if key not in _DET_RUN_CACHE:
        _DET_RUN_CACHE[key] = run_cull_on_copies(_src_files(key), workers=4)
    return _DET_RUN_CACHE[key]


@pytest.mark.precision
@pytest.mark.deterministic
@pytest.mark.parametrize("key", ["jpg", "heif", "arw", "nef"])
def test_deterministic_ratings_cross_platform(key, deterministic_env):
    """Cross-platform gate: deterministic-backend ratings must equal the
    committed truth exactly (70/70 verified mac-vs-win 2026-08-28)."""
    baseline = baseline_section(key)
    actual = _det_run(key)
    for name, (exp_rating, _) in baseline.items():
        act_rating = actual[name][0]
        assert act_rating == exp_rating, \
            f"deterministic/{key}::{name}: rating {act_rating} != truth {exp_rating} " \
            f"(raw {actual[name][1]} vs {baseline[name][1]}, veto={actual[name][2]!r})"


@pytest.mark.precision
@pytest.mark.deterministic
@pytest.mark.parametrize("key", ["jpg", "heif", "arw", "nef"])
def test_deterministic_raw_regression(key, deterministic_env):
    """Platform-internal raw regression: deterministic raw_score stays within
    the per-platform window around the truth (mac ±0.05 / win ±0.005)."""
    baseline = baseline_section(key)
    actual = _det_run(key)
    tol = DET_RAW_TOL[PLATFORM]
    for name, (exp_rating, exp_raw) in baseline.items():
        act_raw = actual[name][1]
        assert abs(act_raw - exp_raw) <= tol, \
            f"deterministic/{key}::{name}: raw_score {act_raw} drifted from truth " \
            f"{exp_raw} by {abs(act_raw-exp_raw):.4f} > {tol} (platform={PLATFORM})"


@pytest.mark.precision
@pytest.mark.alignment
@pytest.mark.parametrize("key", ["jpg", "heif", "arw", "nef"])
def test_backends_align_to_deterministic(key, nondeterministic_env):
    """Default backends align to deterministic truth: rating strict (minus the
    documented ANE knife-edge files), raw within the platform alignment window
    minus the documented P4 cut-boundary files."""
    src_files = _src_files(key)
    baseline = baseline_section(key)
    actual = run_cull_on_copies(src_files, workers=4)
    align_tol = ALIGN_RAW_TOL[PLATFORM]
    for name, (exp_rating, exp_raw) in baseline.items():
        act_rating, act_raw, veto, ndet = actual[name]
        if (key, name) in KNOWN_RATING_DIVERGENCE:
            continue  # documented ANE-vs-CPU rating boundary (±0.016 logit)
        assert act_rating == exp_rating, \
            f"align/{key}::{name}: rating {act_rating} != truth {exp_rating} " \
            f"(raw {act_raw} vs {exp_raw}, veto={veto!r})"
        if (key, name) in KNOWN_CUT_BOUNDARY:
            continue  # P4 cut flip ≈ P4_CUT_PENALTY (0.6); rating unaffected
        assert abs(act_raw - exp_raw) <= align_tol, \
            f"align/{key}::{name}: raw_score {act_raw} drifted from truth {exp_raw} " \
            f"by {abs(act_raw-exp_raw):.4f} > {align_tol} (platform={PLATFORM})"
