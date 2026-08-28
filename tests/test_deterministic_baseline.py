"""
tests/test_deterministic_baseline.py — Deterministic truth vs backend alignment.

The deterministic baseline (tests/baselines/deterministic.json) is the
cross-platform truth: CULL_DETERMINISTIC=1 forces CPU-only ORT, software
decode, and single-threaded FFT so macOS and Windows produce identical
(raw_score, rating) per file.

Gates:
  test_deterministic_matches_baseline  — deterministic run must match the
      baseline exactly (rating) and within 0.005 raw (strict truth).
  test_backends_align_to_deterministic — each non-deterministic backend
      (onnx/CUDA, and coreml on macOS) must stay within alignment tolerance
      of the same truth.  This is the user-requested 'different backends look
      toward the deterministic baseline' gate.

The 0.03 raw + exact-rating alignment budget was chosen from measured
platform deltas (ANE vs CUDA vs CPU): raw drifts up to ~0.02 and a single
boundary rating flip (heif/arw/nef each had 1 rating-edge file).  Large
drifts (e.g. 0.6 from P4 cut-penalty flips on DSC00942.heif) still fail.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent))

from score_gate import assert_scores_match, run_cull_on_copies  # noqa: E402

BASELINE_PATH = Path(__file__).parent / "baselines" / "deterministic.json"

# Alignment tolerance for non-deterministic backends toward the truth.
# 0.03 covers CPU vs CUDA and macOS vs Windows decode LSB on ARW/NEF/HEIF;
# JPG also has the ffmpeg-hw vs cv2 decode path (now disabled unless
# CULL_HW_JPEG=1, so the remaining drift is just ORT backend).
ALIGN_RAW_TOL = 0.03


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(f"deterministic baseline missing: {BASELINE_PATH} (run scripts/generate_deterministic_baseline.py)")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _collect_for_key(key: str, baseline: dict) -> tuple[list[Path], dict]:
    section = baseline.get(key, {})
    if not section:
        pytest.skip(f"baseline has no {key} section")
    # Resolve source dirs (same as existing precision gates)
    dir_map = {
        "jpg": Path("tests/test_img"),
        "heif": Path("test_import"),
        "arw": Path("test_arw"),
        "nef": Path("test_nef"),
    }
    src_dir = dir_map[key]
    if not src_dir.is_dir():
        pytest.skip(f"source dir missing: {src_dir}")
    src_files = [src_dir / name for name in section]
    missing = [p for p in src_files if not p.exists()]
    if missing:
        pytest.skip(f"{key}: {len(missing)} files missing (e.g. {missing[0].name})")
    # Normalize baseline to the shape assert_scores_match expects
    norm = {name: (int(v[0]), float(v[1])) for name, v in section.items()}
    return src_files, norm


def _assert_align(actual: dict, baseline: dict, what: str, raw_tol: float) -> None:
    """Alignment gate toward deterministic truth: rating must match, raw within raw_tol."""
    assert set(actual) == set(baseline), \
        f"{what}: file sets differ — missing={set(baseline)-set(actual)} extra={set(actual)-set(baseline)}"
    for name, (exp_rating, exp_raw) in baseline.items():
        act_rating, act_raw, veto, ndet = actual[name]
        assert act_rating == exp_rating, \
            f"{what}::{name}: rating {act_rating} != deterministic truth {exp_rating} (raw {act_raw} vs {exp_raw}, veto={veto!r})"
        assert abs(act_raw - exp_raw) <= raw_tol, \
            f"{what}::{name}: raw_score {act_raw} drifted from deterministic truth {exp_raw} by {abs(act_raw-exp_raw):.4f} > {raw_tol}"


# --- Deterministic truth: must match exactly (modulo 0.005 rounding) ---

@pytest.mark.parametrize("key", ["jpg", "heif", "arw", "nef"])
def test_deterministic_matches_baseline(key):
    """Deterministic run (CULL_DETERMINISTIC=1) must equal the committed baseline."""
    baseline = _load_baseline()
    src_files, norm = _collect_for_key(key, baseline)
    orig = os.environ.get("CULL_DETERMINISTIC")
    os.environ["CULL_DETERMINISTIC"] = "1"
    try:
        actual = run_cull_on_copies(src_files, workers=4)
    finally:
        if orig is None:
            os.environ.pop("CULL_DETERMINISTIC", None)
        else:
            os.environ["CULL_DETERMINISTIC"] = orig
    assert_scores_match(actual, norm, f"deterministic/{key}")


# --- Backend alignment: non-deterministic runs must stay near the truth ---

@pytest.mark.parametrize("key", ["jpg", "heif", "arw", "nef"])
def test_backends_align_to_deterministic(key):
    """Default backends (onnx/CUDA/coreml) must align to deterministic truth."""
    baseline = _load_baseline()
    src_files, norm = _collect_for_key(key, baseline)
    # Ensure non-deterministic (unset the flag for this run)
    orig = os.environ.pop("CULL_DETERMINISTIC", None)
    try:
        actual = run_cull_on_copies(src_files, workers=4)
    finally:
        if orig is not None:
            os.environ["CULL_DETERMINISTIC"] = orig
    _assert_align(actual, norm, f"align/{key}", ALIGN_RAW_TOL)
