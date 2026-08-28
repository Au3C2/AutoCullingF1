"""
tests/conftest.py — Shared fixtures, baseline loader, and pytest markers.

The deterministic artifact (tests/baselines/deterministic.json) is the
single truth. Old per-file BASELINE dicts in test_cull / HEIF / RAW have
been removed; every precision gate reads through the helpers here so the
old and new gates cannot diverge. Markers keep the fast/expensive axes
selectable without memorizing file paths.

Strictness model (measured mac-vs-win 2026-08-28):

  RATING — the product-level contract: identical star/reject outcome on
      any platform/backend. The deterministic backend achieves 70/70
      (macOS vs the Windows-generated truth), so rating gates are STRICT
      everywhere.
  RAW — decode SIMD (NEON vs AVX) makes bit-identical raw_score across
      platforms impossible (JPG aligns; HEIF/ARW/NEF drift 0.013–0.039).
      Raw is asserted only as a PLATFORM-INTERNAL regression window
      around the truth, never as a cross-platform equality.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "precision: precision / accuracy gates")
    config.addinivalue_line("markers", "deterministic: deterministic truth (CULL_DETERMINISTIC=1)")
    config.addinivalue_line("markers", "alignment: default backends vs deterministic truth")
    config.addinivalue_line("markers", "perf: steady-state throughput gate (slow, needs ~500 files)")
    config.addinivalue_line("markers", "packaged: packaged binary (needs built artifact)")


BASELINE_PATH = Path(__file__).parent / "baselines" / "deterministic.json"

PLATFORM = "mac" if sys.platform == "darwin" else "win"

# Raw regression windows for the DETERMINISTIC backend around the truth.
#   win: the truth platform — bit-aligned decode (±0.005).
#   mac: ARM libav/RAW decode differs from x86 by 0.013–0.050 (measured
#        2026-08-28, ratings still 70/70) — window ±0.06.
DET_RAW_TOL = {"win": 0.005, "mac": 0.06}

# Raw alignment windows for the DEFAULT (accelerated) backends vs the
# deterministic truth:
#   win: ±0.03 (CUDA-vs-CPU measured envelope)
#   mac: ±0.06 (CoreML/ImageIO vs CPU: ARW max drift 0.057)
ALIGN_RAW_TOL = {"win": 0.03, "mac": 0.06}

# Files whose RATING is known to sit on a hardware decision boundary
# (ANE-vs-CPU logit delta ≤0.016 flips them; see performance_baseline.md).
# Backend-alignment gates skip the rating assert for these; the
# deterministic cross-platform gate does NOT skip (70/70 verified
# 2026-08-28 — these files only diverge on accelerated backends).
KNOWN_RATING_DIVERGENCE = {
    ("jpg", "IMG_20260314_160318_240.jpg"),
    ("heif", "DSC00849.heif"),
}

# Files where the P4 cut decision flips between CPU and accelerated
# backends — raw delta ≈ P4_CUT_PENALTY (0.6), rating unaffected. Raw
# alignment skips them; rating stays strict.
KNOWN_CUT_BOUNDARY = {
    ("heif", "DSC00942.heif"),
    ("nef", "IMG_20260315_164133_810.nef"),
}

# Backwards-compat: default backends (CUDA, CoreML, etc.) align to the
# deterministic truth with this raw tolerance on top of the strict
# deterministic window. Value comes from the measured CPU-vs-CUDA /
# decode-LSB envelope (see test_deterministic_baseline.py).
ALIGN_RAW_TOL_SCALAR = ALIGN_RAW_TOL[PLATFORM]

# Source locations for each baseline section (kept in one place so
# test_cull / HEIF / RAW do not re-declare them).
SOURCE_DIRS: dict[str, Path] = {
    "jpg": Path("tests/test_img"),
    "heif": Path("test_import"),
    "arw": Path("test_arw"),
    "nef": Path("test_nef"),
}


def load_deterministic_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.exists():
        pytest.skip(f"deterministic baseline missing: {BASELINE_PATH} "
                    f"(run scripts/generate_deterministic_baseline.py)")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def baseline_section(key: str, baseline: dict[str, Any] | None = None) -> dict[str, tuple[int, float]]:
    """Return {filename: (rating, raw_score)} for *key* (jpg/heif/arw/nef)."""
    baseline = baseline if baseline is not None else load_deterministic_baseline()
    section = baseline.get(key, {})
    if not section:
        pytest.skip(f"baseline has no {key} section")
    return {name: (int(v[0]), float(v[1])) for name, v in section.items()}


def baseline_jpg_ratings(baseline: dict[str, Any] | None = None) -> dict[str, int]:
    """test_cull's {filename: rating} view derived from the jpg section."""
    return {k: v[0] for k, v in baseline_section("jpg", baseline).items()}


@pytest.fixture
def deterministic_baseline() -> dict[str, Any]:
    return load_deterministic_baseline()


@pytest.fixture
def deterministic_env():
    """Set CULL_DETERMINISTIC=1 for the duration of the test."""
    prev = os.environ.get("CULL_DETERMINISTIC")
    os.environ["CULL_DETERMINISTIC"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("CULL_DETERMINISTIC", None)
        else:
            os.environ["CULL_DETERMINISTIC"] = prev


@pytest.fixture
def nondeterministic_env():
    """Ensure the run is non-deterministic (unset the flag)."""
    prev = os.environ.pop("CULL_DETERMINISTIC", None)
    try:
        yield
    finally:
        if prev is not None:
            os.environ["CULL_DETERMINISTIC"] = prev
