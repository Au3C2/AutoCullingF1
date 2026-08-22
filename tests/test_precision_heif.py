"""
tests/test_precision_heif.py — HEIF score-precision gate.

Locks per-file rating/raw_score for 24 real Sony HEIF stills from test_import.
Guards optimization #1 (ffmpeg `-vf scale` / decode-path changes): any pixel
change that alters scoring must fail here. Skipped when the source dataset is
absent (e.g. CI without test_import).
"""
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent))

from score_gate import run_cull_on_copies, assert_scores_match  # noqa: E402

HEIF_DIR = Path("test_import")

# Baseline locked 2026-08-22 (master@CUDA, ffmpeg preview-stream + Pillow resize).
# filename -> (rating, raw_score rounded to 3 decimals)
BASELINE = {
    "DSC00827.heif": (-1, 1.137),
    "DSC00845.heif": (-1, 2.798),
    "DSC00849.heif": (-1, 3.038),
    "DSC00851.heif": (2, 3.233),
    "DSC00879.heif": (-1, 1.898),
    "DSC00880.heif": (-1, 1.399),
    "DSC00886.heif": (-1, 1.727),
    "DSC00887.heif": (-1, 2.008),
    "DSC00888.heif": (-1, 2.463),
    "DSC00890.heif": (-1, 1.272),
    "DSC00892.heif": (-1, 0.911),
    "DSC00893.heif": (-1, 2.225),
    "DSC00894.heif": (-1, 0.732),
    "DSC00895.heif": (-1, 1.679),
    "DSC00896.heif": (-1, 2.974),
    "DSC00897.heif": (-1, 1.309),
    "DSC00942.heif": (-1, 1.961),
    "DSC00951.heif": (-1, 1.174),
    "DSC00952.heif": (-1, 1.974),
    "DSC00958.heif": (-1, 1.091),
    "DSC00959.heif": (-1, 1.086),
    "DSC00960.heif": (-1, 2.275),
    "DSC00961.heif": (-1, 1.713),
    "DSC00962.heif": (-1, 1.432),
}


def _src_files() -> list[Path]:
    if not HEIF_DIR.is_dir():
        return []
    return [HEIF_DIR / name for name in BASELINE if (HEIF_DIR / name).exists()]


def _verify_srcs(files: list[Path]) -> list[Path]:
    if not files:
        pytest.skip("test_import dataset not present")
    if len(files) < len(BASELINE):
        pytest.skip(f"test_import incomplete: {len(files)}/{len(BASELINE)} files")
    return files


def test_heif_rating_precision_matches_baseline():
    """24 HEIF stills must keep their per-file rating and raw_score."""
    files = _verify_srcs(_src_files())
    run_files = files[: len(BASELINE)]  # keep deterministic order subset
    actual = run_cull_on_copies(run_files)
    assert_scores_match(actual, BASELINE, "heif")