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

# Baseline locked 2026-08-23 on the CURRENT pipeline (libjpeg draft decode +
# cv2.INTER_AREA resize + cv2 letterbox/cv2 P4-ROI + cv2.dft sharpness + P4 v2
# model retrained with resize-kernel/camera-jitter augmentation). All
# keep/reject decisions identical to the pre-cv2 pipeline; DSC00849 drifted
# 1->2 stars (kept either way).
BASELINE = {
    "DSC00827.heif": (-1, 1.209),
    "DSC00845.heif": (-1, 2.831),
    "DSC00849.heif": (2, 3.114),
    "DSC00851.heif": (2, 3.393),
    "DSC00879.heif": (-1, 1.93),
    "DSC00880.heif": (-1, 1.399),
    "DSC00886.heif": (-1, 1.838),
    "DSC00887.heif": (-1, 1.996),
    "DSC00888.heif": (-1, 2.466),
    "DSC00890.heif": (-1, 2.337),
    "DSC00892.heif": (-1, 0.924),
    "DSC00893.heif": (-1, 2.474),
    "DSC00894.heif": (-1, 0.775),
    "DSC00895.heif": (-1, 2.3),
    "DSC00896.heif": (-1, 2.974),
    "DSC00897.heif": (-1, 2.1),
    "DSC00942.heif": (-1, 2.128),
    "DSC00951.heif": (-1, 1.301),
    "DSC00952.heif": (-1, 2.652),
    "DSC00958.heif": (-1, 1.09),
    "DSC00959.heif": (-1, 1.718),
    "DSC00960.heif": (-1, 2.277),
    "DSC00961.heif": (-1, 1.757),
    "DSC00962.heif": (-1, 2.118),
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