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

# Baseline locked 2026-08-24 on macOS (Apple Silicon, pyav 17.1.0/libav 61,
# exiftool 13.50) — platform re-lock of the 2026-08-23 Windows baseline.
# All 24 ratings identical to the Windows lock (0 flips); raw_score drifts
# within ±0.035 from HEVC/RGB LSB platform decode differences (DSC00893 is
# the largest at -0.035). Pipeline unchanged: libjpeg draft decode +
# cv2.INTER_AREA resize + cv2 letterbox/cv2 P4-ROI + cv2.dft sharpness with
# cv2 gray/Laplacian internals + P4 v2 model.
BASELINE = {
    "DSC00827.heif": (-1, 1.204),
    "DSC00845.heif": (-1, 2.826),
    "DSC00849.heif": (1, 3.108),
    "DSC00851.heif": (2, 3.381),
    "DSC00879.heif": (-1, 1.926),
    "DSC00880.heif": (-1, 1.399),
    "DSC00886.heif": (-1, 1.834),
    "DSC00887.heif": (-1, 1.985),
    "DSC00888.heif": (-1, 2.458),
    "DSC00890.heif": (-1, 1.736),
    "DSC00892.heif": (-1, 0.923),
    "DSC00893.heif": (-1, 2.433),
    "DSC00894.heif": (-1, 0.772),
    "DSC00895.heif": (-1, 1.697),
    "DSC00896.heif": (-1, 2.974),
    "DSC00897.heif": (-1, 2.095),
    "DSC00942.heif": (-1, 2.719),
    "DSC00951.heif": (-1, 1.299),
    "DSC00952.heif": (-1, 2.047),
    "DSC00958.heif": (-1, 1.09),
    "DSC00959.heif": (-1, 1.117),
    "DSC00960.heif": (-1, 2.278),
    "DSC00961.heif": (-1, 1.753),
    "DSC00962.heif": (-1, 1.513),
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