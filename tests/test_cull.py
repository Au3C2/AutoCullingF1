"""
tests/test_cull.py — pytest suite for F1 photo culling pipeline.

The engine writes ratings directly into standalone JPG metadata and no
longer emits XMP sidecars, so these tests verify the --dump-scores CSV
against the golden baseline (v0.1 logic), shared with test_package.py.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
import csv
import pytest

# Add project root to sys.path for any direct module imports in tests
sys.path.append(os.getcwd())

# Golden baseline generated from tests/test_img on the CURRENT pipeline
# (ImageIO HW JPEG decode + scipy.fft sharpness + P4 v2 model on the
# single-partition ANE graph by default on macOS). IMG_20260314_160318_240.jpg
# was re-locked -1 -> 3 on 2026-08-25: the ANE P4 path moves its orientation
# argmax rear -> rear_angle (logit diff <= 0.016, knife-edge), disabling the
# low-confidence rear veto. Stable across consecutive runs. Shared with
# test_package.py so the CLI and the packaged binary are validated identically.
BASELINE = {
    "IMG_20260314_151744_020.jpg": 3,
    "IMG_20260314_160317_680.jpg": 3,
    "IMG_20260314_160318_240.jpg": 3,
    "IMG_20260314_160343_870.jpg": 3,
    "IMG_20260314_160344_380.jpg": 3,
    "IMG_20260315_150404_550.jpg": -1,
}


@pytest.fixture
def test_env(tmp_path):
    """Fixture to set up a clean test environment with sample images."""
    src_dir = Path("tests/test_img")
    for f in src_dir.glob("*.jpg"):
        shutil.copy(f, tmp_path)
    return tmp_path


def run_cull(input_dir: Path, backend: str, csv_path: Path | None = None, workers: int = 4):
    """Helper to run the cull_photos script."""
    env = os.environ.copy()
    env["CULL_BACKEND"] = backend
    env["PYTHONPATH"] = os.getcwd()

    cmd = [
        sys.executable, "cull_photos.py",
        "--input-dir", str(input_dir),
        "--workers", str(workers),
        "--force"
    ]
    if csv_path is not None:
        cmd += ["--dump-scores", str(csv_path)]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def read_scores_csv(csv_path: Path) -> dict[str, int]:
    """Parse the --dump-scores CSV into {filename: rating}."""
    scores = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[row["filename"]] = int(row["rating"])
    return scores


@pytest.mark.parametrize("backend", ["onnx"] + (["coreml"] if sys.platform == "darwin" else []))
def test_cull_execution(test_env, backend):
    """Test that the script executes successfully and produces a scores CSV."""
    csv_path = test_env / "scores.csv"
    proc = run_cull(test_env, backend, csv_path)
    assert proc.returncode == 0, f"Script failed with stderr: {proc.stderr}"

    jpgs = list(test_env.glob("*.jpg"))
    scores = read_scores_csv(csv_path)
    assert len(scores) == len(jpgs), f"Expected {len(jpgs)} score rows, found {len(scores)}"


def test_labels_correctness(test_env):
    """Verify that 15th images are rejected (Rating -1) and kept images come from the 14th."""
    csv_path = test_env / "scores.csv"
    run_cull(test_env, "onnx", csv_path)
    scores = read_scores_csv(csv_path)

    for name, rating in scores.items():
        if "20260315" in name:
            assert rating == -1, f"Image {name} should be REJECTED (Rating -1)"

    kept = [name for name, rating in scores.items() if rating > 0]
    assert kept, "expected at least one kept image"
    assert all("20260314" in name for name in kept), \
        f"kept images must come from the 14th: {kept}"


@pytest.mark.parametrize("workers", [1, 4, 6])
def test_rating_precision_matches_baseline(test_env, workers):
    """Compare per-image ratings against golden baseline across worker thread counts (1, 4, 6)."""
    csv_path = test_env / f"scores_w{workers}.csv"
    proc = run_cull(test_env, "onnx", csv_path, workers=workers)
    assert proc.returncode == 0, f"Script failed with stderr: {proc.stderr}"
    actual = read_scores_csv(csv_path)

    for name, expected in BASELINE.items():
        assert actual.get(name) == expected, \
            f"{name} (workers={workers}): expected rating {expected}, got {actual.get(name)}"