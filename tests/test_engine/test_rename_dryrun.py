"""Verification test for --rename with and without --dry-run."""

import shutil
import tempfile
from pathlib import Path
from cull.engine import CullingEngine, EngineConfig


def test_rename_dry_run_does_not_modify_disk():
    tmp_dir = Path(tempfile.mkdtemp(prefix="test_rename_dryrun_"))
    try:
        sample = Path("tests/test_img/IMG_20260314_151744_020.jpg")
        if not sample.exists():
            return

        target_file = tmp_dir / "ORIGINAL_PHOTO.jpg"
        shutil.copy(sample, target_file)

        # 1. Run engine with rename=True and dry_run=True
        cfg = EngineConfig(
            input_dir=tmp_dir,
            rename=True,
            dry_run=True,
            force=True,
            workers=1,
        )
        engine = CullingEngine(cfg)
        scores, _ = engine.run()

        # Assert: File on disk MUST remain under original name
        assert target_file.exists(), "Original file should NOT be renamed when dry_run=True"
        assert len(scores) == 1
        assert scores[0].path.name == "ORIGINAL_PHOTO.jpg"

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
