"""
tests/test_gui_worker.py — CullWorker message protocol (no display needed).
"""

import os
import queue
import shutil
import sys
import time
from pathlib import Path

import pytest

sys.path.append(os.getcwd())

from cull.engine import EngineConfig
from cull.gui.worker import (
    CANCELLED, DONE, ERROR, FRAME, LOG_LINE, STAGE, TOTAL, CullWorker,
)

TEST_IMG = Path("tests/test_img")


def make_config(input_dir: Path, **overrides) -> EngineConfig:
    kwargs = dict(input_dir=input_dir, workers=1, force=True, dry_run=True, scale_width=512)
    kwargs.update(overrides)
    return EngineConfig(**kwargs)


def drain(worker: CullWorker, msg_queue: "queue.Queue", want: tuple, timeout: float = 180.0):
    """Consume messages until all *want* kinds are seen, the worker exits, or timeout.

    Returns (last_message_by_kind, counts_by_kind).
    """
    seen: dict = {}
    counts: dict = {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not worker.is_running() and msg_queue.empty():
            break
        try:
            msg = msg_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        seen[msg[0]] = msg
        counts[msg[0]] = counts.get(msg[0], 0) + 1
        if all(k in seen for k in want):
            break
    return seen, counts


def copy_photos(dst: Path, copies: int = 1) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for i in range(copies):
        for src in sorted(TEST_IMG.glob("*.jpg")):
            shutil.copy(src, dst / f"{i:02d}_{src.name}")


def test_worker_completes_with_full_message_stream(tmp_path):
    msg_queue: "queue.Queue" = queue.Queue()
    worker = CullWorker(make_config(TEST_IMG), msg_queue)
    worker.start()
    seen, counts = drain(worker, msg_queue, want=(TOTAL, FRAME, LOG_LINE, DONE))
    worker.wait(timeout=60)

    assert worker.is_running() is False
    assert DONE in seen, f"missing DONE; seen kinds: {sorted(seen)}"
    assert ERROR not in seen
    assert STAGE in seen and seen[STAGE][1] == "Done!" and seen[STAGE][2] == 1.0
    assert seen[TOTAL][1] == 6
    assert counts[FRAME] == 6  # one FRAME message per scored frame
    name, rating, sharp, comp, raw, veto = seen[FRAME][1:]
    assert isinstance(name, str) and name.endswith(".jpg")
    assert isinstance(rating, int)
    assert isinstance(sharp, float) and isinstance(comp, float) and isinstance(raw, float)
    assert isinstance(veto, str)
    result = seen[DONE][1]
    assert result["total"] == 6
    assert result["keep"] + result["reject"] == 6
    assert isinstance(result["elapsed"], float)


def test_worker_cancel(tmp_path):
    dir_with_many = tmp_path / "many"
    copy_photos(dir_with_many, copies=6)  # 36 images -> run takes long enough to cancel
    msg_queue: "queue.Queue" = queue.Queue()
    worker = CullWorker(make_config(dir_with_many), msg_queue)
    worker.start()
    time.sleep(0.5)  # cancel while models are still loading / scoring starts
    worker.stop()
    seen, _ = drain(worker, msg_queue, want=(CANCELLED,))
    worker.wait(timeout=60)

    assert worker.is_running() is False
    assert ERROR not in seen
    assert CANCELLED in seen, f"missing CANCELLED; seen kinds: {sorted(seen)}"
    scores = seen[CANCELLED][1]
    assert isinstance(scores, list)
    assert len(scores) <= 36


def test_worker_error_reports_message(tmp_path, monkeypatch):
    """A failing engine surfaces as an ERROR message instead of a crash."""
    import cull.gui.worker as worker_module

    def boom(self, progress_callback=None, cancel_event=None):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(worker_module.CullingEngine, "run", boom)
    msg_queue: "queue.Queue" = queue.Queue()
    worker = worker_module.CullWorker(make_config(TEST_IMG), msg_queue)
    worker.start()
    seen, _ = drain(worker, msg_queue, want=(ERROR,))
    worker.wait(timeout=30)

    assert ERROR in seen
    assert "engine exploded" in seen[ERROR][1]
