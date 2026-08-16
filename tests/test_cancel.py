"""
tests/test_cancel.py — cancellation semantics for CullingEngine.run().

Cancellation is opt-in: callers pass a threading.Event to run(); once the
flag is set, scoring stops early and all side effects (XMP writes and JPG
metadata sync) are skipped. The CLI never passes the event and is unaffected.
"""

import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.append(os.getcwd())

from cull import engine as engine_module
from cull.engine import CullingEngine, EngineConfig

TEST_IMG = Path("tests/test_img")


def make_config(input_dir: Path, **overrides) -> EngineConfig:
    kwargs = dict(
        input_dir=input_dir,
        workers=2,
        force=True,
        dry_run=False,
        scale_width=512,
    )
    kwargs.update(overrides)
    return EngineConfig(**kwargs)


def test_cancel_before_scoring(monkeypatch):
    """Cancelling before parallel scoring returns no scores and skips all side effects."""
    cancel_event = threading.Event()
    messages: list[tuple[str, float]] = []
    xmp_calls: list[int] = []
    sync_calls: list[int] = []

    monkeypatch.setattr(engine_module, "write_xmp_batch", lambda *a, **k: xmp_calls.append(1))
    monkeypatch.setattr(engine_module, "update_image_metadata", lambda *a, **k: sync_calls.append(1))

    def progress(msg: str, p: float):
        messages.append((msg, p))
        if msg == "Analyzing images...":
            cancel_event.set()

    engine = CullingEngine(make_config(TEST_IMG))
    all_scores, elapsed = engine.run(progress_callback=progress, cancel_event=cancel_event)

    assert cancel_event.is_set()
    assert any(msg == "Cancelled" for msg, _ in messages), messages
    assert all_scores == []
    assert xmp_calls == [], "XMP writes must be skipped on cancel"
    assert sync_calls == [], "metadata sync must be skipped on cancel"
    assert elapsed >= 0


def test_cancel_during_parallel_scoring(monkeypatch):
    """Cancelling while the executor is running returns without exceptions and skips side effects."""
    cancel_event = threading.Event()
    messages: list[tuple[str, float]] = []
    xmp_calls: list[int] = []
    sync_calls: list[int] = []
    errors: list[Exception] = []

    monkeypatch.setattr(engine_module, "write_xmp_batch", lambda *a, **k: xmp_calls.append(1))
    monkeypatch.setattr(engine_module, "update_image_metadata", lambda *a, **k: sync_calls.append(1))

    engine = CullingEngine(make_config(TEST_IMG, workers=1))
    original = engine._process_group_internal

    def slow_process(group, cancel_event=None):
        time.sleep(0.4)  # widen the cancellation window
        return original(group, cancel_event)

    monkeypatch.setattr(engine, "_process_group_internal", slow_process)

    def progress(msg: str, p: float):
        messages.append((msg, p))

    def run_engine():
        try:
            engine.run(progress_callback=progress, cancel_event=cancel_event)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    thread = threading.Thread(target=run_engine)
    thread.start()
    time.sleep(0.1)  # let the executor start and enter the first group
    cancel_event.set()
    thread.join(timeout=120)

    assert errors == [], errors
    assert cancel_event.is_set()
    assert any(msg == "Cancelled" for msg, _ in messages), messages
    assert len(engine.all_scores) <= len(engine.image_paths)
    assert xmp_calls == [], "XMP writes must be skipped on cancel"
    assert sync_calls == [], "metadata sync must be skipped on cancel"


def test_cancel_flag_after_completion_is_noop():
    """Setting the flag after run() finished must not change the result."""
    engine = CullingEngine(make_config(TEST_IMG, dry_run=True))
    cancel_event = threading.Event()
    all_scores, _ = engine.run(progress_callback=None, cancel_event=cancel_event)
    cancel_event.set()  # too late - run already returned
    assert len(all_scores) == len(engine.image_paths)
