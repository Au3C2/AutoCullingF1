"""
tests/test_json_protocol.py — JSON Lines sidecar protocol tests.

Runs ``cull_photos.py --json-lines`` as a subprocess and validates the event
stream on stdout (stage/group/frame/done), cancellation via stdin, and the
post-run preview command loop.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.append(os.getcwd())

TEST_IMG = Path("tests/test_img")
SCRIPT = Path("cull_photos.py")


def read_events(proc: subprocess.Popen, n: int, timeout: float = 120.0) -> list[dict]:
    """Read exactly *n* JSON events from proc.stdout (line-buffered)."""
    events = []
    deadline = time.monotonic() + timeout
    while len(events) < n:
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out after {len(events)}/{n} events")
        line = proc.stdout.readline()
        if not line:
            raise TimeoutError(f"stdout closed after {len(events)}/{n} events")
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def wait_for(proc: subprocess.Popen, predicate, timeout: float = 120.0) -> list[dict]:
    """Read events until *predicate* holds; return all events read so far."""
    events = []
    deadline = time.monotonic() + timeout
    while True:
        if predicate(events):
            return events
        if time.monotonic() > deadline:
            raise TimeoutError(f"predicate unmet after {len(events)} events")
        line = proc.stdout.readline()
        if not line:
            raise TimeoutError(f"stdout closed after {len(events)} events")
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))


def start_sidecar(input_dir: Path, extra: list[str] | None = None) -> subprocess.Popen:
    cmd = [sys.executable, str(SCRIPT), "--input-dir", str(input_dir),
           "--json-lines", "--dry-run", "-f", "--workers", "1",
           "--scale-width", "512"] + (extra or [])
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)


def test_json_lines_full_run(tmp_path):
    proc = start_sidecar(TEST_IMG)
    events = []
    try:
        events = wait_for(proc, lambda ev: any(e.get("type") == "done" for e in ev))
        proc.stdin.write('{"cmd": "quit"}\n')
        proc.stdin.flush()
        assert proc.wait(timeout=60) == 0

        by_type: dict[str, list[dict]] = {}
        for e in events:
            by_type.setdefault(e["type"], []).append(e)

        assert by_type["total"][0]["frames"] == 6
        assert len(by_type["paths"][0]["paths"]) == 6
        assert "Analyzing images..." in {e["msg"] for e in by_type["stage"]}
        assert len(by_type["frame"]) == 6
        done = by_type["done"][0]
        assert done["total"] == 6
        assert done["keep"] + done["reject"] == 6
        assert isinstance(done["stars"], dict)
        assert "error" not in by_type
    finally:
        if proc.poll() is None:
            proc.kill()


def test_json_lines_cancel(tmp_path):
    dir_many = tmp_path / "many"
    dir_many.mkdir()
    for i in range(6):
        for src in sorted(TEST_IMG.glob("*.jpg")):
            shutil.copy(src, dir_many / f"{i:02d}_{src.name}")  # 36 images

    proc = start_sidecar(dir_many)
    try:
        events = wait_for(proc, lambda ev: any(e.get("type") == "frame" for e in ev))
        proc.stdin.write("cancel\n")
        proc.stdin.flush()
        events += wait_for(proc, lambda ev: any(e.get("type") == "cancelled" for e in ev))
        assert proc.wait(timeout=60) == 0
        cancelled = [e for e in events if e["type"] == "cancelled"][0]
        frames = [e for e in events if e["type"] == "frame"]
        assert cancelled["count"] == len(frames)
        assert cancelled["count"] < 36
    finally:
        if proc.poll() is None:
            proc.kill()


def test_json_lines_preview_and_quit(tmp_path):
    proc = start_sidecar(TEST_IMG)
    try:
        events = wait_for(proc, lambda ev: any(e.get("type") == "done" for e in ev))
        paths = [e for e in events if e["type"] == "paths"][0]["paths"]
        first = next(iter(paths.values()))

        proc.stdin.write(json.dumps({"cmd": "preview", "path": first, "size": 300}) + "\n")
        proc.stdin.flush()
        events = wait_for(proc, lambda ev: any(e.get("type") == "preview" for e in ev))
        preview = [e for e in events if e["type"] == "preview"][-1]
        assert preview["path"] == first
        assert preview["png"], "expected base64 PNG data"

        proc.stdin.write('{"cmd": "quit"}\n')
        proc.stdin.flush()
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
