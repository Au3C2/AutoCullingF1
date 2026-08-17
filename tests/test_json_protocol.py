"""
tests/test_json_protocol.py — JSON Lines sidecar protocol tests.

Runs ``cull_photos.py --json-lines`` as a RESIDENT subprocess and validates
the command loop: ``scan`` lists shots, ``run`` streams stage/frame/done
events, ``cancel`` aborts, ``preview`` answers a PNG, and the process stays
alive for repeated scans / reruns until ``quit``.

The sidecar's stdout is read as RAW BYTES with os.read + select. A blocking
``readline()`` on a text pipe proved unreliable here: the sidecar writes
from several threads (engine executor + command loop), and the reader can
miss the wakeup with events already sitting in the pipe (observed on macOS).
Reading raw chunks and splitting lines ourselves sidesteps TextIOWrapper
buffering entirely.
"""

import json
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.append(os.getcwd())

TEST_IMG = Path("tests/test_img")
SCRIPT = Path("cull_photos.py")


class SidecarChannel:
    """Raw stdout reader for a sidecar subprocess (binary, line-split)."""

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.fd = proc.stdout.fileno()
        os.set_blocking(self.fd, False)
        self._buf = b""

    def read_line(self, timeout: float) -> str | None:
        """Return the next complete line, or None on timeout/EOF."""
        deadline = time.monotonic() + timeout
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line = self._buf[:nl].decode("utf-8", errors="replace")
                self._buf = self._buf[nl + 1:]
                return line
            ready, _, _ = select.select([self.fd], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(self.fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    # EOF: drain what's left, then signal closure.
                    if self._buf:
                        line = self._buf.decode("utf-8", errors="replace")
                        self._buf = b""
                        return line
                    return ""
                self._buf += chunk
            if time.monotonic() > deadline:
                return None

    def read_event(self, timeout: float) -> dict | None:
        """Next JSON event, skipping blanks; {} on EOF; None on timeout."""
        while True:
            line = self.read_line(timeout)
            if line is None:
                return None
            if line == "":
                return {}
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"non-JSON line from sidecar: {line[:200]!r}") from exc


def send(proc: subprocess.Popen, obj: dict) -> None:
    proc.stdin.write((json.dumps(obj) + "\n").encode())
    proc.stdin.flush()


def wait_for(chan: SidecarChannel, predicate, timeout: float = 120.0) -> list[dict]:
    """Read events until *predicate* holds; return all events read so far."""
    events: list[dict] = []
    deadline = time.monotonic() + timeout
    while True:
        if predicate(events):
            return events
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"predicate unmet after {len(events)} events")
        e = chan.read_event(remaining)
        if e is None:
            raise TimeoutError(f"predicate unmet after {len(events)} events; "
                                 f"tail types: {[x.get('type') for x in events[-10:]]}")
        if e == {}:  # EOF
            raise TimeoutError(f"stdout closed after {len(events)} events")
        events.append(e)


def start_sidecar(input_dir: Path) -> tuple[subprocess.Popen, SidecarChannel]:
    cmd = [sys.executable, str(SCRIPT), "--input-dir", str(input_dir),
           "--json-lines", "--dry-run", "-f", "--workers", "1",
           "--scale-width", "512"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return proc, SidecarChannel(proc)


def test_json_lines_scan_before_run():
    proc, chan = start_sidecar(TEST_IMG)
    try:
        # scan lists the deduplicated shot list before any run
        send(proc, {"cmd": "scan", "dir": str(TEST_IMG)})
        events = wait_for(chan, lambda ev: any(e.get("type") == "scanned" for e in ev))
        scanned = [e for e in events if e["type"] == "scanned"][-1]
        assert scanned["total"] == 6
        assert len(scanned["paths"]) == 6
        assert all(Path(p).exists() for p in scanned["paths"].values())

        # run streams frames matching the scanned list
        send(proc, {"cmd": "run", "dir": str(TEST_IMG)})
        events = wait_for(chan, lambda ev: any(e.get("type") == "done" for e in ev))
        frames = [e for e in events if e["type"] == "frame"]
        by_type = {}
        for e in events:
            by_type.setdefault(e["type"], []).append(e)
        assert len(frames) == 6
        assert {f["name"] for f in frames} == set(scanned["paths"])
        assert all(f.get("status") == "scored" for f in frames)
        assert "Analyzing images..." in {e["msg"] for e in by_type["stage"]}
        done = by_type["done"][0]
        assert done["total"] == 6
        assert done["keep"] + done["reject"] == 6

        # the process stays resident: a second scan still answers
        send(proc, {"cmd": "scan", "dir": str(TEST_IMG)})
        rescan_events = wait_for(chan, lambda ev: any(e.get("type") == "scanned" for e in ev))
        assert rescan_events[-1]["type"] == "scanned"
        send(proc, {"cmd": "quit"})
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_json_lines_rerun_with_config_overrides():
    proc, chan = start_sidecar(TEST_IMG)
    try:
        # First run with the CLI defaults (top-n 11).
        send(proc, {"cmd": "run", "dir": str(TEST_IMG)})
        events = wait_for(chan, lambda ev: any(e.get("type") == "done" for e in ev))
        first_done = [e for e in events if e["type"] == "done"][0]

        # Rerun on the SAME resident process with overridden parameters —
        # the GUI adjusts settings between runs without a respawn.
        send(proc, {"cmd": "run", "dir": str(TEST_IMG),
                    "config": {"top_n": 1, "min_raw": 9.9}})
        events2 = wait_for(chan, lambda ev: any(e.get("type") == "done" for e in ev))
        events += events2
        second_done = [e for e in events2 if e["type"] == "done"][-1]
        # min_raw 9.9 vetoes everything: no keeps regardless of top-n.
        assert second_done["keep"] == 0
        assert first_done["total"] == second_done["total"] == 6

        send(proc, {"cmd": "quit"})
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_json_lines_cancel(tmp_path):
    dir_many = tmp_path / "many"
    dir_many.mkdir()
    for i in range(6):
        for src in sorted(TEST_IMG.glob("*.jpg")):
            shutil.copy(src, dir_many / f"{i:02d}_{src.name}")  # 36 images

    proc, chan = start_sidecar(dir_many)
    try:
        send(proc, {"cmd": "run", "dir": str(dir_many)})
        events = wait_for(chan, lambda ev: any(e.get("type") == "frame" for e in ev))
        proc.stdin.write(b"cancel\n")
        proc.stdin.flush()
        events += wait_for(chan, lambda ev: any(e.get("type") == "cancelled" for e in ev))
        cancelled = [e for e in events if e["type"] == "cancelled"][0]
        frames = [e for e in events if e["type"] == "frame"]
        assert cancelled["count"] == len(frames)
        assert cancelled["count"] < 36

        # Still resident after a cancel: preview answers, then a fresh run works.
        send(proc, {"cmd": "preview", "path": str(dir_many / f"00_{sorted(TEST_IMG.glob('*.jpg'))[0].name}"),
                    "size": 200})
        events = wait_for(chan, lambda ev: any(e.get("type") == "preview" for e in ev))
        preview = [e for e in events if e["type"] == "preview"][-1]
        assert preview["png"], "expected base64 PNG data"

        send(proc, {"cmd": "run", "dir": str(dir_many), "config": {"scale_width": 256}})
        wait_for(chan, lambda ev: any(e.get("type") == "done" for e in ev))
        send(proc, {"cmd": "quit"})
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_json_lines_scan_error():
    proc, chan = start_sidecar(TEST_IMG)
    try:
        send(proc, {"cmd": "scan", "dir": "no/such/dir"})
        events = wait_for(chan, lambda ev: any(e.get("type") == "scan_error" for e in ev))
        assert events[-1]["type"] == "scan_error"
        send(proc, {"cmd": "quit"})
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_frame_status_mapping():
    from cull.protocol import frame_status, STATUS_MANUAL, STATUS_DECODE_FAILED, STATUS_SCORED
    assert frame_status("") == STATUS_SCORED
    assert frame_status("sharpness=0.03 < 0.05") == STATUS_SCORED
    assert frame_status("manual_metadata") == STATUS_MANUAL
    assert frame_status("decode_failed") == STATUS_DECODE_FAILED
