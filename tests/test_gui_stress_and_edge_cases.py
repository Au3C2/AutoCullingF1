"""Comprehensive Edge Case, Stress and Concurrency Test Suite for GUI Sidecar.

Tests:
1. Empty directory scan and run
2. Non-existent directory handling
3. Corrupted / 0-byte image file robustness
4. Concurrent preview requests while culling run is actively processing
5. Consecutive multi-run execution with dynamic configuration overrides
6. Zero-latency immediate cancellation
7. Mixed format folder (JPG + ARW + NEF + dirty files)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
import pytest

from test_json_protocol import SidecarChannel


@pytest.fixture
def sidecar_proc():
    cmd = [sys.executable, "-u", "cull_photos.py", "--json-lines"]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    channel = SidecarChannel(proc)
    time.sleep(0.3)
    yield channel
    channel.close()


def test_empty_directory_scan_and_run(sidecar_proc: SidecarChannel, tmp_path: Path):
    """Test scan and run behavior on an empty directory."""
    empty_dir = tmp_path / "empty_photos"
    empty_dir.mkdir()

    # 1. Scan empty dir
    sidecar_proc.clear()
    sidecar_proc.send({"cmd": "scan", "dir": str(empty_dir), "recursive": False})
    scanned = sidecar_proc.wait_for_event("scanned", timeout=5.0)
    assert scanned is not None
    assert scanned.get("count") == 0

    # 2. Run on empty dir
    sidecar_proc.clear()
    sidecar_proc.send({"cmd": "run", "dir": str(empty_dir), "config": {"dry_run": True}})
    done = sidecar_proc.wait_for_event("done", timeout=10.0)
    assert done is not None
    assert done.get("total") == 0


def test_nonexistent_directory_error_handling(sidecar_proc: SidecarChannel, tmp_path: Path):
    """Test graceful error reporting on invalid paths."""
    bad_dir = tmp_path / "does_not_exist_12345"

    # 1. Scan bad dir
    sidecar_proc.clear()
    sidecar_proc.send({"cmd": "scan", "dir": str(bad_dir)})
    scan_res = sidecar_proc.wait_for_event("scanned", timeout=3.0) or sidecar_proc.wait_for_event("scan_error", timeout=3.0)
    assert scan_res is not None

    # 2. Run bad dir
    sidecar_proc.clear()
    sidecar_proc.send({"cmd": "run", "dir": str(bad_dir)})
    err = sidecar_proc.wait_for_event("error", timeout=5.0)
    assert err is not None
    assert "not found" in err.get("message", "")


def test_corrupted_file_handling(sidecar_proc: SidecarChannel, tmp_path: Path):
    """Test engine resilience when encountering 0-byte or corrupted files."""
    corrupt_dir = tmp_path / "corrupt_test"
    corrupt_dir.mkdir()

    # 1 valid sample JPG
    valid_sample = Path("tests/test_img/IMG_20260314_151744_020.jpg")
    if valid_sample.exists():
        shutil.copy(valid_sample, corrupt_dir / "valid_01.jpg")

    # 0-byte file
    (corrupt_dir / "zero_byte.jpg").write_bytes(b"")

    # Garbled random byte file pretending to be JPG
    (corrupt_dir / "garbage.jpg").write_bytes(b"\xff\xd8\xff\xe0garbage_data_not_an_image_123456789")

    # Garbled ARW file
    (corrupt_dir / "fake.arw").write_bytes(b"II\x2a\x00corrupted_tiff_header_bytes")

    # Scan
    sidecar_proc.clear()
    sidecar_proc.send({"cmd": "scan", "dir": str(corrupt_dir)})
    scanned = sidecar_proc.wait_for_event("scanned", timeout=5.0)
    assert scanned is not None
    assert scanned.get("count") >= 3

    # Run should not crash, must finish with done event
    sidecar_proc.clear()
    sidecar_proc.send({"cmd": "run", "dir": str(corrupt_dir), "config": {"dry_run": True}})
    done = sidecar_proc.wait_for_event("done", timeout=15.0)
    assert done is not None


def test_zero_latency_immediate_cancel(sidecar_proc: SidecarChannel):
    """Test cancelling immediately upon run command without delay."""
    img_dir = Path("test_arw")
    if not img_dir.exists():
        img_dir = Path("tests/test_img")

    sidecar_proc.clear()
    # Send run and cancel immediately back to back
    sidecar_proc.send({"cmd": "run", "dir": str(img_dir), "config": {"dry_run": True}})
    sidecar_proc.send({"cmd": "cancel"})

    evt = sidecar_proc.wait_for_event("cancelled", timeout=8.0) or sidecar_proc.wait_for_event("done", timeout=8.0)
    assert evt is not None


def test_concurrent_preview_requests_during_active_run(sidecar_proc: SidecarChannel):
    """Test that previews can be served concurrently while scoring pipeline is running."""
    img_dir = Path("test_arw")
    if not img_dir.exists():
        img_dir = Path("tests/test_img")

    # Start run
    sidecar_proc.clear()
    sidecar_proc.send({
        "cmd": "run",
        "dir": str(img_dir),
        "config": {"dry_run": True, "workers": 4}
    })

    # Rapidly fire preview requests while running
    files = list(img_dir.glob("*.ARW")) or list(img_dir.glob("*.jpg"))
    assert len(files) > 0

    preview_results = []
    def request_previews():
        for f in files[:4]:
            sidecar_proc.send({"cmd": "preview", "path": str(f.resolve()), "size": 320})
            time.sleep(0.05)

    th = threading.Thread(target=request_previews)
    th.start()
    th.join()

    # Wait for run completion
    done = sidecar_proc.wait_for_event("done", timeout=30.0)
    assert done is not None

    # Check that preview events arrived
    previews = sidecar_proc.get_events_by_type("preview")
    assert len(previews) > 0


def test_multiple_consecutive_runs_on_same_process(sidecar_proc: SidecarChannel):
    """Test running multiple culling jobs with different configs without restarting sidecar."""
    img_dir = Path("tests/test_img")
    if not img_dir.exists():
        pytest.skip("tests/test_img not found")

    # Round 1: Top-N = 1, dry_run = True
    sidecar_proc.clear()
    sidecar_proc.send({"cmd": "run", "dir": str(img_dir), "config": {"dry_run": True, "top_n": 1}})
    done1 = sidecar_proc.wait_for_event("done", timeout=15.0)
    assert done1 is not None

    # Round 2: Top-N = 5, min_raw = 0.5
    sidecar_proc.clear()
    sidecar_proc.send({"cmd": "run", "dir": str(img_dir), "config": {"dry_run": True, "top_n": 5, "min_raw": 0.5}})
    done2 = sidecar_proc.wait_for_event("done", timeout=15.0)
    assert done2 is not None

    # Round 3: Cancel mid-way
    sidecar_proc.clear()
    sidecar_proc.send({"cmd": "run", "dir": str(img_dir), "config": {"dry_run": True}})
    time.sleep(0.05)
    sidecar_proc.send({"cmd": "cancel"})
    cancel_evt = sidecar_proc.wait_for_event("cancelled", timeout=10.0) or sidecar_proc.wait_for_event("done", timeout=10.0)
    assert cancel_evt is not None

    # Round 4: Normal run after cancellation to verify engine recovers cleanly
    sidecar_proc.clear()
    sidecar_proc.send({"cmd": "run", "dir": str(img_dir), "config": {"dry_run": True, "top_n": 3}})
    done4 = sidecar_proc.wait_for_event("done", timeout=15.0)
    assert done4 is not None
    assert done4.get("total") == 6
