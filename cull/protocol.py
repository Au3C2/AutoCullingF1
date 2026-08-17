"""protocol.py — shared JSON Lines protocol for the Tauri GUI sidecar.

The packaged CLI (``cull_photos.py --json-lines``) talks to the Tauri shell
over its stdio: one JSON object per line on stdout, JSON commands on stdin.
The sidecar process is RESIDENT: it answers ``scan`` / ``run`` / ``cancel`` /
``preview`` commands until ``quit`` arrives, so the GUI can list a directory
and request thumbnails without respawning the engine.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from typing import Any, TextIO

# stdout writes are serialized: the engine logs from ThreadPoolExecutor worker
# threads, and concurrent TextIOWrapper write+flush from several threads can
# interleave partial lines — the GUI then blocks forever on a malformed line.
_STDOUT_LOCK = threading.Lock()


# Engine log lines emitted while scoring (engine.py format).
GROUP_RE = re.compile(r"Processing Group (\d+)/(\d+) \((\d+) frames\)")
# The veto reason may itself contain parentheses (e.g. "raw=2.728 <
# min_raw=3.100 (cut penalty applied)"), so capture greedily up to the
# final closing paren.
FRAME_RE = re.compile(
    r"^  \[(.+?)\]  sharp=([\d.]+)  comp=([\d.]+)  raw=([\d.-]+)  "
    r"Rating=([+-]?\d+)(?:  \((.+)\))?$"
)

# Frame outcomes carried in the frame event's "status" field.
STATUS_SCORED = "scored"
STATUS_MANUAL = "manual"
STATUS_DECODE_FAILED = "decode_failed"

_MANUAL_STATUSES = {"manual_metadata"}
_FAILED_STATUSES = {"decode_failed", "load_failed"}


def frame_status(veto_reason: str) -> str:
    """Map an engine veto reason to a frame-event status."""
    if veto_reason in _MANUAL_STATUSES:
        return STATUS_MANUAL
    if veto_reason in _FAILED_STATUSES:
        return STATUS_DECODE_FAILED
    return STATUS_SCORED


def emit(obj: dict[str, Any], stream: TextIO | None = None) -> None:
    """Write one JSON Lines event, ignoring encoding/short-write errors."""
    stream = stream or sys.stdout
    data = json.dumps(obj, ensure_ascii=False) + "\n"
    try:
        with _STDOUT_LOCK:
            stream.write(data)
            stream.flush()
    except Exception:
        pass


class JsonLinesHandler(logging.Handler):
    """Root-logger handler that re-emits engine progress as JSON Lines events.

    GROUP/FRAME engine log lines become structured events (``group`` /
    ``frame``); all other INFO records are forwarded as ``log`` events so the
    GUI can show a live log panel.
    """

    def __init__(self, stream: TextIO | None = None):
        super().__init__(level=logging.INFO)
        self.stream = stream or sys.stdout

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            group_match = GROUP_RE.search(message)
            if group_match:
                emit({"type": "group", "done": int(group_match.group(1)),
                      "total": int(group_match.group(2)),
                      "frames": int(group_match.group(3))}, self.stream)
            frame_match = FRAME_RE.match(message)
            if frame_match:
                name, sharp, comp, raw, rating, veto = frame_match.groups()
                emit({"type": "frame", "name": name, "rating": int(rating),
                      "sharp": float(sharp), "comp": float(comp),
                      "raw": float(raw), "veto": veto or "",
                      "status": frame_status(veto or "")}, self.stream)
            emit({"type": "log", "line": message}, self.stream)
        except Exception:
            self.handleError(record)
