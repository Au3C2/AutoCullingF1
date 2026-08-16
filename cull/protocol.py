"""protocol.py — shared JSON Lines protocol for the Tauri GUI sidecar.

The packaged CLI (``cull_photos.py --json-lines``) talks to the Tauri shell
over its stdio: one JSON object per line on stdout, commands on stdin. The
regexes below parse the engine's log lines into structured events; the GUI
worker (``cull/gui/worker.py``) uses the same patterns over in-process
logging, so both consumers stay in sync with engine.py's log format.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, TextIO

# Engine log lines emitted while scoring (engine.py format).
GROUP_RE = re.compile(r"Processing Group (\d+)/(\d+) \((\d+) frames\)")
# The veto reason may itself contain parentheses (e.g. "raw=2.728 <
# min_raw=3.100 (cut penalty applied)"), so capture greedily up to the
# final closing paren.
FRAME_RE = re.compile(
    r"^  \[(.+?)\]  sharp=([\d.]+)  comp=([\d.]+)  raw=([\d.-]+)  "
    r"Rating=([+-]?\d+)(?:  \((.+)\))?$"
)


def emit(obj: dict[str, Any], stream: TextIO | None = None) -> None:
    """Write one JSON Lines event, ignoring encoding/short-write errors."""
    stream = stream or sys.stdout
    try:
        stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
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
                      "raw": float(raw), "veto": veto or ""}, self.stream)
            emit({"type": "log", "line": message}, self.stream)
        except Exception:
            self.handleError(record)
