"""worker.py — run CullingEngine off the GUI thread, streaming progress to a queue."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from cull.engine import CullingEngine, EngineConfig
from cull.protocol import FRAME_RE, GROUP_RE

log = logging.getLogger(__name__)

# Message kinds pushed onto the queue as (kind, *payload) tuples.
STAGE = "stage"          # (msg: str, progress: float) — phase progress 0..1
LOG_LINE = "log"         # (line: str) — formatted log line
TOTAL = "total"          # (frames: int) — estimated frame count before scoring
PATHS = "paths"          # ({name: abs_path}) — pre-scan name → path map
GROUP = "group"          # (done: int, total: int, frames: int) — group progress
FRAME = "frame"          # (name, rating, sharp, comp, raw, veto_reason) — one frame scored
DONE = "done"            # (result: dict) — run finished
CANCELLED = "cancelled"  # (scores: list[ImageScore]) — run cancelled
ERROR = "error"          # (message: str) — unexpected failure


class _ParsingHandler(logging.Handler):
    """Forwards log lines and emits structured GROUP/FRAME messages on match."""

    def __init__(self, msg_queue: "queue.Queue"):
        super().__init__(level=logging.INFO)
        self.msg_queue = msg_queue
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            group_match = GROUP_RE.search(message)
            if group_match:
                self.msg_queue.put((GROUP, int(group_match.group(1)), int(group_match.group(2)),
                                    int(group_match.group(3))))
            frame_match = FRAME_RE.match(message)
            if frame_match:
                name, sharp, comp, raw, rating, veto = frame_match.groups()
                self.msg_queue.put((FRAME, name, int(rating), float(sharp), float(comp),
                                    float(raw), veto or ""))
            self.msg_queue.put((LOG_LINE, self.format(record)))
        except Exception:
            self.handleError(record)


class CullWorker:
    """Runs one culling session on a background thread.

    Progress events are pushed onto *msg_queue* as (kind, payload) tuples;
    the GUI polls the queue from the main thread. Call :meth:`start` to run,
    :meth:`stop` to request cancellation, and :meth:`is_running` to query
    the state. The engine instance stays available as ``engine`` after the
    run (e.g. for CSV export).
    """

    def __init__(self, config: EngineConfig, msg_queue: "queue.Queue"):
        self.config = config
        self.msg_queue = msg_queue
        self.cancel_event = threading.Event()
        self.engine: CullingEngine | None = None
        self._thread: threading.Thread | None = None
        self._total_frames = 0

    def start(self) -> None:
        if self.is_running():
            return
        self.cancel_event.clear()
        self._thread = threading.Thread(target=self._run, name="cull-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.cancel_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # --- internals ---

    def _run(self) -> None:
        handler = _ParsingHandler(self.msg_queue)
        root = logging.getLogger()
        previous_level = root.level
        if previous_level > logging.INFO:
            root.setLevel(logging.INFO)  # the GUI needs INFO records from the engine
        root.addHandler(handler)
        try:
            self.engine = CullingEngine(self.config)
            # Cheap pre-scan so the GUI can show a frame count estimate
            # while the (silent) parallel scoring phase is running.
            try:
                pre_paths = self.engine._collect_images(self.config.input_dir, self.config.recursive)
                self._total_frames = len(pre_paths)
                self.msg_queue.put((PATHS, {p.name: str(p) for p in pre_paths}))
            except Exception:
                self._total_frames = 0
            self.msg_queue.put((TOTAL, self._total_frames))

            def progress(msg: str, p: float) -> None:
                self.msg_queue.put((STAGE, msg, p))

            t0 = time.perf_counter()
            scores, scoring_elapsed = self.engine.run(
                progress_callback=progress, cancel_event=self.cancel_event
            )

            if self.cancel_event.is_set():
                self.msg_queue.put((CANCELLED, scores))
                return

            result: dict[str, Any] = {
                "scores": scores,
                "total": len(scores),
                "elapsed": time.perf_counter() - t0,
                "scoring_elapsed": scoring_elapsed,
            }
            if result["total"]:
                result["keep"] = sum(1 for s in scores if s.rating > 0)
                result["reject"] = result["total"] - result["keep"]
                stars: dict[int, int] = {}
                for s in scores:
                    if s.rating > 0:
                        stars[s.rating] = stars.get(s.rating, 0) + 1
                result["stars"] = stars
            self.msg_queue.put((DONE, result))
        except Exception as exc:
            log.exception("Culling failed")
            self.msg_queue.put((ERROR, str(exc)))
        finally:
            root.removeHandler(handler)
            root.setLevel(previous_level)
