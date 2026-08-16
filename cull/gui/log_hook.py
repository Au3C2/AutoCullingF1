"""log_hook.py — forward logging records to a thread-safe queue for the GUI."""

from __future__ import annotations

import logging
import queue


class QueueLogHandler(logging.Handler):
    """logging.Handler that pushes formatted records onto a queue.

    ``logging`` calls handlers from the emitter's thread, so this is safe to
    use from the worker threads; the GUI consumes the queue on the main
    thread.
    """

    def __init__(self, msg_queue: "queue.Queue", level: int = logging.INFO):
        super().__init__(level)
        self.msg_queue = msg_queue
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.msg_queue.put((self.format(record),))
        except Exception:
            self.handleError(record)
