"""
deterministic.py — Cross-platform deterministic execution gate.

When ``CULL_DETERMINISTIC=1`` (or ``--deterministic`` on the CLI) the
pipeline forces a single, platform-independent code path so that the same
input files produce bit-identical ``raw_score``/``rating`` on macOS and
Windows:

* ONNX inference: ``CPUExecutionProvider`` only, ``intra/inter_op_num_threads=1``,
  ``use_deterministic_compute=True`` when the ORT build exposes it.
* Image decode: software-only (no ImageIO / VideoToolbox / ffmpeg -hwaccel /
  DXVA2/CUDA/VAAPI probes) and a single letterbox/resize kernel.
* FFT sharpness: single-threaded ``rfft2`` (``workers=1``) to avoid
  summation-order jitter.

The helper is intentionally tiny so every consumer can import it without
pulling heavy dependencies.  The env var is read on every call so
``CULL_DETERMINISTIC=1 cull_photos.py ...`` and ``--deterministic`` are
interchangeable — the CLI sets the env var early in ``main()``.
"""
from __future__ import annotations

import os


def is_deterministic() -> bool:
    v = os.environ.get("CULL_DETERMINISTIC", "")
    return v not in ("", "0", "false", "False", "FALSE", "off", "OFF")


def set_deterministic(enabled: bool = True) -> None:
    os.environ["CULL_DETERMINISTIC"] = "1" if enabled else "0"
