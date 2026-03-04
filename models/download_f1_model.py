"""
download_f1_model.py — Download F1 YOLO model weights from Roboflow and export to ONNX.

Usage
-----
    # Download and export (requires roboflow & ultralytics installed)
    python models/download_f1_model.py

    # Override API key and output path
    python models/download_f1_model.py \\
        --api-key YOUR_KEY \\
        --out models/f1_yolov8n.onnx

What this script does
---------------------
1. Connect to Roboflow using the provided API key.
2. Download ``jayanths-workspace/formula-one-car-detection`` version 1
   in ``yolov8`` format (downloads dataset + best.pt weights).
3. Export ``best.pt`` → ``best.onnx`` using ultralytics.
4. Copy the ONNX file to the target output path.
5. Run a quick ORT sanity-check to confirm the model loads correctly.

If the download fails due to insufficient permissions (the model belongs to
another workspace), the script prints clear instructions for the cloud-API
fallback (inference_sdk against serverless.roboflow.com).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Default Roboflow model coordinates
_DEFAULT_WORKSPACE = "jayanths-workspace"
_DEFAULT_PROJECT   = "formula-one-car-detection"
_DEFAULT_VERSION   = 1
_DEFAULT_API_KEY   = "NQI4igULp4JqFsyY2L2t"  # provided by user; replace if expired
_DEFAULT_OUT       = Path(__file__).parent / "f1_yolov8n.onnx"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_best_pt(download_dir: Path) -> Path | None:
    """Search for best.pt anywhere inside *download_dir*."""
    candidates = sorted(download_dir.rglob("best.pt"))
    if candidates:
        return candidates[0]
    # Also try last.pt as a fallback
    fallbacks = sorted(download_dir.rglob("*.pt"))
    return fallbacks[0] if fallbacks else None


def _export_to_onnx(pt_path: Path, out_path: Path) -> None:
    """Export a YOLOv8 .pt file to ONNX using ultralytics."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for ONNX export.\n"
            "  Install: pip install ultralytics"
        ) from exc

    log.info("Loading weights from %s …", pt_path)
    model = YOLO(str(pt_path))

    log.info("Exporting to ONNX (dynamic batch, opset 17) …")
    exported = model.export(
        format="onnx",
        dynamic=True,
        opset=17,
        simplify=True,
    )
    # ultralytics returns the path of the exported file as a string
    onnx_src = Path(str(exported))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(onnx_src), str(out_path))
    log.info("ONNX model saved to %s", out_path)


def _ort_sanity_check(onnx_path: Path) -> None:
    """Load the ONNX with ORT and run a dummy forward pass."""
    try:
        import numpy as np
        import onnxruntime as ort  # type: ignore
    except ImportError:
        log.warning("onnxruntime not installed — skipping sanity check.")
        return

    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    inp = sess.get_inputs()[0]
    # Build dummy input matching the model's expected shape (replace dynamic dims with 1/640)
    shape = [d if isinstance(d, int) else (1 if i == 0 else 640) for i, d in enumerate(inp.shape)]
    dummy = np.zeros(shape, dtype=np.float32)
    sess.run(None, {inp.name: dummy})
    log.info("ORT sanity check passed  (provider: %s)", sess.get_providers()[0])


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------


def download_and_export(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    out_path: Path,
) -> None:
    """Download model weights from Roboflow and export to ONNX.

    Parameters
    ----------
    api_key:
        Roboflow API key.
    workspace:
        Roboflow workspace slug.
    project:
        Roboflow project slug.
    version:
        Dataset/model version number.
    out_path:
        Destination path for the exported ONNX file.
    """
    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "roboflow SDK is required.\n"
            "  Install: pip install roboflow"
        ) from exc

    rf = Roboflow(api_key=api_key)

    log.info("Connecting to %s/%s (version %d) …", workspace, project, version)
    try:
        proj = rf.workspace(workspace).project(project)
        ver  = proj.version(version)
    except Exception as exc:
        _print_permission_error(exc)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        log.info("Downloading YOLOv8 weights to %s …", tmpdir)
        try:
            ver.download("yolov8", location=tmpdir, overwrite=True)
        except Exception as exc:
            _print_permission_error(exc)
            sys.exit(1)

        pt_path = _find_best_pt(Path(tmpdir))
        if pt_path is None:
            log.error(
                "Download completed but no .pt file found in %s.\n"
                "Directory contents:\n  %s",
                tmpdir,
                "\n  ".join(str(p) for p in Path(tmpdir).rglob("*")),
            )
            sys.exit(1)

        log.info("Found weights: %s", pt_path)
        _export_to_onnx(pt_path, out_path)

    _ort_sanity_check(out_path)
    print(f"\n✓  F1 model ready: {out_path}")
    print(
        "\nNext step: run the culling pipeline with\n"
        f"  python cull_photos.py --input-dir /path/to/photos "
        f"--f1-model {out_path}"
    )


def _print_permission_error(exc: Exception) -> None:
    print(
        "\n[ERROR] Could not download model weights from Roboflow.\n"
        f"  Reason: {exc}\n\n"
        "This likely means the model is owned by another workspace and\n"
        "the API key does not have download permission.\n\n"
        "Fallback option — use the Roboflow cloud API instead:\n"
        "  The culling pipeline (cull_photos.py) will automatically fall back\n"
        "  to inference_sdk when no local ONNX is present.  Pass:\n"
        "    --rf-api-key NQI4igULp4JqFsyY2L2t\n"
        "  and it will call serverless.roboflow.com for F1 detection.\n",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download F1 YOLO weights from Roboflow and export to ONNX.\n"
            "Requires: roboflow, ultralytics, onnxruntime"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-key",
        default=_DEFAULT_API_KEY,
        help="Roboflow API key.",
    )
    parser.add_argument(
        "--workspace",
        default=_DEFAULT_WORKSPACE,
        help="Roboflow workspace slug.",
    )
    parser.add_argument(
        "--project",
        default=_DEFAULT_PROJECT,
        help="Roboflow project slug.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=_DEFAULT_VERSION,
        help="Dataset/model version number.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output path for the exported ONNX file.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    download_and_export(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
