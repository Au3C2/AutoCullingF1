"""
Export trained checkpoints to ONNX format.

For each architecture the script:
1. Builds the model via ``build_model`` (same arch / finetune settings used
   during training).
2. Loads ``best.pt`` from the corresponding checkpoint directory.
3. Exports the model to ONNX with a fixed or dynamic batch dimension.
4. Runs a quick ONNX sanity check with onnxruntime.

Usage
-----
    # Export all four v2 models (default):
    python export_onnx.py

    # Export a single architecture:
    python export_onnx.py --arch resnext50

    # Custom checkpoint dir pattern and output dir:
    python export_onnx.py \\
        --ckpt-pattern "checkpoints/{arch}_finetune_v2/best.pt" \\
        --output-dir onnx_models \\
        --img-size 224
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported architectures and their default checkpoint paths
# ---------------------------------------------------------------------------

ARCH_CKPT: dict[str, str] = {
    "resnet18":    "checkpoints/resnet18_finetune_v2/best.pt",
    "resnet50":    "checkpoints/resnet50_finetune_v2/best.pt",
    "resnext50":   "checkpoints/resnext50_finetune_v2/best.pt",
    "mobilenetv3": "checkpoints/mobilenetv3_finetune_v2/best.pt",
}


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------


def export_model(
    arch: str,
    ckpt_path: Path,
    output_path: Path,
    img_size: int = 224,
    dynamic_batch: bool = True,
    opset: int = 17,
) -> None:
    """Load a checkpoint, export to ONNX, and verify with onnxruntime.

    Parameters
    ----------
    arch:
        Architecture name recognised by ``build_model``.
    ckpt_path:
        Path to the ``.pt`` checkpoint produced by ``train.py``.
    output_path:
        Destination ``.onnx`` file path.
    img_size:
        Spatial resolution used during training (default 224).
    dynamic_batch:
        When ``True`` the batch dimension is exported as dynamic (``-1``),
        allowing arbitrary batch sizes at runtime.
    opset:
        ONNX opset version.
    """
    # Lazy import so users without onnx installed get a clear error only here
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnx and onnxruntime are required for export.\n"
            "Install with: pip install onnx onnxruntime-gpu"
        ) from exc

    from auto_culling.model import build_model

    log.info("─── %s ─────────────────────────────────────────────", arch)
    log.info("  checkpoint : %s", ckpt_path)
    log.info("  output     : %s", output_path)

    # ---- Build model (finetune=True keeps the same head structure) ----------
    model = build_model(arch, finetune=True, pretrained=False)

    # ---- Load checkpoint ----------------------------------------------------
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    log.info("  Loaded checkpoint (step=%d, val_f1=%.4f)",
             state.get("step", -1), state.get("val_f1", float("nan")))

    # ---- Set eval mode + disable Dropout for deterministic export -----------
    model.eval()

    # ---- Dummy input --------------------------------------------------------
    dummy_input = torch.randn(1, 3, img_size, img_size, requires_grad=False)

    # ---- Export to ONNX -----------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    log.info("  ONNX export done → %s", output_path)

    # ---- ONNX model check ---------------------------------------------------
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    log.info("  ONNX model check passed")

    # ---- Quick onnxruntime inference sanity check ---------------------------
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(output_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: dummy_input.numpy()})
    import numpy as np
    log.info(
        "  ORT inference OK — output shape %s, logit=%.4f",
        ort_out[0].shape,
        float(ort_out[0].flat[0]),
    )

    # ---- PyTorch vs ORT output comparison -----------------------------------
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()
    max_diff = float(np.abs(pt_out - ort_out[0]).max())
    log.info("  Max |PT − ORT| diff = %.6f", max_diff)
    if max_diff > 1e-2:
        log.warning("  Large numerical difference between PyTorch and ORT outputs!")
    else:
        log.info("  Numerical match OK (diff < 1e-2)")

    # ---- File size ----------------------------------------------------------
    size_mb = output_path.stat().st_size / 1024 / 1024
    log.info("  ONNX file size: %.1f MB", size_mb)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="export_onnx",
        description="Export F1 culling model checkpoints to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arch",
        nargs="+",
        default=list(ARCH_CKPT.keys()),
        choices=list(ARCH_CKPT.keys()),
        help="Architecture(s) to export.  Defaults to all four.",
    )
    parser.add_argument(
        "--ckpt-pattern",
        type=str,
        default=None,
        help=(
            "Override checkpoint path pattern.  Use {arch} as placeholder, "
            "e.g. 'checkpoints/{arch}_finetune_v2/best.pt'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("onnx_models"),
        help="Directory to write .onnx files.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image resolution (square).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--no-dynamic-batch",
        dest="dynamic_batch",
        action="store_false",
        default=True,
        help="Fix batch size to 1 instead of exporting a dynamic batch axis.",
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

    errors: list[str] = []

    for arch in args.arch:
        # Determine checkpoint path
        if args.ckpt_pattern:
            ckpt_path = Path(args.ckpt_pattern.format(arch=arch))
        else:
            ckpt_path = Path(ARCH_CKPT[arch])

        output_path = args.output_dir / f"{arch}.onnx"

        try:
            export_model(
                arch=arch,
                ckpt_path=ckpt_path,
                output_path=output_path,
                img_size=args.img_size,
                dynamic_batch=args.dynamic_batch,
                opset=args.opset,
            )
        except Exception as exc:
            log.error("Failed to export %s: %s", arch, exc)
            errors.append(arch)

    if errors:
        log.error("Export FAILED for: %s", ", ".join(errors))
        return 1

    log.info("All exports complete → %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
