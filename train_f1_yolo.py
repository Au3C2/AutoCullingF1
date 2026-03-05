"""
train_f1_yolo.py — Train a YOLOv8n model for F1 car detection.

Uses the merged F1 dataset (10 teams, from three Roboflow sources)
at ``datasets/f1_merged/``.  See ``merge_datasets.py`` to regenerate.

Usage
-----
    python train_f1_yolo.py

    # Custom epochs / image size
    python train_f1_yolo.py --epochs 150 --imgsz 640

    # Resume from a checkpoint
    python train_f1_yolo.py --resume

After training, the best model is exported to ONNX at
``models/f1_yolov8n.onnx`` for use with ``cull_photos.py --f1-model``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
_DATA_YAML    = _PROJECT_ROOT / "datasets" / "f1_merged" / "data.yaml"
_TRAIN_DIR    = _PROJECT_ROOT / "runs" / "f1_detect"
_ONNX_OUT     = _PROJECT_ROOT / "models" / "f1_yolov8n.onnx"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_f1_yolo",
        description="Train YOLOv8n on F1 car detection dataset and export ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size for training.",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (adjust for VRAM).  -1 for auto-batch.",
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Device to train on ('0' for first GPU, 'cpu' for CPU).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint.",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip ONNX export after training.",
    )
    parser.add_argument(
        "--data", type=Path, default=_DATA_YAML,
        help="Path to data.yaml.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.data.exists():
        print(f"ERROR: data.yaml not found: {args.data}")
        print("  Run merge_datasets.py --merge first to create the merged dataset.")
        return 1

    # --- Train ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training YOLOv8n on F1 car detection dataset")
    print(f"  data:   {args.data}")
    print(f"  epochs: {args.epochs}")
    print(f"  imgsz:  {args.imgsz}")
    print(f"  batch:  {args.batch}")
    print(f"  device: {args.device}")
    print(f"{'='*60}\n")

    if args.resume:
        # Resume from last run
        model = YOLO(str(_TRAIN_DIR / "train" / "weights" / "last.pt"))
        results = model.train(resume=True)
    else:
        # Train from pretrained YOLOv8n
        model = YOLO("yolov8n.pt")
        results = model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(_TRAIN_DIR),
            name="train",
            exist_ok=True,
            # Augmentation (sensible defaults for object detection)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,       # no rotation — F1 cars are always upright
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.0,        # no vertical flip — cars don't fly
            mosaic=1.0,
            mixup=0.1,
            # Training params
            patience=15,       # early stopping patience
            lr0=0.01,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            cos_lr=True,
            amp=True,
            # Save
            save=True,
            save_period=-1,    # only save best + last
            plots=True,
            verbose=True,
        )

    # --- Evaluate on test set ------------------------------------------------
    best_pt = _TRAIN_DIR / "train" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\nBest model: {best_pt}")
        best_model = YOLO(str(best_pt))

        print("\nValidation metrics:")
        val_results = best_model.val(data=str(args.data), split="val", device=args.device)

        print("\nTest metrics:")
        test_results = best_model.val(data=str(args.data), split="test", device=args.device)
    else:
        print(f"WARNING: best.pt not found at {best_pt}")
        best_model = model

    # --- Export to ONNX ------------------------------------------------------
    if not args.no_export:
        print(f"\nExporting to ONNX: {_ONNX_OUT}")
        _ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)

        exported_path = best_model.export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=True,     # dynamic batch axis
            simplify=True,
            opset=17,
        )

        # Move to target location
        exported = Path(exported_path)
        if exported.exists() and exported != _ONNX_OUT:
            shutil.copy2(exported, _ONNX_OUT)
            print(f"  Copied to {_ONNX_OUT}")

        print(f"  ONNX export complete: {_ONNX_OUT}")
        print(f"  Size: {_ONNX_OUT.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("\nSkipping ONNX export (--no-export)")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  Best weights:  {best_pt}")
    if not args.no_export:
        print(f"  ONNX model:    {_ONNX_OUT}")
    print(f"  Use with:      python cull_photos.py --f1-model {_ONNX_OUT}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
