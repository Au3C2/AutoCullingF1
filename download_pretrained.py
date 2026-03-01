"""
Pre-trained weight downloader for all supported backbone architectures.

This script uses torchvision's built-in weight registry to download and cache
the ImageNet pre-trained checkpoints that will be used during fine-tuning.
Running it once ensures the weights are available offline (useful in
restricted-network training environments).

Cached weights are stored in the directory specified by ``TORCH_HOME``
(default: ``~/.cache/torch/hub/checkpoints/``).

Usage
-----
    uv run download_pretrained.py [--weights-dir PATH] [--arch ARCH [ARCH ...]]

Examples
--------
    # Download all four architectures
    uv run download_pretrained.py

    # Download only ResNet-50 and MobileNetV3
    uv run download_pretrained.py --arch resnet50 mobilenetv3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNeXt50_32X4D_Weights,
    mobilenet_v3_large,
    resnet18,
    resnet50,
    resnext50_32x4d,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple] = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnext50": (resnext50_32x4d, ResNeXt50_32X4D_Weights.IMAGENET1K_V2),
    "mobilenetv3": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2),
}


def download_weights(
    archs: list[str],
    weights_dir: Path | None,
) -> None:
    """Download and cache pre-trained weights for the specified architectures.

    Parameters
    ----------
    archs:
        List of architecture names to download.
    weights_dir:
        Optional custom cache directory.  When provided, ``TORCH_HOME`` is set
        to this path so torchvision stores the weights there.
    """
    if weights_dir is not None:
        weights_dir = Path(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(weights_dir)
        log.info("TORCH_HOME set to: %s", weights_dir)

    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    log.info("Weights will be cached in: %s", cache_dir)

    for arch in archs:
        if arch not in _REGISTRY:
            log.warning("Unknown architecture %r — skipping.", arch)
            continue

        builder_fn, weights_enum = _REGISTRY[arch]
        log.info("Downloading weights for: %s  (%s)", arch, weights_enum)
        try:
            _ = builder_fn(weights=weights_enum)
            log.info("  OK — %s", arch)
        except Exception as exc:
            log.error("  FAILED — %s: %s", arch, exc)
            raise

    log.info("All requested weights downloaded successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="download_pretrained",
        description="Download ImageNet pre-trained weights for supported backbones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arch",
        nargs="+",
        default=list(_REGISTRY.keys()),
        choices=list(_REGISTRY.keys()),
        metavar="ARCH",
        help="Architecture(s) to download. Defaults to all four.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Custom directory for cached weights (overrides TORCH_HOME).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry-point."""
    args = parse_args(argv)
    try:
        download_weights(archs=args.arch, weights_dir=args.weights_dir)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
