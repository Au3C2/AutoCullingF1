"""
Model definitions for the F1 motorsport photo auto-culling binary classifier.

Supported architectures
-----------------------
- resnet18      : ResNet-18  (torchvision)
- resnet50      : ResNet-50  (torchvision)
- resnext50     : ResNeXt-50-32x4d (torchvision)
- mobilenetv3   : MobileNetV3-Large (torchvision)

Fine-tuning strategy
--------------------
In fine-tune mode (``finetune=True``) all parameters are **frozen** except:

* ResNet-18/50 and ResNeXt-50  → ``layer4``  + ``fc``
* MobileNetV3-Large            → ``features[-2:]`` + ``classifier``

The final fully-connected / classifier head is always replaced with a
single-output linear layer for binary classification (BCEWithLogitsLoss).

Usage
-----
    from auto_culling.model import build_model

    model = build_model("resnet50", finetune=True, pretrained=True)
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNeXt50_32X4D_Weights,
)

log = logging.getLogger(__name__)

# Public type alias for supported architecture names
ArchName = Literal["resnet18", "resnet50", "resnext50", "mobilenetv3"]

_ARCH_NAMES: tuple[str, ...] = ("resnet18", "resnet50", "resnext50", "mobilenetv3")


def _freeze_all(model: nn.Module) -> None:
    """Set ``requires_grad = False`` for every parameter in *model*."""
    for param in model.parameters():
        param.requires_grad = False


def _unfreeze(module: nn.Module) -> None:
    """Set ``requires_grad = True`` for every parameter in *module*."""
    for param in module.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Per-architecture builders
# ---------------------------------------------------------------------------


def _build_resnet18(finetune: bool, pretrained: bool) -> nn.Module:
    """Build ResNet-18 adapted for binary classification.

    Parameters
    ----------
    finetune:
        Freeze all layers except ``layer3``, ``layer4`` and ``fc``.
    pretrained:
        Load ImageNet-1k weights when ``True``.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if finetune:
        _freeze_all(model)
        _unfreeze(model.layer3)
        _unfreeze(model.layer4)

    # Replace classifier head: 512 → Dropout(0.3) → 1
    in_features = model.fc.in_features
    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )

    return model


def _build_resnet50(finetune: bool, pretrained: bool) -> nn.Module:
    """Build ResNet-50 adapted for binary classification.

    Parameters
    ----------
    finetune:
        Freeze all layers except ``layer3``, ``layer4`` and ``fc``.
    pretrained:
        Load ImageNet-1k V2 weights when ``True``.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if finetune:
        _freeze_all(model)
        _unfreeze(model.layer3)
        _unfreeze(model.layer4)

    # Replace classifier head: 2048 → Dropout(0.3) → 1
    in_features = model.fc.in_features
    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )

    return model


def _build_resnext50(finetune: bool, pretrained: bool) -> nn.Module:
    """Build ResNeXt-50-32x4d adapted for binary classification.

    Parameters
    ----------
    finetune:
        Freeze all layers except ``layer3``, ``layer4`` and ``fc``.
    pretrained:
        Load ImageNet-1k V2 weights when ``True``.
    """
    weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnext50_32x4d(weights=weights)

    if finetune:
        _freeze_all(model)
        _unfreeze(model.layer3)
        _unfreeze(model.layer4)

    # Replace classifier head: 2048 → Dropout(0.3) → 1
    in_features = model.fc.in_features
    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )

    return model


def _build_mobilenetv3(finetune: bool, pretrained: bool) -> nn.Module:
    """Build MobileNetV3-Large adapted for binary classification.

    Parameters
    ----------
    finetune:
        Freeze all layers except the last two feature blocks and
        the classifier.
    pretrained:
        Load ImageNet-1k V2 weights when ``True``.
    """
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    if finetune:
        _freeze_all(model)
        # Unfreeze last two inverted-residual blocks (index -2 and -1)
        _unfreeze(model.features[-1])
        _unfreeze(model.features[-2])

    # Replace classifier: [dropout, linear(960,1280), hardswish, dropout, linear(1280,1000)]
    # Keep the existing feature extraction up to the penultimate linear, swap final head.
    last_linear = model.classifier[-1]
    assert isinstance(last_linear, nn.Linear), "Expected nn.Linear as last classifier layer"
    in_features: int = last_linear.in_features
    model.classifier[-1] = nn.Linear(in_features, 1)

    return model


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

_BUILDERS = {
    "resnet18": _build_resnet18,
    "resnet50": _build_resnet50,
    "resnext50": _build_resnext50,
    "mobilenetv3": _build_mobilenetv3,
}


def build_model(
    arch: ArchName,
    *,
    finetune: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """Construct a binary-classification model for the given architecture.

    The returned model outputs a single un-activated logit per sample,
    compatible with ``torch.nn.BCEWithLogitsLoss``.

    Parameters
    ----------
    arch:
        Architecture name. One of ``"resnet18"``, ``"resnet50"``,
        ``"resnext50"``, ``"mobilenetv3"``.
    finetune:
        When ``True``, freeze the backbone and only train the last block(s)
        plus the classification head.
    pretrained:
        When ``True``, initialise the backbone with ImageNet weights.

    Returns
    -------
    nn.Module
        Model instance (not moved to any device — caller is responsible).

    Raises
    ------
    ValueError
        If *arch* is not one of the supported names.

    Examples
    --------
    >>> model = build_model("resnet50", finetune=True, pretrained=True)
    >>> model.cuda()
    >>> logits = model(torch.randn(4, 3, 224, 224).cuda())
    >>> logits.shape
    torch.Size([4, 1])
    """
    arch = arch.lower()  # type: ignore[assignment]
    if arch not in _BUILDERS:
        raise ValueError(
            f"Unknown architecture {arch!r}. "
            f"Choose from: {list(_BUILDERS.keys())}"
        )

    model = _BUILDERS[arch](finetune=finetune, pretrained=pretrained)

    # ---------- logging summary ----------
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    log.info(
        "Built model: arch=%s  pretrained=%s  finetune=%s  "
        "total_params=%s  trainable=%s  frozen=%s",
        arch,
        pretrained,
        finetune,
        f"{total:,}",
        f"{trainable:,}",
        f"{frozen:,}",
    )

    return model


def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    """Return the list of parameters that require gradient updates.

    Parameters
    ----------
    model:
        Any ``nn.Module``.

    Returns
    -------
    list[nn.Parameter]
        Parameters with ``requires_grad == True``.
    """
    return [p for p in model.parameters() if p.requires_grad]


def available_archs() -> tuple[str, ...]:
    """Return a tuple of supported architecture names."""
    return _ARCH_NAMES
