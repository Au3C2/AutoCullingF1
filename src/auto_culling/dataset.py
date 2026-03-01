"""
Dataset and DataLoader utilities for the F1 motorsport photo culling task.

Design notes
------------
* Original HIF/HEIF files have been deleted.  Images are now read from a
  pre-decoded JPEG cache in ``dataset/cache/<stem>.jpg``.
* **Early resize in __getitem__**: immediately after opening the JPEG the
  image is resized to ``img_size × img_size`` via SquarePad + Pillow resize
  *before* handing it to the torchvision transform pipeline.  This keeps the
  in-process PIL image small (e.g. 224×224 × 3 bytes ≈ 150 KB vs the
  original 4672×7008 ≈ 94 MB), which is the key fix for the DataLoader
  worker memory / GPU starvation issue.
* **No spatial crop transforms are applied** because the culling decision
  depends on the full-frame composition.  Instead the pipeline is:
      Train : (early SquarePad+Resize) → RandAugment → colour jitter →
              grayscale → blur → H-flip → normalise
      Val/Test: (early SquarePad+Resize) → normalise
* ``CullingDataset`` reads a CSV produced by ``prepare_dataset.py`` and
  returns ``(image_tensor, label_tensor)`` pairs.
* A helper ``build_dataloaders`` constructs train / val / test loaders with
  class-aware weighted sampling on the train split to counter imbalance.

CSV expected columns
--------------------
    filename, img_path, label, session, datetime_str, ...

Usage
-----
    from auto_culling.dataset import build_dataloaders

    train_loader, val_loader, test_loader = build_dataloaders(
        train_csv=Path("dataset/train_info.csv"),
        test_csv=Path("dataset/test_info.csv"),
        img_size=224,
        batch_size=32,
        num_workers=4,
        val_fraction=0.1,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SquarePad: letterbox the full image to a square without any crop
# ---------------------------------------------------------------------------


class SquarePad:
    """Pad a PIL image to a square by adding black borders on the shorter side.

    This preserves every pixel of the original image — no content is ever
    discarded.  The shorter dimension is centred so the aspect ratio is
    maintained and the composition cues that drive the culling decision remain
    intact.

    After ``SquarePad`` the image is ``max(W, H) × max(W, H)`` and can be
    passed to ``transforms.Resize((n, n))`` without any further cropping.

    Parameters
    ----------
    fill:
        Pixel fill value for the padding area (default: 0 = black).
    """

    def __init__(self, fill: int = 0) -> None:
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        max_side = max(w, h)
        pad_left = (max_side - w) // 2
        pad_top = (max_side - h) // 2
        pad_right = max_side - w - pad_left
        pad_bottom = max_side - h - pad_top
        # ImageOps.expand adds (left, top, right, bottom) borders
        return ImageOps.expand(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fill={self.fill})"


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

# ImageNet normalisation statistics
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int = 224) -> transforms.Compose:
    """Return the training augmentation pipeline.

    The image arriving here is already a square PIL image at ``img_size``
    (resized by ``CullingDataset.__getitem__`` before this transform runs).
    So this pipeline only applies colour / flip augmentations and the final
    tensor conversion — no spatial resize needed here.

    Pipeline:
    1. ``RandomHorizontalFlip``  — racing scenes are left/right symmetric.
    2. ``RandAugment(n=2, m=9)`` — diverse colour/geometric perturbations;
       ``m=9`` is moderate (scale 0–30).
    3. ``ColorJitter``           — additional exposure/WB variation on top of
       RandAugment for robustness to real-world lighting.
    4. ``RandomGrayscale``       — occasionally remove colour cues so the
       model can't rely solely on car livery.
    5. ``GaussianBlur``          — simulate camera out-of-focus / motion blur.
    6. ``RandomRotation(5°)``    — handle slight camera tilt.
    7. ``ToTensor + Normalize``  — standard ImageNet normalisation.

    Parameters
    ----------
    img_size:
        Unused; kept for API consistency with ``build_eval_transform``.

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def build_eval_transform(img_size: int = 224) -> transforms.Compose:
    """Return the deterministic evaluation / inference transform pipeline.

    The image arriving here is already a square PIL image at ``img_size``
    (resized by ``CullingDataset.__getitem__``).  Only tensor conversion and
    normalisation are needed.

    Parameters
    ----------
    img_size:
        Unused; kept for API consistency.

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CullingDataset(Dataset):
    """PyTorch Dataset for the F1 culling binary-classification task.

    Parameters
    ----------
    csv_path:
        Path to a CSV file with at least ``img_path`` and ``label`` columns.
        ``img_path`` values may point to the original ``dataset/img/`` path;
        the dataset automatically resolves them to ``dataset/cache/<stem>.jpg``.
    transform:
        Optional torchvision transform applied *after* the early resize.
        When ``None``, ``build_eval_transform(img_size)`` is used.
    img_size:
        Target square size.  In ``__getitem__`` each image is SquarePad'd and
        resized to ``img_size × img_size`` *before* the transform pipeline,
        so each DataLoader worker holds only a small PIL image (~150 KB at
        224×224) rather than the full 4672×7008 original (~94 MB).

    Notes
    -----
    Images are opened lazily (per ``__getitem__`` call).
    """

    def __init__(
        self,
        csv_path: Path,
        transform: Optional[transforms.Compose] = None,
        img_size: int = 224,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.img_size = img_size
        self.transform = transform or build_eval_transform(img_size)

        if "img_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(
                f"{csv_path} must contain 'img_path' and 'label' columns."
            )

        log.info(
            "Loaded %d samples from %s  (label-1: %d, label-0: %d)",
            len(self.df),
            self.csv_path.name,
            int(self.df["label"].sum()),
            int((self.df["label"] == 0).sum()),
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = Path(row["img_path"])

        # Resolve to the JPEG cache file.
        # img_path may still reference the old dataset/img/<stem>.HIF path
        # (the CSV was written before HIF files were deleted), so we derive
        # the cache path from the stem regardless of the stored extension.
        cache_path = img_path.parent.parent / "cache" / (img_path.stem + ".jpg")
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache JPEG not found: {cache_path}\n"
                "Run cache_images.py to generate the JPEG cache."
            )

        try:
            image = Image.open(cache_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open image: {cache_path}") from exc

        # --- Early resize: SquarePad + Resize to img_size BEFORE transform ---
        # This is the critical memory fix: a 4672×7008 JPEG decoded to ~94 MB
        # is immediately shrunk to img_size×img_size (~150 KB at 224).
        # DataLoader workers therefore only hold small tensors, preventing the
        # RAM exhaustion / GPU starvation that occurred with full-res images.
        w, h = image.size
        max_side = max(w, h)
        pad_l = (max_side - w) // 2
        pad_t = (max_side - h) // 2
        pad_r = max_side - w - pad_l
        pad_b = max_side - h - pad_t
        image = ImageOps.expand(image, (pad_l, pad_t, pad_r, pad_b), fill=0)
        image = image.resize(
            (self.img_size, self.img_size), resample=Image.Resampling.BILINEAR
        )

        image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image, label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def labels(self) -> list[int]:
        """Return all integer labels as a plain list."""
        return self.df["label"].tolist()

    @property
    def pos_weight(self) -> torch.Tensor:
        """Compute ``pos_weight`` for ``BCEWithLogitsLoss``.

        ``pos_weight = num_neg / num_pos`` so the loss is re-weighted to
        compensate for class imbalance.
        """
        num_pos = int(self.df["label"].sum())
        num_neg = len(self.df) - num_pos
        if num_pos == 0:
            return torch.tensor(1.0)
        return torch.tensor(num_neg / num_pos, dtype=torch.float32)


# ---------------------------------------------------------------------------
# WeightedRandomSampler factory
# ---------------------------------------------------------------------------


def build_weighted_sampler(dataset: CullingDataset) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` that balances positive/negative draws.

    Each sample is assigned a weight inverse to its class frequency so that
    in expectation each mini-batch contains a balanced mix.

    Parameters
    ----------
    dataset:
        A ``CullingDataset`` instance.

    Returns
    -------
    WeightedRandomSampler
    """
    labels = dataset.labels
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos

    weight_for_1 = 1.0 / num_pos if num_pos > 0 else 1.0
    weight_for_0 = 1.0 / num_neg if num_neg > 0 else 1.0

    sample_weights = [
        weight_for_1 if lbl == 1 else weight_for_0 for lbl in labels
    ]
    weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_dataloaders(
    train_csv: Path,
    test_csv: Path,
    *,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 42,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train, validation, and test ``DataLoader`` instances.

    The validation set is split off from the training CSV using a deterministic
    random subset (stratified by label).

    Parameters
    ----------
    train_csv:
        Path to ``train_info.csv``.
    test_csv:
        Path to ``test_info.csv``.
    img_size:
        Input resolution (square).
    batch_size:
        Mini-batch size for all loaders.
    num_workers:
        Worker processes for data loading.
    val_fraction:
        Fraction of training data to hold out as a validation set.
    seed:
        Random seed for the train/val split.
    pin_memory:
        Enable pinned memory for faster GPU transfers.
    use_weighted_sampler:
        Apply ``WeightedRandomSampler`` on the training loader to handle class
        imbalance; has no effect when ``val_fraction=0``.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit

    train_df_full = pd.read_csv(train_csv)

    # ---------- train / val split ----------
    if val_fraction > 0:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=seed
        )
        train_idx, val_idx = next(
            sss.split(train_df_full, train_df_full["label"])
        )
        df_train = train_df_full.iloc[train_idx].reset_index(drop=True)
        df_val = train_df_full.iloc[val_idx].reset_index(drop=True)
    else:
        df_train = train_df_full
        df_val = pd.DataFrame(columns=train_df_full.columns)

    # Write temp CSVs that CullingDataset can read from
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="culling_"))
    tmp_train = tmp_dir / "split_train.csv"
    tmp_val = tmp_dir / "split_val.csv"
    df_train.to_csv(tmp_train, index=False)
    df_val.to_csv(tmp_val, index=False)

    # ---------- datasets ----------
    ds_train = CullingDataset(tmp_train, transform=build_train_transform(img_size))
    ds_val = CullingDataset(tmp_val, transform=build_eval_transform(img_size))
    ds_test = CullingDataset(test_csv, transform=build_eval_transform(img_size))

    # ---------- samplers ----------
    sampler = build_weighted_sampler(ds_train) if use_weighted_sampler else None

    # ---------- loaders ----------
    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    log.info(
        "DataLoaders ready — train: %d batches, val: %d batches, test: %d batches",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    return train_loader, val_loader, test_loader
