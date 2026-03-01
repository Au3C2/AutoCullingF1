"""
Training script for the F1 motorsport photo auto-culling binary classifier.

Features
--------
* Configurable total training steps (``--steps``) rather than epochs, with
  progress tracked via a global step counter.
* Periodic evaluation on the validation set every ``--eval-interval`` steps,
  and a final evaluation on the test set at the end of training.
* TensorBoard logging:
    - ``train/loss``  (per step)
    - ``val/loss``, ``val/acc``, ``val/f1``, ``val/auc``  (per eval)
    - ``test/loss``, ``test/acc``, ``test/f1``, ``test/auc``  (once, at end)
    - ``lr``          (per step)
* Fine-tune mode (``--finetune``): only the last backbone block + head are
  trained; the backbone uses a lower learning rate (``--backbone-lr``).
* Automatic mixed precision (AMP) via ``torch.amp``.
* Checkpoint saving: best validation F1 and last checkpoint.
* Reproducibility via ``--seed``.
* Label smoothing via ``--label-smoothing`` (default 0.1) to reduce
  overconfidence and mitigate overfitting.

Usage
-----
    uv run train.py \\
        --arch resnet50 \\
        --train-csv dataset/train_info.csv \\
        --test-csv  dataset/test_info.csv  \\
        --steps 5000 \\
        --eval-interval 500 \\
        --batch-size 32 \\
        --lr 1e-3 \\
        --finetune \\
        --pretrained \\
        --checkpoint-dir checkpoints/resnet50
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project imports – resolved via pyproject.toml [tool.uv.sources] / PYTHONPATH
from auto_culling.dataset import build_dataloaders
from auto_culling.model import build_model, get_trainable_params

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label-smoothed BCE loss
# ---------------------------------------------------------------------------


class LabelSmoothingBCELoss(nn.Module):
    """Binary cross-entropy with label smoothing.

    Soft targets are computed as::

        y_smooth = y * (1 - smoothing) + 0.5 * smoothing

    where 0.5 is used for the "off" class so that the model is never pushed to
    predict a probability of exactly 0 or 1.  This reduces overconfidence and
    acts as a regulariser.

    Parameters
    ----------
    smoothing:
        Label-smoothing factor in [0, 1).  Set to 0.0 to disable.
    pos_weight:
        Optional positive-class weight tensor (same as ``BCEWithLogitsLoss``).
    reduction:
        ``'mean'`` (default) or ``'sum'``.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing
        self.reduction = reduction
        self._bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0.0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self._bce(logits, targets)


# ---------------------------------------------------------------------------
# Reproducibility helper
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDNN (may reduce throughput slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on a data loader.

    Parameters
    ----------
    model:
        Model in eval mode.
    loader:
        DataLoader to iterate over.
    criterion:
        Loss function.
    device:
        Device to run inference on.

    Returns
    -------
    dict[str, float]
        Keys: ``loss``, ``acc``, ``f1``, ``auc``.
    """
    model.eval()
    total_loss = 0.0
    all_probs: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy().tolist()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.long().cpu().numpy().tolist())

    n = len(all_labels)
    avg_loss = total_loss / n if n > 0 else float("nan")
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / n if n > 0 else 0.0

    # F1 (binary, positive class = 1)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

    # AUC-ROC (gracefully handle degenerate splits)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {"loss": avg_loss, "acc": acc, "f1": f1, "auc": auc}


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Monitors validation F1 (higher = better).  After ``patience`` consecutive
    evaluations with no improvement of at least ``min_delta``, ``should_stop``
    is set to ``True``.

    Parameters
    ----------
    patience:
        Number of consecutive non-improving evaluations before stopping.
        Set to 0 or a negative value to disable early stopping entirely.
    min_delta:
        Minimum absolute improvement in val F1 required to count as an
        improvement and reset the patience counter.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = -float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def step(self, val_f1: float) -> bool:
        """Update state with the latest validation F1.

        Parameters
        ----------
        val_f1:
            Current validation F1 score.

        Returns
        -------
        bool
            ``True`` if training should stop now, ``False`` otherwise.
        """
        if not self.enabled:
            return False

        if val_f1 >= self.best + self.min_delta:
            self.best = val_f1
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------


def build_optimizer(
    model: nn.Module,
    arch: str,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
    finetune: bool,
) -> torch.optim.Optimizer:
    """Construct an AdamW optimiser with per-group learning rates.

    In fine-tune mode the classification head uses ``lr`` and the unfrozen
    backbone layers use ``backbone_lr`` (typically 10× smaller).  In full
    training mode all parameters share ``lr``.

    Parameters
    ----------
    model:
        The model whose parameters are optimised.
    arch:
        Architecture name (used to identify head vs backbone layers).
    lr:
        Learning rate for the classification head.
    backbone_lr:
        Learning rate for the unfrozen backbone layers (fine-tune only).
    weight_decay:
        L2 regularisation coefficient.
    finetune:
        Whether fine-tune mode is active.

    Returns
    -------
    torch.optim.Optimizer
    """
    if not finetune:
        params = get_trainable_params(model)
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Separate head parameters from backbone parameters
    head_names = {"fc", "classifier"}
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top_module = name.split(".")[0]
        if top_module in head_names:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": head_params, "lr": lr},
        {"params": backbone_params, "lr": backbone_lr},
    ]
    log.info(
        "Optimiser param groups — head: %d tensors (lr=%.2e), "
        "backbone: %d tensors (lr=%.2e)",
        len(head_params),
        lr,
        len(backbone_params),
        backbone_lr,
    )
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# LR scheduler factory
# ---------------------------------------------------------------------------


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a linear-warmup + cosine-decay LR scheduler.

    Parameters
    ----------
    optimizer:
        The optimiser to schedule.
    total_steps:
        Total number of training steps.
    warmup_steps:
        Number of linear warm-up steps.
    last_epoch:
        The index of the last completed step (``-1`` means start from
        scratch).  Pass ``start_step - 1`` when resuming a checkpoint so the
        LR schedule continues from the correct position without triggering
        PyTorch's "step before optimizer" warning.

    Returns
    -------
    LambdaLR
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    state: dict,
    path: Path,
) -> None:
    """Save a training checkpoint to *path*.

    Parameters
    ----------
    state:
        Dictionary with at least ``model_state_dict`` and ``step``.
    path:
        Output file path (parent directory is created if necessary).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    log.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> int:
    """Load a checkpoint and restore model (and optionally optimiser/scheduler) state.

    Parameters
    ----------
    path:
        Path to the ``.pt`` checkpoint file.
    model:
        Model instance to restore weights into.
    optimizer:
        Optional optimiser to restore state.
    scheduler:
        Optional scheduler to restore state.

    Returns
    -------
    int
        Global step at which the checkpoint was saved.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    step = state.get("step", 0)
    log.info("Checkpoint loaded from %s  (step=%d)", path, step)
    return step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Run the full training procedure according to *args*.

    Parameters
    ----------
    args:
        Parsed CLI arguments (see :func:`parse_args`).
    """
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if (not args.cpu) and torch.cuda.is_available() else "cpu"
    )
    log.info("Using device: %s", device)

    # ---- DataLoaders -------------------------------------------------------
    train_loader, val_loader, test_loader = build_dataloaders(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model -------------------------------------------------------------
    model = build_model(
        arch=args.arch,
        finetune=args.finetune,
        pretrained=args.pretrained,
    )
    model = model.to(device)

    # ---- Loss (with pos_weight for imbalance + optional label smoothing) ---
    # Compute pos_weight from training CSV directly
    import pandas as pd

    train_df = pd.read_csv(args.train_csv)
    num_pos = int(train_df["label"].sum())
    num_neg = len(train_df) - num_pos
    pos_weight = torch.tensor(num_neg / max(num_pos, 1), dtype=torch.float32).to(device)
    log.info("BCEWithLogitsLoss pos_weight = %.4f", pos_weight.item())
    criterion = LabelSmoothingBCELoss(
        smoothing=args.label_smoothing,
        pos_weight=pos_weight,
    )
    if args.label_smoothing > 0.0:
        log.info("Label smoothing enabled: smoothing=%.2f", args.label_smoothing)

    # ---- Optimiser ---------------------------------------------------------
    optimizer = build_optimizer(
        model=model,
        arch=args.arch,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        finetune=args.finetune,
    )

    # ---- AMP ---------------------------------------------------------------
    scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

    # ---- Optionally resume (must happen before scheduler so last_epoch is known) ---
    start_step = 0
    ckpt_dir = Path(args.checkpoint_dir)
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            # Load model + optimizer state; scheduler is built below with last_epoch
            start_step = load_checkpoint(resume_path, model, optimizer)
        else:
            log.warning("--resume path not found: %s", resume_path)

    # ---- Scheduler (built after start_step is known) -----------------------
    warmup_steps = max(1, int(args.steps * args.warmup_ratio))
    scheduler = build_scheduler(
        optimizer,
        total_steps=args.steps,
        warmup_steps=warmup_steps,
        last_epoch=start_step - 1,  # Ensures LR resumes correctly after a checkpoint
    )
    # If resuming, also restore the scheduler's internal state from the checkpoint
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            state = torch.load(resume_path, map_location="cpu", weights_only=False)
            if "scheduler_state_dict" in state:
                scheduler.load_state_dict(state["scheduler_state_dict"])
                log.info("Scheduler state restored from checkpoint.")

    # ---- TensorBoard -------------------------------------------------------
    tb_dir = ckpt_dir / "tb_logs"
    writer = SummaryWriter(log_dir=str(tb_dir))
    log.info("TensorBoard logs → %s", tb_dir)

    # ---- Early stopping ----------------------------------------------------
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )
    if early_stopping.enabled:
        log.info(
            "Early stopping enabled: patience=%d  min_delta=%.1e",
            args.early_stopping_patience,
            args.early_stopping_min_delta,
        )
    else:
        log.info("Early stopping disabled (patience <= 0).")

    # ---- Training loop -----------------------------------------------------
    global_step = start_step
    best_val_f1 = 0.0
    train_iter = iter(train_loader)

    log.info(
        "Starting training: arch=%s  steps=%d  eval_interval=%d  finetune=%s",
        args.arch,
        args.steps,
        args.eval_interval,
        args.finetune,
    )

    model.train()
    t0 = time.time()

    while global_step < args.steps:
        # Refresh iterator when exhausted (step-based loop)
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1

        # ---- TensorBoard: train loss & lr ----------------------------------
        writer.add_scalar("train/loss", loss.item(), global_step)
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("lr", current_lr, global_step)

        # ---- Periodic logging to stdout ------------------------------------
        if global_step % max(1, args.eval_interval // 10) == 0:
            elapsed = time.time() - t0
            log.info(
                "step %5d/%d  train_loss=%.4f  lr=%.2e  elapsed=%.1fs",
                global_step,
                args.steps,
                loss.item(),
                current_lr,
                elapsed,
            )

        # ---- Evaluation ----------------------------------------------------
        if global_step % args.eval_interval == 0 or global_step == args.steps:
            val_metrics = evaluate(model, val_loader, criterion, device)
            log.info(
                "step %5d  VAL  loss=%.4f  acc=%.4f  f1=%.4f  auc=%.4f",
                global_step,
                val_metrics["loss"],
                val_metrics["acc"],
                val_metrics["f1"],
                val_metrics["auc"],
            )
            writer.add_scalar("val/loss", val_metrics["loss"], global_step)
            writer.add_scalar("val/acc", val_metrics["acc"], global_step)
            writer.add_scalar("val/f1", val_metrics["f1"], global_step)
            writer.add_scalar("val/auc", val_metrics["auc"], global_step)

            # Save best checkpoint
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                save_checkpoint(
                    {
                        "step": global_step,
                        "arch": args.arch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_f1": best_val_f1,
                    },
                    ckpt_dir / "best.pt",
                )

            # Early stopping check
            if early_stopping.step(val_metrics["f1"]):
                log.info(
                    "Early stopping triggered at step %d "
                    "(no improvement in val F1 for %d evaluations, best=%.4f).",
                    global_step,
                    args.early_stopping_patience,
                    early_stopping.best,
                )
                model.train()
                break

            model.train()

    # ---- Save last checkpoint ----------------------------------------------
    save_checkpoint(
        {
            "step": global_step,
            "arch": args.arch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        ckpt_dir / "last.pt",
    )

    # ---- Final test evaluation ---------------------------------------------
    test_metrics = evaluate(model, test_loader, criterion, device)
    log.info(
        "TEST  loss=%.4f  acc=%.4f  f1=%.4f  auc=%.4f",
        test_metrics["loss"],
        test_metrics["acc"],
        test_metrics["f1"],
        test_metrics["auc"],
    )
    writer.add_scalar("test/loss", test_metrics["loss"], global_step)
    writer.add_scalar("test/acc", test_metrics["acc"], global_step)
    writer.add_scalar("test/f1", test_metrics["f1"], global_step)
    writer.add_scalar("test/auc", test_metrics["auc"], global_step)

    writer.close()
    log.info(
        "Training complete. Best val F1=%.4f  Test F1=%.4f",
        best_val_f1,
        test_metrics["f1"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        prog="train",
        description="Train / fine-tune a binary image classifier for photo culling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data --------------------------------------------------------------
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--train-csv",
        type=Path,
        default=Path("dataset/train_info.csv"),
        help="Path to train_info.csv.",
    )
    data.add_argument(
        "--test-csv",
        type=Path,
        default=Path("dataset/test_info.csv"),
        help="Path to test_info.csv.",
    )
    data.add_argument("--img-size", type=int, default=224, help="Input image size (square).")
    data.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    data.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    data.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of training data held out as validation.",
    )

    # ---- Model -------------------------------------------------------------
    mdl = parser.add_argument_group("Model")
    mdl.add_argument(
        "--arch",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50", "resnext50", "mobilenetv3"],
        help="Backbone architecture.",
    )
    mdl.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Load ImageNet pre-trained weights.",
    )
    mdl.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train from scratch (no pre-trained weights).",
    )
    mdl.add_argument(
        "--finetune",
        action="store_true",
        default=True,
        help="Freeze early backbone layers (fine-tune mode).",
    )
    mdl.add_argument(
        "--no-finetune",
        dest="finetune",
        action="store_false",
        help="Train all layers (full training mode).",
    )

    # ---- Training schedule -------------------------------------------------
    sched = parser.add_argument_group("Training schedule")
    sched.add_argument("--steps", type=int, default=20000, help="Total training steps (upper bound; early stopping may halt sooner).")
    sched.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Run validation every N steps.",
    )
    sched.add_argument("--lr", type=float, default=1e-3, help="Head / full-model learning rate.")
    sched.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-4,
        help="Backbone learning rate (fine-tune only).",
    )
    sched.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    sched.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help=(
            "Label-smoothing factor for BCE loss.  Soft targets are "
            "y*(1-s) + 0.5*s.  Set to 0.0 to disable."
        ),
    )
    sched.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Fraction of total steps used for linear LR warm-up.",
    )
    sched.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help=(
            "Stop training if val F1 does not improve for this many consecutive "
            "evaluations.  Each evaluation is --eval-interval steps apart, so the "
            "effective patience in steps is patience × eval_interval.  "
            "Set to 0 to disable early stopping."
        ),
    )
    sched.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum absolute improvement in val F1 to reset the patience counter.",
    )

    # ---- Misc --------------------------------------------------------------
    misc = parser.add_argument_group("Misc")
    misc.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/run"),
        help="Directory for checkpoints and TensorBoard logs.",
    )
    misc.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    misc.add_argument("--seed", type=int, default=42, help="Random seed.")
    misc.add_argument("--cpu", action="store_true", help="Force CPU (disable CUDA).")
    misc.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG-level logging."
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry-point for the training script.

    Parameters
    ----------
    argv:
        Argument list (``sys.argv`` is used when ``None``).

    Returns
    -------
    int
        Exit code (0 = success).
    """
    import sys

    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    try:
        train(args)
    except KeyboardInterrupt:
        log.info("Training interrupted by user.")
        return 130

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
