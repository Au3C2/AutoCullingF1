"""
eval_orientation_classifier.py — Evaluate the trained orientation classifier.

Computes per-class metrics (precision, recall, F1) and generates confusion matrix.

Usage
-----
    python eval_orientation_classifier.py --checkpoint orientation_checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from train_orientation_classifier import OrientationDataset, get_transforms

log = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device):
    """Load trained model from checkpoint."""
    model = models.resnet50()
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 4),
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def evaluate(model, loader, device):
    """Run evaluation on test set."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate orientation classifier")
    parser.add_argument("--checkpoint", type=str, default="orientation_checkpoints/best.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, default="vehicle_orientation_labels",
                        help="Data directory with class subdirectories")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        log.error(f"Checkpoint not found: {checkpoint_path}")
        return 1
    
    log.info(f"Loading checkpoint from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    
    # Load dataset
    log.info(f"Loading dataset from {args.data_dir}...")
    dataset = OrientationDataset(args.data_dir, transform=get_transforms("val"))
    
    if len(dataset) == 0:
        log.error("No images found in dataset!")
        return 1
    
    # Create loader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    # Evaluate
    log.info("Running evaluation...")
    preds, labels = evaluate(model, loader, device)
    
    # Metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=range(4)
    )
    
    cm = confusion_matrix(labels, preds, labels=range(4))
    
    # Print results
    log.info("\n" + "="*70)
    log.info("Per-Class Metrics")
    log.info("="*70)
    
    class_names = ["正前方", "侧身", "正后方", "侧后方"]
    for i, class_name in enumerate(class_names):
        log.info(f"{class_name:6s}  Precision: {precision[i]:.4f}  Recall: {recall[i]:.4f}  F1: {f1[i]:.4f}  (n={support[i]})")
    
    # Macro and weighted averages
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support).sum() / support.sum()
    
    log.info("\n" + "="*70)
    log.info(f"Macro F1:    {macro_f1:.4f}")
    log.info(f"Weighted F1: {weighted_f1:.4f}")
    log.info("="*70)
    
    # Confusion matrix
    log.info("\nConfusion Matrix:")
    log.info("             Pred: 正前方 侧身 正后方 侧后方")
    for i, class_name in enumerate(class_names):
        row_str = " ".join(f"{cm[i, j]:5d}" for j in range(4))
        log.info(f"True: {class_name:6s}  {row_str}")
    
    # Save results
    results = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    
    results_path = Path(args.checkpoint).parent / "eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {results_path}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
