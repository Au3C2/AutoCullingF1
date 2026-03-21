"""
train_orientation_classifier.py — Train 4-class vehicle orientation classifier.

This script trains a ResNet50 model to classify vehicle heading in F1 photos.

4 Classes (based on degree of car rotation):
  - 正前方 (Head-on, 0°)
  - 侧身 (Side, 90°)
  - 正后方 (Rear, 180°)
  - 侧后方 (Diagonal, 45-135°)

The model is trained on the annotated ROI images in:
  vehicle_orientation_labels/{正前方,侧身,正后方,侧后方}/

Usage
-----
    python train_orientation_classifier.py --epoch 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

log = logging.getLogger(__name__)


class OrientationDataset(Dataset):
    """Custom dataset for orientation classification."""
    
    CLASSES = ["正前方", "侧身", "正后方", "侧后方"]  # Head-on, Side, Rear, Diagonal
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
    IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
    
    def __init__(self, img_dir: Path, transform=None):
        """
        Initialize dataset from directory structure.
        
        Expected structure:
            img_dir/
            ├── 正前方/
            ├── 侧身/
            ├── 正后方/
            └── 侧后方/
        """
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples = []
        
        # Scan each class directory
        for class_name, class_idx in self.CLASS_TO_IDX.items():
            class_dir = self.img_dir / class_name
            if not class_dir.exists():
                log.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Collect all JPG files
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, class_idx))
            for img_path in sorted(class_dir.glob("*.JPG")):
                self.samples.append((img_path, class_idx))
            for img_path in sorted(class_dir.glob("*.jpeg")):
                self.samples.append((img_path, class_idx))
        
        log.info(f"Loaded {len(self.samples)} images from {self.img_dir}")
        
        # Log per-class counts
        counts = defaultdict(int)
        for _, idx in self.samples:
            counts[self.IDX_TO_CLASS[idx]] += 1
        for class_name in self.CLASSES:
            log.info(f"  {class_name:6s}: {counts[class_name]:4d} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        # Load image
        try:
            import cv2
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                log.warning(f"Failed to load {img_path}, returning black image")
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            log.warning(f"Error loading {img_path}: {e}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            img = self.transform(img)
        
        return img, class_idx


def build_model(num_classes: int = 4, pretrained: bool = True):
    """Build ResNet50 for orientation classification."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    
    return model


def get_transforms(split: str = "train"):
    """Get augmentation transforms."""
    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, preds = logits.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total
    
    return avg_loss, acc


def eval_epoch(model, loader, criterion, device):
    """Evaluate one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, preds = logits.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total
    
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train orientation classifier")
    parser.add_argument("--data-dir", type=str, default="vehicle_orientation_labels",
                        help="Root directory with class subdirectories")
    parser.add_argument("--output-dir", type=str, default="orientation_checkpoints",
                        help="Output checkpoint directory")
    parser.add_argument("--epoch", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoints will be saved to: {output_dir}")
    
    # Load dataset
    log.info(f"Loading dataset from {args.data_dir}...")
    dataset = OrientationDataset(args.data_dir, transform=get_transforms("train"))
    
    if len(dataset) == 0:
        log.error("No images found! Check directory structure:")
        log.error(f"  {args.data_dir}/")
        log.error(f"    ├── 正前方/ (0-180 images expected)")
        log.error(f"    ├── 侧身/")
        log.error(f"    ├── 正后方/")
        log.error(f"    └── 侧后方/")
        return 1
    
    # Split into train/val
    n_train = int(len(dataset) * (1 - args.val_split))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    # Create loaders (note: val_set needs separate transform)
    val_transform = get_transforms("val")
    val_set_with_transform = type(dataset)(args.data_dir, transform=val_transform)
    val_indices = val_set.indices
    val_set_subset = torch.utils.data.Subset(val_set_with_transform, val_indices)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    log.info(f"Train: {len(train_set)} | Val: {len(val_set)}")
    
    # Model
    log.info("Building ResNet50 model...")
    model = build_model(num_classes=4, pretrained=True).to(device)
    
    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Early stopping
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    
    # Training loop
    log.info("Starting training...")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(args.epoch):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        log.info(f"Epoch {epoch+1:3d}/{args.epoch} | "
                f"Loss: {train_loss:.4f} / {val_loss:.4f} | "
                f"Acc: {train_acc:.1f}% / {val_acc:.1f}%")
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_path = output_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
            log.info(f"  → Saved best checkpoint to {best_path} (val_acc={val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"Early stopping at epoch {epoch+1} (patience={patience} reached)")
                break
        
        scheduler.step()
    
    # Save history
    history_path = output_dir / "history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    log.info(f"Training history saved to {history_path}")
    
    log.info(f"\nTraining complete!")
    log.info(f"Best validation accuracy: {best_val_acc:.1f}%")
    log.info(f"Best model saved to: {output_dir / 'best.pt'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
