"""
Wire fence binary classifier training.

Models: ResNet18, ResNet50, MobileNetV2 (all pretrained)
Loss: BCEWithLogitsLoss for binary classification
Early stopping + extended epochs as requested
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import cv2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================================================================
# Dataset
# ===========================================================================

class FenceDataset(Dataset):
    """Binary classification: 1=fence, 0=no fence"""
    def __init__(self, img_dir: Path, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples = []
        
        # Class 0: no fence
        for img_file in (self.img_dir / '没有铁丝网').glob('*.jpg'):
            self.samples.append((img_file, 0))
        
        # Class 1: fence
        for img_file in (self.img_dir / '有铁丝网').glob('*.jpg'):
            self.samples.append((img_file, 1))
        
        log.info(f"Loaded {len(self.samples)} images: {sum(1 for _, y in self.samples if y==0)} no-fence, {sum(1 for _, y in self.samples if y==1)} fence")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning(f"Failed to load {img_path}")
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# ===========================================================================
# Model builders
# ===========================================================================

def build_model(arch: str) -> nn.Module:
    """Build pretrained model with binary classification head."""
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    elif arch == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model.to(device)

# ===========================================================================
# Training
# ===========================================================================

@dataclass
class TrainConfig:
    arch: str = "resnet18"
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 200  # Extended as requested
    patience: int = 20  # Early stopping
    val_split: float = 0.2
    output_dir: str = "fence_classifier_checkpoints"

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(val_loader), correct / total

def train_model(config: TrainConfig, train_loader, val_loader):
    model = build_model(config.arch)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    log.info(f"Training {config.arch} for up to {config.epochs} epochs (patience={config.patience})")
    
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        if (epoch + 1) % 10 == 0:
            log.info(f"Epoch {epoch+1}/{config.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            checkpoint_dir = Path(config.output_dir) / config.arch
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                log.info(f"Early stopping at epoch {epoch+1} (no improvement for {config.patience} epochs)")
                break
    
    log.info(f"Training completed. Best model at epoch {best_epoch+1} with val_loss={best_val_loss:.4f}")
    return model, best_val_loss

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="fence_label", help="Directory with 有铁丝网 and 没有铁丝网")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    
    # Data transform
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = FenceDataset(args.data_dir, transform=train_transform)
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    # Update transforms for val_dataset
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    log.info(f"Train: {n_train}, Val: {n_val}")
    
    # Train all 3 models
    results = {}
    for arch in ["resnet18", "resnet50", "mobilenetv2"]:
        log.info(f"\n{'='*60}")
        log.info(f"Training {arch}")
        log.info(f"{'='*60}")
        
        config = TrainConfig(arch=arch, batch_size=args.batch_size, learning_rate=args.lr, 
                           epochs=args.epochs, patience=args.patience)
        model, best_val_loss = train_model(config, train_loader, val_loader)
        results[arch] = {"best_val_loss": best_val_loss}
    
    log.info(f"\n{'='*60}")
    log.info("FINAL RESULTS")
    log.info(f"{'='*60}")
    for arch, res in results.items():
        log.info(f"{arch}: best_val_loss={res['best_val_loss']:.4f}")

if __name__ == "__main__":
    main()
