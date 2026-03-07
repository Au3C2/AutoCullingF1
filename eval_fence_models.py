"""
Evaluate trained fence classifiers on test set.

Loads best.pt checkpoints for ResNet18, ResNet50, MobileNetV2 and compares:
- Accuracy, Precision, Recall, F1
- Inference speed (img/s)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import cv2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================================================================
# Dataset (same as training)
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
        # Use cv2.imdecode with absolute path to handle Chinese characters on Windows
        import os
        try:
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                log.warning(f"Failed to load {img_path}")
                return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            log.warning(f"Failed to load {img_path}: {e}")
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

def load_checkpoint(arch: str, checkpoint_dir: str = "fence_classifier_checkpoints") -> nn.Module:
    """Load best.pt checkpoint for architecture."""
    model = build_model(arch)
    checkpoint_path = Path(checkpoint_dir) / arch / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate(model, test_loader, arch: str):
    """Evaluate model on test set and measure inference speed."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    # Measure inference time (warm-up + measurement)
    warmup_batches = 2
    batch_times = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Time this batch
            if batch_idx >= warmup_batches:
                start = time.time()
            
            outputs = model(images).squeeze()
            
            if batch_idx >= warmup_batches:
                elapsed = time.time() - start
                batch_times.append(elapsed)
            
            scores = torch.sigmoid(outputs)  # Convert logits to probabilities
            preds = (scores > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # Convert to numpy
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    scores = np.array(all_scores)
    
    # Compute metrics
    accuracy = (preds == labels).mean()
    
    # Precision, Recall, F1
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    # Inference speed
    if batch_times:
        avg_batch_time = np.mean(batch_times)
        batch_size = test_loader.batch_size
        throughput = batch_size / avg_batch_time  # images per second
    else:
        throughput = 0
    
    results = {
        "arch": arch,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "throughput_img_per_s": float(throughput),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }
    
    return results

def main():
    # Data transform (no augmentation for evaluation)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = FenceDataset("fence_label", transform=val_transform)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    log.info(f"Evaluating on {len(dataset)} images")
    
    # Evaluate all 3 models
    all_results = []
    
    for arch in ["resnet18", "resnet50", "mobilenetv2"]:
        log.info(f"\n{'='*70}")
        log.info(f"Evaluating {arch}")
        log.info(f"{'='*70}")
        
        try:
            model = load_checkpoint(arch)
            results = evaluate(model, test_loader, arch)
            all_results.append(results)
            
            log.info(f"Accuracy:    {results['accuracy']:.4f}")
            log.info(f"Precision:   {results['precision']:.4f}")
            log.info(f"Recall:      {results['recall']:.4f}")
            log.info(f"F1:          {results['f1']:.4f}")
            log.info(f"Specificity: {results['specificity']:.4f}")
            log.info(f"Throughput:  {results['throughput_img_per_s']:.1f} img/s")
            log.info(f"Confusion matrix: TP={results['tp']}, FP={results['fp']}, FN={results['fn']}, TN={results['tn']}")
        
        except FileNotFoundError as e:
            log.error(f"Failed to load {arch}: {e}")
    
    # Summary
    log.info(f"\n{'='*70}")
    log.info("SUMMARY")
    log.info(f"{'='*70}")
    
    for res in all_results:
        log.info(f"\n{res['arch']}:")
        log.info(f"  F1:        {res['f1']:.4f}")
        log.info(f"  Accuracy:  {res['accuracy']:.4f}")
        log.info(f"  Throughput: {res['throughput_img_per_s']:.1f} img/s")
    
    # Save results to JSON
    output_path = Path("fence_eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to {output_path}")
    
    # Recommend best model
    if all_results:
        best_by_f1 = max(all_results, key=lambda x: x['f1'])
        best_by_speed = max(all_results, key=lambda x: x['throughput_img_per_s'])
        
        log.info(f"\n{'='*70}")
        log.info("RECOMMENDATIONS")
        log.info(f"{'='*70}")
        log.info(f"Best F1 score:     {best_by_f1['arch']} ({best_by_f1['f1']:.4f})")
        log.info(f"Fastest inference: {best_by_speed['arch']} ({best_by_speed['throughput_img_per_s']:.1f} img/s)")

if __name__ == "__main__":
    main()
