"""
Train a Multi-Task MobileNetV3 model for P4:
Task 1: Object Orientation (5 classes)
Task 2: Object Integrity (Binary: Full vs Cut/Occluded)

Incorporates:
- Dynamic "Cut" augmentation (crops >1/3 of "Full" images to generate infinite "Cut" data)
- YOLO native context simulation (crops away the 15% extraction padding for pure YOLO box validation)
"""

import argparse
import logging
import random
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ORIENT_MAP = {
    'front': 0,
    'front_angle': 1,
    'side': 2,
    'rear_angle': 3,
    'rear': 4
}
NUM_ORIENT_CLASSES = 5

class MultiTaskCarDataset(Dataset):
    def __init__(self, img_paths, labels_orient, labels_integ, is_train=True):
        self.img_paths = img_paths
        self.labels_orient = labels_orient
        self.labels_integ = labels_integ
        self.is_train = is_train

        self.transform_base = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = str(self.img_paths[idx])
        orient = self.labels_orient[idx]
        integ = self.labels_integ[idx]  # 1 for full, 0 for cut
        
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
        h, w = img.shape[:2]
        
        if not self.is_train:
            # === Validation / Testing Logic ===
            # The user extracted ROIs with 15% padding on all sides. 
            # In production, testing runs on YOLO native boxes (0% padding).
            # To recover the YOLO crop: 1 / 1.3 ≈ 76.92% width/height.
            new_w, new_h = int(w / 1.3), int(h / 1.3)
            x1 = (w - new_w) // 2
            y1 = (h - new_h) // 2
            img = img[y1:y1+new_h, x1:x1+new_w]
            
        else:
            # === Training Logic ===
            # 1. Random Crop to simulate YOLO box bounding variance
            # Since YOLO is 76.9% of the pad, we randomly scale [0.70, 0.95]
            scale = random.uniform(0.70, 0.95)
            new_w, new_h = int(w * scale), int(h * scale)
            x1 = random.randint(0, w - new_w)
            y1 = random.randint(0, h - new_h)
            img = img[y1:y1+new_h, x1:x1+new_w]
            
            # 2. Dynamic Cut Augmentation
            # If the image is full (1) and we trigger the probability, we forcefully cut it >1/3
            # Generating infinite scenarios of "cut" vehicles.
            if integ == 1 and random.random() < 0.45:
                # Discard 35% to 55% of the image
                cut_ratio = random.uniform(0.35, 0.55)
                # Pick edge to cut (0: left, 1: right, 2: top, 3: bottom)
                side = random.randint(0, 3)
                nh, nw = img.shape[:2]
                
                if side == 0:   # cut left away -> keep right part
                    img = img[:, int(nw * cut_ratio):]
                elif side == 1: # cut right away -> keep left part
                    img = img[:, :int(nw * (1 - cut_ratio))]
                elif side == 2: # cut top away -> keep bottom part
                    img = img[int(nh * cut_ratio):, :]
                else:           # cut bottom away -> keep top part
                    img = img[:int(nh * (1 - cut_ratio)), :]
                    
                integ = 0 # Label is now CUT
                
        # Resize to typical network input size
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        if self.is_train and random.random() < 0.5:
            img = cv2.flip(img, 1) # Horizontal flip is orientation-invariant
            
        pil_img = Image.fromarray(img)
        if self.is_train:
            pil_img = self.transform_base(pil_img)
        
        tensor_img = self.to_tensor(pil_img)
        return tensor_img, torch.tensor(orient, dtype=torch.long), torch.tensor(integ, dtype=torch.float32)

class MultiTaskMobileNet(nn.Module):
    def __init__(self, num_orient_classes=5):
        super().__init__()
        # Use MobileNetV3-Large for high accuracy and fast ONNX deployment
        backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Output channels for MobileNetV3-Large features is 960
        in_features = 960 
        
        # Dual Heads
        self.orient_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_orient_classes)
        )
        self.integ_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        orient_logits = self.orient_head(x)
        integ_logits = self.integ_head(x).squeeze(1)
        return orient_logits, integ_logits

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="p4_data/labeled")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    img_paths, labels_orient, labels_integ = [], [], []
    
    # Class distribution tracking
    dist_integ = {0: 0, 1: 0}
    dist_orient = [0] * NUM_ORIENT_CLASSES
    
    for cat_dir in data_dir.iterdir():
        if not cat_dir.is_dir() or cat_dir.name in ["无效数据", "ignore"]:
            continue
            
        # Parse 'front_angle_full' into 'front_angle' and 'full'
        parts = cat_dir.name.rsplit('_', 1)
        if len(parts) != 2:
            continue
        orient_str, integ_str = parts[0], parts[1]
        
        if orient_str not in ORIENT_MAP:
            log.warning(f"Unknown orientation in category {cat_dir.name}")
            continue
            
        o_label = ORIENT_MAP[orient_str]
        i_label = 1 if integ_str == 'full' else 0
        
        for p in cat_dir.glob("*.jpg"):
            img_paths.append(p)
            labels_orient.append(o_label)
            labels_integ.append(i_label)
            
            dist_integ[i_label] += 1
            dist_orient[o_label] += 1

    total = len(img_paths)
    if total == 0:
        log.error("No valid labeled images found!")
        return

    log.info(f"Loaded {total} images.")
    log.info(f"Original Integrity Dist: Full: {dist_integ[1]} | Cut: {dist_integ[0]}")
    log.info(f"Orientation Dist: {dist_orient}")

    # Train/Val Split
    indices = np.arange(total)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split = int(0.8 * total)
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_dataset = MultiTaskCarDataset(
        [img_paths[i] for i in train_idx],
        [labels_orient[i] for i in train_idx],
        [labels_integ[i] for i in train_idx],
        is_train=True
    )
    val_dataset = MultiTaskCarDataset(
        [img_paths[i] for i in val_idx],
        [labels_orient[i] for i in val_idx],
        [labels_integ[i] for i in val_idx],
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = MultiTaskMobileNet(NUM_ORIENT_CLASSES).to(device)
    
    # Weighted Cross Entropy for Imbalanced Orientation
    orient_counts = np.array(dist_orient)
    orient_weights = total / (NUM_ORIENT_CLASSES * (orient_counts + 1))
    class_weights = torch.FloatTensor(orient_weights).to(device)
    log.info(f"Orientation Weights: {class_weights}")
    
    criterion_orient = nn.CrossEntropyLoss(weight=class_weights)
    # The BCE weight can be flat, because dynamic augmentation heavily pumps Cut class up to ~40-50%
    criterion_integ = nn.BCEWithLogitsLoss() 
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    out_dir = Path("p4_model_checkpoints")
    out_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        train_loss, train_o_loss, train_i_loss = 0, 0, 0
        for imgs, o_tgt, i_tgt in train_loader:
            imgs, o_tgt, i_tgt = imgs.to(device), o_tgt.to(device), i_tgt.to(device)
            
            optimizer.zero_grad()
            o_pred, i_pred = model(imgs)
            
            loss_o = criterion_orient(o_pred, o_tgt)
            loss_i = criterion_integ(i_pred, i_tgt)
            # Both are equally important
            loss = loss_o + loss_i  
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_o_loss += loss_o.item()
            train_i_loss += loss_i.item()
            
        # EVAL
        model.eval()
        val_loss, val_o_loss, val_i_loss = 0, 0, 0
        o_correct, i_correct, total_val = 0, 0, 0
        tp, fp, fn = 0, 0, 0
        
        with torch.no_grad():
            for imgs, o_tgt, i_tgt in val_loader:
                imgs, o_tgt, i_tgt = imgs.to(device), o_tgt.to(device), i_tgt.to(device)
                
                o_pred, i_pred = model(imgs)
                loss_o = criterion_orient(o_pred, o_tgt)
                loss_i = criterion_integ(i_pred, i_tgt)
                val_loss += (loss_o + loss_i).item()
                val_o_loss += loss_o.item()
                val_i_loss += loss_i.item()
                
                # Accuracies
                o_preds = torch.argmax(o_pred, dim=1)
                o_correct += (o_preds == o_tgt).sum().item()
                
                i_preds = (torch.sigmoid(i_pred) > 0.5).float()
                i_correct += (i_preds == i_tgt).sum().item()
                total_val += imgs.size(0)
                
                # Integrity stats (detecting full)
                tp += ((i_preds == 1) & (i_tgt == 1)).sum().item()
                fp += ((i_preds == 1) & (i_tgt == 0)).sum().item()
                fn += ((i_preds == 0) & (i_tgt == 1)).sum().item()
                
        t_l = train_loss / len(train_loader)
        v_l = val_loss / len(val_loader)
        o_acc = o_correct / total_val
        i_acc = i_correct / total_val
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        
        log.info(f"Epoch [{epoch+1}/{args.epochs}] TL:{t_l:.3f} VL:{v_l:.3f} | OrientAcc:{o_acc:.3f} IntegAcc:{i_acc:.3f} IntegF1:{f1:.3f}")
        
        if v_l < best_loss:
            best_loss = v_l
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / "p4_best.pt")
            log.info(f"  -> Saved new best model at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                log.info(f"Early stopping triggered after {epoch+1} epochs; no improvement for {args.patience} epochs.")
                break
                
        scheduler.step()
        
    log.info("Training complete!")
    
    # Export to ONNX
    log.info("Exporting to ONNX...")
    model.load_state_dict(torch.load(out_dir / "p4_best.pt", map_location=device))
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    onnx_path = out_dir / "p4_car_model.onnx"
    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['orient_logits', 'integ_logits'],
        dynamic_axes={'input': {0: 'batch_size'}, 'orient_logits': {0: 'batch_size'}, 'integ_logits': {0: 'batch_size'}}
    )
    log.info(f"ONNX model saved to {onnx_path}")

if __name__ == "__main__":
    train()
