"""
Train a Multi-Task MobileNetV3 model for P4:
Task 1: Object Orientation (5 classes)
Task 2: Object Integrity (Binary: Full vs Cut/Occluded)

Incorporates:
- Dynamic "Cut" augmentation (crops >1/3 of "Full" images to generate infinite "Cut" data)
- YOLO native context simulation (crops away the 15% extraction padding for pure YOLO box validation)
- Resize-kernel randomization (cv2/PIL interpolation zoo) so the integrity head is
  invariant to the decode/ROI resampling chain — v1 trained on a single cv2.INTER_AREA
  kernel flipped verdicts on ~5% of real photos when the pipeline kernel changed
- Camera-pipeline jitter (gamma, per-channel gain, JPEG recompression, sensor noise)
  for cross-camera robustness
"""

import argparse
import logging
import random
from pathlib import Path
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

# Interpolation zoo spanning the kernels seen in production (PIL BILINEAR for
# ROI prep, cv2.INTER_AREA for decode) plus the ones we want to unlock
# (libjpeg draft ~= BOX, cv2 letterbox = LINEAR).
_CV2_KERNELS = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_CUBIC]
_PIL_KERNELS = [Image.BILINEAR, Image.BICUBIC, Image.BOX, Image.HAMMING, Image.LANCZOS]


def rand_resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize with a randomly drawn interpolation kernel."""
    if random.random() < 0.6:
        return cv2.resize(img, size, interpolation=random.choice(_CV2_KERNELS))
    pil = Image.fromarray(img).resize(size, random.choice(_PIL_KERNELS))
    return np.array(pil)


def pixel_jitter(img: np.ndarray) -> np.ndarray:
    """Emulate +-LSB decoder differences and sensor/readout noise."""
    sigma = random.uniform(0.0, 1.5)
    if sigma > 0:
        noise = np.random.default_rng().normal(0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def gamma_jitter(img: np.ndarray) -> np.ndarray:
    """Random gamma response curve (cross-camera tonal robustness)."""
    gamma = random.uniform(0.75, 1.35)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def channel_gain_jitter(img: np.ndarray) -> np.ndarray:
    """Small per-channel multiplicative gain (white-balance variation)."""
    gains = np.random.default_rng().uniform(0.94, 1.06, 3).astype(np.float32)
    out = img.astype(np.float32) * gains[None, None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg_roundtrip(img: np.ndarray, quality_range=(60, 95)) -> np.ndarray:
    """Re-encode/decode as JPEG to emulate camera codec differences."""
    q = int(random.uniform(*quality_range))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img if dec is None else dec

class MultiTaskCarDataset(Dataset):
    def __init__(self, img_paths, labels_orient, labels_integ, is_train=True,
                 pixel_jitter=True, jpeg_jitter=True, cache_dir: Path | None = None):
        self.img_paths = img_paths
        self.labels_orient = labels_orient
        self.labels_integ = labels_integ
        self.is_train = is_train
        self.pixel_jitter = pixel_jitter
        self.jpeg_jitter = jpeg_jitter
        self.cache_dir = cache_dir

        self.transform_base = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_roi(self, path: str) -> np.ndarray | None:
        """Decoded ROI: from the .npy cache when enabled, else JPEG decode."""
        if self.cache_dir is not None:
            cpath = self.cache_dir / (Path(path).stem + ".npy")
            if cpath.exists():
                arr = np.load(cpath)
                return arr if arr.ndim == 3 and arr.size else None
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = str(self.img_paths[idx])
        orient = self.labels_orient[idx]
        integ = self.labels_integ[idx]  # 1 for full, 0 for cut
        
        try:
            img = self._load_roi(path)
        except Exception:
            img = None
        if img is None:
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
                
        # Resize to typical network input size. Training draws a random
        # interpolation kernel so the integrity head becomes invariant to the
        # production decode/ROI resampling chain; validation keeps the
        # cv2.INTER_AREA reference.
        img = rand_resize(img, (224, 224)) if self.is_train \
            else cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        if self.is_train:
            if random.random() < 0.5:
                img = cv2.flip(img, 1) # Horizontal flip is orientation-invariant
            if self.pixel_jitter:
                img = pixel_jitter(img)
            if random.random() < 0.3:
                img = gamma_jitter(img)
            if random.random() < 0.3:
                img = channel_gain_jitter(img)
            if self.jpeg_jitter and random.random() < 0.35:
                img = jpeg_roundtrip(img)

        pil_img = Image.fromarray(img)
        if self.is_train:
            pil_img = self.transform_base(pil_img)
        
        tensor_img = self.to_tensor(pil_img)
        return tensor_img, torch.tensor(orient, dtype=torch.long), torch.tensor(integ, dtype=torch.float32)

class MultiTaskMobileNet(nn.Module):
    def __init__(self, num_orient_classes=5, arch: str = "large"):
        super().__init__()
        # MobileNetV3-Large for high accuracy and fast ONNX deployment;
        # 'small' is the speed variant (A/B'd — see perf docs).
        assert arch in ("large", "small"), arch
        if arch == "large":
            backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            in_features = 960
        else:
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            in_features = 576
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

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

@torch.no_grad()
def kernel_flip_rate(model, paths, integ_labels) -> tuple[int, int]:
    """Fraction of val images whose binary integrity verdict changes across
    resize kernels (cv2.INTER_AREA / PIL.BILINEAR / cv2.INTER_LINEAR).

    This is the production-critical metric: v1 flipped ~5% of real photos when
    the pipeline kernel changed, blocking every decode-path optimization."""
    model.eval()
    kernels = [cv2.INTER_AREA, Image.BILINEAR, cv2.INTER_LINEAR]
    per_kernel = [[] for _ in kernels]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    for path in paths:
        try:
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            nw, nh = int(w / 1.3), int(h / 1.3)
            x1, y1 = (w - nw) // 2, (h - nh) // 2
            roi = img[y1:y1 + nh, x1:x1 + nw]
            batch = []
            for k in kernels:
                if isinstance(k, int):
                    r = cv2.resize(roi, (224, 224), interpolation=k)
                else:
                    r = np.array(Image.fromarray(roi).resize((224, 224), k))
                x = torch.from_numpy(np.ascontiguousarray(r)).float().permute(2, 0, 1) / 255.0
                batch.append((x - mean) / std)
            xb = torch.stack(batch).to(device)
            _, i_logits = model(xb)
            probs = torch.sigmoid(i_logits).cpu().numpy()
            for j in range(len(kernels)):
                per_kernel[j].append(int(probs[j] > 0.5))
        except Exception:
            continue
    n = len(per_kernel[0])
    flips = sum(1 for t in zip(*per_kernel) if len(set(t)) > 1)
    return flips, n


def train():
    torch.backends.cudnn.benchmark = True  # fixed 224x224 input: one autotune pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="p4_data/labeled")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--arch", type=str, default="large", choices=["large", "small"])
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader workers; the per-sample pipeline is CPU-heavy "
                             "(decode + resize-kernel zoo + JPEG recompression) and "
                             "num_workers=0 starves the GPU")
    parser.add_argument("--cache-dir", type=str, default="p4_data/cache_raw",
                        help="pre-decoded ROI .npy cache (built once, then reads "
                             "raw pixels instead of JPEG-decode every epoch; "
                             "pass '' to disable)")
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
    
    cache_dir: Path | None = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # One pre-decode pass: raw-pixel .npy per ROI, so DataLoader workers
        # never JPEG-decode again (GPU starvation fix — see perf docs).
        import time as _t
        missing = [p for p in img_paths
                   if not (cache_dir / (Path(p).stem + ".npy")).exists()]
        if missing:
            log.info(f"Building decode cache for {len(missing)} ROIs...")
            t0 = _t.perf_counter()
            n_ok = 0
            for p in missing:
                im = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                if im is not None:
                    np.save(cache_dir / (Path(p).stem + ".npy"), im[:, :, ::-1])
                    n_ok += 1
            log.info(f"Cache built: {n_ok}/{len(missing)} in {_t.perf_counter()-t0:.0f}s")

    train_dataset = MultiTaskCarDataset(
        [img_paths[i] for i in train_idx],
        [labels_orient[i] for i in train_idx],
        [labels_integ[i] for i in train_idx],
        is_train=True,
        cache_dir=cache_dir,
    )
    val_paths = [img_paths[i] for i in val_idx]
    val_integ = [labels_integ[i] for i in val_idx]
    val_dataset = MultiTaskCarDataset(
        [img_paths[i] for i in val_idx],
        [labels_orient[i] for i in val_idx],
        [labels_integ[i] for i in val_idx],
        is_train=False,
        cache_dir=cache_dir,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers if args.num_workers <= 2 else 2,
                            persistent_workers=args.num_workers > 0, pin_memory=True)
    
    # Model
    model = MultiTaskMobileNet(NUM_ORIENT_CLASSES, arch=args.arch).to(device)
    
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
    out_dir = Path("models")
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
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            flips, n = kernel_flip_rate(model, val_paths, val_integ)
            log.info(f"  KernelFlipRate: {flips}/{n} = {flips / max(1, n):.2%} (target ~0)")
        
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
