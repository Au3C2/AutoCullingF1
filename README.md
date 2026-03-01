# F1 Motorsport Photo Auto-Culling

A deep learning pipeline that learns which burst-shot photos a motorsport photographer would **keep** (label=1) vs **discard** (label=0). Trained on F1 race images using fine-tuned CNN classifiers, exported to ONNX for fast production inference.

## Results

| Model | Test Acc | Test F1 | Test AUC | Peak img/s (ONNX) |
|---|---|---|---|---|
| **resnext50** | **81.9%** | **0.7680** | 0.8639 | 42 |
| resnet18 | 80.2% | 0.7554 | 0.8611 | 165 |
| resnet50 | 80.0% | 0.7555 | 0.8577 | 57 |
| mobilenetv3 | 75.4% | 0.7279 | **0.8710** | **218** |

> resnext50 has the best accuracy. resnet18 offers the best speed/accuracy trade-off. Benchmarked on RTX 4070 Ti.

## Project Structure

```
auto_culling/
├── src/auto_culling/
│   ├── model.py          # ResNet-18/50, ResNeXt-50, MobileNetV3-Large builders
│   ├── dataset.py        # CullingDataset, augmentation pipelines, DataLoader factory
│   └── train.py          # Training loop, LabelSmoothingBCELoss, EarlyStopping, TensorBoard
├── export_onnx.py         # Export best.pt checkpoints → ONNX + ORT verification
├── infer_onnx.py          # Batch inference on a photo directory → keep/ / discard/
├── benchmark_onnx.sh      # One-shot: export all models + measure throughput
├── prepare_dataset.py     # Build train/test CSVs from raw data
├── cache_images.py        # Decode HIF/HEIF → 512×512 JPEG cache
├── download_pretrained.py # Pre-download ImageNet weights to torch hub cache
├── run_finetune_v2.sh     # Training launcher (v2 settings)
├── onnx_models/           # Exported ONNX files (resnet18/50, resnext50, mobilenetv3)
└── REPORT.md              # Full experiment log (Chinese)
```

## Quick Start

### Setup

```bash
# Python 3.10 + uv required
/home/au3c2/.local/bin/uv sync
source .venv/bin/activate

# Install PyTorch (CUDA 12.4)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install ONNX runtime
uv pip install onnx onnxruntime-gpu
```

### Inference on your photos

```bash
# Sort a directory of JPEGs into keep/ and discard/
python infer_onnx.py \
    --model onnx_models/resnext50.onnx \
    --input-dir /path/to/photos \
    --output-dir /path/to/sorted \
    --threshold 0.5

# Copy instead of move
python infer_onnx.py --model onnx_models/resnet18.onnx \
    --input-dir /path/to/photos --output-dir /path/to/sorted --copy
```

Output structure:
```
sorted/
├── keep/       ← predicted label=1 (photographer would keep)
├── discard/    ← predicted label=0 (photographer would discard)
└── scores.csv  ← per-image sigmoid score
```

### Export checkpoints to ONNX

```bash
# Export all four v2 checkpoints
python export_onnx.py

# Export a single model
python export_onnx.py --arch resnext50
```

### Run benchmark

```bash
bash benchmark_onnx.sh
```

### Train from scratch

```bash
# Prepare dataset CSVs
python prepare_dataset.py

# Cache images (HIF/HEIF → 512×512 JPEG)
python cache_images.py

# Fine-tune all four architectures (v2 settings)
bash run_finetune_v2.sh
```

## Model Architecture

All models output a single logit for `BCEWithLogitsLoss` (binary classification).

**Fine-tune strategy (v2):**
- ResNet-18/50, ResNeXt-50: freeze all → unfreeze `layer3` + `layer4` + `fc`
- MobileNetV3-Large: freeze all → unfreeze `features[-2:]` + `classifier`
- Classification head: `Dropout(0.3)` → `Linear(in_features, 1)`

**Anti-overfitting measures:**
- Label smoothing (`--label-smoothing 0.1`)
- Dropout(0.3) before FC head
- RandAugment(n=2, m=9) + GaussianBlur + RandomGrayscale
- Weight decay 5e-4
- No spatial crop augmentation (preserves full-frame composition)

## Dataset

- 7469 images (512×512 JPEG cache), split 5975 train / 1494 test
- Label distribution: 38.9% keep, 61.1% discard (~1.6:1 imbalance)
- Imbalance handled via `WeightedRandomSampler` + `pos_weight` in loss

## Requirements

- Python 3.10
- PyTorch 2.6.0 + CUDA 12.4
- torchvision, onnx, onnxruntime-gpu
- pandas, scikit-learn, Pillow, pillow-heif, tqdm
