# Auto-Culling 🏎️📸

[中文版](README_zh.md) | **English**

An automated photo culling tool for F1 & motorsport photography. It systematically processes thousands of burst-shot images (HIF/RAW), identifies the best shots using deep learning and heuristic rules, and generates Lightroom-compatible XMP sidecars with ratings and auto-crops.

---

## 🌟 Key Features

- **Burst Grouping**: Automatically groups rapid-fire sequences based on EXIF timestamps.
- **Multi-Stage Scoring Pipeline**:
  - **P0 Sharpness**: High-frequency detail analysis (HF Ratio) to filter out-of-focus shots.
  - **P1 Composition**: YOLO-based object detection (F1 specific + COCO) to evaluate subject size and centering.
  - **P4 Orientation & Integrity**: MobileNetV3 multi-task model to classify car orientation (rejecting rear shots) and detect cut/occluded subjects.
- **Top-N Selection**: Intelligently selects the best $N$ frames from each burst sequence.
- **Auto-Cropping**: Automatically calculates and writes optimal crops to XMP based on subject position and target aspect ratio (3:2/2:3).
- **Lightroom Integration**: Generates `.xmp` files that Lightroom Classic reads instantly for ratings (1-5 stars) and flags.

---

## 🚀 End-to-End Performance

Measured on a sample of 1000 HEIF images (1280px decode scale). **"End-to-End"** throughput represents the entire workflow: file loading, decoding, multi-stage AI inference, and XMP generation.

### macOS (Apple Silicon M4 Pro)
Optimized for the Apple Neural Engine (ANE) using CoreML.

| Backend | Hardware | End-to-End Throughput |
| :--- | :--- | :--- |
| **ONNX Runtime** | M-Series CPU | ~13.8 img/s |
| **CoreML** | **Neural Engine (ANE)** | **~18.6 img/s (+35%)** |

### Windows (Intel i9 + RTX 4070 Ti)
Leverages CUDA acceleration and massively parallel prefetching.

| Backend | Hardware | End-to-End Throughput |
| :--- | :--- | :--- |
| **CUDA** | **NVIDIA RTX 4070 Ti** | **~35.0 img/s** |
| **CUDA** | **NVIDIA RTX 4090** | **~52.0+ img/s** |

---

## 🛠️ Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **FFmpeg**: Required for high-speed HIF decoding.
  - **macOS**: `brew install ffmpeg`
  - **Windows**: [Download](https://ffmpeg.org/download.html) and add to `PATH`.

### 2. Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

**macOS / Linux:**
```bash
uv sync
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
uv sync
.venv\Scripts\activate.ps1
```

### 3. Basic Usage

Analyze a directory of images and generate XMP sidecars:

**macOS:**
```bash
python cull_photos.py --input-dir /path/to/photos --workers 8 --scale-width 1280
```

**Windows:**
```powershell
python cull_photos.py --input-dir C:\Photos\F1 --workers 12 --scale-width 1280
```

**Common Options:**
- `--workers N`: Number of parallel prefetch workers.
- `--scale-width 1280`: Downscale images during decode for faster processing.
- `--top-n 11`: Max keepers per burst group.
- `--force`: Re-analyze even if XMP/Ratings already exist.

---

## 📂 Project Structure

```text
auto_culling/
├── cull/                  # Core package (Sharpness, Composition, Detectors, Scorer)
├── eval/                  # Evaluation & benchmarking scripts
├── train/                 # Model training pipelines (YOLO, Classifiers)
├── utils/                 # Utility scripts (Autocrop, EXIF tools, Model download)
├── models/                # Model weights (Local ONNX/CoreML)
├── results/               # Benchmark reports and experiment logs
├── tests/                 # Automated test suite
└── cull_photos.py         # Main entry point
```

---

## 📊 Scoring Logic

The final `raw_score` is calculated as:
$$score = 1.5 \times S_{sharp} + 2.5 \times S_{comp} - Penalty_{cut}$$

**Veto Rules (Automatic Rejection):**
- No target detected.
- Sharpness below threshold (0.05).
- Car orientation is "Rear" (back view).
- Low overall score (below 3.1).

---

## 🧪 Testing

Run the integration test suite to verify backend execution and XMP accuracy:

```bash
pytest tests/test_cull.py
```

---

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).
