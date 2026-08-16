# Auto-Culling (F1 Exclusive) 🏎️📸

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
- **Lite & Portable**: Completely removed heavy dependencies (Torch, OpenCV). The entire engine is now powered by **ONNX Runtime** and **Pillow**, enabling a <50MB compressed distribution.

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

## 📦 Binary Distribution (LITE)

The **LITE version** is a standalone executable that does not require Python or any heavy AI frameworks installed on your system. It is optimized for speed and portability.

### 1. Download & Use (Pre-compiled)
1. Download the latest `cull_photos_lite.zip` from our releases.
2. Unzip to any folder.
3. Run the command directly:
   - **macOS:** `./dist/cull_photos/cull_photos --input-dir /path/to/photos`
   - **Windows:** `.\dist\cull_photos\cull_photos.exe --input-dir C:\Photos`

### 2. Build Your Own (Packaging)
If you want to compile the binary yourself using `pyinstaller`:

**Step 1: Install PyInstaller**
```bash
uv pip install pyinstaller
```

**Step 2: Build the Lite Binary**
```bash
# Using the provided spec file (optimized to exclude Torch/CV2)
pyinstaller cull_photos.spec --noconfirm
```
The output will be available in `dist/cull_photos/`.

---

## 🖥️ Desktop GUI

The primary desktop app is a **Tauri 2** shell (`ui/` static frontend + Rust in `src-tauri/`): a modern dark UI rendered by the OS webview (WebView2 on Windows, WKWebView on macOS), with an installer-sized shell of ~10MB. The culling engine runs as a bundled Python sidecar (`cull_photos.py --json-lines`) that streams events over stdio — the GUI spawns the engine once per job and keeps it alive afterwards to decode previews, so there is no per-click process startup.

- **Streaming results**: rows appear in the table the moment each frame is scored — no need to wait for the whole job.
- **Live progress**: remapped phase bar (the scoring phase spans most of the bar) plus per-frame counters ("scored X/Y, keep A / discard B").
- **Full control**: every CLI option in the parameter panel (basic + advanced); settings persist in the browser's localStorage between sessions.
- **Result review**: sortable/filterable table of ratings, scores and veto reasons; click a row to preview the photo in a **fixed-size preview pane** that never reflows the layout.
- **Cancellation**: stop a running job at any time — already-scored frames stay visible and nothing is written to disk.
- **Logs & export**: live log panel, summary statistics (throughput, keep/discard, star distribution) and one-click CSV export via the native save dialog.

**Hardware backend (auto-selected per platform):** Windows prefers CUDA (`onnxruntime-gpu` + `nvidia-cudnn-cu12`), macOS prefers MLX then CoreML, everything else uses CPU. Unavailable providers degrade gracefully to CPU. Running from source on Windows uses CUDA automatically; the packaged binaries ship CPU-only (the CUDA runtime cannot be bundled without adding ~1GB) and fall back to CPU.

Run from source (dev mode):
```bash
python cull_gui.py                    # lightweight CustomTkinter fallback GUI
cd src-tauri && cargo tauri dev       # Tauri UI (requires Rust toolchain)
```

Launch the packaged Tauri app without stealing focus (e.g. from a fullscreen game):
```powershell
Start-Process -FilePath "...\auto-culling-gui.exe" -WindowStyle Minimized
```
Rust-side diagnostics are written to `gui.log` next to the executable.

Build the Tauri app:
```bash
pyinstaller cull_sidecar.spec --noconfirm          # 1. windowed sidecar
cp dist/cull_sidecar.exe src-tauri/binaries/cull-sidecar-x86_64-pc-windows-gnu.exe
tauri build --no-bundle                            # 2. shell + sidecar
# Windows: src-tauri/target/release/auto-culling.exe (+ cull-sidecar.exe beside it)
# macOS:   src-tauri/target/release/auto-culling
```

The legacy CustomTkinter GUI can still be built with `pyinstaller cull_gui.spec --noconfirm` (output `dist/auto_cull_gui_v0.1_win_x64.exe`). Packaged binaries launch subprocesses with `CREATE_NO_WINDOW`, so no command-line windows flash while a job is starting.

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

Run the full suite (CLI pipeline, GUI logic + view layer, cancellation semantics):

```bash
pytest tests/
```

Notes:
- `tests/test_package.py` additionally validates the packaged binary; it requires the built executable in the project root and is skipped otherwise.
- The GUI view-layer tests instantiate the real window and drive it by pumping the event loop. They need a display and skip automatically without one; on headless Linux CI use `xvfb-run pytest tests/`.
- `pyproject.toml` sets `--capture=sys`: fd-level capture makes Tcl/Tk initialization fail intermittently on Windows.

---

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).
