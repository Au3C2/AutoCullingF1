# Auto-Culling (F1 Exclusive) 🏎️📸

**English** | [中文版](README_zh.md)

An automated photo culling tool for F1 & motorsport photography. It processes thousands of
burst-shot images (HEIF/RAW/JPEG), groups them into bursts, scores every frame with a
multi-stage AI pipeline, keeps the best shots per burst, and writes Lightroom-compatible
ratings (stars / reject flags) plus auto-crops — no manual triage required.

- **Input**: a folder straight off the camera (Sony ARW, Nikon NEF, Canon CR2/CR3, Fuji RAF,
  Olympus ORF, Panasonic RW2, Canon/Sony HEIF `.hif/.heif/.heic`, JPEG, PNG, TIFF)
- **Output**: Lightroom `.xmp` sidecars (RAW/HEIF) or in-file XMP metadata (JPEG/HIF) with
  star ratings, reject flags and crop parameters
- **Runtime**: pure ONNX Runtime inference — no PyTorch required at runtime

---

## Key Features

- **Burst grouping** from EXIF timestamps (with gap-based fallback).
- **Multi-stage scoring pipeline**:
  - **P0 Sharpness** — high-frequency energy ratio (FFT-based) with subject-ROI weighting;
    out-of-focus frames are vetoed.
  - **P1 Composition** — YOLO detection (F1-specific 14-class model at 640 px, with a COCO
    `yolov8n` cascade fallback when the F1 model misses) scores subject size, placement and
    lead room across the burst.
  - **P4 Orientation & Integrity** — MobileNetV3 multi-task classifier (224 px) rejects
    rear-view shots and penalizes cut / occluded subjects.
  - **P3 Fence veto** — optional fence classifier (disabled by default).
- **Top-N selection** — keeps the best *N* frames (default 11) per burst by raw score.
- **Auto-cropping** — computes a Lightroom crop around the detected subject (3:2 / 2:3)
  and writes `crs:` crop parameters.
- **Lightroom integration** — ratings appear instantly on import; already-rated files are
  skipped unless `--force`.
- **Deterministic mode** — `--deterministic` (or `CULL_DETERMINISTIC=1`) pins CPU-only ONNX
  + software decode for bit-identical results across macOS / Windows / Linux; this is the
  platform-independent truth all precision gates are locked against.

## End-to-End Performance

Gate protocol: ~500-file per-format sets, `--dry-run`, steady-state throughput measured on
real camera files (decode + burst grouping + AI inference + metadata write). Baselines and
methodology: [`results/performance_baseline.md`](results/performance_baseline.md).

### macOS — Apple M4 (10 cores), workers = 4

| Format | Throughput |
| :--- | ---: |
| JPEG | 83.5 img/s |
| HEIF | 65.5 img/s |
| Sony ARW | 49.9 img/s |
| Nikon NEF | 70.0 img/s |

Hardware acceleration: VideoToolbox HEIF decode, ImageIO JPEG decode, CoreML (+ANE) YOLO
static graphs, sharded exiftool EXIF scan. Serial scoring chain ≈ 37.5 fps.

### Windows — Ryzen 7 5700X + RTX 4070 Ti, workers = 4 (default 8)

| Format | workers = 4 | workers = 8 (default) |
| :--- | ---: | ---: |
| JPEG | 28.1 img/s | 40–42 img/s |
| HEIF | 38.0 img/s | 38.5 img/s |
| Sony ARW | 31.9 img/s | 35 img/s |
| Nikon NEF | 45.0 img/s | 46 img/s |

Inference runs on **DirectML** (onnxruntime-directml; scores the chain at 6.8 ms/frame vs
12.6 ms on the CUDA EP). HEVC 4:2:2 and JPEG have no hardware decoders on consumer NVIDIA
GPUs, so decode is libjpeg-turbo / libav software (measured and documented in
[`results/performance_baseline.md`](results/performance_baseline.md)).

> Per-frame cost is dominated by decode for JPEG/RAW (~115 ms for a 24 MP frame — Huffman
> entropy coding is resolution-independent) and by the inference chain for HEIF/NEF.
> `--workers` scales decode parallelism; ratings are identical across worker counts.

## Quick Start

### Option 1 — Standalone executable (no Python required)

Grab the prebuilt binary from the GitHub releases (or build it yourself — see
[Packaging](#packaging-standalone-binary)):

- Windows: `auto_cull_v0.1_win_x64.exe`
- macOS (Apple Silicon): `auto_cull_v0.1_macos_arm64`

```powershell
# Windows
.\auto_cull_v0.1_win_x64.exe --input-dir C:\Photos\F1 --recursive --force
```

```bash
# macOS
./auto_cull_v0.1_macos_arm64 --input-dir /path/to/photos --recursive --force
```

Omit `--input-dir` to open a folder picker. All [options](#useful-options) work the same
as the source CLI. The binary bundles the ONNX models, exiftool and ffmpeg runtime pieces —
nothing else needs to be installed.

### Option 2 — Run from source

Prerequisites:

- **Python 3.10+** with [uv](https://github.com/astral-sh/uv)
- **ffmpeg** on PATH (macOS: `brew install ffmpeg`; Windows: bundled in `external/ffmpeg/`)
- **exiftool** — bundled under `external/exiftool/` (macOS uses the system perl that ships
  with macOS); Windows CI installs it via `choco install exiftool`
- GPU optional: NVIDIA (DirectML/CUDA EP) or Apple Silicon (CoreML) accelerate inference;
  everything falls back to CPU automatically.

```bash
uv sync
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python cull_photos.py --input-dir /path/to/photos --recursive --force
```

#### Useful options

| Option | Meaning |
| :--- | :--- |
| `--workers N` | Decode-pool size (default 8; ratings are worker-invariant) |
| `--top-n 11` | Max keepers per burst group |
| `--scale-width 1280` | Decode resolution for the scoring chain |
| `--p4-policy` | `always` (default) / `never` / `auto` (only for F1/GP folders) |
| `--crop-off` | Disable auto-crop writing |
| `--dry-run` | Score and report without writing any metadata |
| `--dump-scores FILE` | Export per-image CSV (sharp/comp/raw/rating) |
| `--force` | Re-analyze files that already carry ratings |
| `--deterministic` | Bit-identical cross-platform CPU path (slower) |

There is also a small **customtkinter GUI** (`cull/gui`) when `--input-dir` is omitted.

## Packaging (standalone binary)

```bash
uv pip install pyinstaller

python packaging/build.py            # onefile  -> dist/ + copy to repo root
python packaging/build.py --onedir   # directory form (recommended: no per-launch
                                     # extraction/signature tax)
```

Artifacts: `auto_cull_v0.1_win_x64.exe` (Windows, ~478 MiB onedir) and
`auto_cull_v0.1_macos_arm64` (macOS, 161 MiB onefile). The spec bundles the frozen ONNX
graphs, exiftool (perl form) and ffmpeg runtime pieces; no Python installation is needed on
the target machine. Unified regression suite for packaging:
`python packaging/guards.py` (precision → perf → build → packaged precision → packaged perf).

## Project Structure

```text
auto_culling/
├── cull/                  # Core engine: loader (decode), detector (YOLO), sharpness,
│                          # composition, scorer (P0-P4 + ratings), p4_classifier,
│                          # fence_classifier, engine (process pool + consumer),
│                          # exif_reader, xmp_writer, gui
├── models/                # ONNX weights (f1_yolov8n, yolov8n, p4_car_model + static variants)
├── train/                 # Training pipelines (YOLO fine-tune, P4 multi-task, fence, tuning)
├── packaging/             # build.py + guards.py (PyInstaller, unified regression suite)
├── benchmarks/            # run_benchmarks.py — per-format steady-state perf gate
├── tests/                 # Precision gates + deterministic truth + CI harness
├── scripts/               # Utilities (baseline generation, profilers, precision report)
├── eval/                  # Offline evaluation tooling
├── docs/                  # Optimization plans, labeling guides
├── results/               # performance_baseline.md (authoritative numbers & history)
├── external/              # Vendored exiftool (+ ffmpeg on Windows)
└── cull_photos.py         # CLI entry point
```

## Scoring Logic

```
raw_score = 1.5 × S_sharp + 2.5 × S_comp − 0.6 × [P4 integrity = cut/occluded]

rating: raw < 3.11 → 1★, < 3.40 → 2★, < 3.80 → 3★, < 4.20 → 4★, else 5★
```

Automatic rejection (veto, rating = −1):

- No subject detected in the frame
- Sharpness below threshold (0.05)
- Car orientation classified as **rear**
- Raw score below the minimum (3.1)

After per-frame scoring, `select_best_n` keeps the top *N* frames per burst and adjusts
star ratings within the group.

## Testing & Gates

```bash
# Precision: ratings + raw scores vs the committed deterministic truth (70 gate files)
pytest tests/ -m deterministic                       # truth platform check (strict)
pytest tests/test_cull.py tests/test_precision_heif.py tests/test_precision_raw.py

# Performance: per-format steady-state gate (needs the ~1.3 GB camera datasets)
python benchmarks/run_benchmarks.py

# Everything (precision → perf → build → packaged gates): ~12-15 min
python packaging/guards.py

# Regenerate the deterministic truth after intentional scoring changes
CULL_DETERMINISTIC=1 python scripts/generate_deterministic_baseline.py
```

CI (`.github/workflows/`) runs the same gates on GitHub-hosted macOS/Windows runners using
committed seed samples (`tests/ci/`). Precision is consistency-based: the packaged binary
must match the source pipeline per-file (±0.002 raw tolerance), and ratings must be equal.

Further reading: [`results/performance_baseline.md`](results/performance_baseline.md)
(measured numbers, optimization history, platform baselines),
[`docs/P4_LABELING.md`](docs/P4_LABELING.md) (P4 labeling guide).

## License

Licensed under the [Apache License 2.0](LICENSE).
