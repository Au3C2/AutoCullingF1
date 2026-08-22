# Auto-Culling — Agent Rules

## Project Overview

F1 motorsport burst-photo binary classifier. Learns which shots a photographer keeps (label=1) vs discards (label=0). PyTorch fine-tuning pipeline + ONNX export for production inference.

## Language

- All code, comments, and docstrings: **English**
- All communication with the user: **Chinese**

## Environment

- Python 3.10, managed by `uv` at `/home/au3c2/.local/bin/uv` (NOT on PATH)
- Always activate venv before running Python: `source .venv/bin/activate`
- Or use full path: `/home/au3c2/.local/bin/uv run <script>`
- GPU: NVIDIA RTX 4070 Ti (12 GB VRAM), CUDA 12.4, PyTorch 2.6.0+cu124
- Project root: `/home/au3c2/auto_culling/`

## Tooling Conventions

- Path operations: always use `pathlib.Path`, never string concatenation
- CLI arguments: always use `argparse` with `ArgumentDefaultsHelpFormatter`
- No `cd <dir> && <cmd>` — use `workdir` parameter or activate venv instead
- Shell scripts: must activate `.venv` first (`source .venv/bin/activate`) since `uv` is not on PATH

## Architecture & Training

- Supported archs: `resnet18`, `resnet50`, `resnext50`, `mobilenetv3`
- All models output a single logit → `BCEWithLogitsLoss` (binary)
- Fine-tune mode: freeze all → unfreeze last 2 backbone blocks + head
  - ResNet/ResNeXt: `layer3` + `layer4` + `fc`
  - MobileNetV3: `features[-2]` + `features[-1]` + `classifier`
- Classification head always has `Dropout(0.3)` before `Linear(in_features, 1)`
- Training is **step-based** (not epoch-based)
- AMP (`torch.amp`), `WeightedRandomSampler`, gradient clipping (`max_norm=1.0`)
- Loss: `LabelSmoothingBCELoss` with `pos_weight` support (defined in `train.py`)

## Anti-Overfitting (v2 settings — do not regress)

- `--label-smoothing 0.1`
- `--weight-decay 5e-4`
- `Dropout(p=0.3)` in head
- Unfreeze `layer3` + `layer4` (not just `layer4`)
- Train augmentation: `RandAugment(n=2, m=9)` + `GaussianBlur(k=5)` + `RandomGrayscale(p=0.1)`
- **No spatial crop augmentation** — culling depends on full-frame composition

## Dataset

- Images live in `dataset/cache/*.jpg` (512×512 JPEG, pre-decoded from HIF)
- `dataset/img/` is empty — original HIF files deleted
- CSVs: `dataset/train_info.csv` (5975 rows), `dataset/test_info.csv` (1494 rows)
- `img_path` column in CSVs still has old `.HIF` paths — `CullingDataset` resolves to cache automatically
- Label distribution: 38.9% keep / 61.1% discard

## Checkpoints

All v2 best checkpoints are in `checkpoints/<arch>_finetune_v2/best.pt`.
Checkpoint dict keys: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `step`, `val_f1`, `arch`.

## ONNX

- Exported with opset 17, dynamic batch axis
- Output files: `onnx_models/<arch>.onnx`
- Inference provider priority: `CUDAExecutionProvider` → `CPUExecutionProvider`
- Preprocessing must match eval transform: SquarePad → Resize(224) → ToTensor → ImageNet Normalize

## Key Files

| File | Purpose |
|---|---|
| `src/auto_culling/model.py` | `build_model(arch, finetune, pretrained)` factory |
| `src/auto_culling/dataset.py` | `CullingDataset`, `build_dataloaders`, `SquarePad` |
| `src/auto_culling/train.py` | Training loop, `LabelSmoothingBCELoss`, `EarlyStopping` |
| `export_onnx.py` | Export `.pt` → `.onnx` + ORT sanity check |
| `infer_onnx.py` | Batch inference on photo dir → `keep/` / `discard/` + `scores.csv` |
| `benchmark_onnx.sh` | One-shot export + throughput benchmark |
| `run_finetune_v2.sh` | Training launcher (v2 hyper-parameters) |
| `REPORT.md` | Full experiment log (Chinese) |

## TensorBoard Logging

Scalars logged per step: `train/loss`, `val/loss`, `val/acc`, `val/f1`, `val/auc`, `test/*`, `lr`.
Logs saved to `checkpoints/<run>/tb_logs/`.

## v2 Benchmark Results (do not overwrite without re-running experiments)

| Model | Test F1 | Peak ONNX img/s |
|---|---|---|
| resnext50 | 0.7680 (best) | 42 |
| resnet18 | 0.7554 | 165 |
| resnet50 | 0.7555 | 57 |
| mobilenetv3 | 0.7279 | 218 (fastest) |

## Runtime Performance Baseline (master/develop @ CUDA, 2026-08-22)

Authoritative detail in `results/performance_baseline.md`. Benchmarks: `--workers 4 --dry-run`,
`onnxruntime-gpu 1.23.2`, CUDAExecutionProvider, RTX 4070 Ti. Keep this baseline.

End-to-end: JPG 5.9 / HEIF 6.4 / ARW 4.4 / NEF 3.3 img/s. Serial budget ≈300 ms/frame:
decode 186–218 ms (JPG-Pillow / HEIF-ffmpeg+resize) is the dominant bottleneck (62%+);
RAW 462/484 ms via per-file exiftool spawn. Inference is NOT the bottleneck: CUDA
session.run 8.2 ms, CPU 28 ms. Real runs also pay 399 ms/file exiftool metadata sync
(dry-run hides it).

Established facts:
- yolo batch inference is a LOSS (batch=8 = 0.5×; model too small, H2D bound) — do not batch YOLO.
- All 3 ONNX (f1/yolov8n 640px + p4 224px, opset 17) have dynamic batch dims.
- README's 35/52 img/s is UNVERIFIED legacy (no artifacts; unreproducible at any commit; max 7.8 img/s).
- Engine: ThreadPoolExecutor over burst groups; Python postprocess (8400-row argmax, 12.4 ms)
  is GIL-amplified at workers=4 (detect 49.6 vs 33 ms single-thread).
- CUDA support needs `ensure_nvidia_runtime_on_path()` (detector.py) for nvidia wheel DLLs.

Optimization order (difficulty ↑ / benefit ↓): 1 ffmpeg `-vf scale` (HEIF −45%),
2 vectorize postprocess (33→13 ms), 3 decode process pool (JPG/HEIF ~2–4×),
4 RAW batch exiftool `-stay_open` (462→~150 ms), 5 sharpness into decode workers,
6 batch metadata sync (399→30 ms/file). Target after #1–#6: 25–35 img/s JPG/HEIF.
Use the 4-dataset benchmark protocol as the regression gate alongside the CSV-baseline
pytest suite.

Gates (built 2026-08-22): precision = `tests/test_cull.py` +
`tests/test_precision_heif.py` (24 HEIF) + `tests/test_precision_raw.py`
(20 ARW + 20 NEF), ALL pinned to `--workers 1`; performance =
`benchmarks/run_benchmarks.py` (gate thresholds for its small samples: JPG
5.26 / HEIF 3.76 / ARW 2.86 / NEF 2.39 img/s). KNOWN ISSUE: the CUDA +
shared-session + concurrent-workers engine is NON-DETERMINISTIC (intermittent
detection drops → rating flips / raw-score drift, ~2/3 of runs at workers=4);
workers=1 is deterministic and is why gates pin it. Item #3 (single-consumer
inference + decode process pool) is required to make workers>1 trustworthy;
do not fix by loosening gates.
