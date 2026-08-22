# Performance Baseline — master@CUDA (2026-08-22)

Measured on the `develop`/`master` line after rebase, with `onnxruntime-gpu 1.23.2`
+ CUDAExecutionProvider active (env: Windows 10, RTX 4070 Ti, driver CUDA 13.1,
Python 3.10, AMD Zen3 CPU). All end-to-end numbers below are `--workers 4 --dry-run`.

## End-to-end throughput

| Dataset | Count | Throughput | Engine time | Decode path |
|---|---|---|---|---|
| JPG (24MP → 1280px, Pillow) | 60 | 5.9 img/s | 10.2 s | Pillow full-decode + resize |
| HEIF (1664x1088 preview stream) | 100 | 6.4 img/s | 15.5 s | ffmpeg spawn + Pillow resize |
| ARW (exiftool embed) | 100 | 4.4 img/s | 22.5 s | exiftool `-JpgFromRaw` per file |
| NEF (exiftool embed) | 100 | 3.3 img/s | 29.6 s | exiftool `-PreviewImage` per file |

Real (non-dry-run) runs additionally pay ~399 ms per file in metadata sync
(exiftool spawn per file) — not visible in dry-run benchmarks.

## Per-frame stage profile (single thread, warm)

Serial budget ≈ 300 ms/frame, decode dominates everywhere.

| Stage | ms/img | Notes |
|---|---|---|
| Decode JPG | 186 | 62% of budget |
| Decode HEIF | 218 | ffmpeg spawn 112 + Pillow resize 106 |
| Decode ARW / NEF | 462 / 484 | subprocess spawn dominates |
| EXIF scan | ~30 | scan phase, amortized |
| letterbox (PIL) | 6.2 | |
| numpy preprocess | 2.0 | |
| session.run (CUDA) | 8.2 | stays 5.7 ms even at 8 concurrent threads |
| Python postprocess (8400-row argmax + NMS) | 12.4 | pure-Python hotspot; GIL-amplified to 49.6 ms detect at 4 threads |
| Sharpness FFT | 21.7 | |
| P4 classifier | 6 | only when policy active |
| Composition | ~0.01 | |
| Metadata sync (real runs) | 399 /file | exiftool spawn per file |

## Key facts established

- CUDA batch inference is a LOSS for yolov8n: batch=8 = 0.5x of 8 single runs
  (model too small; H2D bandwidth dominated). Do not batch the YOLO stage.
- Inference backend is irrelevant to end-to-end throughput — decode is 62%+ of
  the budget. CPU (28 ms) vs CUDA (8.2 ms) session.run barely moves the total.
- All 3 ONNX models have dynamic batch dims (opset 17) — batching is legal but
  not beneficial for YOLO; P4 (224x224) could batch opportunistically (minor).
- README's documented 35/52 img/s (CUDA) has no supporting artifacts in git and
  cannot be reproduced at any commit (max recorded: 7.8 img/s CPU @master tip,
  7.5 img/s @6694dd6). Treat README performance table as unverified legacy.
- Current engine architecture: ThreadPoolExecutor over burst groups, every
  worker runs decode→detect→score synchronously. Multi-thread contention on the
  Python postprocess makes detect slower at workers=4 (49.6 ms) than 1-thread
  (33 ms). Decode is the actual serial bottleneck, not inference.

## Optimization roadmap (ordered by difficulty ↑, change size ↑, benefit ↓)

| # | Item | Difficulty | Change | Expected gain |
|---|---|---|---|---|
| 1 | ffmpeg `-vf scale=1280:-1` instead of Pillow resize | low | ~10 lines | HEIF decode 218→~120 ms (−45%) |
| 2 | Vectorize postprocess (numpy mask, no 8400-row loop) | low | ~15 lines | detect 33→~13 ms; removes GIL amplification |
| 3 | Decode process pool (freeze_support pattern, verified on gui b955fcf) | med-low | ~25+3 lines | decode off critical path; JPG/HEIF ~2–4x |
| 4 | RAW batch extract (single exiftool `-stay_open` pass) | medium | loader fn | ARW/NEF 462/484→~150 ms |
| 5 | Fetch sharpness in decode workers | low (needs #3) | ~10 lines | −22 ms serial |
| 6 | Batch metadata write (one exiftool session) | medium | batch API | 399→~30 ms/file real runs |
| 7 | pyav / resident ffmpeg HEVC decode | med-high | medium | HEIF −50% on top of #1 |
| 8 | Re-train f1+coco → single YOLO (unified classes) | high | training + re-validate | −13 ms/frame; 3→2 sessions |

Target after #1–#6: serial critical path ≈ 21 ms/frame → 25–35 img/s JPG/HEIF,
15–25 img/s RAW.

## Regression gate

Precision gates (`tests/test_cull.py`, `tests/test_precision_heif.py`,
`tests/test_precision_raw.py`) must stay green after any change; they run the
CLI at **workers=1 for determinism** — see below. The performance gate is
`benchmarks/run_benchmarks.py` (4-dataset protocol; gate thresholds locked to
the GATE sample sizes: JPG 60/HEIF 24/ARW 20/NEF 20 → 5.26/3.76/2.86/2.39 img/s
measured, thresholds 4.2/3.0/2.3/1.9; the 100-image baselines above run ~1.3–2×
faster because per-job overhead amortizes better).

### Known issue: CUDA concurrency is non-deterministic (pre-optimization)

With the master engine (ThreadPoolExecutor over burst groups, one shared CUDA
session, NO lock) the pipeline intermittently drops detections under
concurrent workers: measured rating flips and raw_score drift (e.g.
DSC00827.ARW raw 2.436 → 0.394; IMG_20260314_151744_020 rating 3 → -1) in
~2/3 of repeated runs. Single-threaded (workers=1) runs are fully
deterministic (3/3 clean across all gates). Precision gates therefore pin
workers=1; restoring deterministic concurrency (decode process pool +
single-consumer inference, roadmap item #3) is required before workers>1 can
be trusted again. Track this in the optimization gate, not by loosening tests.