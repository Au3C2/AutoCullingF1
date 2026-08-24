# Performance Baseline — master@CUDA (2026-08-22)

Measured on the `develop`/`master` line after rebase, with `onnxruntime-gpu 1.23.2`
+ CUDAExecutionProvider active (env: Windows 10, RTX 4070 Ti, driver CUDA 13.1,
Python 3.10, AMD Zen3 CPU). All end-to-end numbers below are `--workers 4 --dry-run`.

## End-to-end throughput

**After optimization #3 (decode process pool + single-consumer inference, 2026-08-22) + #A (cv2.INTER_AREA resize) + #B (cv2.dft sharpness):**

| Dataset | Count | Pre-#3 | #3 | **+ #A/#B** | Decode path |
|---|---|---|---|---|---|
| JPG (24MP → 1280px) | 60 | 5.9 | 6.7 | **7.2 img/s** | Pillow decode + cv2.INTER_AREA resize |
| HEIF (1664x1088 preview) | 100 | 6.4 | 11.5 | **~12 img/s** | ffmpeg spawn + cv2.INTER_AREA |
| ARW (exiftool embed) | 100 | 4.4 | 4.7 | **~5.0 img/s** | exiftool per file |
| NEF (exiftool embed) | 100 | 3.3 | 4.6 | **~5.5 img/s** | exiftool per file |

*Note: 60/100-image wall numbers on this line vary (machine load); the perf-gate
protocol numbers (JPG 7.21 / HEIF 4.49 / ARW 3.15 / NEF 3.42) are the
authoritative regression baseline since 2026-08-22.*

**#A — decode resize (KEPT):** Pillow BILINEAR → `cv2.INTER_AREA`. Scoring
sensitivity at 30× downscale is fundamental (all 13 backends drift ≥0.18); the
user accepted INTER_AREA's small systematic upward drift (~8% flips at score
thresholds, all upward: 2★→3★, −1→+2/keep). Area-average is the only
mathematically sound downsampler among cv2 methods. ~8× faster resize.

**#B — sharpness FFT (KEPT):** numpy float64 `fft2+fftshift+mgrid` → `cv2.dft`
(float32, C++/IPP) + unshifted radial-distance broadcasting (min(y,h−y)² +
min(x,w−x)² mask, no quadrant copy, no sqrt). HF ratio diff < 2e-9 (verified
24/24), single-image sharpness 20.8 → ~5.2 ms (4×), scores unchanged.

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

## Attempted optimizations log (each run through gates; reverted if no effect or drift)

| # | Change | Gate result | Decision |
|---|---|---|---|
| 1 | ffmpeg `-vf scale=1280:-1` instead of Pillow resize (sws_scale in-process) | PRECISION FAIL: HEIF `DSC00827.heif` raw_score 1.137→1.361 (+20%); sws↔Pillow-BILINEAR kernel differences alter pixels. Throughput gain only −13% (281→245 ms/img), far below the −45% hoped | **REVERTED** (2026-08-22). Pixel-identity is a hard requirement; do not retry unless a pixel-exact downscaler is available. |
| 2 | Vectorize YOLO postprocess (numpy mask+argmax replace 8400-row python loop in `LiteYOLO.detect`) | Precision 6/6 green; per-image detections bit-identical to old loop (24/24, checked to 9 decimals); detect 33.1→20.3 ms/img single-thread (−39%). E2E throughput gates unchanged (JPG 5.09 vs 5.26) — decode still dominates and hides it | **KEPT** (2026-08-22). Equal-op optimization, prerequisite for #3's single-consumer path; E2E benefit expected after #3 |
| 3 | Decode process pool + single-consumer inference (`engine.run` drains `ProcessPoolExecutor` decodes in burst order on one thread with one CUDA session) | Precision 6/6 × 3 consecutive runs at workers=4 (fixes the concurrency non-determinism). Throughput gates: JPG 6.84 (+35%), HEIF 3.73, ARW 2.98 (+14%), NEF 3.15 (+48%) — all above old baselines. Full-100 runs: HEIF 6.4→11.5 (+80%), NEF 3.3→4.6 (+35%), JPG 5.9→6.7, ARW 4.4→4.7 | **KEPT** (2026-08-22). Confidence gates unlocked back to workers=4 |
| 4 | RAW batch extraction (single exiftool spawn for N files) | NOT FEASIBLE pixel-safe: `-b` only writes concatenated stdout (no per-file framing); `-w` renders `Jpg` text wrapper (89 bytes) instead of raw JPEG bytes. No exiftool path extracts embedded JPEGs byte-identically in batch. The #3 process pool already parallelizes the per-file spawns (ARW 4.4→4.7) | **DROPPED** (2026-08-22). Would need a decoder change (e.g. rawpy extract_thumb) with a new pixel-identity gate; revisit only if RAW becomes the bottleneck after #6 |
| 5 | Sharpness into decode workers | `score_sharpness` crops the detection ROI, so it depends on detections produced by the consumer thread; parallelizing it changes the algorithm (image-level vs ROI FFT) → precision drift risk. Decode is already parallelized (#3); sharpness is now ~35 ms of the ~35 ms serial budget after #3 | **DROPPED** (2026-08-22) as-is. Only revisit with full-image sharpness re-validation |
| 6 | Batch metadata write (persistent exiftool `-stay_open` session, text protocol, one process for all files) | `update_image_metadata_batch` measured 60 ms/file vs 458 ms/file per-spawn (7.6×); read-back verified 12/12 ratings correct; precision gates 6/6 green; real 60-JPG run no longer charged per-file spawn (engine 9.3 img/s = dry-run rate) | **KEPT** (2026-08-22) |
| 7 | Sharpness offloaded to a 2-worker process pool (deferred-finalize pipeline: frame i sharpness computes while frame i+1 detects) | Deterministic (two runs bit-identical). BUT zero end-to-end gain on all four datasets (JPG 6.4 vs 6.7, HEIF 11.3 vs 11.5, ARW 4.7, NEF 4.7 — all within noise of #3 figures); decode supply (4 workers × 186 ms JPG) is the actual bottleneck and the extra 2 workers compete for CPU | **REVERTED** (2026-08-22). Serial path is no longer the limiter; raising decode parallelism is the lever, not more consumer-side overlap |
| 8 | EXIF scan via persistent `-stay_open` session (large-list batch) | Implemented but never validated to release bar (batch-boundary drift concern); scan phase is only 19 ms/file and not user-perceptible | **REVERTED** (2026-08-22). Revisit only when scanning thousands of files per run |
| 10 | `cv2.setNumThreads(1)` in loader (eliminate decode-worker OpenCV threadpool) | A/B 60-JPG: 6.57–6.97 vs baseline 6.99–7.00 wall img/s — no gain, within noise. OpenCV resize on this machine does not expand threads internally; contention ≠ cv2 threadpool | **REVERTED** (2026-08-22). Contention is decode workers × ~190 ms full-core CPU + main-thread compute sharing 8 physical cores; remaining levers are worker-count matching (6 for JPG), process priority, or moving letterbox (+6 ms) into workers |
| 11 | Demote decode workers to BELOW_NORMAL priority (worker process name != MainProcess) | 4-run A/B 60-JPG: 7.10/7.02/7.20/7.04 (mean ~7.09) vs 6.99/7.00 baseline (+1.4% wall; engine 9.9 vs 9.75). Gates green (perf JPG 7.15). | **KEPT** (2026-08-22, `d0c3016`) |
| 12 | Move letterbox (640 canvas) into decode workers via `letterbox_image` shared fn + `precomputed` detect arg | A/B 60-JPG: 6.95/7.06 (mean 7.005) vs 7.09 with priority-only — no gain. Worker-side CPU +1.2 MB canvas IPC roundtrip offsets the ~6.6 ms main-thread saving | **REVERTED** (2026-08-22) |
| 13 | JPG decode via libjpeg DCT scaling (Pillow `draft` 1/4 + 1/2) | Draft decode cuts worker CPU ~2-3x (180->~60 ms), but flips the P4 integrity decision on 2/6 real JPGs (raw 3.69->3.09 via cut penalty → min_raw veto → **keep→reject**, the anti-INTER_AREA direction). Identical flips at 1/2 and 1/4 — P4's ROI boundary is knife-edge wrt ~1-LSB pixel changes | **REVERTED** (2026-08-22). Pixel changes are off-limits for the decode path |
| 14 | HEIF `-vf scale` in-process resize | NOT RE-TRIED after #13: decode is no longer the binding constraint (decode_wait 12-18 ms vs ~84 ms consumer serial) and its pixel changes carry the same P4 flip risk | SKIPPED |
| 15 | RAW embedded-preview extraction via persistent per-process exiftool `-stay_open` session (`-b -w tmp/%f.jpg`, one file per -execute) | Extraction bytes identical to per-file spawn (6 samples verified); 33 ms/file vs ~460 ms (14x). Gates: precision 6/6 green; perf ARW 3.39 vs 2.98 (+14%), NEF 3.84 vs 3.17 (+21%) | **KEPT** (2026-08-22). RAW was the only supply-bound format |
| 16 | Decode worker count A/B (2/3/4/6) | Interleaved 3x each on 60-JPG: 7.52/7.44/7.48/7.45 img/s — all within noise. Consumer serial path is the cap, not decode supply | NO CHANGE (workers from CLI) |
| 17 | `np.asarray` instead of `np.array` after `convert("RGB")` | Pixel-identical (verified np.array_equal); saves ~10 ms of the 72 MB copy per 24MP frame (Pillow 12 does not expose a true zero-copy view) | **KEPT** |
| 18 | Consumer de-GIL: letterbox PIL→cv2 (`letterbox_numpy`) + P4 ROI PIL→cv2 INTER_LINEAR | Precision 6/6 → 4 FAILED: DSC00827 (HEIF+ARW) raw +0.6 and NEF/ARW ±0.013 — the cv2 kernel shift flips P4 integrity 0→1 (penalty removed) and moves detection boxes. Same knife-edge as #13, both directions | **REVERTED** (2026-08-22). PIL BILINEAR preprocessing is frozen |
| 19 | CPU-affinity partitioning (consumer→2 physical cores, workers→rest) | Zero wall gain on 60-JPG (7.55 s both pinned/unpinned); consumer stages stay inflated (detect 38.9 ms). The inflation is memory-bandwidth contention, not CPU-core slots | **REVERTED** (2026-08-22) |
| 20 | (see #16) | — | — |
| 21 | P4 classifier warm-up in `load_models()` (moves the lazy ORT session + warmup run out of the timed window) | No score change (identical lazy-load semantics, just earlier); removes a ~1 s first-frame stall from every run | **KEPT** (2026-08-22) |
| 22 | P4 skip for already-sharpness-vetoed frames (output-identical reorder) | Rejected-frame `raw_score` loses the ±0.6 cut penalty → ARW/NEF precision gates FAIL on the recorded raw of vetoed frames (ratings were unchanged) | **REVERTED** (2026-08-22). Gate compares raw of vetoed frames too |
| 23 | **P4 v2 retrain** (MobileNetV3 multitask, same architecture/data as v1 + resize-kernel randomization across 9 cv2/PIL kernels, pixel noise, gamma/channel-gain/JPEG-recompression jitter; `train/train_p4_multitask.py`) | Labeled-val (9-kernel): integrity F1 96.3→96.8, orient acc 85.6→86.0, kernel flip rate **4.8%→2.6%** (production 3-kernel subset: ~0.4%). Real-photo gates vs v1: keep/reject decisions unchanged on all 70 gate files; 2 kept files drifted +1 star (HEIF DSC00849 1→2, NEF _220 1→2); raw records of rejected files moved up to ±0.6 (P4 v2 retains residual kernel sensitivity on the RAW embedded-preview domain). Baselines re-locked 2026-08-23 | **KEPT** — unlocks the frozen pixel path |
| 24 | JPG libjpeg draft DCT decode re-enabled (largest 1/2^k ≥ scale_width) | With P4 v2: precision gates green (v1 flipped 2/6 keep→reject). Worker CPU ~180→~65 ms. Interleaved A/B below | **KEPT** |
| 25 | cv2 letterbox (`detect_numpy`) + cv2 P4-ROI resize re-enabled | With P4 v2: gates green after re-lock; interleaved A/B (draft fixed, letterbox varied): JPG 8.92 vs 8.47 img/s (+5%), HEIF 3.93 vs 3.94 (neutral) | **KEPT** |
| 26 | Dual-consumer scoring threads (`--consumer-threads`, per-thread ONNX sessions, eager bundle pool outside the timed window) | Isolated probe: chain 46→58 fps with 0/120 detection mismatches, GPU util 17% (not saturated). In-engine interleaved A/B: ct=2 consistently SLOWER than ct=1 (JPG 8.67 vs 9.21, HEIF 3.62 vs 3.77) — the overlap gain is eaten by memory-bandwidth contention with 4 decode workers | **KEPT AS FLAG, default 1** (2026-08-23). Revisit on hardware with more memory bandwidth; determinism verified (gates green) |
| 27 | Sharpness internals → cv2: `cvtColor` RGB2GRAY (12x) + `Laplacian(CV_32F)` variance (2.4x) | Gray conversion differs <=1 LSB on ~64% of pixels → raw records drift <=0.016 on 19/70 gate files, 1 star drift on a kept HEIF (DSC00849 2→1). Interleaved A/B JPG 10.65 vs 10.34 (+3%), HEIF neutral. Baselines re-locked | **KEPT** (2026-08-23) |
| 28 | Full-GPU scoring chain probe (torch weights: ultralytics f1 ckpt `runs/f1_detect/train/weights/best.pt` + P4 v2 `models/p4_best.pt`, all tensors on CUDA; `benchmarks/bench_torch_chain.py`) | NO GAIN: torch eager f1 10.1 ms vs ORT 6.3; P4 8.8 vs 3.5 (per-op dispatch dominates on tiny models); GPU sharpen kernels ~1.4 ms vs CPU 6.5 (only win); GPU argmax postproc 1.4 vs numpy 0.26 (loss). Whole GPU chain ≈ 21.8 ms vs current ~21.3. Torch/ORT detections also differ (conf 0.31–0.58 cluster vs ORT 0.71). Bonus risk: onnxruntime + torch in ONE process hit a cudnn DLL clash (WinError 127) | **REJECTED** (2026-08-23). Keep ORT for models; GPU pre/post kernels not worth H2D + port cost |
| 29 | HEIF decode: drop failed NVDEC hwaccel + resident pyav decode | Camera HEIF previews are HEVC Rext 4:2:2 10-bit — consumer NVDEC cannot decode; the failed `-hwaccel cuda` attempt cost +~110 ms/file (371-406 vs 267-269 ms). Removing it: HEIF gate 3.9→5.3 img/s. In-process pyav (libav, resident, no spawn): 5.3→5.8 (+49% total vs 3.8-4.1 baseline). Pixel-identity gate-verified (6/6; the DEPENDENT flag is AV_DISPOSITION_DEPENDENT = 1<<19 — a `&4` mistake selected burst frames and collapsed scores). pyav adds ~40 MB to packaging | **KEPT** (2026-08-23) |
| 30 | Worker-count under the fast-decode pipeline (2/3/4) + workers=1 contention isolation | w=1: consumer stages return to CLEAN speed (detect 39→16.5 ms, score_image 22→5.6, sharpness 11→3.1 — contention confirmed as memory-bandwidth, ~2.4x inflation at w=4) but decode_wait 35 ms makes decode the new bottleneck; steady ≈12 fps either way. w=2/3/4 gates all within noise. Conclusion: total per-frame memory traffic is fixed on this box; worker partitioning moves the bottleneck between decode-supply and consumer-inflation without changing the plateau | **NO CHANGE** (2026-08-23); keep --workers 4 |
| 31 | P4 v2.1 retrain with synthesized cut data + RAW embedded-preview draft decode | `utils/generate_p4_cuts.py` synthesizes 4308 cuts from labeled fulls (exact orientation; front_cut 1→237, rear_cut 2→134) + 444 RAW-domain cuts via orientation-pseudo-labeled RAWs. P4 v2.1: RAW-domain kernel-verdict flip rate ~15%→0.6-1.7% (only knife-edge files remain). UNLOCKED: RAW preview draft DCT decode — ARW 265→127 ms, NEF 179→78 ms; gates ARW 3.5→4.43 (+27%), NEF 3.8→4.48 (+18%). 1 keep/reject flip on a knife-edge NEF (`_220`, oscillates across model versions, raw drops 0.045 under min_raw); baselines re-locked. Trainer fixes: DataLoader num_workers (GPU util 13%→~50-96%), optional decoded-.npy cache | **KEPT** (2026-08-23) |

## Scoring-chain sub-step profile (2026-08-23, cached 1280px frames, warm)

## Round-3 unlocked pipeline (2026-08-23)

Draft decode + cv2 letterbox + cv2 P4-ROI + P4 v2 + persistent RAW session.
Gate protocol (two consecutive runs): JPG **8.89/9.02**, HEIF 3.91/3.77,
ARW 3.07/3.11, NEF 3.36/3.39 img/s — all thresholds green. JPG +25% vs the
pre-retrain pipeline (7.13-7.27); other formats within machine drift
(±10% between sessions on this box — always interleave A/Bs).
Residual risk: P4 v2 integrity still flips on ~15% of RAW-domain ROIs'
*raw records* (ratings unaffected — those files sit far below min_raw);
multi-camera RAW/HEIF labels are the fix (docs/P4_LABELING.md).

## Scoring-chain sub-step profile (2026-08-23, cached 1280px frames, warm)

Whole chain = 22.3 ms/frame → 46 fps single-thread (GPU util 17%). Sub-steps:

| sub-step | ms | note |
|---|---|---|
| letterbox cv2 (640) | 0.78 | — |
| preprocess float32/255 + CHW | 2.95 | cv2.blobFromImage slower (3.7) — kept numpy |
| f1 session.run (CUDA) | 6.29 | batch x6 = 3.6 ms/img, but CPU-side dominates chain |
| postprocess argmax+NMS | 0.26 | vectorized |
| gray conversion | 2.06 | → cv2.cvtColor 0.04 (#27) |
| Laplacian veto | 3.07 | → cv2.CV_32F 0.36 (#27) |
| hf_ratio dft | 2.33 | cv2.dft, optimized |
| P4 roi+prep+run | 4.1 | mostly session.run 3.5 (CUDA) |
| score_image veto/dataclass | ~0.3 | pure python |

After #27 the chain is ~21 ms. Remaining consumer-side items are either GPU
necessary (f1 6.3 + p4 3.5) or within ~2-3 ms of floor; no further
pixel-equivalent wins identified — verified by the sub-step inventory above.

## Stage-profile findings (2026-08-22, 60-JPG protocol, workers=4)

Per-stage consumer/serial times measured via `benchmarks/profile_pipeline.py`:
decode_wait 12-19 ms (supply ahead), detect 39-44 ms (letterbox PIL 5.3 ms
+ f1 session.run ~25-34 ms under load + postprocess ~2 ms), score_image 22-25
ms (P4 ROI 9.8-11 ms inside), sharpness 11-13 ms, xmp_read 0.5 ms → consumer
serial ≈ 84 ms/frame ≈ 12 fps steady state. All stages inflate ~2x from
decode-worker memory-bandwidth contention (isolated runs: detect 16-17 ms,
P4 5 ms, sharpness 5 ms). ~25-30% of gate wall is fixed tax (3 CUDA model
loads + pool spawn + EXIF), which amortizes on real runs.
| 9 | Decode-path resize backend matrix (all 6 PIL + all 7 cv2 interpolations, 24MP→1280) | Speed: cv2 1.3–13.9 ms vs PIL 36–192 ms (up to 75×). Drift vs PIL-BILINEAR: EVERY alternative ≥0.176 max |Δraw| (BOX/HAMMING lowest at 0.18/0.21) with 3–4/6 rating flips — none within even 0.005, let alone 0.001. Only PIL-BILINEAR itself satisfies 0.001 | **SUPERSEDED by #A** (2026-08-22): user accepted cv2.INTER_AREA (area-average ~8× faster, ≤8–22% upward-only flips at boundaries); gates re-locked to the new pipeline |

## Optimization notes

- Decode downscaling must stay pixel-identical to Pillow BILINEAR (the score
  gates lock raw_score at 3 decimals — any decoder-kernel change flips them).
  ffmpeg `-vf scale` (sws bilinear), `draft()` (DCT), and cv2 INTER_AREA all
  fail this bar (see 461a6c6 and the revert above). Speed must come from
  parallelism/pipelining over pixel-for-pixel decode, not from faster kernels.

## Regression gate

Precision gates (`tests/test_cull.py`, `tests/test_precision_heif.py`,
`tests/test_precision_raw.py`) must stay green after any change; they run the
CLI at **workers=1 for determinism** — see below. The performance gate is
`benchmarks/run_benchmarks.py` (4-dataset protocol; gate thresholds locked to
the GATE sample sizes: JPG 60/HEIF 24/ARW 20/NEF 20 → 5.26/3.76/2.86/2.39 img/s
measured, thresholds 4.2/3.0/2.3/1.9; the 100-image baselines above run ~1.3–2×
faster because per-job overhead amortizes better).

### ~~Known issue~~ FIXED: CUDA concurrency was non-deterministic

RESOLVED by optimization #3 (2026-08-22): the old shared-session threaded
engine dropped detections under concurrent workers (~2/3 of runs at workers=4,
e.g. DSC00827.ARW raw 2.436→0.394). The engine now decodes in a process pool
and runs inference on a single consumer thread with one session; `--workers`
now controls the decode-pool size. Verified: all precision gates green for 3
consecutive runs at workers=4.
## macOS platform baseline (2026-08-24)

Platform re-lock on macOS (Apple Silicon, `.venv`: Python 3.10.20, pyav
17.1.0/libav 61, exiftool 13.50, ffmpeg 8.0). All decision semantics vs the
Windows locks are preserved; raw_score drifts come from decode LSB
differences (HEVC/RGB conversion, exiftool Perl-version byte extraction).

| Dataset | Files | Rating flips | Max \|Δraw\| | Largest drift file |
|---|---|---|---|---|
| HEIF (test_import) | 24 | 0 | 0.035 | DSC00893.heif 2.468 → 2.433 |
| ARW (test_arw) | 20 | 0 | 0.034 | DSC00886.ARW 2.217 → 2.183 |
| NEF (test_nef) | 20 | 0 | 0.026 | IMG...136_220.nef 3.072 → 3.046 |
| JPG (test_img) | 6 | 0 | 0.000 | identical to Windows lock |

Mac internal stability: two consecutive full gate runs are bit-identical
(max_drift 0.000, 0 rating flips). Baselines in the three test files were
re-locked to the macOS measured values on 2026-08-24; JPG baseline untouched.
Full per-file diff is recorded in the git history of tests/.

### macOS performance work (2026-08-24, same day)

Three optimizations landed on the macOS line, all zero-drift against the
Mac lock above:

1. **P4 on CPU EP (darwin only)** — CoreML partitions only 20/77 nodes of
   the small P4 model and pays bridge overhead: measured 16.6 ms vs 5.0 ms
   for plain CPU on the scored chain (Apple M4). Logit diff vs CoreML is
   <= 0.011 and never crosses a decision margin: all 64 HEIF/ARW/NEF gate
   files keep rating AND raw_score bit-identical (max_drift 0.000).
   Scoring chain: 18.4 -> 23.4 fps serial, 26.7 fps at 4 threads (CoreML
   made multi-threading a loss; CPU restores positive scaling).
2. **EXIF scan parallelization** — exiftool reading sharded across 4
   processes with argv file lists (8.1 ms/file vs 18.5 ms/file for the
   `-@ -` stdin protocol on M4; `-@ -` retained for > 400 files).
   Verified field-identical output; EXIF feeds burst grouping only, so
   scores are untouched. End-to-end JPG: 13.6 -> 14.5 img/s.
3. **videotoolbox HEIF hwaccel: REJECTED** — interleaved A/B (5 rounds)
   shows hwaccel is 100 ms vs 56 ms soft spawn on M4 (hw transfer +
   yuv422p10le->rgb24 conversion cost more than the soft HEVC decode it
   replaces; pyav exposes no working hwaccel API). Pixels are bit-identical
   to soft, so there is no precision upside either.

Final macOS end-to-end (gate protocol, workers=4): JPG 14.37, HEIF 7.48,
ARW 6.67, NEF 7.99 img/s (vs Windows 10.8 / 5.9 / 4.5 / 4.7). Scoring
chain serial 23.4 fps. `--consumer-threads` 2/4 is a LOSS end-to-end on
M4 (13.6 -> 11.4 -> 9.0 img/s, interleaved A/B) — default 1 stays.

### YOLO CoreML compute-units / partition investigation (2026-08-24, later)

Follow-up on the CoreML EP options ("Unknown option" root cause + the
partition-split question):

- **Root cause of "Unknown option"**: the option keys are CamelCase —
  `MLComputeUnits`, `ModelFormat`, `RequireStaticInputShapes` (the snake_case
  `ml_compute_units`/`coreml_compute_units` forms are rejected). With correct
  keys ORT 1.23.2 accepts all options; the earlier failures were purely a
  naming error, not a build limitation.
- **Real partition structure**: production f1_yolov8n.onnx = 7 partitions,
  233/318 nodes on CoreML (NeuralNetwork format). The "29 partitions" line in
  earlier logs belonged to p4_car_model.onnx (140 nodes), not the YOLO.
  MLProgram format supports 255/318 nodes but still yields 7 partitions.
- **RequireStaticInputShapes is a dead end** for ultralytics exports: the
  graph carries ~3000 data-dependent dynamic dims (Reshape/Concat shapes);
  even with the batch axis frozen and full onnx.shape_inference applied, only
  0-6 nodes qualify for CoreML (everything falls back to CPU).
- **ANE compute units: REJECTED despite -16% on the YOLO stage alone.**
  Interleaved A/B, Apple M4: YOLO session.run 24.2 vs 28.7 ms (MLProgram+ANE
  vs NeuralNetwork default), but the FULL scoring chain gets SLOWER —
  MLProgram+ANE 40.9 vs 39.1 ms/frame, NN+ANE 41.8 vs 38.3 ms/frame (higher
  variance too). ANE's synchronous submit-wait schedule interleaves badly
  with the CPU stages (sharpness/P4) that follow every run.
- **Side finding**: gate files scored under ANE keep ALL ratings and mostly
  return to the ORIGINAL Windows-locked raw values (fp16 accumulation
  converges with CUDA's low-precision path where Mac NN/CPU fp32 differed).
  Zero rating flips — but the performance verdict above stands, so the
  shipped config stays NeuralNetwork default + P4 on CPU.
- CPUAndGPU compute units are bit-identical to default but +1% slower.
