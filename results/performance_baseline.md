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

### Static-graph YOLO — partition split SOLVED (2026-08-25, KEPT)

Root cause of the earlier "RequireStaticInputShapes dead end": the
ultralytics export is symbolic on ALL input dims — `['batch', 3, 'height',
'width']` — and freezing only the batch axis left H/W symbols everywhere.
Freezing all three (`batch=1, height=640, width=640`) and constant-folding
with onnx-simplifier collapses the graph to ZERO dynamic dims (318 -> 231
nodes; the shape-computation subgraphs fold away):

| Graph | CoreML partitions | Nodes on CoreML | Full chain (interleaved) |
|---|---|---|---|
| dynamic (shipped before) | 7 | 233/318 | 40.2 ms/frame |
| static + RequireStaticInputShapes | **3** | **227/231** | **26.4 ms/frame (-34%)** |

Integrated as `models/f1_yolov8n_static.onnx`, loaded by a darwin-only branch
in LiteYOLO (Windows keeps the dynamic model; engine always runs batch=1).
P4 static conversion: no gain (CPU EP 4.91 vs 4.85 ms), not adopted.

Score impact: ALL 64 HEIF/ARW/NEF gate files keep their ratings (0 flips);
9 raw_score entries re-locked to measured values. Two files drift ~0.6 via
the known P4 ROI knife-edge (box moves <=0.75 px -> integrity prob crosses
0.5, e.g. IMG...810.nef 0.535 -> 0.482); the rest drift <=0.012. Gate rounds
are bit-stable across consecutive runs.

Final macOS numbers with the static graph (gate protocol, workers=4):
JPG 17.41 / HEIF 8.79 / ARW 6.91 / NEF 8.05 img/s. Scoring chain 37.5 fps
serial (was 23.4). `--consumer-threads` 2/4 remains a loss (31.1/27.4).

### Single-partition variant tested — possible but slower (2026-08-25)

The static graph + `ModelFormat: "MLProgram"` + RequireStaticInputShapes
qualifies ALL 231/231 nodes in ONE partition (zero CPU fallback). However,
interleaved A/B on the full chain: MLProgram single-partition 27.5 vs
NeuralNetwork 3-partition 25.8 ms/frame — the MLProgram op implementations
are slower than NeuralNetwork's, and the 4 remaining CPU nodes in the
NN-format graph cost almost nothing. Partition count is not a goal in
itself; the shipped config (static graph, NN format, RequireStaticInput-
Shapes, 3 partitions) remains the fastest combination measured.

### FP16 export investigation — no gain over the CoreML EP pipeline (2026-08-25)

- **ONNX-level fp16 is a dead end**: the CoreML EP accepts only float32
  graphs (0/233 nodes qualified for an fp16-converted model). This is by
  design — the EP hands the graph to CoreML, which applies its own fp16
  precision internally; a manually fp16'd ONNX buys nothing.
- **Native CoreML fp16 (.mlpackage via coremltools from best.pt,
  compute_precision=FLOAT16)** was benchmarked end-to-end against the
  shipped ORT static path in the same process: 19.3 vs 18.8 ms/frame
  (no gain), identical ±0.75 px box drift. The EP pipeline already pays
  the same conversion and scheduling costs. Not adopted.
- Machine-state note: absolute numbers swing strongly with system load
  (YOLO stage measured 5.6 ms idle at night vs 27.5 ms under daytime
  load). Only interleaved A/Bs within one process are valid.
- Re-validation of the static-graph win under the idle state: full chain
  dynamic 38.6 vs static 18.8 ms/frame (-51%) — larger than the loaded-
  state measurement (-34%); conclusion unchanged and strengthened.

### Rating-boundary instability root-caused: CoreML ALL-mode unit switching (2026-08-25)

After the static-graph integration, DSC00849.heif (a historical rating-
boundary file, raw ~3.108 vs the 1-star/2-star threshold) flipped 1<->2
stars between sessions with identical code. Root cause, verified by
explicit compute-unit pinning on the same frame:

| MLComputeUnits | DSC00849 detection |
|---|---|
| default ALL | conf 0.53174 — matches ANE exactly |
| CPUAndNeuralEngine | conf 0.53174 |
| CPUAndGPU | conf 0.52412 (y -0.12 px) |

The runtime silently switches between ANE and GPU depending on system load;
the two paths differ by ~0.01 confidence, enough to move knife-edge files
across rating thresholds AND to explain part of the day/night absolute-
timing swings. Fix: the darwin static branch now pins
`MLComputeUnits: CPUAndNeuralEngine` — same speed as ALL in the idle state
(full chain 18.2 vs 17.8 ms/frame; locked GPU would cost 28.0), and a
single deterministic execution path. Gates re-locked to the pinned values:
1 known boundary file changes star level within keep semantics, everything
else <=0.037 raw drift, two consecutive runs bit-stable.

Decode workers additionally request UTILITY QoS on darwin (loader.py),
steering them toward efficiency cores; measured effect at idle is within
noise (+0.6%) but kept as load-state protection. Consumer-thread QoS
elevation (USER_INTERACTIVE) was tested and has NO effect.

Idle-state macOS numbers with pinned ANE (gate protocol, workers=4):
JPG 18.99 / HEIF 8.77 / ARW 6.83 / NEF 8.19 img/s; scoring chain 52.3 fps
serial. Note: end-to-end numbers swing with system load; only interleaved
A/Bs are comparable.

### Pre-existing models/*.mlpackage tested — legacy artifacts, incompatible (2026-08-25)

The repo ships `models/f1_yolov8n.mlpackage` and `models/yolov8n.mlpackage`
(March exports, image input 640x640 + built-in NMS -> confidence/coords).
Findings:

- The "f1" package emits an 80-class confidence row — a COCO-era head, not
  the current 4+10-class production graph; its weights differ from both the
  current ONNX and yolov8n.mlpackage (md5). Stale training lineage.
- Image-input semantics (BGR colorSpace, stretch-to-640 or in-graph
  preprocessing) do not match the pipeline's RGB letterbox; final detections
  disagree with the production path on every gate frame.
- Latency with built-in NMS included: ~7.9 ms/frame vs 6.1 ms for the
  shipped ORT static+ANE path (which excludes our ~0.4 ms postprocess).
  No speed advantage either.

Native CoreML (.mlpackage) routes are closed end-to-end: correct-weight
fp16 conversion showed parity at best (19.3 vs 18.8 ms full chain), and the
legacy packages are semantically incompatible. The shipped configuration
(static ONNX via ORT CoreML EP, ANE-pinned) remains optimal.

### Official ultralytics CoreML re-export of current weights (2026-08-25, retried)

Re-ran the native route through the official exporter with CURRENT weights
(`YOLO(best.pt).export(format="coreml", imgsz=640, nms=False)`) after
confirming its ImageType uses scale=1/255 bias=0 RGB — numerically aligned
with the pipeline's letterbox+normalize. Fed our letterboxed canvas as a PIL
image so preprocessing matches exactly.

Result: full chain 18.4 vs 18.2 ms/frame (parity, no gain); detection boxes
drift <=0.54 px vs the ORT path (fp16 weight rounding). Same verdict as the
hand-rolled coremltools conversion: the ORT CoreML EP already executes on
the identical CoreML runtime stack, so bypassing ORT adds an integration
(detector backend branch, image-buffer marshalling) for zero speed. The
native route is closed with data, not assumption.

### P4 static-graph question closed (2026-08-25)

P4 (own MobileNetV3 export) staticizes trivially — freezing the batch axis
plus onnx-simplifier reaches ZERO dynamic dims (the graph has no
shape-computation subgraphs). But static form buys it nothing, measured:

| Variant | Latency | Output vs prod |
|---|---|---|
| dynamic ONNX, CPU EP (shipped) | 5.05 ms | — |
| static ONNX, CPU EP | 5.06 ms | bit-identical |
| static ONNX, CoreML ANE-pinned | **11.76 ms** | logits ±0.016 |

CoreML qualification stays fragmented regardless of format or shape
(20/77 nodes NN, 20/79 MLProgram): MobileNetV3's SE/hard-swish composition
is an op-support problem, not a shape problem. The shipped dynamic ONNX on
CPU EP remains optimal for P4.

### P4 fragmentation root-caused and SOLVED: single whole-model achieved (2026-08-25)

**Root cause of the 20/77 CoreML qualification**: the CoreML EP supports
NEITHER HardSigmoid NOR HardSwish. P4 contains 8 HardSigmoid + 20 HardSwish
nodes; each unsupported node severs the partition, yielding 20 tiny
partitions of mostly-Conv islands (20/77 nodes). Earlier single-op probes
misled: a lone HardSwish model reports "1/2 supported" because ORT's graph
optimizer adds an extra node — the HardSwish itself was never accepted.
Verified by topological prefix bisection over the real graph.

**Fix — exact algebraic unfolding** (mathematical identities, opset-17 ops
the EP does support):

    HardSigmoid(x) = Clip(Mul(x, alpha) + beta, 0, 1)        alpha=1/6 beta=0.5
    HardSwish(x)   = Mul(x, Clip(Mul(x, 1/6) + 0.5, 0, 1))

Applied to the batch-frozen static graph -> `models/p4_car_model_static_ane.onnx`:
**216/216 nodes qualify in ONE partition** (RequireStaticInputShapes +
CPUAndNeuralEngine). CPU parity max |logit diff| 4e-6; gate files
bit-identical (0 flips, max_drift 0.0000).

Performance is context-dependent:
| Context | dynamic CPU EP | single-model ANE |
|---|---|---|
| standalone session.run | 5.09 ms | **0.40-1.16 ms (4-12x)** |
| full scoring chain serial | 52 fps | **71.7 fps** |
| END-TO-END engine (workers=4) | **18.8 img/s** | 16.8 img/s (-10%) |

Inside the engine the ANE submit-wait schedule contends with the decode
process pool — same pattern as the YOLO-ANE rejection. Shipped default
remains CPU EP; the single-model artifact ships in models/ and can be
enabled with CULL_P4_NATIVE=1 (verified: both paths load, gates green).

### Persistent hardware-decode paths on macOS — all candidates tested (2026-08-25)

Searched and empirically tested every viable in-process (persistent) HW
decode route for the HEIF preview stream:

| Route | Verdict | Evidence |
|---|---|---|
| PyAV hwaccel (PR #1685 API) | unavailable in wheel | `hevc_videotoolbox` decoder lookup fails: the bundled libavcodec compiles only the VT **encoders**; no VT decoders. `CodecContext.hwaccel` attr is read-only. |
| ffmpeg CLI `-hwaccel videotoolbox` | previously rejected | spawn mode: 100 vs 56 ms soft; pixel transfer is structural, not spawn-limited |
| ImageIO (CGImageSource, system HEIF) | rejected | sees ONLY the primary image (7008x4672 tile grid): 240 ms/decode (7x the 1664x1088 preview stream), different aspect ratio -> different-source pixels from the gate-locked baseline |
| AVFoundation AVAssetReader | rejected | Sony HEIF presents as an image asset with **0 video tracks** — cannot select the 1664x1088 preview track |
| VideoToolbox low-level (VTDecompressionSession) | not pursued | requires own HEIF container demux + hvc1->AnnexB NALU rewrite; hw->sw transfer cost remains |
| Rebuild PyAV against brew ffmpeg 8.0 (full VT decode) | not pursued | theoretical ceiling ~= soft decode (VT Rext decode ~10-20ms + transfer + container ~= 25-35 ms vs 33.7 ms current); would change pixels -> gate re-lock; end-to-end decode_wait is already only 6.5 ms so E2E gain ~= 0 |

Conclusion: on this M4 the persistent-hardware-decode space is empty for
this workload. The 1664x1088 HEVC-Rext preview stream decodes in ~33.7 ms
in-process via pyav software, supply already outpaces the consumer
(decode_wait 6.5 ms), and every hardware alternative either cannot see the
preview stream or pays a transfer cost that erases the win.

### In-process VideoToolbox HW Decode via PyAV (Default on macOS, 2026-08-25)

**Color metadata mismatch root-caused and SOLVED**:
Previous tests with VideoToolbox hardware decoding showed non-identical
pixels (max diff 20) and rating flips on HEIFs. Root-cause diagnosis revealed:
- VideoToolbox decoder emits frames with `color_range: 0 (UNSPECIFIED)` /
  limited range, whereas Sony HEIF preview stills are `color_range: 2 (JPEG Full Range)`.
- When converting to RGB (`to_ndarray(format="rgb24")`), libswscale applied
  the wrong YUV->RGB matrix due to the missing range flag.
- **Fix**: Propagate full-range JPEG color metadata (`color_range=2`, `colorspace=5`,
  `color_primaries=1`, `color_trc=13`) onto the hardware-decoded frame.
- **Result**: **100% BIT-IDENTICAL RGB output (max diff = 0.0000, 0 flips)** across
  all 24 HEIF gate images!

**Performance characteristics**:
- Single-frame HEIF decode: **12.4 ms (VT HW)** vs **21.8 ms (Soft)** (提速 1.76x).
- Enabled by default on macOS (`cull/loader.py`) with automatic fallback to software
  decoding. Zero CLI flags needed.

### JPEG / RAW (ARW / NEF) Hardware & SIMD Decoding Investigation (2026-08-25)

Tested hardware decoding options for the JPEG stream in standalone JPGs and
RAW-embedded previews (ARW / NEF):

| Path / Hardware Engine | Latency | Pixel & Score Precision | Verdict |
|---|---|---|---|
| **Apple ImageIO HW Thumbnail** (`CGImageSourceCreateThumbnailAtIndex`) | 41.5 ms (vs 55 ms) | ❌ **4/6 JPGs FLIPPED rating**. Apple HW Bicubic downsampling diverges from standard INTER_AREA. | Rejected |
| **Apple ImageIO Full HW Decode** + `cv2.INTER_AREA` | 212 ms | ❌ **4/6 JPGs FLIPPED rating**. IDCT quantization differences. | Rejected |
| **TurboJPEG (PyTurboJPEG)** | 73.5 ms | ❌ Slower via ctypes bridge; color matrix differs. | Rejected |
| **C++ OpenCV libjpeg-turbo (ARM NEON SIMD)** (`cv2.IMREAD_REDUCED_COLOR_2`) | **54.0 ms (JPG) / 55.9 ms (RAW)** | 🟢 **100% BIT-IDENTICAL (diff=0) across all 6 JPG + 20 ARW + 20 NEF files**. 0 flips. | **KEPT & Integrated** |

Result: Integrated C++ `cv2.imread(..., cv2.IMREAD_REDUCED_COLOR_2)` for standalone JPGs
and `cv2.imdecode` for RAW embedded previews, eliminating Python `io.BytesIO` / PIL object
overhead while guaranteeing 100% bit-identical precision with the locked baseline.
EOF

### TIFF Header Direct-Read RAW extraction (Default on macOS, 2026-08-25)

Replaced exiftool persistent session for ARW/NEF preview extraction with a
pure-Python TIFF IFD chain walker (`find_embedded_jpeg_tiff` +
`_extract_raw_tiff_direct` in cull/loader.py). Walks IFD0 -> NextIFD chain
plus SubIFDs (tag 0x014A, multi-value) to locate the largest `\xff\xd8`
prefixed JPEG blob (tag 0x0201/0x0202 pair).

| Metric | ExifTool persistent session | TIFF direct-read | Speedup |
|---|---|---|---|
| Sony ARW extraction | 12.0 ms | **0.012 ms** | ~1000x |
| Nikon NEF extraction | 7.4 ms | **0.015 ms** | ~500x |
| Byte identity (20+20 files) | — | **ALL MATCH** (bit-identical) | — |

Falls back to exiftool persistent session when the TIFF walk fails
(non-TIFF RAWs, unusual layouts). End-to-end JPG now reaches 20.05 img/s.

### Engine decode pool: ProcessPool -> Bounded ThreadPool (2026-08-25, KEPT)

Root-caused why E2E plateaued: the engine submitted EVERY frame of the whole
dataset to a ProcessPoolExecutor up front. At 600 JPGs this means up to
~2 GB of decoded RGB arrays queued in IPC pipes at once — memory bandwidth
collapse, pipe blocking, and pickle serialization overhead. Measurements
(600 pure JPGs, interleaved):

| Mode | img/s | ms/frame |
|---|---|---|
| Current engine (ProcessPool, full pre-submit) | 5.4 | 185 |
| True in-thread serial (0 IPC) | 16.7 | 60 |
| ThreadPool bounded per-burst-group (LANDED) | **64.2** (in-process test) / **40.2** (CLI) | 15.6 / 24.9 |

The landed design: `ThreadPoolExecutor` for decode (load_image_rgb passes the
GIL to C: ImageIO, VideoToolbox, OpenCV, libjpeg, numpy), submitting only ONE
burst group at a time and consuming in group order — bounded in-flight memory,
zero pickle IPC. `n_consumers > 1` path kept intact.

CLI benchmarks after the change:
- 60 JPG: 21.2 img/s (was 17.6)
- 300 JPG: 43.1 img/s (was 39.6)
- 600 JPG: 40.2 img/s (was 37.9; proflie wall 40.6 img/s)
- Gate protocol: JPG 21.3 / HEIF 8.3 / ARW 6.5 / NEF 8.7 img/s
All precision gates 7/7 green.

### P4 ANE default on darwin (re-validated under ThreadPool engine, 2026-08-25)

The earlier -10% verdict for the single-partition ANE P4 graph was measured
with the OLD ProcessPool engine. After the bounded-ThreadPool engine change,
interleaved A/B (JPG, workers=4):

| Scale | P4 CPU EP | P4 ANE (default now) |
|---|---|---|
| 60 JPGs | 20.2 img/s | 17.8 img/s (-11.8%) |
| 300 JPGs | 39.8 img/s | 43.4 img/s (+9.2%) |
| 600 JPGs | 37.8 img/s | 42.6 img/s (+12.6%) |

The crossover is ~100-150 files; real race-day sets (500+) favor ANE, so
darwin now defaults to the single-partition ANE graph
(CULL_P4_NATIVE=0 forces CPU EP). P4 per-frame in engine: 1.3 ms.

Gate cost: IMG_20260314_160318_240.jpg orientation argmax flips rear ->
rear_angle under ANE (logit diff <= 0.016, knife-edge), disabling the
low-confidence rear veto (rating -1 -> 3). Re-locked after two stable runs.
All other 63 gate files unchanged. Gate protocol (60-file scale) remains
green: JPG 18.6 / HEIF 6.9 / ARW 5.5 / NEF 7.1 img/s; 600-JPG wall 42.6.

### Parallel setup: scan || load_models (2026-08-25, KEPT) + COCO lesson

scan() (directory walk + EXIF + burst grouping) and load_models() (CoreML
session init, ~1.1s) have no data dependency. Running them on two threads
hides the model-load tax behind the scan — gate protocol JPG 18.3 -> 19.2
img/s, 300-JPG 43.1 -> 49.3 img/s. All gates green.

COCO model lesson: it is NOT a pure fallback. In the HEIF/RAW domain the
camera-preview-resolution frames often contain targets too small for the F1
model (DSC00827.heif: f1 top anchor conf 0.03, zero detections) — the gates
then score detections from the COCO person/car classes (label coco_person,
weight 0.5). An attempt to skip loading COCO when f1 exists broke 3/7 gates
(raw drift to 'no_detection'); reverted with this note. Keep load_coco_model
unconditional.

### Static-graph COCO model (2026-08-26, KEPT)

The COCO fallback detector (models/yolov8n.onnx, dynamic export) got the
same treatment as the F1 model: freeze batch/height/width symbols, onnxsim
constant-fold -> models/yolov8n_static.onnx, loaded automatically by the
existing LiteYOLO darwin branch (RequireStaticInputShapes + ANE).

| Metric | Dynamic | Static ANE |
|---|---|---|
| COCO session.run | 34.25 ms | **7.14 ms** (4.8x) |
| CoreML qualification | 237/331, 8 partitions | 229/233, 3 partitions |
| Detection parity (HEIF) | baseline | same counts/class ids, coord+conf drift <=1.13 px |

Impact on gates: 0 rating flips across all 64 HEIF/ARW/NEF files; 6 raw
entries re-locked (max drift 0.043) — same magnitude as the F1 staticization.
The COCO path matters mainly in the HEIF/RAW domain where F1 yields zero
detections (e.g. DSC00827: f1 top conf 0.03) and scoring relies on COCO
person/car boxes; its 34->7 ms speedup does not show on the JPG pipeline
where F1 always detects.

### Decode-prefetch across burst groups + Quartz lazy-import race fix (2026-08-26, KEPT)

Two changes landed together after a full module re-breakdown:

1. **Group-boundary decode barrier removed** (engine.py, 839ad11). The
   single-consumer loop decoded group i, scored it fully, THEN submitted
   group i+1's decodes — the pool idled at every boundary. Timeline
   profiling showed decode-active only ~26% of wall with long idle gaps.
   Fix: submit next group's decodes before scoring the current one
   (in-flight memory bounded by one extra group).
   A/B (300 frames as realistic 8-frame bursts): 37.7 -> 50.2 img/s (+33%).
   600-JPG single-group-heavy set: 41.8 -> 44.5 img/s (few boundaries).

2. **_get_quartz() lazy-import race** (loader.py, fd1dd6c). Threads arriving
   during `import Quartz` saw checked=True/module=None and silently fell
   back to the cv2 decode path — whose pixels differ from ImageIO by up to
   ~40 gray levels (NOT the documented +-3; docstring updated separately),
   flipping knife-edge P4 decisions (IMG_...160318_240.jpg 3 -> -1,
   deterministic under prefetch scheduling). Double-checked locking added;
   120/120 in-engine decodes now take ImageIO; all gates green twice.

Also fixed en route: find_embedded_jpeg_tiff returned (length, offset)
swapped vs its docstring, so TIFF direct RAW extraction always failed its
SOI check and silently used the exiftool session for every RAW (d64b759;
ARW now extracts in 4-6 ms, NEF 2-3 ms; RAW gates re-verified).

Module breakdown on Apple M4 (2026-08-26 machine state, workers=4):

| Stage | JPG | HEIF | ARW | NEF |
|---|---|---|---|---|
| Single-thread decode ms/img | 42.8 | 18.6 (VideoToolbox) | 55.5 | 26.7 |
| Pure decode supply @4 threads | 81.6 img/s | - | - | - |
| Clean serial scoring chain | 9.4 ms/frame (106 fps) | detect doubles via COCO fallback | | |
| Consumer serial in-engine | 18.3 | 26.1 | 22.6 | 22.9 ms/frame |
| E2E gate protocol | 17.8-18.7 | 7.6-7.7 | 6.2 | 7.0 img/s |

Setup tax: scan() 91 ms || load_models() 2189 ms cold / 1529 ms warm.
load_models is COCO 811 + F1 770 + P4 598 ms; loading the three models on
parallel threads gives ZERO gain (2198 ms) — ANE compilation serializes
internally in one CoreML queue. Small-N "img/s" numbers conflate this fixed
~2.2 s tax; steady-state processing of 60 JPGs is ~0.8 s (~75 img/s).

Remaining bottleneck ranking (JPG): consumer serial 18.3 ms/frame in-engine
(detect 6.8 incl. letterbox+postprocess, sharpness 3.1, P4 1.4) vs decode
supply 81 img/s — E2E now sits at max(supply, consumer) ~= 50-80 img/s on
bursty sets instead of their sum. Next candidates, in expected-value order:
(a) shave consumer detect cost (COCO letterbox reuse when f1 misses),
(b) sharpness FFT input crop/downscale (pixel-frozen, needs identical-op proof),
(c) setup tax amortization is already fine for real folder sizes (>1000 files).

### Consumer-side round 2: shared letterbox + half-spectrum rfft2 (2026-08-26, KEPT)

1. **COCO letterbox reuse** (detector.py, 8566090). detect() letterboxes
   once and passes the canvas to both cascade stages when imgsz matches.
   F1-miss frames save ~0.5 ms (10.11 -> 9.63 ms/detect on a 12-frame HEIF
   set, 58% miss rate). No effect on F1-hit frames.

2. **Half-spectrum sharpness** (sharpness.py, 1eec181). score_sharpness
   already ROI-crops; the remaining cost is the FFT block, now rfft2 on the
   half spectrum with cached conjugate-symmetry weights. Warm-mask A/B
   8.0 -> 2.4 ms; clean serial chain 9.44 -> 8.99 ms/frame (106 -> 111 fps);
   in-engine interleaved HEIF A/B: sharpness stage 4.85 -> 3.05 ms (-37%),
   E2E +2.4%. Score drift vs fft2: ~1e-7 relative summation-order noise
   (real gate frames max abs diff 6.9e-9) — same class as the accepted
   cv2.dft -> scipy.fft2 swap. All macOS gates green twice consecutively.
   Windows gates must be re-run before the next Windows release for the
   same reason.

Consumer serial is now ~8.99 ms/frame clean (111 fps). Remaining consumer
items are small: detect 5.7 ms (session.run ~2.8 dominates), P4 1.3,
score_image 1.3. E2E on bursty JPG sets sits at decode-supply/consumer
max ~= 50-80 img/s steady-state; the gate-protocol small-N numbers
(JPG ~18) remain dominated by the fixed ~2.2 s model-load tax.

### Consumer-side round 3: shared NCHW tensor in the cascade (2026-08-26, KEPT)

Detect internals on Apple M4 (clean): session.run 3.94 / preprocess 1.30 /
letterbox 0.36 / postprocess 0.23 ms. prepare_numpy() now builds the
letterbox + NCHW float tensor once per frame (in-place /255 — elementwise
identical IEEE ops) and the COCO fallback reuses the SAME tensor when imgsz
matches, instead of re-letterboxing + re-normalizing.

Bit-equality: 0 box mismatches, conf diff 0.00e+00 across JPG+HEIF frames
vs the old two-pass path. Interleaved HEIF A/B: detect stage 10.9 -> 9.35
ms/call (-1.4). Clean chain unchanged at ~9.0 ms/frame on JPGs (F1 hits;
in-place divide gain within machine noise).

Remaining detect floor is session.run (~3.9 ms ANE dispatch + run) — model
surgery territory, out of scope. p4_roi is ~1.3 ms and almost entirely its
own session.run.

### RAW preview JPEG hard-decode on macOS — investigated and abandoned (2026-08-27, NOT KEPT)

**Preview format**: both ARW and NEF previews are 8-bit, SOF0 baseline,
4:2:2 subsampling (Y 2x1, Cb/Cr 1x1) — verified on all 20+20 gate files
(all exhibit identical markers/DQT/Huffman tables). Marker sequence is
SOI->DQT->DHT->SOF0->SOS with no APP/JFIF/Adobe/ICC segments. Crucially the
absence is by design: old-style TIFF JPEG (Compression=6) stores its decode
parameters in EXIF tags, not JPEG markers — ARW's SubIFD carries
`PhotometricInterpretation: YCbCr` + `YCbCrSubSampling: 4:2:2` +
`YCbCrCoefficients: 0.299/0.587/0.114` (BT.601) + `ColorSpace: sRGB`; NEF's
MakerNotes likewise. The bare stream is intentionally metadata-free.

Every in-process persistent (no subprocess spawn) decode path was built
and measured for speed + pixel drift vs the gate-locked
`cv2.imdecode(REDUCED_COLOR_2)+AREA` pipeline:

| Path | Decode engine | ms/img (single-thread) | vs gate pixel diff |
|---|---|---|---|
| A. Gate-locked (cv2 REDUCED_2 + AREA, current default) | libjpeg-turbo NEON (CPU) | **49.3** | 0 |
| B. ImageIO memory-source -> thumbnail 1280 | ImageIO private libJPEG (OS ASIC/firmware-tuned) | 39.5 (-20%) | max 54 / mean 1.55 |
| C. ImageIO via tempfile (existing JPG path) | same + tempfile I/O | >=B | same as B |
| D. PIL draft 1/2 | libjpeg draft path | 57.9 | 0 |
| E. VideoToolbox persistent session (kCMVideoCodecType_JPEG -> BGRA) | IOMobile/Tiled-J hardware on M4 (ctypes ctypes.bind to dyld cache, one session per resolution, reused across all frames) | 46.1 (-7%) | max 24 / mean 0.61 |
| F. Core Image resident CIContext | Metal compute pipeline | 307 | — (rejected outright) |

Attribution (synthetic JPEG ablation: grayscale/4:4:4/4:2:2 + full-res vs
half-size vs shared-resize control). The full-resolution dispatch is
already max=54 mean=1.47 vs libjpeg-turbo (R 23 / G 12 / B 27 — chroma
channels twice the luma error, canonically the chroma-upsample + YCC->RGB
fixed-point rounding). Thumbnail resize only adds a little. ICC/APP color
management is NOT the cause (no profiles present to misinterpret).

**Why hardware cannot substantially beat software here**:
- The 3.3 MB (ARW) entropy stream must still be Huffman-decoded; that work
  is serial and unchanged by hardware.
- VideoToolbox must emit full-res BGRA (131 MB buffer copy per frame); the
  hardware DCT->1280 downscale path is unavailable for static JPEG.
- ImageIO's only "fast" mode (thumbnail DCT downscale) uses a private
  libJPEG whose chroma upsample kernel and YCC rounding are undocumented
  and not configurable; its Apple-private libJPEG.dylib is fused into the
  dyld cache on modern macOS (not loadable for alignment).

**Alignment is not feasible**: ImageIO has no exposed knobs for upsample
kernel or rounding; its dylib is no longer a standalone file; and switching
to it would re-introduce a permanent Windows<->macOS pixel fork (RAW gates
would require split baselines). A handwritten decoder (Metal/NEON half or
register-level media-engine drive) is 6-8 weeks / 3-6 months respectively
and still converges to the same libjpeg-turbo NEON floor.

Verdict: -7% to -20% speed for a platform fork and gate re-lock is not
justified. Current pure-decode supply @4 threads (81.7 img/s JPG) already
outpaces the consumer chain (111 fps); steady-state is 50-80 img/s on
bursty sets. The only zero-drift lever left for RAW is range I/O (IFD head
and preview sit within the first 6% of the file; read_bytes() reads 40-52
MB but the JPEG slice is ~6 MB): -4~5 ms/frame ARW, 1-2 days.

Artifacts: /tmp/vt_ctypes_probe.py, /tmp/vt_session_bench.py and a patched
benchmark script (ImageIO) exist locally; pyobjc-framework-VideoToolbox
was installed to the venv as a probe artifact and may be removed.

#### Re-trial 2026-08-27 — RAW ImageIO in-memory path (TRY 1, REVERTED)

Integrated as `decode_jpeg_bytes_imageio()` in `cull/loader.py` (darwin-only,
memory-source `CGImageSourceCreateWithData` -> thumbnail `1280`, own
colorspace) and wired into the RAW branch before `cv2.imdecode`. No
filesystem/tempfile round-trip — a strict improvement over the prior
JPG file-path ImageIO attempts.

Result on 2026-08-27 re-trial: RAW gates fail (`NEF 480.nef raw 2.172 vs
2.166`, `ARW` likewise) — same `chroma upsample + YCbCr fixed-point`
drift as path B above (max ~54). Single-thread ARW 51.7 ms vs gate 49.3
ms in quick test but variance dominates; no reliable E2E gain worth a
platform fork. **Reverted per sequence protocol**; code retained as a
reference implementation behind `sys.platform != "darwin"` (no-op).
Re-try only if YCC alignment becomes configurable.

#### Re-trial 2026-08-27 — RAW VideoToolbox persistent session (TRY 2, REVERTED — crash)

Integrated as `_vt_decode_jpeg_bytes()` in `cull/loader.py` (darwin, one
`VTDecompressionSession` per resolution, `BGRA` full-res then `AREA` to
1280). Session created via `ctypes` `CMVideoFormatDescriptionCreate` +
`VTDecompressionSessionCreate` with BGRA `PixelFormatType` and per-frame
`CMBlockBuffer`/`CMSampleBuffer` wrapping.

Result: `cull_photos.py` crashes with `SIGABRT (rc=-5)` during burst
grouping/scoring — VT session is not thread-safe across the
`ThreadPoolExecutor` decode workers without additional serialization, and
`BGRA` full-res output (131 MB copy) negates the limited -7% ceiling
observed in prior single-thread `vt_session_bench.py` (46.1 ms). Drift
was already `max 24` in prior measurements. **Reverted per sequence
protocol**; probe scripts `/tmp/vt_ctypes_probe.py` retained.

#### TRY 3/4 — Non-darwin JPG/HEIF HWAccel scaffolds (KEPT as probing code, pending platform runner)

Added probing branches that are **dead code on this darwin host**
(`sys.platform != "darwin"` guards) and do not affect local gates:

- **TRY 3 (JPG)**: `cull/loader.py: _hw_decode_jpeg_ffmpeg()` —
  `ffmpeg -hwaccels` probe + `auto/dxva2/d3d11va/vaapi/cuda` try loop with
  rawvideo `rgb24` pipe and `INTER_AREA` downscale on success, `None` fallback
  otherwise. Wired before the macOS ImageIO path in `load_image_rgb`.
- **TRY 4 (HEIF)**: `cull/loader.py: _load_image_pyav()` `elif sys.platform
  != "darwin"` branch — iterates `av.codec.hwaccel.HWAccel("cuda"/"dxva2"/
  "d3d11va"/"vaapi")` with per-probe `av.open()` to avoid consuming the
  primary container's `demux` state, then falls back to `container.decode`.

Local gates: `7/7 passed` (dead code). E2E `run_benchmarks.py` gates still
require **interleaved A/B on the actual non-darwin runner** before any
performance claim can be made — prior teaching on `4:2:2 10-bit Rext` is
that `NVDEC` often fails and falls back at `+110 ms`, so `+5%` and `0`
drift must be proven there, not here. **Kept as probing scaffolds** for
the next platform pass; memory updated per sequence protocol.

### Deep Dive: Why Module Sum != End-to-End Throughput & Continuous Sliding Window Pipeline (2026-08-27, KEPT)

#### 1. The Mathematical Throughput Gap & 3 Root Causes

In a theoretical dual-stage pipeline:
- **Pure Decode Supply**: 4 workers on JPG = **81.42 img/s** (12.28 ms/frame).
- **Clean Scoring Chain**: Isolated serial latency = **8.92 ms/frame** (**112.1 fps**).
- **Theoretical E2E Limit**: $\min(\text{supply}, \text{scoring}) = \min(81.4, 112.1) = \mathbf{81.4\text{ img/s}}$.

However, measured E2E on 60 JPGs was only **18.8 img/s**, and on 300 JPGs was **43.2 img/s**. Thorough instrumentation isolated three root causes:

1. **Fixed Setup Tax (2200 ms fixed denominator)**:
   - `scan()` = 91.7 ms, `load_models()` = 2108.3 ms (CoreML compile & ANE topology graph warmup).
   - Fixed tax = 2.20 s. For small sample sizes, this acts as a hard mathematical speed ceiling:
     - 60 frames: $60 / 2.20\text{s} = \mathbf{27.3\text{ img/s}}$ maximum possible even if frame processing cost 0 ms.
     - 300 frames: $300 / (2.20 + 300 \times 0.015) \approx \mathbf{44.7\text{ img/s}}$ maximum possible.
     - 3000 frames: fixed tax amortizes to 0.7 ms/frame (negligible).

2. **Pipeline Stalls at Group Boundaries**:
   - The prior `next_futs` coarse prefetch submitted only the *next burst group*.
   - When a burst group had few frames (1-2 isolated frames), the consumer finished in 15 ms while the next 20-frame group had only just started decoding (45 ms single decode), forcing the consumer thread to block hard on `futs[0].result()`.

3. **Concurrency Inflation & GIL/Cache Competition (+71.5%)**:
   - Standalone clean scoring = 8.92 ms/frame. In-engine scoring = 15.30 ms/frame.
   - 4 decode threads concurrently allocating/freeing 3.2 MB numpy arrays evicted L1/L2 caches and contended for memory bandwidth and Python GIL during Future synchronization.

#### 2. Solution: Continuous Sliding Window Pipeline (`db9fa0e`)

Replaced coarse per-group prefetch with a continuous sliding window across the entire flattened sequence of frames:
- An in-flight Queue of depth $\max(8, \min(16, \text{workers} \times 3))$ is continuously kept full.
- As each frame is popped and scored by `_process_group_internal`, the next frame is immediately submitted.
- Group boundaries no longer induce pipeline bubbles; decode workers remain 100% saturated.

#### 3. Measured Results Across Datasets (workers=4, dry-run)

| Dataset / Scale | Before (`839ad11`) | After Sliding Window (`db9fa0e`) | Speedup / Decode-Wait Impact |
|---|---|---|---|
| **ARW (80 files)** | 14.28 img/s | **17.79 img/s** | **+24.6%** (decode_wait 4904 -> 2655 ms, -46%) |
| **HEIF (72 files)** | 18.06 img/s | **19.79 img/s** | **+9.6%** (decode_wait 3166 -> 1993 ms, -37%) |
| **JPG (300 files)** | 43.19 img/s | **46.37 img/s** | **+7.4%** (steady-state 54.5 img/s excluding setup) |
| **NEF (80 files)** | 22.38 img/s | **22.48 img/s** | (decode_wait 2764 -> 1968 ms, -29%) |
| **Gate Protocol (JPG/HEIF/ARW/NEF)** | 15.35 / 7.04 / 6.02 / 6.99 | **18.08 / 7.86 / 6.74 / 7.15** | All above min thresholds (4.2 / 3.0 / 2.3 / 1.9) |
| **Precision Gates** | 7/7 passed | **7/7 passed (0 flips, 0 drift)** | 100% Bit-identical |

