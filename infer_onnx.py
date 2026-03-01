"""
ONNX-based inference script for F1 motorsport photo culling.

Given a directory of JPEG files, this script:
1. Runs inference with an ONNX model (GPU via CUDAExecutionProvider, or CPU).
2. Moves / copies each image into a subdirectory named after its predicted label:
       <output_dir>/keep/    (label = 1, sigmoid > threshold)
       <output_dir>/discard/ (label = 0, sigmoid <= threshold)
3. Writes a CSV summary with per-image scores.
4. Optionally benchmarks throughput (images/sec) across multiple batch sizes.

Usage
-----
    # Basic: infer all JPEGs in ./photos, sort into ./results
    python infer_onnx.py \\
        --model onnx_models/resnext50.onnx \\
        --input-dir photos \\
        --output-dir results

    # Copy instead of move, custom threshold
    python infer_onnx.py \\
        --model onnx_models/resnext50.onnx \\
        --input-dir photos \\
        --output-dir results \\
        --threshold 0.45 \\
        --copy \\
        --batch-size 16

    # Benchmark only (no file operations)
    python infer_onnx.py \\
        --model onnx_models/resnext50.onnx \\
        --input-dir photos \\
        --benchmark
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet normalisation constants (must match training)
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Image preprocessing (mirrors dataset.py eval transform, numpy-only)
# ---------------------------------------------------------------------------


def preprocess_image(img_path: Path, img_size: int = 224) -> np.ndarray:
    """Load and preprocess a single image to a (3, H, W) float32 array.

    Steps mirror the eval-time transform in ``dataset.py``:
      1. Open RGB
      2. SquarePad (black borders)
      3. Resize to img_size × img_size (bilinear)
      4. ToTensor (uint8 → float32, divide by 255)
      5. Normalize with ImageNet mean/std

    Parameters
    ----------
    img_path:
        Path to any image file readable by Pillow.
    img_size:
        Target square resolution (default 224).

    Returns
    -------
    np.ndarray
        Shape (3, img_size, img_size), dtype float32.
    """
    img = Image.open(img_path).convert("RGB")

    # SquarePad
    w, h = img.size
    max_side = max(w, h)
    pad_l = (max_side - w) // 2
    pad_t = (max_side - h) // 2
    pad_r = max_side - w - pad_l
    pad_b = max_side - h - pad_t
    img = ImageOps.expand(img, (pad_l, pad_t, pad_r, pad_b), fill=0)

    # Resize
    img = img.resize((img_size, img_size), resample=Image.Resampling.BILINEAR)

    # HWC uint8 → CHW float32 in [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0      # (H, W, 3)
    arr = (arr - _MEAN) / _STD                          # normalise
    arr = arr.transpose(2, 0, 1)                        # (3, H, W)
    return arr


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------


def batched(items: list, batch_size: int):
    """Yield successive batches of *batch_size* from *items*."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ---------------------------------------------------------------------------
# ONNX session factory
# ---------------------------------------------------------------------------


def build_session(model_path: Path, use_gpu: bool = True, use_coreml: bool = False):
    """Create an onnxruntime InferenceSession.

    Provider priority:
      - ``use_coreml=True``  → CoreMLExecutionProvider → CPUExecutionProvider
      - ``use_gpu=True``     → CUDAExecutionProvider   → CPUExecutionProvider
      - otherwise            → CPUExecutionProvider only

    On Apple-Silicon machines install ``onnxruntime-silicon`` and pass
    ``--coreml`` to enable Apple Neural Engine / GPU acceleration via CoreML.

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` file.
    use_gpu:
        When ``True``, try CUDAExecutionProvider first and fall back to CPU.
    use_coreml:
        When ``True``, try CoreMLExecutionProvider first and fall back to CPU.
        Requires ``onnxruntime-silicon`` (``pip install onnxruntime-silicon``).

    Returns
    -------
    onnxruntime.InferenceSession
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required.\n"
            "  GPU (CUDA):  pip install onnxruntime-gpu\n"
            "  Apple M-series: pip install onnxruntime-silicon\n"
            "  CPU only:    pip install onnxruntime"
        ) from exc

    if use_coreml:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    elif use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(model_path), providers=providers)
    active = sess.get_providers()[0]
    log.info("ORT session created: model=%s  provider=%s", model_path.name, active)
    return sess


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------


def run_inference(
    sess,
    image_paths: list[Path],
    img_size: int = 224,
    batch_size: int = 32,
    threshold: float = 0.5,
) -> list[dict]:
    """Run inference over *image_paths* and return per-image result dicts.

    Parameters
    ----------
    sess:
        onnxruntime InferenceSession.
    image_paths:
        List of image file paths to score.
    img_size:
        Input resolution expected by the model.
    batch_size:
        Number of images per ORT forward pass.
    threshold:
        Sigmoid threshold for positive (keep) class.

    Returns
    -------
    list[dict]
        Each dict has keys: ``path``, ``score``, ``label``.
        ``score`` is the raw sigmoid probability (float).
        ``label`` is 1 (keep) or 0 (discard).
    """
    input_name = sess.get_inputs()[0].name
    results: list[dict] = []

    n = len(image_paths)
    processed = 0

    for batch_paths in batched(image_paths, batch_size):
        # ---- Preprocess batch -----------------------------------------------
        arrays = []
        valid_paths = []
        for p in batch_paths:
            try:
                arrays.append(preprocess_image(p, img_size))
                valid_paths.append(p)
            except Exception as exc:
                log.warning("Skipping %s: %s", p.name, exc)

        if not arrays:
            continue

        batch_np = np.stack(arrays, axis=0)   # (B, 3, H, W)

        # ---- ORT forward pass -----------------------------------------------
        logits = sess.run(None, {input_name: batch_np})[0]  # (B, 1)
        scores = 1.0 / (1.0 + np.exp(-logits.squeeze(1)))   # sigmoid

        for path, score in zip(valid_paths, scores):
            label = 1 if float(score) > threshold else 0
            results.append({"path": path, "score": float(score), "label": label})

        processed += len(valid_paths)
        if processed % max(batch_size * 4, 100) == 0 or processed == n:
            log.info("  Processed %d / %d images …", processed, n)

    return results


# ---------------------------------------------------------------------------
# File organisation
# ---------------------------------------------------------------------------


def organise_files(
    results: list[dict],
    output_dir: Path,
    copy: bool = False,
) -> None:
    """Move or copy images into label subdirectories.

    Creates:
        <output_dir>/keep/    ← label == 1
        <output_dir>/discard/ ← label == 0

    Parameters
    ----------
    results:
        List of dicts from :func:`run_inference`.
    output_dir:
        Root output directory.
    copy:
        When ``True``, copy files; when ``False`` (default), move them.
    """
    keep_dir    = output_dir / "keep"
    discard_dir = output_dir / "discard"
    keep_dir.mkdir(parents=True, exist_ok=True)
    discard_dir.mkdir(parents=True, exist_ok=True)

    action = shutil.copy2 if copy else shutil.move
    action_name = "Copied" if copy else "Moved"

    keep_count = discard_count = 0
    for r in results:
        src: Path = r["path"]
        dst_dir = keep_dir if r["label"] == 1 else discard_dir
        dst = dst_dir / src.name
        # Avoid overwriting if destination already exists
        if dst.exists():
            stem, suffix = src.stem, src.suffix
            dst = dst_dir / f"{stem}_{int(time.time() * 1000)}{suffix}"
        action(str(src), str(dst))
        if r["label"] == 1:
            keep_count += 1
        else:
            discard_count += 1

    log.info("%s %d images → keep/", action_name, keep_count)
    log.info("%s %d images → discard/", action_name, discard_count)


# ---------------------------------------------------------------------------
# CSV summary writer
# ---------------------------------------------------------------------------


def write_csv(results: list[dict], csv_path: Path) -> None:
    """Write inference results to *csv_path*.

    Columns: ``filename``, ``score``, ``label``
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "score", "label"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename": r["path"].name,
                "score": f"{r['score']:.6f}",
                "label": r["label"],
            })
    log.info("CSV summary → %s", csv_path)


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------


def benchmark(
    sess,
    image_paths: list[Path],
    img_size: int = 224,
    batch_sizes: list[int] | None = None,
    warmup_batches: int = 3,
    measure_batches: int = 20,
) -> dict[int, float]:
    """Measure inference throughput (images/sec) for several batch sizes.

    Parameters
    ----------
    sess:
        onnxruntime InferenceSession.
    image_paths:
        Pool of images to sample from (at least ``max(batch_sizes)`` needed).
    img_size:
        Input resolution.
    batch_sizes:
        Batch sizes to benchmark.  Defaults to [1, 4, 8, 16, 32].
    warmup_batches:
        Number of warm-up forward passes (not timed).
    measure_batches:
        Number of timed forward passes.

    Returns
    -------
    dict[int, float]
        Mapping batch_size → images_per_second.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    input_name = sess.get_inputs()[0].name
    throughputs: dict[int, float] = {}

    # Pre-load a pool of preprocessed images (up to 64 to avoid RAM pressure)
    pool_size = min(128, len(image_paths))
    log.info("Pre-loading %d images for benchmark …", pool_size)
    pool: list[np.ndarray] = []
    for p in image_paths[:pool_size]:
        try:
            pool.append(preprocess_image(p, img_size))
        except Exception:
            pass

    if not pool:
        log.error("No valid images in pool for benchmarking.")
        return {}

    for bs in batch_sizes:
        if bs > len(pool):
            log.warning("Skipping batch_size=%d (pool only has %d images)", bs, len(pool))
            continue

        # Build a fixed batch from the pool (cycling if needed)
        indices = [i % len(pool) for i in range(bs)]
        batch_np = np.stack([pool[i] for i in indices], axis=0)

        # Warm-up
        for _ in range(warmup_batches):
            sess.run(None, {input_name: batch_np})

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(measure_batches):
            sess.run(None, {input_name: batch_np})
        elapsed = time.perf_counter() - t0

        imgs_per_sec = (bs * measure_batches) / elapsed
        throughputs[bs] = imgs_per_sec
        log.info(
            "  batch_size=%-3d  %.1f images/sec  (%.2f ms/img)",
            bs,
            imgs_per_sec,
            1000.0 / imgs_per_sec,
        )

    return throughputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="infer_onnx",
        description="Run ONNX inference on a directory of JPEGs and sort by label.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Required -----------------------------------------------------------
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the .onnx model file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing JPEG images to classify.",
    )

    # ---- Output -------------------------------------------------------------
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for keep/ and discard/ subdirs.  "
            "Defaults to <input-dir>_sorted."
        ),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=False,
        help="Copy files instead of moving them.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write a CSV score summary to this path (optional).",
    )

    # ---- Inference settings -------------------------------------------------
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input resolution expected by the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for the keep (label=1) class.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force CPU execution (default: try GPU first).",
    )
    parser.add_argument(
        "--coreml",
        action="store_true",
        default=False,
        help=(
            "Use CoreMLExecutionProvider for Apple-Silicon acceleration. "
            "Requires onnxruntime-silicon (pip install onnxruntime-silicon). "
            "Takes precedence over --cpu / GPU auto-detection."
        ),
    )

    # ---- Benchmark ----------------------------------------------------------
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help=(
            "Run throughput benchmark instead of file classification.  "
            "No files are moved/copied."
        ),
    )
    parser.add_argument(
        "--benchmark-batch-sizes",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32],
        metavar="BS",
        help="Batch sizes to test in benchmark mode.",
    )

    # ---- Extensions ---------------------------------------------------------
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"],
        help="Image file extensions to glob for in --input-dir.",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # ---- Collect images -----------------------------------------------------
    input_dir: Path = args.input_dir
    if not input_dir.exists():
        log.error("Input directory not found: %s", input_dir)
        return 1

    image_paths: list[Path] = []
    for ext in args.extensions:
        image_paths.extend(sorted(input_dir.glob(f"*{ext}")))

    if not image_paths:
        log.error("No images found in %s (extensions: %s)", input_dir, args.extensions)
        return 1

    log.info("Found %d images in %s", len(image_paths), input_dir)

    # ---- Build ORT session --------------------------------------------------
    sess = build_session(args.model, use_gpu=not args.cpu, use_coreml=args.coreml)

    # ---- Benchmark mode -----------------------------------------------------
    if args.benchmark:
        log.info("=== Benchmark mode — model: %s ===", args.model.name)
        throughputs = benchmark(
            sess,
            image_paths,
            img_size=args.img_size,
            batch_sizes=args.benchmark_batch_sizes,
        )
        print("\n--- Benchmark results ---")
        print(f"Model: {args.model.name}")
        print(f"{'batch_size':>12}  {'img/sec':>12}  {'ms/img':>10}")
        print("-" * 40)
        for bs, ips in sorted(throughputs.items()):
            print(f"{bs:>12}  {ips:>12.1f}  {1000/ips:>10.2f}")
        return 0

    # ---- Inference ----------------------------------------------------------
    t_start = time.perf_counter()
    results = run_inference(
        sess,
        image_paths,
        img_size=args.img_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    elapsed = time.perf_counter() - t_start

    n = len(results)
    keep_n    = sum(1 for r in results if r["label"] == 1)
    discard_n = n - keep_n
    ips = n / elapsed if elapsed > 0 else float("inf")

    log.info(
        "Inference complete: %d images in %.2fs (%.1f img/sec)",
        n, elapsed, ips,
    )
    log.info("  keep=%d (%.1f%%)  discard=%d (%.1f%%)",
             keep_n,    100 * keep_n / max(n, 1),
             discard_n, 100 * discard_n / max(n, 1))

    # ---- Organise files -----------------------------------------------------
    output_dir = args.output_dir or (input_dir.parent / (input_dir.name + "_sorted"))
    organise_files(results, output_dir, copy=args.copy)

    # ---- Optional CSV -------------------------------------------------------
    if args.csv:
        write_csv(results, args.csv)
    else:
        # Always write a default CSV alongside the output dir
        default_csv = output_dir / "scores.csv"
        write_csv(results, default_csv)

    log.info("Done.  Results in: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
