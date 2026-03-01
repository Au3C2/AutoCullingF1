"""
Pre-decode all HIF/HEIF images in the dataset to JPEG for fast training.

Why
---
pillow-heif decodes a single HIF file in ~1.6 s on this machine.  With 7 469
images one full epoch of on-the-fly decoding takes roughly 3 hours.  Writing
each image to a JPEG cache file once reduces subsequent reads to < 20 ms per
image (typical SSD + libjpeg), making multi-epoch training practical.

Output layout
--------------
    dataset/cache/<stem>.jpg   — one JPEG per source HIF file

The stem is identical to the source file stem so that ``dataset.py`` can
locate the cache file from the ``img_path`` column of the CSV:

    img_path  = dataset/img/20250322_143512.HIF
    cache file = dataset/cache/20250322_143512.jpg

Usage
-----
    # Convert all images (uses all CPU cores by default):
    uv run cache_images.py

    # Limit parallelism:
    uv run cache_images.py --workers 4

    # Custom paths:
    uv run cache_images.py \\
        --img-dir dataset/img \\
        --cache-dir dataset/cache \\
        --quality 95

    # Dry-run (count work without writing):
    uv run cache_images.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker function (runs in subprocess — must be importable at module level)
# ---------------------------------------------------------------------------


def _convert_one(args: tuple[Path, Path, int]) -> tuple[str, bool, str]:
    """Convert a single HIF image to JPEG.

    Parameters
    ----------
    args:
        Tuple of ``(src_path, dst_path, quality)``.

    Returns
    -------
    tuple[str, bool, str]
        ``(src filename, success, error message or empty string)``
    """
    src, dst, quality = args
    try:
        # Import here so each worker process loads the plugin independently
        import pillow_heif  # type: ignore
        from PIL import Image

        pillow_heif.register_heif_opener()

        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            rgb.save(dst, format="JPEG", quality=quality, subsampling=0)
        return (src.name, True, "")
    except Exception as exc:  # noqa: BLE001
        return (src.name, False, str(exc))


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def collect_sources(img_dir: Path, cache_dir: Path, force: bool) -> list[tuple[Path, Path, int]]:
    """Return list of (src, dst, quality) tuples for images that need conversion.

    Parameters
    ----------
    img_dir:
        Directory containing source HIF/HEIF files.
    cache_dir:
        Destination directory for JPEG files.
    force:
        If True, include images whose JPEG already exists (re-encode them).

    Returns
    -------
    list of (src, dst, quality) — quality is filled in by the caller.
    """
    extensions = {".hif", ".heif", ".heic"}
    sources = []
    for src in sorted(img_dir.iterdir()):
        if src.suffix.lower() not in extensions:
            continue
        dst = cache_dir / (src.stem + ".jpg")
        if not force and dst.exists():
            continue
        sources.append(src)
    return sources


def run_conversion(
    img_dir: Path,
    cache_dir: Path,
    quality: int,
    workers: int,
    force: bool,
    dry_run: bool,
) -> int:
    """Execute the batch conversion and return an exit code.

    Parameters
    ----------
    img_dir:
        Source directory of HIF files.
    cache_dir:
        Destination directory for JPEG files.
    quality:
        JPEG quality (1–95; 95 is near-lossless at much smaller size).
    workers:
        Number of parallel worker processes.
    force:
        Re-encode even if JPEG already exists.
    dry_run:
        Print the plan but do not write any files.

    Returns
    -------
    int
        Exit code (0 = all OK, 1 = some failures).
    """
    if not img_dir.is_dir():
        log.error("Image directory not found: %s", img_dir)
        return 1

    sources = collect_sources(img_dir, cache_dir, force)

    log.info(
        "Found %d images to convert  (cache_dir=%s, quality=%d, workers=%d)",
        len(sources),
        cache_dir,
        quality,
        workers,
    )

    if dry_run:
        for src in sources:
            print(f"  {src.name}  →  {cache_dir / (src.stem + '.jpg')}")
        print(f"\nDry-run: {len(sources)} files would be written.")
        return 0

    if not sources:
        log.info("Nothing to do — all JPEGs already exist.  Use --force to re-encode.")
        return 0

    cache_dir.mkdir(parents=True, exist_ok=True)
    work = [(src, cache_dir / (src.stem + ".jpg"), quality) for src in sources]

    t0 = time.time()
    n_ok = 0
    n_fail = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_convert_one, item): item[0] for item in work}
        for i, future in enumerate(as_completed(futures), 1):
            name, ok, err = future.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                log.warning("FAILED %s: %s", name, err)

            if i % 100 == 0 or i == len(work):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(work) - i) / rate if rate > 0 else 0
                log.info(
                    "[%d/%d]  %.1f img/s  ETA %.0fs  (ok=%d  fail=%d)",
                    i, len(work), rate, eta, n_ok, n_fail,
                )

    elapsed = time.time() - t0
    log.info(
        "Done in %.1fs — %d succeeded, %d failed.",
        elapsed, n_ok, n_fail,
    )
    return 0 if n_fail == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="cache_images",
        description="Pre-decode HIF/HEIF images to JPEG for fast training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=Path("dataset/img"),
        help="Directory containing source HIF/HEIF files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("dataset/cache"),
        help="Output directory for JPEG cache files.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (1–95).  95 = near-lossless, ~5–10× smaller than HIF.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode images even if the JPEG cache file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the conversion plan without writing any files.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry-point for the cache_images script."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    return run_conversion(
        img_dir=args.img_dir,
        cache_dir=args.cache_dir,
        quality=args.quality,
        workers=args.workers,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
