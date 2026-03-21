#!/usr/bin/env python3
"""
autocrop.py — Scan a directory for rated images and update XMP with auto-cropping.
"""

import sys
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from cull.detector import load_f1_model, load_coco_model, detect, Detection
from cull.xmp_reader import read_xmp_rating
from cull.cropper import calculate_crop, update_xmp_with_crop, has_crop_info
from cull.loader import load_image_rgb, RAW_EXTS, COOKED_EXTS

log = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

def process_image(img_path: Path, f1_model, coco_model, dry_run=False, overwrite=False):
    """Detect subject and update crop in XMP."""
    xmp_path = img_path.with_suffix(".xmp")
    if not xmp_path.exists():
        return False
        
    if not dry_run and not overwrite and has_crop_info(xmp_path):
        log.debug("  [%s]  Already has crop information, skipping", img_path.name)
        return False
    
    img_rgb = load_image_rgb(img_path, scale_width=1280)
    if img_rgb is None:
        log.warning("  [%s]  Failed to load image", img_path.name)
        return False
        
    # Use the unified detector from cull.detector
    detections = detect(img_rgb, f1_model, coco_model, conf=0.3)
    
    # Unified detector returns labels: "f1_car", "coco_car", "coco_person", etc.
    targets = [d for d in detections if d.label in ("f1_car", "coco_car")]
    
    if not targets:
        log.warning("  [%s]  Found 0 cars, skipping crop", img_path.name)
        return False

    h_img, w_img = img_rgb.shape[:2]
    img_ar = w_img / h_img
    
    # Subject prioritization (Aggressive mode)
    if len(targets) > 1:
        # Largest one first
        targets.sort(key=lambda d: d.area(), reverse=True)
        area0 = targets[0].area()
        area1 = targets[1].area()
        
        if area1 > area0 * 0.9:
            # Very close in size, pick most central
            targets.sort(key=lambda d: d.center_proximity(w_img, h_img), reverse=True)

    det = targets[0]
    
    x1, y1, x2, y2 = det.x1 / w_img, det.y1 / h_img, det.x2 / w_img, det.y2 / h_img
    
    crop = calculate_crop(x1, y1, x2, y2, img_ar=img_ar)
    if crop:
        if dry_run:
            log.info("  [%s]  [dry-run] Calculated crop: top=%.3f, left=%.3f, bottom=%.3f, right=%.3f (AR=%.2f)", 
                     img_path.name, *crop, img_ar)
        else:
            update_xmp_with_crop(xmp_path, crop)
            log.info("  [%s]  Updated crop: top=%.3f, left=%.3f, bottom=%.3f, right=%.3f", 
                     img_path.name, *crop)
        return True
    
    return False

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Auto-crop images based on F1 detections.")
    parser.add_argument("dir", type=Path, nargs='?', help="Directory to process")
    parser.add_argument("--min-rating", type=int, default=1, help="Min rating to process (default: 1)")
    parser.add_argument("--f1-model", type=str, default="models/f1_yolov8n.mlpackage", help="F1 model path")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, onnx, coreml)")
    parser.add_argument("--overwrite", action="store_true", help="Force recalculate crop even if it exists")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to XMP, only log")
    parser.add_argument("--file", type=str, help="Process a single file")
    args = parser.parse_args()

    # Load models first if we have anything to do
    f1_model = load_f1_model(Path(args.f1_model))
    coco_model = load_coco_model()

    if args.file:
        img_path = Path(args.file)
        if not img_path.exists():
            log.error("File not found: %s", img_path)
            sys.exit(1)
        log.info("Processing single file: %s", img_path)
        # For single file, overwrite=True by default or if requested
        ok = process_image(img_path, f1_model, coco_model, dry_run=args.dry_run, overwrite=(True or args.overwrite))
        sys.exit(0 if ok else 1)

    if not args.dir or not args.dir.is_dir():
        log.error("Valid directory required if --file is not specified.")
        sys.exit(1)
    
    # 1. Collect all candidates
    all_files = list(args.dir.iterdir())
    xmp_files = [f for f in all_files if f.suffix.lower() == ".xmp"]
    
    candidates: list[Path] = []
    for xmp in xmp_files:
        rating, pick = read_xmp_rating(xmp.with_suffix(".jpg"))
        if rating is not None and rating >= args.min_rating:
            # Found a rated image. Find the source file.
            base = xmp.stem
            found = False
            for ext in COOKED_EXTS:
                p = args.dir / f"{base}{ext}"
                if p.exists():
                    candidates.append(p)
                    found = True
                    break
            if not found:
                for ext in RAW_EXTS:
                    p = args.dir / f"{base}{ext}"
                    if p.exists():
                        candidates.append(p)
                        found = True
                        break

    if not candidates:
        log.info("No rated images found to process.")
        return

    if args.dry_run:
        log.info("--- DRY RUN MODE (limit to 10 images, no XMP writing) ---")
        candidates = candidates[:10]

    log.info("Found %d rated images. Processing with %d workers...", len(candidates), args.workers)
    
    t_start = time.perf_counter()
    count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_image, c, f1_model, coco_model, dry_run=args.dry_run, overwrite=args.overwrite): c for c in candidates}
        for future in futures:
            if future.result():
                count += 1
                
    elapsed = time.perf_counter() - t_start
    log.info("Done. Updated crop for %d images in %.1fs", count, elapsed)

if __name__ == "__main__":
    main()
