"""
evaluate_fence_veto_impact.py — Evaluate fence veto impact on existing scores.

This script:
1. Loads existing scores (without fence veto) from scores_multi_hfr_v4.csv
2. Runs fence classifier on all images
3. Applies fence veto to get new scores
4. Compares metrics before/after fence veto
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np

from cull.fence_classifier import FenceClassifier
from cull.scorer import ImageScore, _raw_to_rating

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

device_str = "cuda"

def load_image_paths_from_csv(csv_path: Path) -> dict[str, Path]:
    """Load image paths from scores CSV (maps filename to path in error_review dir)."""
    image_paths = {}
    
    # Try to find images in error_review directory
    error_review_dir = Path("error_review")
    if error_review_dir.exists():
        for img_path in error_review_dir.glob("**/*.HIF"):
            image_paths[img_path.name] = img_path
    
    log.info(f"Found {len(image_paths)} images in error_review directory")
    return image_paths


def apply_fence_veto(rows: list[dict], classifier: FenceClassifier, image_paths: dict[str, Path]) -> list[dict]:
    """Apply fence veto to existing scores."""
    
    new_rows = []
    fence_preds = []
    fence_confs = []
    num_fence_vetoed = 0
    num_skipped = 0
    
    for i, row in enumerate(rows):
        filename = row['filename']
        
        # Try to find image
        if filename not in image_paths:
            # Try without extension
            base_name = Path(filename).stem
            found = False
            for name, path in image_paths.items():
                if Path(name).stem == base_name:
                    img_path = path
                    found = True
                    break
            if not found:
                log.warning(f"Image not found: {filename}")
                num_skipped += 1
                new_rows.append(row)  # keep original
                fence_preds.append(0)
                fence_confs.append(0.0)
                continue
        else:
            img_path = image_paths[filename]
        
        # Predict fence
        try:
            fence_pred, fence_conf = classifier.predict_image(img_path)
            fence_preds.append(fence_pred)
            fence_confs.append(fence_conf)
            
            # Create new row with fence veto
            new_row = row.copy()
            new_row['fence_pred'] = fence_pred
            new_row['fence_confidence'] = f"{fence_conf:.6f}"
            
            # Apply fence veto if fence detected
            if fence_pred == 1:
                new_row['rating'] = -1
                new_row['vetoed'] = 1
                old_reason = new_row['veto_reason']
                new_row['veto_reason'] = f"fence_detected (conf={fence_conf:.3f})" if not old_reason else f"{old_reason} + fence"
                num_fence_vetoed += 1
            
            new_rows.append(new_row)
        except Exception as e:
            log.error(f"Error processing {filename}: {e}")
            num_skipped += 1
            new_rows.append(row)
            fence_preds.append(0)
            fence_confs.append(0.0)
        
        if (i + 1) % 100 == 0:
            log.info(f"Processed {i+1}/{len(rows)} images")
    
    log.info(f"Fence veto applied: {num_fence_vetoed} images detected with fence")
    log.info(f"Skipped (not found): {num_skipped} images")
    return new_rows


def compute_metrics(rows: list[dict], label: str = "Overall") -> dict:
    """Compute evaluation metrics."""
    
    # Count by rating
    rating_counts = defaultdict(int)
    vetoed_counts = defaultdict(int)
    
    for row in rows:
        rating = int(row['rating'])
        vetoed = int(row['vetoed'])
        rating_counts[rating] += 1
        if vetoed:
            vetoed_counts[row['veto_reason']] += 1
    
    # Simple stats
    total = len(rows)
    n_kept = sum(1 for r in rows if int(r['rating']) >= 1)
    n_rejected = sum(1 for r in rows if int(r['rating']) == -1)
    
    log.info(f"\n{'='*70}")
    log.info(f"{label} Metrics")
    log.info(f"{'='*70}")
    log.info(f"Total images: {total}")
    log.info(f"Kept (Rating 1-5): {n_kept} ({100*n_kept/total:.1f}%)")
    log.info(f"Rejected (Rating -1): {n_rejected} ({100*n_rejected/total:.1f}%)")
    log.info(f"\nRating distribution:")
    for rating in sorted(rating_counts.keys()):
        count = rating_counts[rating]
        log.info(f"  Rating {rating:2d}: {count:4d} ({100*count/total:5.1f}%)")
    
    log.info(f"\nVeto reasons:")
    for reason, count in sorted(vetoed_counts.items(), key=lambda x: -x[1]):
        if reason:
            log.info(f"  {reason:40s}: {count:4d}")
    
    return {
        'total': total,
        'kept': n_kept,
        'rejected': n_rejected,
        'kept_pct': 100 * n_kept / total,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fence veto impact")
    parser.add_argument("--input", type=str, default="scores_multi_hfr_v4.csv", help="Input scores CSV")
    parser.add_argument("--output", type=str, default="scores_multi_with_fence_veto.csv", help="Output scores CSV")
    parser.add_argument("--arch", type=str, default="mobilenetv2", help="Fence classifier architecture")
    args = parser.parse_args()
    
    # Load classifier
    log.info(f"Loading fence classifier ({args.arch})...")
    classifier = FenceClassifier(arch=args.arch)
    
    # Load existing scores
    log.info(f"Loading scores from {args.input}...")
    with open(args.input) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    log.info(f"Loaded {len(rows)} image scores")
    
    # Compute baseline metrics
    log.info("Computing baseline metrics (without fence veto)...")
    baseline_metrics = compute_metrics(rows, label="Baseline (without fence veto)")
    
    # Load image paths
    log.info("Loading image paths...")
    image_paths = load_image_paths_from_csv(Path(args.input))
    
    # Apply fence veto
    log.info("Applying fence veto...")
    new_rows = apply_fence_veto(rows, classifier, image_paths)
    
    # Compute new metrics
    log.info("Computing metrics with fence veto...")
    new_metrics = compute_metrics(new_rows, label="With fence veto (P3)")
    
    # Compare
    log.info(f"\n{'='*70}")
    log.info("Impact Summary")
    log.info(f"{'='*70}")
    log.info(f"Baseline kept:       {baseline_metrics['kept']:4d} ({baseline_metrics['kept_pct']:5.1f}%)")
    log.info(f"With fence veto:     {new_metrics['kept']:4d} ({new_metrics['kept_pct']:5.1f}%)")
    log.info(f"Additional filtered: {baseline_metrics['kept'] - new_metrics['kept']:4d}")
    
    # Save new scores
    log.info(f"\nSaving new scores to {args.output}...")
    with open(args.output, 'w', newline='') as f:
        if new_rows:
            fieldnames = list(new_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
    log.info("Done!")


if __name__ == "__main__":
    main()
