"""
Analyze fence veto impact using synthetic/simulated fence detection.

Since we don't have the original HIF files, we simulate fence detection
based on the fence classifier's performance characteristics on the labeled data:
- MobileNetV2: F1=0.9796, TP=120, FP=5, FN=0, TN=483 (on 608 samples)
- False positive rate: 5/488 = 1.03%
- False negative rate: 0/120 = 0%

We'll apply this to the existing test set to estimate fence veto impact.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def simulate_fence_detection(rows: list[dict]) -> list[dict]:
    """
    Simulate fence detection based on MobileNetV2 performance characteristics.
    
    Assumptions:
    - Estimated false positive rate on general images: ~1.0%
    - We'll apply fence veto to ~1% of non-rejected and ~2% of rejected images
      (more likely to have fence issues)
    """
    
    new_rows = []
    num_fence_detected = 0
    
    # Set random seed for reproducibility
    random.seed(42)
    
    for row in rows:
        new_row = row.copy()
        rating = int(row['rating'])
        
        # Estimate fence probability based on current rating
        # Images already rejected are more likely to have fence
        if rating == -1:
            # Probability of fence in rejected images: ~2%
            fence_prob = 0.02
        else:
            # Probability of fence in kept images: ~1%
            fence_prob = 0.01
        
        if random.random() < fence_prob:
            # Fence detected
            new_row['fence_pred'] = 1
            new_row['fence_confidence'] = f"{random.uniform(0.6, 0.95):.6f}"
            new_row['rating'] = -1
            new_row['vetoed'] = 1
            old_reason = new_row['veto_reason']
            conf_str = new_row['fence_confidence']
            new_row['veto_reason'] = f"fence_detected (conf={conf_str})" if not old_reason else f"{old_reason} + fence"
            num_fence_detected += 1
        else:
            new_row['fence_pred'] = 0
            new_row['fence_confidence'] = f"{random.uniform(0.0, 0.3):.6f}"
        
        new_rows.append(new_row)
    
    log.info(f"Simulated fence detection: {num_fence_detected} images detected with fence")
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
    
    # Top veto reasons
    log.info(f"\nTop veto reasons:")
    top_reasons = sorted(vetoed_counts.items(), key=lambda x: -x[1])[:10]
    for reason, count in top_reasons:
        if reason:
            log.info(f"  {reason[:50]:50s}: {count:4d}")
    
    return {
        'total': total,
        'kept': n_kept,
        'rejected': n_rejected,
        'kept_pct': 100 * n_kept / total,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze fence veto impact (simulated)")
    parser.add_argument("--input", type=str, default="scores_multi_hfr_v4.csv", help="Input scores CSV")
    parser.add_argument("--output", type=str, default="scores_with_fence_veto_simulated.csv", help="Output scores CSV")
    args = parser.parse_args()
    
    # Load existing scores
    log.info(f"Loading scores from {args.input}...")
    with open(args.input) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    log.info(f"Loaded {len(rows)} image scores")
    
    # Compute baseline metrics
    log.info("Computing baseline metrics (without fence veto)...")
    baseline_metrics = compute_metrics(rows, label="Baseline (without fence veto)")
    
    # Simulate fence veto
    log.info("Simulating fence veto (based on MobileNetV2 performance)...")
    new_rows = simulate_fence_detection(rows)
    
    # Compute new metrics
    log.info("Computing metrics with simulated fence veto...")
    new_metrics = compute_metrics(new_rows, label="With fence veto (P3 simulated)")
    
    # Compare
    log.info(f"\n{'='*70}")
    log.info("Impact Summary (Simulated)")
    log.info(f"{'='*70}")
    log.info(f"Baseline kept:           {baseline_metrics['kept']:4d} ({baseline_metrics['kept_pct']:5.1f}%)")
    log.info(f"With fence veto:         {new_metrics['kept']:4d} ({new_metrics['kept_pct']:5.1f}%)")
    log.info(f"Additional filtered:     {baseline_metrics['kept'] - new_metrics['kept']:4d}")
    log.info(f"Reduction in kept rate:  {baseline_metrics['kept_pct'] - new_metrics['kept_pct']:5.2f} pp")
    
    # Save new scores
    log.info(f"\nSaving simulated scores to {args.output}...")
    with open(args.output, 'w', newline='') as f:
        if new_rows:
            fieldnames = list(new_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(new_rows)
    log.info("Done!")


if __name__ == "__main__":
    main()
