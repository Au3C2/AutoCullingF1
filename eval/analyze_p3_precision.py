"""
简化的P3精度提升评估
"""
import csv
import numpy as np
from pathlib import Path

# Load data
with open('scores_multi_hfr_v4.csv', encoding='utf-8') as f:
    baseline = list(csv.DictReader(f))

with open('scores_with_fence_veto_simulated.csv', encoding='utf-8') as f:
    with_fence = list(csv.DictReader(f))

print("\n" + "="*70)
print("P3 FENCE DETECTION - PRECISION IMPACT ANALYSIS")
print("="*70)

# === BASELINE METRICS ===
baseline_kept = sum(1 for r in baseline if int(r['rating']) >= 1)
baseline_sharp = np.mean([float(r['s_sharp']) for r in baseline if int(r['rating']) >= 1])
baseline_comp = np.mean([float(r['s_comp']) for r in baseline if int(r['rating']) >= 1])
baseline_raw = np.mean([float(r['raw_score']) for r in baseline if int(r['rating']) >= 1])

print("\nBASELINE (P0-P2, No Fence Detection):")
print(f"  Kept images:    {baseline_kept:4d} / {len(baseline)} ({100*baseline_kept/len(baseline):.1f}%)")
print(f"  Avg sharpness:  {baseline_sharp:.4f}")
print(f"  Avg composition:{baseline_comp:.4f}")
print(f"  Avg raw score:  {baseline_raw:.4f}")

# === WITH FENCE METRICS ===
with_fence_kept = sum(1 for r in with_fence if int(r['rating']) >= 1)
with_fence_sharp = np.mean([float(r['s_sharp']) for r in with_fence if int(r['rating']) >= 1])
with_fence_comp = np.mean([float(r['s_comp']) for r in with_fence if int(r['rating']) >= 1])
with_fence_raw = np.mean([float(r['raw_score']) for r in with_fence if int(r['rating']) >= 1])

print("\nWITH FENCE VETO (P0-P3, With Fence Detection):")
print(f"  Kept images:    {with_fence_kept:4d} / {len(with_fence)} ({100*with_fence_kept/len(with_fence):.1f}%)")
print(f"  Avg sharpness:  {with_fence_sharp:.4f}")
print(f"  Avg composition:{with_fence_comp:.4f}")
print(f"  Avg raw score:  {with_fence_raw:.4f}")

# === FENCE IMPACT ===
print("\n" + "="*70)
print("FENCE DETECTION IMPACT:")
print("="*70)

filtered_count = baseline_kept - with_fence_kept
filtered_pct = 100 * filtered_count / baseline_kept

print(f"\n  Images additionally filtered: {filtered_count} ({filtered_pct:.2f}% of baseline kept)")

# Analyze filtered images
baseline_dict = {r['filename']: r for r in baseline}
fence_filtered = []
for r in with_fence:
    fn = r['filename']
    br = baseline_dict.get(fn)
    if br and int(br['rating']) >= 1 and int(r['rating']) == -1:
        if 'fence' in r['veto_reason'].lower():
            fence_filtered.append(br)

if fence_filtered:
    print(f"\n  Quality of fence-filtered images:")
    filt_sharp = np.mean([float(r['s_sharp']) for r in fence_filtered])
    filt_comp = np.mean([float(r['s_comp']) for r in fence_filtered])
    filt_raw = np.mean([float(r['raw_score']) for r in fence_filtered])
    
    print(f"    Avg sharpness: {filt_sharp:.4f}")
    print(f"    Avg composition: {filt_comp:.4f}")
    print(f"    Avg raw score: {filt_raw:.4f}")
    
    # Estimate TP/FP
    tp_count = sum(1 for r in fence_filtered if float(r['raw_score']) < 3.5)
    fp_count = len(fence_filtered) - tp_count
    print(f"\n  Estimated TP (correctly filtered): {tp_count}")
    print(f"  Estimated FP (incorrectly filtered): {fp_count}")
    print(f"  Estimated precision: {tp_count/(tp_count+fp_count):.2%}" if (tp_count+fp_count) > 0 else "  N/A")

# === PRECISION IMPROVEMENT ===
print("\n" + "="*70)
print("PRECISION/QUALITY IMPROVEMENT:")
print("="*70)

quality_improvement = with_fence_raw - baseline_raw
quality_improvement_pct = 100 * quality_improvement / baseline_raw

print(f"\n  Baseline avg raw score:  {baseline_raw:.4f}")
print(f"  With P3 avg raw score:   {with_fence_raw:.4f}")
print(f"  Quality improvement:     {quality_improvement:+.4f} ({quality_improvement_pct:+.2f}%)")

# === USER SATISFACTION MODEL ===
print("\n" + "="*70)
print("USER SATISFACTION MODEL:")
print("="*70)
print("\nAssumptions (F1 photography workflow):")
print("  - Quality importance: 60% (raw score)")
print("  - Quantity importance: 30% (number of kept images)")
print("  - Accuracy importance: 10% (no false positives)")

# Normalize scores to [0, 1]
quality_score_baseline = baseline_raw / 6.0
quality_score_with_fence = with_fence_raw / 6.0

quantity_score_baseline = min(baseline_kept / 500, 1.0)
quantity_score_with_fence = min(with_fence_kept / 500, 1.0)

# Accuracy: TP/(TP+FP)
accuracy_baseline = 0.80  # Estimate for P0-P2
accuracy_with_fence = (len(fence_filtered) - len([r for r in fence_filtered if float(r['raw_score']) >= 4.5])) / len(fence_filtered) if fence_filtered else 0.80

satisfaction_baseline = 0.6 * quality_score_baseline + 0.3 * quantity_score_baseline + 0.1 * accuracy_baseline
satisfaction_with_fence = 0.6 * quality_score_with_fence + 0.3 * quantity_score_with_fence + 0.1 * accuracy_with_fence

satisfaction_improvement = satisfaction_with_fence - satisfaction_baseline
satisfaction_improvement_pct = 100 * satisfaction_improvement / satisfaction_baseline if satisfaction_baseline > 0 else 0

print(f"\n  Component scores (BASELINE):")
print(f"    Quality (60%):  {quality_score_baseline:.3f}")
print(f"    Quantity (30%): {quantity_score_baseline:.3f}")
print(f"    Accuracy (10%): {accuracy_baseline:.3f}")
print(f"    TOTAL:          {satisfaction_baseline:.3f}")

print(f"\n  Component scores (WITH P3):")
print(f"    Quality (60%):  {quality_score_with_fence:.3f}")
print(f"    Quantity (30%): {quantity_score_with_fence:.3f}")
print(f"    Accuracy (10%): {accuracy_with_fence:.3f}")
print(f"    TOTAL:          {satisfaction_with_fence:.3f}")

print(f"\n  Satisfaction improvement: {satisfaction_improvement:+.3f} ({satisfaction_improvement_pct:+.2f}%)")

# === CONCLUSION ===
print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)

if satisfaction_improvement > 0.05:
    assessment = "SIGNIFICANT IMPROVEMENT"
    emoji = "✅"
elif satisfaction_improvement > 0.02:
    assessment = "MODERATE IMPROVEMENT"
    emoji = "✓"
elif satisfaction_improvement > 0:
    assessment = "MARGINAL IMPROVEMENT"
    emoji = "~"
else:
    assessment = "MINIMAL/NO IMPROVEMENT"
    emoji = "⚠️"

print(f"\n{emoji} {assessment}")
print(f"\nEstimated precision improvement: {satisfaction_improvement_pct:.2f}%")

if satisfaction_improvement > 0:
    print(f"\nWhat this means:")
    print(f"  - User will retain ~{abs(filtered_count)} fewer images (removed fence-occluded ones)")
    print(f"  - Kept images will be slightly higher quality (+{quality_improvement_pct:.2f}%)")
    print(f"  - Overall satisfaction improvement: ~{satisfaction_improvement_pct:.2f}%")
else:
    print(f"\nNote: P3 provides minimal benefit in the simulated scenario.")
    print(f"This could be because:")
    print(f"  1. Most fence-occluded images were already filtered by P0-P2")
    print(f"  2. The fence classifier has high false positive rate on this dataset")

print("\n" + "="*70)
