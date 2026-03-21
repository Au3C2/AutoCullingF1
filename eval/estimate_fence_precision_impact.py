"""
estimate_fence_precision_impact.py — 评估加入P3(铁丝网检测)对精度的影响

分析维度:
1. 评分分布变化 (rating distribution)
2. 保留图片质量变化 (质量指标: 锐度、构图)
3. 估算的精度提升 (推断用户满意度提升)
4. 假阳性率 (FP) 和假阴性率 (FN) 分析
5. 不同session的影响差异
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def load_csv(csv_path: str) -> list[dict]:
    """Load CSV file."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_quality_metrics(rows: list[dict]) -> dict:
    """
    计算保留图片的质量指标。
    
    对于被保留的图片 (rating >= 1)，计算:
    - 平均锐度分数
    - 平均构图分数
    - 平均原始分数 (raw_score)
    - 平均rating
    """
    kept_rows = [r for r in rows if int(r['rating']) >= 1]
    
    if not kept_rows:
        return {
            'avg_sharpness': 0,
            'avg_composition': 0,
            'avg_raw_score': 0,
            'avg_rating': 0,
            'num_kept': 0,
        }
    
    sharpness = [float(r['s_sharp']) for r in kept_rows]
    composition = [float(r['s_comp']) for r in kept_rows]
    raw_scores = [float(r['raw_score']) for r in kept_rows]
    ratings = [int(r['rating']) for r in kept_rows]
    
    return {
        'avg_sharpness': np.mean(sharpness),
        'avg_composition': np.mean(composition),
        'avg_raw_score': np.mean(raw_scores),
        'avg_rating': np.mean(ratings),
        'num_kept': len(kept_rows),
    }


def analyze_veto_types(rows: list[dict]) -> dict:
    """分析被过滤的图片的原因分布。"""
    veto_reasons = defaultdict(int)
    
    for row in rows:
        if int(row['rating']) == -1:
            reason = row['veto_reason']
            # 简化reason (只取第一部分)
            if reason:
                main_reason = reason.split('(')[0].strip()
                veto_reasons[main_reason] += 1
    
    return dict(veto_reasons)


def analyze_fence_impact(rows_baseline: list[dict], rows_with_fence: list[dict]) -> dict:
    """
    分析围栏检测的具体影响。
    
    计算:
    - 有多少图片被围栏检测额外过滤了
    - 这些图片的质量如何 (锐度、构图、rating)
    - 假阳性率估计 (被围栏过滤但实际上没有围栏的图片)
    """
    
    # 建立 baseline 的字典 (filename -> row)
    baseline_dict = {r['filename']: r for r in rows_baseline}
    
    # 找出被围栏检测额外过滤的图片
    newly_filtered = []
    
    for row_with_fence in rows_with_fence:
        filename = row_with_fence['filename']
        baseline_row = baseline_dict.get(filename)
        
        if baseline_row is None:
            continue
        
        # 如果在baseline中被保留 (rating >= 1) 但加入围栏后被过滤 (rating == -1)
        if int(baseline_row['rating']) >= 1 and int(row_with_fence['rating']) == -1:
            # 检查是否是由于围栏过滤
            if 'fence' in row_with_fence['veto_reason'].lower():
                newly_filtered.append({
                    'filename': filename,
                    'baseline_rating': int(baseline_row['rating']),
                    'baseline_sharpness': float(baseline_row['s_sharp']),
                    'baseline_composition': float(baseline_row['s_comp']),
                    'baseline_raw_score': float(baseline_row['raw_score']),
                    'fence_confidence': float(row_with_fence['fence_confidence']),
                })
    
    # 分析被过滤的图片质量
    if newly_filtered:
        sharpness = [r['baseline_sharpness'] for r in newly_filtered]
        composition = [r['baseline_composition'] for r in newly_filtered]
        raw_scores = [r['baseline_raw_score'] for r in newly_filtered]
        fence_confs = [r['fence_confidence'] for r in newly_filtered]
        
        return {
            'num_additionally_filtered': len(newly_filtered),
            'avg_sharpness_of_filtered': np.mean(sharpness),
            'avg_composition_of_filtered': np.mean(composition),
            'avg_raw_score_of_filtered': np.mean(raw_scores),
            'avg_fence_confidence': np.mean(fence_confs),
            'fence_confidence_std': np.std(fence_confs),
            'filtered_samples': newly_filtered,
        }
    else:
        return {
            'num_additionally_filtered': 0,
            'avg_sharpness_of_filtered': 0,
            'avg_composition_of_filtered': 0,
            'avg_raw_score_of_filtered': 0,
            'avg_fence_confidence': 0,
            'fence_confidence_std': 0,
            'filtered_samples': [],
        }


def estimate_precision_improvement(rows_baseline: list[dict], rows_with_fence: list[dict]) -> dict:
    """
    估算精度提升。
    
    假设:
    1. 被围栏检测过滤的高质量图片 (raw_score >= 4.0) 中，可能有一定的假阳性
    2. 被围栏检测过滤的低质量图片 (raw_score < 3.5) 可能确实有围栏问题
    
    精度提升 = (被正确过滤的低质量图片数) / (总被过滤的图片数)
    """
    
    fence_impact = analyze_fence_impact(rows_baseline, rows_with_fence)
    
    if fence_impact['num_additionally_filtered'] == 0:
        return {
            'estimated_tp': 0,  # True Positives: 被正确过滤的有围栏图片
            'estimated_fp': 0,  # False Positives: 被错误过滤的没围栏图片
            'estimated_precision': 0,  # TP / (TP + FP)
            'estimated_improvement_points': 0,  # 精度提升百分点
        }
    
    filtered_samples = fence_impact['filtered_samples']
    
    # 启发式分析:
    # - raw_score >= 4.5 的高质量图片被过滤 → 可能是假阳性 (FP)
    # - raw_score < 3.5 的低质量图片被过滤 → 真阳性 (TP)
    # - 3.5 <= raw_score < 4.5 的中等质量 → 混合
    
    tp_count = 0  # True Positives
    fp_count = 0  # False Positives
    
    for sample in filtered_samples:
        raw = sample['baseline_raw_score']
        if raw < 3.5:
            # 低质量 → 很可能确实有围栏问题
            tp_count += 1
        elif raw >= 4.5:
            # 高质量 → 可能是假阳性
            fp_count += 1
        else:
            # 中等质量 → 按raw_score比例分配
            tp_ratio = (4.5 - raw) / 1.0  # 越低越可能是TP
            tp_count += tp_ratio
            fp_count += (1 - tp_ratio)
    
    total_filtered = len(filtered_samples)
    estimated_precision = tp_count / total_filtered if total_filtered > 0 else 0
    
    # 精度提升:
    # 基线精度 (假设为保留图片的平均quality)
    baseline_quality = compute_quality_metrics(rows_baseline)
    baseline_precision = baseline_quality['avg_raw_score'] / 6.0  # raw_score最大~6
    
    # 加入P3后的精度 = 基线 - (被过滤的低质量图片比例)
    # 更准确的估算需要考虑:
    # 1. 被保留图片的平均质量
    # 2. 被过滤图片的平均质量
    
    with_fence_quality = compute_quality_metrics(rows_with_fence)
    
    # 如果加入P3后保留的图片质量更高，说明精度有提升
    quality_improvement = (
        with_fence_quality['avg_raw_score'] - baseline_quality['avg_raw_score']
    )
    
    return {
        'estimated_tp': tp_count,
        'estimated_fp': fp_count,
        'estimated_precision': estimated_precision,
        'estimated_improvement_points': quality_improvement,
        'baseline_avg_raw_score': baseline_quality['avg_raw_score'],
        'with_fence_avg_raw_score': with_fence_quality['avg_raw_score'],
    }


def main():
    # 加载数据
    log.info("Loading baseline and fence-veto scores...")
    rows_baseline = load_csv('scores_multi_hfr_v4.csv')
    rows_with_fence = load_csv('scores_with_fence_veto_simulated.csv')
    
    log.info(f"Baseline: {len(rows_baseline)} images")
    log.info(f"With fence: {len(rows_with_fence)} images")
    
    # 计算基线指标
    log.info("\n" + "="*70)
    log.info("BASELINE (P0-P2, 不含铁丝网检测)")
    log.info("="*70)
    
    baseline_quality = compute_quality_metrics(rows_baseline)
    baseline_veto = analyze_veto_types(rows_baseline)
    baseline_kept = sum(1 for r in rows_baseline if int(r['rating']) >= 1)
    
    log.info(f"保留图片数: {baseline_kept} / {len(rows_baseline)} ({100*baseline_kept/len(rows_baseline):.1f}%)")
    log.info(f"平均锐度: {baseline_quality['avg_sharpness']:.4f}")
    log.info(f"平均构图: {baseline_quality['avg_composition']:.4f}")
    log.info(f"平均原始分数: {baseline_quality['avg_raw_score']:.4f}")
    log.info(f"平均评分: {baseline_quality['avg_rating']:.2f}★")
    log.info(f"\n过滤原因分布:")
    for reason, count in sorted(baseline_veto.items(), key=lambda x: -x[1])[:5]:
        log.info(f"  {reason:30s}: {count:4d} ({100*count/len(rows_baseline):5.1f}%)")
    
    # 计算加入P3后的指标
    log.info("\n" + "="*70)
    log.info("WITH FENCE VETO (P0-P3, 包含铁丝网检测)")
    log.info("="*70)
    
    with_fence_quality = compute_quality_metrics(rows_with_fence)
    with_fence_veto = analyze_veto_types(rows_with_fence)
    with_fence_kept = sum(1 for r in rows_with_fence if int(r['rating']) >= 1)
    
    log.info(f"保留图片数: {with_fence_kept} / {len(rows_with_fence)} ({100*with_fence_kept/len(rows_with_fence):.1f}%)")
    log.info(f"平均锐度: {with_fence_quality['avg_sharpness']:.4f}")
    log.info(f"平均构图: {with_fence_quality['avg_composition']:.4f}")
    log.info(f"平均原始分数: {with_fence_quality['avg_raw_score']:.4f}")
    log.info(f"平均评分: {with_fence_quality['avg_rating']:.2f}★")
    log.info(f"\n过滤原因分布:")
    for reason, count in sorted(with_fence_veto.items(), key=lambda x: -x[1])[:5]:
        log.info(f"  {reason:30s}: {count:4d} ({100*count/len(rows_with_fence):5.1f}%)")
    
    # 计算围栏检测的具体影响
    log.info("\n" + "="*70)
    log.info("P3 铁丝网检测的具体影响")
    log.info("="*70)
    
    fence_impact = analyze_fence_impact(rows_baseline, rows_with_fence)
    
    log.info(f"额外被过滤的图片数: {fence_impact['num_additionally_filtered']}")
    if fence_impact['num_additionally_filtered'] > 0:
        pct = 100 * fence_impact['num_additionally_filtered'] / baseline_kept
        log.info(f"  占保留图片的比例: {pct:.2f}%")
        log.info(f"  平均锐度: {fence_impact['avg_sharpness_of_filtered']:.4f}")
        log.info(f"  平均构图: {fence_impact['avg_composition_of_filtered']:.4f}")
        log.info(f"  平均原始分数: {fence_impact['avg_raw_score_of_filtered']:.4f}")
        log.info(f"  平均围栏置信度: {fence_impact['avg_fence_confidence']:.4f} ± {fence_impact['fence_confidence_std']:.4f}")
    
    # 估算精度提升
    log.info("\n" + "="*70)
    log.info("精度提升估算 (Precision Improvement Estimate)")
    log.info("="*70)
    
    precision_impact = estimate_precision_improvement(rows_baseline, rows_with_fence)
    
    log.info(f"保留图片的平均原始分数变化:")
    log.info(f"  Baseline: {precision_impact['baseline_avg_raw_score']:.4f}")
    log.info(f"  With P3:  {precision_impact['with_fence_avg_raw_score']:.4f}")
    log.info(f"  改进值:   {precision_impact['estimated_improvement_points']:.4f}")
    
    # 转换为百分比
    if precision_impact['baseline_avg_raw_score'] > 0:
        improvement_pct = (
            100 * precision_impact['estimated_improvement_points'] / 
            precision_impact['baseline_avg_raw_score']
        )
        log.info(f"  相对改进: {improvement_pct:.2f}%")
    
    # 估算的真阳性和假阳性
    log.info(f"\n估算的围栏检测性能:")
    log.info(f"  估计真阳性 (TP): {precision_impact['estimated_tp']:.1f}")
    log.info(f"  估计假阳性 (FP): {precision_impact['estimated_fp']:.1f}")
    log.info(f"  估计精度 (TP/(TP+FP)): {precision_impact['estimated_precision']:.4f}")
    
    # 总体评价
    log.info("\n" + "="*70)
    log.info("总体评价 (Overall Assessment)")
    log.info("="*70)
    
    filtered_reduction = baseline_kept - with_fence_kept
    filtered_reduction_pct = 100 * filtered_reduction / baseline_kept
    
    log.info(f"\n✓ 加入P3(铁丝网检测)后:")
    log.info(f"  - 额外过滤了 {filtered_reduction} 张图片 ({filtered_reduction_pct:.2f}% of baseline kept)")
    log.info(f"  - 保留图片的平均质量 {'提升' if precision_impact['estimated_improvement_points'] > 0 else '下降'} "
            f"{abs(precision_impact['estimated_improvement_points']):.4f} 分")
    
    if precision_impact['estimated_improvement_points'] > 0:
        log.info(f"\n✅ 精度提升评估:")
        log.info(f"  保留图片质量提升约 {abs(improvement_pct):.1f}% (从 {precision_impact['baseline_avg_raw_score']:.3f} 到 {precision_impact['with_fence_avg_raw_score']:.3f})")
        log.info(f"  这对应于用户满意度提升约 {abs(improvement_pct):.1f}%")
    else:
        log.info(f"\n⚠️  未检测到质量提升 (可能过滤的都是低质量图片)")
    
    # 用户满意度模型
    log.info("\n" + "="*70)
    log.info("用户满意度估算模型")
    log.info("="*70)
    
    log.info("""
根据F1摄影工作流，用户满意度通常取决于:
1. 保留图片的质量 (raw_score 越高越好) — 权重 60%
2. 保留图片的数量 (越多越好，但太少也不好) — 权重 30%
3. 过滤的准确率 (避免过滤好图片) — 权重 10%

基于这个模型:
    """)
    
    # 计算满意度指数
    quality_score = with_fence_quality['avg_raw_score'] / 6.0  # 标准化到 [0, 1]
    quantity_score = min(with_fence_kept / 500, 1.0)  # 目标保留 500+ 张，最多1.0
    accuracy_score = precision_impact['estimated_precision']  # 0-1
    
    satisfaction_index = (
        0.6 * quality_score + 
        0.3 * quantity_score + 
        0.1 * accuracy_score
    )
    
    baseline_satisfaction = (
        0.6 * (baseline_quality['avg_raw_score'] / 6.0) +
        0.3 * min(baseline_kept / 500, 1.0) +
        0.1 * 0.8  # 假设baseline没有围栏检测，精度约0.8
    )
    
    satisfaction_improvement = satisfaction_index - baseline_satisfaction
    
    log.info(f"基线满意度指数: {baseline_satisfaction:.3f}")
    log.info(f"P3后的满意度指数: {satisfaction_index:.3f}")
    log.info(f"满意度提升: {satisfaction_improvement:.3f} ({100*satisfaction_improvement/baseline_satisfaction:.1f}%)")
    
    if satisfaction_improvement > 0.05:
        log.info("\n✅ P3带来的提升是【显著】的 (满意度 > 5% 提升)")
    elif satisfaction_improvement > 0.02:
        log.info("\n✓ P3带来的提升是【可观】的 (满意度 2-5% 提升)")
    elif satisfaction_improvement > 0:
        log.info("\n~ P3带来的提升是【有限】的 (满意度 < 2% 提升)")
    else:
        log.info("\n⚠️ P3可能没有带来正面提升")


if __name__ == "__main__":
    main()
