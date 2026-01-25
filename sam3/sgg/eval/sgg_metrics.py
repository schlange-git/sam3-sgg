"""
Scene Graph Generation Evaluation Metrics
参照: https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch

实现标准 SGG 评测指标：
- Recall@K (R@K): 前 K 个预测中正确关系的比例
- Mean Recall@K (mR@K): 每个 predicate 类别的平均 recall
"""
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict


def compute_recall_at_k(
    pred_triplets: List[Tuple[int, int, int]],  # [(s, o, p), ...]
    gt_triplets: List[Tuple[int, int, int]],    # [(s, o, p), ...]
    k: int = 20,
) -> float:
    """
    计算 Recall@K
    
    Args:
        pred_triplets: 预测的关系三元组列表（按置信度排序，前 K 个）
        gt_triplets: GT 关系三元组列表
        k: Top-K
        
    Returns:
        Recall@K (0.0 ~ 1.0)
    """
    if len(gt_triplets) == 0:
        return 0.0 if len(pred_triplets) > 0 else 1.0
    
    # 取前 K 个预测
    pred_topk = pred_triplets[:k]
    
    # 转换为 set 以便快速查找
    gt_set = set(gt_triplets)
    
    # 计算正确预测的数量
    correct = sum(1 for t in pred_topk if t in gt_set)
    
    return correct / len(gt_triplets)


def compute_mean_recall_at_k(
    pred_triplets: List[Tuple[int, int, int]],  # [(s, o, p), ...]
    pred_scores: List[float],                    # 对应的置信度分数
    gt_triplets: List[Tuple[int, int, int]],    # [(s, o, p), ...]
    num_predicates: int,                        # 包括 background
    k: int = 20,
) -> float:
    """
    计算 Mean Recall@K
    
    对每个 predicate 类别分别计算 Recall@K，然后取平均
    
    Args:
        pred_triplets: 预测的关系三元组列表
        pred_scores: 对应的置信度分数（用于排序）
        gt_triplets: GT 关系三元组列表
        num_predicates: Predicate 类别数（包括 background）
        k: Top-K
        
    Returns:
        Mean Recall@K (0.0 ~ 1.0)
    """
    if len(gt_triplets) == 0:
        return 0.0
    
    # 按 predicate 分组 GT
    gt_by_pred = defaultdict(list)
    for s, o, p in gt_triplets:
        if p > 0:  # 排除 background
            gt_by_pred[p].append((s, o, p))
    
    # 如果没有正样本，返回 0
    if len(gt_by_pred) == 0:
        return 0.0
    
    # 按 predicate 分组预测（带分数）
    pred_with_scores = list(zip(pred_triplets, pred_scores))
    pred_by_pred = defaultdict(list)
    for (s, o, p), score in pred_with_scores:
        if p > 0:  # 排除 background
            pred_by_pred[p].append(((s, o, p), score))
    
    # 对每个 predicate 类别计算 Recall@K
    recalls = []
    for pred_id in range(1, num_predicates):  # 从 1 开始，排除 background
        if pred_id not in gt_by_pred:
            # 如果 GT 中没有这个 predicate，跳过（不参与平均）
            continue
        
        gt_triplets_p = gt_by_pred[pred_id]
        
        # 获取该 predicate 的预测，按分数排序
        if pred_id in pred_by_pred:
            pred_with_scores_p = pred_by_pred[pred_id]
            pred_with_scores_p.sort(key=lambda x: x[1], reverse=True)  # 按分数降序
            pred_triplets_p = [t for t, _ in pred_with_scores_p]
        else:
            pred_triplets_p = []
        
        # 计算该 predicate 的 Recall@K
        recall_p = compute_recall_at_k(pred_triplets_p, gt_triplets_p, k)
        recalls.append(recall_p)
    
    # 返回平均 recall
    if len(recalls) == 0:
        return 0.0
    return np.mean(recalls)


def evaluate_predcls_batch(
    pred_triplets_list: List[List[Tuple[int, int, int]]],  # 每个图像的预测
    pred_scores_list: List[List[float]],                    # 每个图像的预测分数
    gt_triplets_list: List[List[Tuple[int, int, int]]],    # 每个图像的 GT
    num_predicates: int,
    k_list: List[int] = [20, 50, 100],
) -> Dict[str, float]:
    """
    批量评测 PredCls 任务
    
    Args:
        pred_triplets_list: 每个图像的预测三元组列表
        pred_scores_list: 每个图像的预测分数列表
        gt_triplets_list: 每个图像的 GT 三元组列表
        num_predicates: Predicate 类别数
        k_list: K 值列表
        
    Returns:
        评测结果字典，包含 R@K 和 mR@K
    """
    assert len(pred_triplets_list) == len(gt_triplets_list)
    assert len(pred_triplets_list) == len(pred_scores_list)
    
    results = {}
    
    # 对每个图像，按分数对预测排序
    sorted_pred_list = []
    sorted_scores_list = []
    for pred_triplets, pred_scores in zip(pred_triplets_list, pred_scores_list):
        # 按分数降序排序
        combined = list(zip(pred_triplets, pred_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        sorted_pred, sorted_scores = zip(*combined) if combined else ([], [])
        sorted_pred_list.append(list(sorted_pred))
        sorted_scores_list.append(list(sorted_scores))
    
    # 计算每个 K 的指标
    for k in k_list:
        # Recall@K
        recalls = []
        for pred_triplets, gt_triplets in zip(sorted_pred_list, gt_triplets_list):
            recall = compute_recall_at_k(pred_triplets, gt_triplets, k)
            recalls.append(recall)
        results[f"R@{k}"] = np.mean(recalls) * 100.0  # 转换为百分比
        
        # Mean Recall@K
        mrecalls = []
        for pred_triplets, pred_scores, gt_triplets in zip(
            sorted_pred_list, sorted_scores_list, gt_triplets_list
        ):
            mrecall = compute_mean_recall_at_k(
                pred_triplets, pred_scores, gt_triplets, num_predicates, k
            )
            mrecalls.append(mrecall)
        results[f"mR@{k}"] = np.mean(mrecalls) * 100.0  # 转换为百分比
    
    return results

