#!/usr/bin/env python3
"""
ROI Refine vs Original 小物体检测对比可视化脚本

功能：
1. 加载指定 checkpoint，在 AG_val 上推理
2. 一次前向同时获取 ROI refine 和原始（non-ROI）两套预测
3. 针对 5 个最小类别 (dish, light, medicine, groceries, doorknob)
   逐帧计算 recall 和 AP
4. 找出 ROI 正确但原始预测错误的小物体帧（体现 ROI refine 效果）
5. 按 AP + recall 排序，保留高分对比图
6. 输出目录: OUTPUT_DIR/roi_comparison/

用法:
  cd /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
  python3 tools/eval_roi_comparison.py \
    --config-file configs/speaq_actiongenome_minimal.yaml \
    --model-weights z_outputs/train_fulltask_v3_8gpu_bs96_80k/model_0079999.pth \
    --output-dir z_outputs/train_fulltask_v3_8gpu_bs96_80k/roi_comparison \
    --num-images -1 \
    --top-k-vis 50 \
    --small-area-thresh 0.001
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import default_setup
from detectron2.structures import Boxes, Instances, pairwise_iou

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.data.dataset_mapper import DetrDatasetMapper
from SpeaQ.data.tools.utils import register_datasets
from SpeaQ.engine import JointTransformerTrainer
from SpeaQ.modeling import Detr  # noqa: F401
from visualization.visualizer import SceneGraphVisualizer

# 5 个最小 / 长尾目标类别
SMALL_CLASS_NAMES = ["dish", "light", "medicine", "groceries", "doorknob"]


def build_cfg(args: argparse.Namespace):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    if cfg.is_frozen():
        cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.model_weights:
        cfg.MODEL.WEIGHTS = args.model_weights
    # 确保 ROI_REFINE 启用
    cfg.MODEL.ROI_REFINE.ENABLED = True
    cfg.MODEL.ROI_REFINE.EVAL_DUAL = False
    cfg.freeze()
    register_datasets(cfg)
    if not hasattr(args, "num_gpus"):
        args.num_gpus = 1
    if not hasattr(args, "num_machines"):
        args.num_machines = 1
    if not hasattr(args, "machine_rank"):
        args.machine_rank = 0
    default_setup(cfg, args)
    return cfg


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)


def compute_iou(box1_xyxy, box2_xyxy):
    """Compute pairwise IoU between two sets of boxes in xyxy format."""
    x1 = torch.max(box1_xyxy[:, None, 0], box2_xyxy[None, :, 0])
    y1 = torch.max(box1_xyxy[:, None, 1], box2_xyxy[None, :, 1])
    x2 = torch.min(box1_xyxy[:, None, 2], box2_xyxy[None, :, 2])
    y2 = torch.min(box1_xyxy[:, None, 3], box2_xyxy[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1_xyxy[:, 2] - box1_xyxy[:, 0]) * (box1_xyxy[:, 3] - box1_xyxy[:, 1])
    area2 = (box2_xyxy[:, 2] - box2_xyxy[:, 0]) * (box2_xyxy[:, 3] - box2_xyxy[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def compute_per_image_ap_recall(
    gt_boxes_xyxy: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes_xyxy: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    target_class_indices: List[int],
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """
    计算单张图片上指定类别的 recall 和 AP。
    Recall: GT 中被匹配到的比例 (greedy matching, score desc)
    AP: 简化版 — 每类中匹配到的 pred 占该类所有 pred (above thresh+correct class) 的比例
    """
    result = {"recall": 0.0, "ap": 0.0, "n_gt": 0, "n_matched": 0, "n_pred_correct": 0}

    if len(gt_boxes_xyxy) == 0 or len(pred_boxes_xyxy) == 0:
        return result

    # Filter to target classes only
    gt_target_mask = torch.isin(gt_labels, torch.tensor(target_class_indices))
    if not gt_target_mask.any():
        return result

    gt_boxes_target = gt_boxes_xyxy[gt_target_mask]
    gt_labels_target = gt_labels[gt_target_mask]
    result["n_gt"] = int(len(gt_boxes_target))

    # Sort predictions by score desc
    score_order = torch.argsort(pred_scores, descending=True)
    pred_boxes_sorted = pred_boxes_xyxy[score_order]
    pred_labels_sorted = pred_labels[score_order]

    used_pred = set()
    matched = 0
    for gt_i in range(len(gt_boxes_target)):
        gt_cls = int(gt_labels_target[gt_i])
        gt_box = gt_boxes_target[gt_i : gt_i + 1]

        for p_i in range(len(pred_boxes_sorted)):
            if p_i in used_pred:
                continue
            pred_cls = int(pred_labels_sorted[p_i])
            if pred_cls != gt_cls:
                continue
            iou = compute_iou(gt_box, pred_boxes_sorted[p_i : p_i + 1])[0, 0]
            if iou >= iou_thresh:
                matched += 1
                used_pred.add(p_i)
                break

    result["n_matched"] = matched
    result["recall"] = float(matched) / float(result["n_gt"]) if result["n_gt"] > 0 else 0.0

    # Simplified AP: proportion of target-class predictions that matched a GT
    n_pred_target = int(torch.isin(pred_labels_sorted, torch.tensor(target_class_indices)).sum())
    if n_pred_target > 0:
        result["ap"] = float(matched) / float(n_pred_target)
        result["n_pred_correct"] = matched

    return result


def extract_predictions_from_instances(
    instances: Instances, image_size: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """从 Instances 提取归一化到 [0,1] 的 xyxy boxes, labels, scores."""
    if instances is None or len(instances) == 0:
        return (
            torch.zeros((0, 4)),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,)),
        )
    boxes_xyxy = instances.pred_boxes.tensor.clone()
    # Normalize to [0,1]
    h, w = image_size
    boxes_xyxy[:, [0, 2]] /= float(w)
    boxes_xyxy[:, [1, 3]] /= float(h)
    labels = instances.pred_classes.clone()
    scores = instances.scores.clone()
    return boxes_xyxy, labels, scores


def run_dual_inference(model, cfg, dataset_name: str, small_area_thresh: float, small_class_indices: List[int]):
    """
    对数据集逐帧推理，同时返回 ROI 和原始两套预测的逐帧指标。
    通过修改输出字典中的 logits 来模拟 ROI/非ROI 两条路径。
    """
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    mapper = DetrDatasetMapper(cfg, False)
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    metadata = MetadataCatalog.get(dataset_name)
    thing_classes = list(metadata.thing_classes)

    model.eval()
    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # 前向传播 — 获取包含 ROI logits 的原始输出
            outputs = model(batch)

            for b_i, (input_per_image, output_per_image) in enumerate(zip(batch, outputs)):
                file_name = input_per_image.get("file_name", "")
                image_id = input_per_image.get("image_id", b_i)

                # 获取图像尺寸
                height = input_per_image.get("height")
                width = input_per_image.get("width")
                if height is None or width is None:
                    if file_name and os.path.exists(file_name):
                        with Image.open(file_name) as img:
                            width, height = img.size
                    else:
                        continue

                # --- GT 数据 ---
                gt_instances = input_per_image.get("instances")
                if gt_instances is None or not hasattr(gt_instances, "gt_boxes"):
                    continue

                gt_boxes_t = gt_instances.gt_boxes.tensor.clone()
                # GT boxes 在 transformed image space, 需要转回原始图像
                tfm_h, tfm_w = gt_instances.image_size
                scale_x = float(width) / float(tfm_w)
                scale_y = float(height) / float(tfm_h)
                gt_boxes_xyxy = gt_boxes_t.clone()
                gt_boxes_xyxy[:, [0, 2]] *= scale_x
                gt_boxes_xyxy[:, [1, 3]] *= scale_y
                gt_labels = gt_instances.gt_classes.clone()

                # Normalize GT to [0,1]
                gt_boxes_norm = gt_boxes_xyxy.clone()
                gt_boxes_norm[:, [0, 2]] /= float(width)
                gt_boxes_norm[:, [1, 3]] /= float(height)

                # --- 获取模型原始输出中的关系预测 ---
                rel_sub_logits = output_per_image.get("relation_subject_logits")
                rel_obj_logits = output_per_image.get("relation_object_logits")
                rel_logits = output_per_image.get("relation_logits")
                rel_sub_boxes = output_per_image.get("relation_subject_boxes")
                rel_obj_boxes = output_per_image.get("relation_object_boxes")

                # 检查是否有 ROI refined logits
                rel_sub_logits_roi = output_per_image.get("relation_subject_logits_roi")
                rel_obj_logits_roi = output_per_image.get("relation_object_logits_roi")
                has_roi = rel_sub_logits_roi is not None and rel_obj_logits_roi is not None

                if rel_sub_logits is None or rel_obj_logits is None:
                    continue

                # 取最后一层
                if rel_sub_logits.dim() == 3:
                    rel_sub_logits = rel_sub_logits[-1]
                if rel_obj_logits.dim() == 3:
                    rel_obj_logits = rel_obj_logits[-1]
                if rel_logits is not None and rel_logits.dim() == 3:
                    rel_logits = rel_logits[-1]
                if rel_sub_boxes is not None and rel_sub_boxes.dim() == 3:
                    rel_sub_boxes = rel_sub_boxes[-1]
                if rel_obj_boxes is not None and rel_obj_boxes.dim() == 3:
                    rel_obj_boxes = rel_obj_boxes[-1]

                def build_predictions(sub_logits, obj_logits, sub_boxes, obj_boxes):
                    """从 subject/object logits 构建统一的检测预测"""
                    sub_prob = F.softmax(sub_logits, dim=-1)
                    obj_prob = F.softmax(obj_logits, dim=-1)

                    sub_score, sub_label = sub_prob[..., :-1].max(-1)
                    obj_score, obj_label = obj_prob[..., :-1].max(-1)

                    all_boxes = []
                    all_labels = []
                    all_scores = []

                    # Subject predictions
                    for i in range(len(sub_label)):
                        bx = sub_boxes[i] if sub_boxes is not None else torch.zeros(4)
                        all_boxes.append(box_cxcywh_to_xyxy(bx.unsqueeze(0))[0])
                        all_labels.append(sub_label[i])
                        all_scores.append(sub_score[i])

                    # Object predictions
                    for i in range(len(obj_label)):
                        bx = obj_boxes[i] if obj_boxes is not None else torch.zeros(4)
                        all_boxes.append(box_cxcywh_to_xyxy(bx.unsqueeze(0))[0])
                        all_labels.append(obj_label[i])
                        all_scores.append(obj_score[i])

                    if all_boxes:
                        return (
                            torch.stack(all_boxes),
                            torch.stack(all_labels),
                            torch.stack(all_scores),
                        )
                    else:
                        return (
                            torch.zeros((0, 4)),
                            torch.zeros((0,), dtype=torch.long),
                            torch.zeros((0,)),
                        )

                # --- 原始预测（无 ROI） ---
                pred_boxes_norm_orig, pred_labels_orig, pred_scores_orig = build_predictions(
                    rel_sub_logits, rel_obj_logits, rel_sub_boxes, rel_obj_boxes
                )

                metrics_orig = compute_per_image_ap_recall(
                    gt_boxes_norm, gt_labels,
                    pred_boxes_norm_orig, pred_labels_orig, pred_scores_orig,
                    small_class_indices, iou_thresh=0.5,
                )

                # --- ROI 预测 ---
                metrics_roi = dict(metrics_orig)
                pred_boxes_norm_roi = pred_boxes_norm_orig
                pred_labels_roi = pred_labels_orig
                pred_scores_roi = pred_scores_orig

                if has_roi:
                    # 取最后一层 ROI logits
                    sub_logits_roi = rel_sub_logits_roi
                    obj_logits_roi = rel_obj_logits_roi
                    if sub_logits_roi.dim() == 3:
                        sub_logits_roi = sub_logits_roi[-1]
                    if obj_logits_roi.dim() == 3:
                        obj_logits_roi = obj_logits_roi[-1]

                    # 计算面积以确定哪些预测使用 ROI logits
                    sub_area = rel_sub_boxes[..., 2].clamp(min=0) * rel_sub_boxes[..., 3].clamp(min=0)
                    obj_area = rel_obj_boxes[..., 2].clamp(min=0) * rel_obj_boxes[..., 3].clamp(min=0)
                    sub_small = sub_area < small_area_thresh
                    obj_small = obj_area < small_area_thresh

                    # 对小目标替换为 ROI logits
                    mixed_sub_logits = rel_sub_logits.clone()
                    mixed_obj_logits = rel_obj_logits.clone()
                    mixed_sub_logits[sub_small] = sub_logits_roi[sub_small]
                    mixed_obj_logits[obj_small] = obj_logits_roi[obj_small]

                    pred_boxes_norm_roi, pred_labels_roi, pred_scores_roi = build_predictions(
                        mixed_sub_logits, mixed_obj_logits, rel_sub_boxes, rel_obj_boxes
                    )

                    metrics_roi = compute_per_image_ap_recall(
                        gt_boxes_norm, gt_labels,
                        pred_boxes_norm_roi, pred_labels_roi, pred_scores_roi,
                        small_class_indices, iou_thresh=0.5,
                    )

                # 检查哪些 GT 目标类别存在于该帧
                gt_small_mask = torch.isin(gt_labels, torch.tensor(small_class_indices))
                gt_small_classes = gt_labels[gt_small_mask].unique().tolist()
                gt_small_names = [thing_classes[c] for c in gt_small_classes]

                # ROI correct but original wrong: 判断标准
                #   ROI recall > original recall (ROI 匹配到更多小物体 GT)
                roi_better = metrics_roi["recall"] > metrics_orig["recall"]

                combined_score = metrics_roi["recall"] + metrics_roi["ap"]

                all_results.append({
                    "image_id": image_id,
                    "file_name": file_name,
                    "image_size": (int(height), int(width)),
                    "gt_small_classes": gt_small_names,
                    "gt_small_class_ids": gt_small_classes,
                    "n_gt_small": metrics_roi["n_gt"],
                    "n_matched_orig": metrics_orig["n_matched"],
                    "n_matched_roi": metrics_roi["n_matched"],
                    "recall_orig": float(metrics_orig["recall"]),
                    "recall_roi": float(metrics_roi["recall"]),
                    "ap_orig": float(metrics_orig["ap"]),
                    "ap_roi": float(metrics_roi["ap"]),
                    "roi_better": roi_better,
                    "combined_score": float(combined_score),
                    "pred_boxes_orig": pred_boxes_norm_orig.tolist(),
                    "pred_labels_orig": pred_labels_orig.tolist(),
                    "pred_scores_orig": pred_scores_orig.tolist(),
                    "pred_boxes_roi": pred_boxes_norm_roi.tolist(),
                    "pred_labels_roi": pred_labels_roi.tolist(),
                    "pred_scores_roi": pred_scores_roi.tolist(),
                    "gt_boxes_norm": gt_boxes_norm.tolist(),
                    "gt_labels": gt_labels.tolist(),
                })

            if (batch_idx + 1) % 50 == 0:
                print(f"[eval] processed {batch_idx + 1} batches, {len(all_results)} images so far...")

    return all_results, thing_classes


def save_comparison_visualization(
    file_name: str,
    image_size: Tuple[int, int],
    gt_boxes_norm: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes_orig: torch.Tensor,
    pred_labels_orig: torch.Tensor,
    pred_scores_orig: torch.Tensor,
    pred_boxes_roi: torch.Tensor,
    pred_labels_roi: torch.Tensor,
    pred_scores_roi: torch.Tensor,
    metadata,
    output_dir: str,
    index: int,
    score: float,
):
    """保存一组对比可视化：原始 vs ROI refine。"""
    os.makedirs(output_dir, exist_ok=True)
    vis_orig = SceneGraphVisualizer(metadata, output_dir=os.path.join(output_dir, "original"))
    vis_roi = SceneGraphVisualizer(metadata, output_dir=os.path.join(output_dir, "roi_refine"))

    h, w = image_size
    # Denormalize boxes
    def denorm(boxes_norm):
        boxes = boxes_norm.clone()
        boxes[:, [0, 2]] *= float(w)
        boxes[:, [1, 3]] *= float(h)
        return boxes

    gt_boxes_denorm = denorm(gt_boxes_norm)
    pred_boxes_orig_denorm = denorm(pred_boxes_orig)
    pred_boxes_roi_denorm = denorm(pred_boxes_roi)

    stem = os.path.splitext(os.path.basename(file_name))[0]
    vis_name = f"{index:04d}_score{score:.3f}_{stem}"

    # Original prediction visualization
    if len(pred_boxes_orig) > 0:
        vis_orig.visualize_scene_graph(
            file_name,
            pred_boxes_orig_denorm.numpy(),
            pred_labels_orig.numpy(),
            np.zeros((0, 3), dtype=np.int64),
            scores=pred_scores_orig.numpy(),
            rel_scores=None,
            output_name=vis_name,
            top_k_relations=0,
            score_threshold=0.0,
        )

    # ROI refined prediction visualization
    if len(pred_boxes_roi) > 0:
        vis_roi.visualize_scene_graph(
            file_name,
            pred_boxes_roi_denorm.numpy(),
            pred_labels_roi.numpy(),
            np.zeros((0, 3), dtype=np.int64),
            scores=pred_scores_roi.numpy(),
            rel_scores=None,
            output_name=vis_name,
            top_k_relations=0,
            score_threshold=0.0,
        )


def main():
    parser = argparse.ArgumentParser(description="ROI Refine vs Original small-object comparison")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="AG_val")
    parser.add_argument("--num-images", type=int, default=-1)
    parser.add_argument("--top-k-vis", type=int, default=50, help="最多保存多少组对比图")
    parser.add_argument("--small-area-thresh", type=float, default=0.001)
    parser.add_argument("--min-score", type=float, default=0.0, help="最低 combined_score 阈值")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    print(f"Config: {args.config_file}")
    print(f"Checkpoint: {args.model_weights}")
    print(f"Output: {args.output_dir}")
    print(f"Small area threshold: {args.small_area_thresh}")
    print(f"Top-K vis: {args.top_k_vis}")

    cfg = build_cfg(args)
    metadata = MetadataCatalog.get(args.dataset_name)
    thing_classes = list(metadata.thing_classes)

    # 获取 5 个小目标类别的 ID
    small_class_indices = []
    for name in SMALL_CLASS_NAMES:
        if name in thing_classes:
            small_class_indices.append(thing_classes.index(name))
    print(f"Target small classes: {SMALL_CLASS_NAMES} → indices: {small_class_indices}")

    # 构建模型
    model = JointTransformerTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    model.to(cfg.MODEL.DEVICE)

    print("Starting dual inference (ROI + original per frame)...")
    t_start = time.time()
    all_results, _ = run_dual_inference(
        model, cfg, args.dataset_name, args.small_area_thresh, small_class_indices
    )
    t_elapsed = time.time() - t_start
    print(f"Inference done: {len(all_results)} images in {t_elapsed:.1f}s ({t_elapsed/len(all_results)*1000:.1f} ms/img)")

    # 保存全量结果
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "per_image_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Full per-image results saved to: {results_path}")

    # 筛选：有小物体 GT 的帧
    has_small = [r for r in all_results if r["n_gt_small"] > 0]
    print(f"Frames with small-object GT: {len(has_small)} / {len(all_results)}")

    # 筛选：ROI better (ROI 匹配数 > 原始匹配数)
    roi_better_results = [r for r in has_small if r["roi_better"]]
    print(f"Frames where ROI is better than original: {len(roi_better_results)}")

    # 筛选：ROI 正确 (recall > 0) 且 原始错误 (recall == 0)
    roi_correct_orig_wrong = [
        r for r in has_small
        if r["recall_roi"] > 0 and r["recall_orig"] == 0
    ]
    print(f"Frames where ROI correct but original wrong: {len(roi_correct_orig_wrong)}")

    # 按 combined_score 排序
    if roi_correct_orig_wrong:
        roi_correct_orig_wrong.sort(key=lambda r: r["combined_score"], reverse=True)
    elif roi_better_results:
        roi_better_results.sort(key=lambda r: r["combined_score"], reverse=True)
        roi_correct_orig_wrong = roi_better_results
    else:
        has_small.sort(key=lambda r: r["combined_score"], reverse=True)
        roi_correct_orig_wrong = has_small

    # 保存 top-K 可视化
    top_k = min(args.top_k_vis, len(roi_correct_orig_wrong))
    print(f"\nSaving top {top_k} comparison visualizations...")

    # 保存筛选后的高分结果 JSON
    top_results = roi_correct_orig_wrong[:top_k]
    top_json_path = os.path.join(args.output_dir, "top_comparison_results.json")
    with open(top_json_path, "w") as f:
        json.dump(top_results, f, indent=2, ensure_ascii=False)

    # 打印排名表
    print(f"\n{'Rank':<6} {'Score':<10} {'R_orig':<10} {'R_roi':<10} {'AP_orig':<10} {'AP_roi':<10} {'GT classes':<30} {'File'}")
    print("-" * 120)
    for i, r in enumerate(top_results):
        fname = os.path.basename(r["file_name"]) if r["file_name"] else "N/A"
        gt_cls_str = ",".join(r["gt_small_classes"])
        print(f"{i+1:<6} {r['combined_score']:<10.4f} {r['recall_orig']:<10.4f} {r['recall_roi']:<10.4f} "
              f"{r['ap_orig']:<10.4f} {r['ap_roi']:<10.4f} {gt_cls_str:<30} {fname}")

    # 保存对比可视化
    vis_dir = os.path.join(args.output_dir, "comparison_vis")
    for i, r in enumerate(top_results):
        file_name = r["file_name"]
        if not file_name or not os.path.exists(file_name):
            continue

        gt_boxes_norm = torch.tensor(r["gt_boxes_norm"])
        gt_labels = torch.tensor(r["gt_labels"])
        pred_boxes_orig = torch.tensor(r["pred_boxes_orig"])
        pred_labels_orig = torch.tensor(r["pred_labels_orig"], dtype=torch.long)
        pred_scores_orig = torch.tensor(r["pred_scores_orig"])
        pred_boxes_roi = torch.tensor(r["pred_boxes_roi"])
        pred_labels_roi = torch.tensor(r["pred_labels_roi"], dtype=torch.long)
        pred_scores_roi = torch.tensor(r["pred_scores_roi"])

        save_comparison_visualization(
            file_name, tuple(r["image_size"]),
            gt_boxes_norm, gt_labels,
            pred_boxes_orig, pred_labels_orig, pred_scores_orig,
            pred_boxes_roi, pred_labels_roi, pred_scores_roi,
            metadata,
            vis_dir,
            i,
            r["combined_score"],
        )

    print(f"\nComparison visualizations saved to: {vis_dir}")
    print(f"Top results JSON: {top_json_path}")

    # 统计信息
    print(f"\n===== Summary =====")
    print(f"Total images processed: {len(all_results)}")
    print(f"Images with small-object GT: {len(has_small)}")
    print(f"ROI better than original: {len(roi_better_results) if 'roi_better_results' in dir() else 0}")
    print(f"ROI correct, original wrong: {len([r for r in has_small if r['recall_roi'] > 0 and r['recall_orig'] == 0])}")

    avg_recall_orig = np.mean([r["recall_orig"] for r in has_small]) if has_small else 0
    avg_recall_roi = np.mean([r["recall_roi"] for r in has_small]) if has_small else 0
    print(f"Avg recall (original) on small-object frames: {avg_recall_orig:.4f}")
    print(f"Avg recall (ROI refined) on small-object frames: {avg_recall_roi:.4f}")


if __name__ == "__main__":
    main()
