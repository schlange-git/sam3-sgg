#!/usr/bin/env python3
"""
ROI Refine vs Original 小物体检测对比可视化脚本 (EVAL_DUAL 版本)

功能：
1. 启用 EVAL_DUAL 模式，一次前向获取 ROI refine 和原始两套预测
2. 针对 5 个最小类别逐帧计算 recall 和 AP
3. 找出 ROI 正确但原始预测错误的小物体帧
4. 按 AP + recall 排序，保存对比图（含三元组关系）
"""

import argparse
import json
import math
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
from detectron2.structures import Boxes, Instances

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.data.dataset_mapper import DetrDatasetMapper
from SpeaQ.data.tools.utils import register_datasets
from SpeaQ.engine import JointTransformerTrainer
from SpeaQ.modeling import Detr  # noqa: F401
from visualization.visualizer import SceneGraphVisualizer

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
    cfg.MODEL.ROI_REFINE.ENABLED = True
    cfg.MODEL.ROI_REFINE.EVAL_DUAL = True
    cfg.MODEL.ROI_REFINE.SMALL_AREA_THRESH = args.small_area_thresh
    cfg.MODEL.DETR.PERSON_SCORE_SCALE = args.person_score_scale
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


def compute_iou(box1_xyxy: torch.Tensor, box2_xyxy: torch.Tensor) -> torch.Tensor:
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
    result = {"recall": 0.0, "ap": 0.0, "n_gt": 0, "n_matched": 0}

    gt_boxes_xyxy = gt_boxes_xyxy.cpu()
    gt_labels = gt_labels.cpu()
    pred_boxes_xyxy = pred_boxes_xyxy.cpu()
    pred_labels = pred_labels.cpu()
    pred_scores = pred_scores.cpu()

    if len(gt_boxes_xyxy) == 0 or len(pred_boxes_xyxy) == 0:
        return result

    gt_target_mask = torch.isin(gt_labels, torch.tensor(target_class_indices))
    if not gt_target_mask.any():
        return result

    gt_boxes_target = gt_boxes_xyxy[gt_target_mask]
    gt_labels_target = gt_labels[gt_target_mask]
    result["n_gt"] = int(len(gt_boxes_target))

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
            if int(pred_labels_sorted[p_i]) != gt_cls:
                continue
            iou = compute_iou(gt_box, pred_boxes_sorted[p_i : p_i + 1])[0, 0]
            if iou >= iou_thresh:
                matched += 1
                used_pred.add(p_i)
                break

    result["n_matched"] = matched
    result["recall"] = float(matched) / float(result["n_gt"]) if result["n_gt"] > 0 else 0.0
    n_pred_target = int(torch.isin(pred_labels_sorted, torch.tensor(target_class_indices)).sum())
    if n_pred_target > 0:
        result["ap"] = float(matched) / float(n_pred_target)

    return result


def extract_preds_from_instances(instances: Instances, image_size: Tuple[int, int],
                                 score_thresh: float = 0.0):
    """从 Instances 提取归一化 boxes/labels/scores，返回 (boxes, labels, scores, keep_mask).
    keep_mask 为 None 表示无过滤（全部保留），否则为 bool tensor 标记哪些原始索引被保留。"""
    if instances is None or len(instances) == 0:
        return (
            torch.zeros((0, 4)),
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0,)),
            torch.zeros((0,), dtype=torch.bool),
        )
    boxes_xyxy = instances.pred_boxes.tensor.clone()
    h, w = image_size
    boxes_norm = boxes_xyxy.clone()
    boxes_norm[:, [0, 2]] /= float(w)
    boxes_norm[:, [1, 3]] /= float(h)
    labels = instances.pred_classes.clone()
    scores = instances.scores.clone()
    n_total = len(scores)

    if score_thresh > 0:
        keep = scores >= score_thresh
        if keep.sum() == 0:
            max_idx = int(scores.argmax())
            keep[max_idx] = True
        boxes_norm = boxes_norm[keep]
        labels = labels[keep]
        scores = scores[keep]
        return boxes_norm, labels, scores, keep
    else:
        keep_all = torch.ones(n_total, dtype=torch.bool)
        return boxes_norm, labels, scores, keep_all


def _remap_relations(rel_pair_idxs, rel_labels, rel_scores, keep_mask):
    """根据 keep_mask 重映射关系索引，过滤掉包含被移除 box 的关系。"""
    n_old = len(keep_mask)
    old_to_new = torch.full((n_old,), -1, dtype=torch.long)
    kept_indices = torch.where(keep_mask)[0]
    for new_idx, old_idx in enumerate(kept_indices.tolist()):
        old_to_new[old_idx] = new_idx

    if rel_pair_idxs is None or len(rel_pair_idxs) == 0:
        return (np.zeros((0, 3), dtype=np.int64),
                np.zeros((0,), dtype=np.float32) if rel_scores is not None else None)

    pair = rel_pair_idxs.cpu().numpy() if torch.is_tensor(rel_pair_idxs) else np.asarray(rel_pair_idxs)
    lbl = rel_labels.cpu().numpy() if torch.is_tensor(rel_labels) else np.asarray(rel_labels)
    if rel_scores is not None:
        sc = rel_scores.cpu().numpy() if torch.is_tensor(rel_scores) else np.asarray(rel_scores)
        if sc.ndim == 2:
            sc = sc.max(axis=1)
    else:
        sc = None

    new_relations = []
    new_rel_scores = []
    for k in range(len(pair)):
        s_old = int(pair[k, 0])
        o_old = int(pair[k, 1])
        if s_old >= n_old or o_old >= n_old:
            continue
        s_new = int(old_to_new[s_old])
        o_new = int(old_to_new[o_old])
        if s_new >= 0 and o_new >= 0:
            new_relations.append([s_new, o_new, int(lbl[k])])
            if sc is not None:
                new_rel_scores.append(float(sc[k]))

    if new_relations:
        rel_arr = np.asarray(new_relations, dtype=np.int64)
        sc_arr = np.asarray(new_rel_scores, dtype=np.float32) if new_rel_scores else None
    else:
        rel_arr = np.zeros((0, 3), dtype=np.int64)
        sc_arr = None
    return rel_arr, sc_arr


def run_dual_inference(model, cfg, dataset_name: str, small_class_indices: List[int],
                       score_thresh: float = 0.0, num_images: int = -1):
    """EVAL_DUAL 模式推理，返回 per-image 结果（含检测 + 关系）。"""
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
            outputs = model(batch)

            if not isinstance(outputs, dict) or not outputs.get("__roi_dual__"):
                print(f"[error] batch {batch_idx}: EVAL_DUAL not active, skipping")
                continue

            roi_results = outputs["override"]
            orig_results = outputs["raw"]

            for b_i, (input_per_image, roi_r, orig_r) in enumerate(zip(batch, roi_results, orig_results)):
                if num_images > 0 and len(all_results) >= num_images:
                    break

                file_name = input_per_image.get("file_name", "")
                image_id = input_per_image.get("image_id", b_i)

                height = input_per_image.get("height")
                width = input_per_image.get("width")
                if height is None or width is None:
                    if file_name and os.path.exists(file_name):
                        with Image.open(file_name) as img:
                            width, height = img.size
                    else:
                        continue
                image_size = (int(height), int(width))

                # GT
                gt_instances = input_per_image.get("instances")
                if gt_instances is None or not hasattr(gt_instances, "gt_boxes"):
                    continue

                gt_boxes_t = gt_instances.gt_boxes.tensor.clone()
                tfm_h, tfm_w = gt_instances.image_size
                scale_x = float(width) / float(tfm_w)
                scale_y = float(height) / float(tfm_h)
                gt_boxes_xyxy = gt_boxes_t.clone()
                gt_boxes_xyxy[:, [0, 2]] *= scale_x
                gt_boxes_xyxy[:, [1, 3]] *= scale_y
                gt_labels = gt_instances.gt_classes.clone()

                gt_boxes_norm = gt_boxes_xyxy.clone()
                gt_boxes_norm[:, [0, 2]] /= float(width)
                gt_boxes_norm[:, [1, 3]] /= float(height)

                # ---- ROI 预测 + 关系 ----
                roi_inst = roi_r.get("instances")
                pred_boxes_norm_roi, pred_labels_roi, pred_scores_roi, keep_roi = \
                    extract_preds_from_instances(roi_inst, image_size, score_thresh)

                roi_rel_pair = roi_r.get("rel_pair_idxs")
                roi_rel_labels = roi_r.get("pred_rel_labels")
                roi_rel_scores = roi_r.get("pred_rel_scores")
                roi_relations, roi_rel_scores_arr = _remap_relations(
                    roi_rel_pair, roi_rel_labels, roi_rel_scores, keep_roi)

                metrics_roi = compute_per_image_ap_recall(
                    gt_boxes_norm, gt_labels,
                    pred_boxes_norm_roi, pred_labels_roi, pred_scores_roi,
                    small_class_indices, iou_thresh=0.5,
                )

                # ---- 原始预测 + 关系 ----
                orig_inst = orig_r.get("instances")
                pred_boxes_norm_orig, pred_labels_orig, pred_scores_orig, keep_orig = \
                    extract_preds_from_instances(orig_inst, image_size, score_thresh)

                orig_rel_pair = orig_r.get("rel_pair_idxs")
                orig_rel_labels = orig_r.get("pred_rel_labels")
                orig_rel_scores = orig_r.get("pred_rel_scores")
                orig_relations, orig_rel_scores_arr = _remap_relations(
                    orig_rel_pair, orig_rel_labels, orig_rel_scores, keep_orig)

                metrics_orig = compute_per_image_ap_recall(
                    gt_boxes_norm, gt_labels,
                    pred_boxes_norm_orig, pred_labels_orig, pred_scores_orig,
                    small_class_indices, iou_thresh=0.5,
                )

                # GT 小物体类别
                gt_small_mask = torch.isin(gt_labels, torch.tensor(small_class_indices))
                gt_small_classes = gt_labels[gt_small_mask].unique().tolist()
                gt_small_names = [thing_classes[c] for c in gt_small_classes]

                roi_better = metrics_roi["recall"] > metrics_orig["recall"]
                combined_score = metrics_roi["recall"] + metrics_roi["ap"]

                all_results.append({
                    "image_id": image_id,
                    "file_name": file_name,
                    "image_size": [int(height), int(width)],
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
                    # 检测结果
                    "pred_boxes_orig": pred_boxes_norm_orig.tolist(),
                    "pred_labels_orig": pred_labels_orig.tolist(),
                    "pred_scores_orig": pred_scores_orig.tolist(),
                    "pred_boxes_roi": pred_boxes_norm_roi.tolist(),
                    "pred_labels_roi": pred_labels_roi.tolist(),
                    "pred_scores_roi": pred_scores_roi.tolist(),
                    # 关系结果 (三元组 [sub, obj, pred])
                    "pred_relations_orig": orig_relations.tolist() if hasattr(orig_relations, 'tolist') else orig_relations,
                    "pred_rel_scores_orig": orig_rel_scores_arr.tolist() if orig_rel_scores_arr is not None and hasattr(orig_rel_scores_arr, 'tolist') else (orig_rel_scores_arr.tolist() if orig_rel_scores_arr is not None else None),
                    "pred_relations_roi": roi_relations.tolist() if hasattr(roi_relations, 'tolist') else roi_relations,
                    "pred_rel_scores_roi": roi_rel_scores_arr.tolist() if roi_rel_scores_arr is not None and hasattr(roi_rel_scores_arr, 'tolist') else (roi_rel_scores_arr.tolist() if roi_rel_scores_arr is not None else None),
                    # GT
                    "gt_boxes_norm": gt_boxes_norm.tolist(),
                    "gt_labels": gt_labels.tolist(),
                })

            if num_images > 0 and len(all_results) >= num_images:
                print(f"[eval] reached {num_images} images, stopping.")
                break

            if (batch_idx + 1) % 50 == 0:
                print(f"[eval] processed {batch_idx + 1} batches, {len(all_results)} images so far...")

    return all_results, thing_classes


def main():
    parser = argparse.ArgumentParser(description="ROI Refine vs Original small-object comparison (EVAL_DUAL)")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="AG_val")
    parser.add_argument("--num-images", type=int, default=-1)
    parser.add_argument("--top-k-vis", type=int, default=50)
    parser.add_argument("--small-area-thresh", type=float, default=0.001)
    parser.add_argument("--person-score-scale", type=float, default=400.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--score-thresh", type=float, default=0.2,
                        help="检测框分数阈值（默认: 0.2）")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    print(f"Config: {args.config_file}")
    print(f"Checkpoint: {args.model_weights}")
    print(f"Output: {args.output_dir}")
    print(f"Small area threshold: {args.small_area_thresh}")
    print(f"Person score scale: {args.person_score_scale}")
    print(f"Score threshold: {args.score_thresh}")
    print(f"Top-K vis: {args.top_k_vis}")

    cfg = build_cfg(args)
    metadata = MetadataCatalog.get(args.dataset_name)
    thing_classes = list(metadata.thing_classes)

    small_class_indices = []
    for name in SMALL_CLASS_NAMES:
        if name in thing_classes:
            small_class_indices.append(thing_classes.index(name))
    print(f"Target small classes: {SMALL_CLASS_NAMES} -> indices: {small_class_indices}")

    print("Building model...")
    model = JointTransformerTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    model.to(cfg.MODEL.DEVICE)

    print("Starting dual inference (EVAL_DUAL mode)...")
    t_start = time.time()
    all_results, _ = run_dual_inference(model, cfg, args.dataset_name, small_class_indices,
                                         score_thresh=args.score_thresh,
                                         num_images=args.num_images)
    t_elapsed = time.time() - t_start
    if all_results:
        print(f"Inference done: {len(all_results)} images in {t_elapsed:.1f}s "
              f"({t_elapsed/len(all_results)*1000:.1f} ms/img)")
    else:
        print("No results!")
        return

    # 保存全量结果
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "per_image_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Full per-image results saved to: {results_path}")

    # 统计
    has_small = [r for r in all_results if r["n_gt_small"] > 0]
    print(f"Frames with small-object GT: {len(has_small)} / {len(all_results)}")

    roi_better_results = [r for r in has_small if r["roi_better"]]
    print(f"Frames where ROI is better than original: {len(roi_better_results)}")

    roi_correct_orig_wrong = [
        r for r in has_small
        if r["recall_roi"] > 0 and r["recall_orig"] == 0
    ]
    print(f"Frames where ROI correct but original wrong: {len(roi_correct_orig_wrong)}")

    # 排序
    if roi_correct_orig_wrong:
        roi_correct_orig_wrong.sort(key=lambda r: r["combined_score"], reverse=True)
    elif roi_better_results:
        roi_better_results.sort(key=lambda r: r["combined_score"], reverse=True)
        roi_correct_orig_wrong = roi_better_results
    else:
        has_small.sort(key=lambda r: r["combined_score"], reverse=True)
        roi_correct_orig_wrong = has_small

    top_k = min(args.top_k_vis, len(roi_correct_orig_wrong))
    top_results = roi_correct_orig_wrong[:top_k]

    top_json_path = os.path.join(args.output_dir, "top_comparison_results.json")
    with open(top_json_path, "w") as f:
        json.dump(top_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'Rank':<6} {'Score':<10} {'R_orig':<10} {'R_roi':<10} "
          f"{'AP_orig':<10} {'AP_roi':<10} {'GT classes':<30} {'File'}")
    print("-" * 120)
    for i, r in enumerate(top_results):
        fname = os.path.basename(r["file_name"]) if r["file_name"] else "N/A"
        gt_cls_str = ",".join(r["gt_small_classes"])
        print(f"{i+1:<6} {r['combined_score']:<10.4f} {r['recall_orig']:<10.4f} "
              f"{r['recall_roi']:<10.4f} {r['ap_orig']:<10.4f} {r['ap_roi']:<10.4f} "
              f"{gt_cls_str:<30} {fname}")

    print(f"\n===== Summary =====")
    print(f"Total images processed: {len(all_results)}")
    print(f"Images with small-object GT: {len(has_small)}")
    print(f"ROI correct, original wrong: {len(roi_correct_orig_wrong)}")

    avg_recall_orig = np.mean([r["recall_orig"] for r in has_small]) if has_small else 0
    avg_recall_roi = np.mean([r["recall_roi"] for r in has_small]) if has_small else 0
    print(f"Avg recall (original) on small-object frames: {avg_recall_orig:.4f}")
    print(f"Avg recall (ROI refined) on small-object frames: {avg_recall_roi:.4f}")

    print(f"\nTop results JSON: {top_json_path}")
    print(f"Run recompute script to generate visualizations.")


if __name__ == "__main__":
    main()
