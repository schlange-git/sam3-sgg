#!/usr/bin/env python3
"""
后处理脚本：读取 per_image_results.json，应用 score 阈值 + GIoU NMS 过滤，
重算指标、重排、带三元组关系的可视化。不需要重新推理，秒级完成。
"""
import json
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.data.tools.utils import register_datasets
from visualization.visualizer import SceneGraphVisualizer

SMALL_CLASS_NAMES = ["dish", "light", "medicine", "groceries", "doorknob"]
TRIPLET_LIMIT_CLASS_NAMES = ["dish", "light", "medicine", "groceries", "doorknob", "food", "sandwich"]
CROSS_NMS_CLASS_GROUPS = [["sandwich", "dish", "food", "book"]]


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


def compute_giou(boxes_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """计算一组 box 与单个 box 的 GIoU.
    boxes_a: (N, 4) xyxy, box_b: (4,) xyxy
    returns: (N,) GIoU values in [-1, 1].
    """
    x1 = np.maximum(box_b[0], boxes_a[:, 0])
    y1 = np.maximum(box_b[1], boxes_a[:, 1])
    x2 = np.minimum(box_b[2], boxes_a[:, 2])
    y2 = np.minimum(box_b[3], boxes_a[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    union = area_a + area_b - inter

    # smallest enclosing box
    ex1 = np.minimum(box_b[0], boxes_a[:, 0])
    ey1 = np.minimum(box_b[1], boxes_a[:, 1])
    ex2 = np.maximum(box_b[2], boxes_a[:, 2])
    ey2 = np.maximum(box_b[3], boxes_a[:, 3])
    c_area = np.maximum(0, ex2 - ex1) * np.maximum(0, ey2 - ey1)

    iou = inter / (union + 1e-6)
    giou = iou - (c_area - union) / (c_area + 1e-6)
    return giou


def batchedNmsCpuGiou(boxes, scores, labels, giou_thresh=0.5):
    """Per-class GIoU NMS on CPU. 相比 IoU NMS，对包含关系更敏感。"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        cls_i = labels[i]
        rest = order[1:]
        same_cls = labels[rest] == cls_i
        if not same_cls.any():
            order = rest
            continue
        same_idx = np.where(same_cls)[0]
        rest_boxes_same = boxes[rest[same_cls]]
        giou = compute_giou(rest_boxes_same, boxes[i])
        cls_suppressed = same_idx[giou >= giou_thresh]
        survive = np.setdiff1d(np.arange(len(rest)), cls_suppressed)
        order = rest[survive]
    return np.array(keep, dtype=np.int64)


def crossClassNmsGiou(boxes, scores, labels, groups_indices, giou_thresh=0.25):
    """跨类别 GIoU NMS: groups_indices 中每组内的所有类别按同类处理，高分抑制低分。
    用于互斥的易混淆类别（如 sandwich/dish/food/book）。
    返回 keep 索引（在原数组中的位置）。"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    suppress_mask = np.zeros(len(boxes), dtype=bool)
    for group in groups_indices:
        group_arr = np.array(group, dtype=np.int64)
        in_group = np.isin(labels, group_arr)
        group_idx = np.where(in_group)[0]
        if len(group_idx) <= 1:
            continue
        # 组内按分数降序做 NMS
        order = group_idx[scores[group_idx].argsort()[::-1]]
        while len(order) > 0:
            i = order[0]
            rest = order[1:]
            if len(rest) == 0:
                break
            giou = compute_giou(boxes[rest], boxes[i])
            suppressed = rest[giou >= giou_thresh]
            suppress_mask[suppressed] = True
            survive = rest[giou < giou_thresh]
            order = survive

    if not suppress_mask.any():
        return np.arange(len(boxes), dtype=np.int64)
    return np.where(~suppress_mask)[0]


def filter_preds(boxes, labels, scores, score_thresh):
    """按 score 阈值过滤预测，返回 (boxes, labels, scores, keep_mask)."""
    if len(scores) == 0:
        return boxes, labels, scores, np.zeros(0, dtype=np.int64)
    scores_t = torch.tensor(scores)
    keep = scores_t >= score_thresh
    if keep.sum() == 0:
        max_idx = int(scores_t.argmax())
        keep[max_idx] = True
    keep = keep.numpy()
    return boxes[keep], labels[keep], scores[keep], keep


def _remap_relations(rel_pair_idxs, rel_labels, rel_scores, keep_mask):
    """根据 keep_mask 重映射关系索引，丢弃引用已删除 box 的关系。"""
    n_old = len(keep_mask)
    old_to_new = np.full(n_old, -1, dtype=np.int64)
    kept_idx = np.where(keep_mask)[0]
    for new_i, old_i in enumerate(kept_idx):
        old_to_new[old_i] = new_i

    if rel_pair_idxs is None or len(rel_pair_idxs) == 0:
        return np.zeros((0, 3), dtype=np.int64), None

    pair = np.asarray(rel_pair_idxs, dtype=np.int64)
    lbl = np.asarray(rel_labels, dtype=np.int64)
    sc = np.asarray(rel_scores, dtype=np.float32) if rel_scores is not None else None

    new_rels, new_scs = [], []
    for k in range(len(pair)):
        s_old, o_old = int(pair[k, 0]), int(pair[k, 1])
        if s_old >= n_old or o_old >= n_old:
            continue
        s_new = int(old_to_new[s_old])
        o_new = int(old_to_new[o_old])
        if s_new >= 0 and o_new >= 0:
            new_rels.append([s_new, o_new, int(lbl[k])])
            if sc is not None:
                new_scs.append(float(sc[k]))

    if new_rels:
        rel_arr = np.asarray(new_rels, dtype=np.int64)
        sc_arr = np.asarray(new_scs, dtype=np.float32) if new_scs else None
    else:
        rel_arr = np.zeros((0, 3), dtype=np.int64)
        sc_arr = None
    return rel_arr, sc_arr


def applyPerClassTripletLimit(relations, rel_scores, box_labels, small_class_indices, person_class_idx):
    """对每个小目标类别，限制 person→small_class 三元组数量 ≤ num_persons。
    不影响 object 不是 small class 的三元组。
    relations: (N, 3) [sub_idx, obj_idx, pred_label], 已 remap 到过滤后的 box 索引
    rel_scores: (N,) or None
    box_labels: (M,) 过滤后的 box 类别标签
    """
    if len(relations) == 0:
        return relations, rel_scores

    num_persons = int((box_labels == person_class_idx).sum())

    # 没有 person 检出时，没有主体能发出关系，丢弃所有涉及小目标的关系
    if num_persons == 0:
        n_total = len(relations)
        # 检查每个三元组的 object 是否是小目标
        obj_labels = box_labels[relations[:, 1]]
        is_small_obj = np.isin(obj_labels, small_class_indices)
        keep = ~is_small_obj
        if not keep.any():
            return np.zeros((0, 3), dtype=np.int64), None
        return relations[keep], rel_scores[keep] if rel_scores is not None else None

    keep_mask = np.ones(len(relations), dtype=bool)
    for sc_idx in small_class_indices:
        obj_indices = relations[:, 1]
        obj_labels = box_labels[obj_indices]
        is_small = obj_labels == sc_idx
        small_rel_idx = np.where(is_small)[0]

        if len(small_rel_idx) <= num_persons:
            continue

        # 按 relation score 降序，仅保留 top num_persons
        if rel_scores is not None:
            order = np.argsort(rel_scores[small_rel_idx])[::-1]
        else:
            order = np.arange(len(small_rel_idx))
        suppress = small_rel_idx[order[num_persons:]]
        keep_mask[suppress] = False

    if not keep_mask.any():
        return np.zeros((0, 3), dtype=np.int64), None
    return relations[keep_mask], rel_scores[keep_mask] if rel_scores is not None else None


def filterBoxesByRelations(boxes, labels, scores, relations):
    """只保留参与至少一个 triplet 的检测框，并重映射关系索引。
    relations: (N, 3) [sub, obj, pred]，已映射到当前 box 索引。
    返回 (filtered_boxes, filtered_labels, filtered_scores, remapped_relations)。"""
    if len(relations) == 0 or len(boxes) == 0:
        return boxes[:0], labels[:0], scores[:0], relations[:0]

    used = np.unique(np.concatenate([relations[:, 0], relations[:, 1]]))
    used = used.astype(np.int64)
    if len(used) == 0:
        return boxes[:0], labels[:0], scores[:0], relations[:0]

    boxes_f = boxes[used]
    labels_f = labels[used]
    scores_f = scores[used]

    old_to_new = np.full(len(boxes), -1, dtype=np.int64)
    for new_i, old_i in enumerate(used):
        old_to_new[old_i] = new_i
    remapped = relations.copy()
    remapped[:, 0] = old_to_new[relations[:, 0]]
    remapped[:, 1] = old_to_new[relations[:, 1]]

    return boxes_f, labels_f, scores_f, remapped


def compute_iou_single(box_a, box_b):
    """计算两个单 box 的 IoU。box_a/b: (4,) xyxy numpy arrays。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def findClassCorrections(gt_boxes, gt_labels, orig_boxes, orig_labels, orig_scores,
                          roi_boxes, roi_labels, roi_scores, small_class_indices,
                          thing_classes, iou_thresh=0.5):
    """检测类别纠错：original 预测了错误类别但 ROI 纠正为正确类别。
    返回 list of dict，每个 dict 包含纠错详情。"""
    corrections = []
    for gt_i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        gt_label = int(gt_label)
        if gt_label not in small_class_indices:
            continue

        best_orig_idx, best_orig_iou, best_orig_label = -1, 0.0, -1
        for p_i, (p_box, p_label) in enumerate(zip(orig_boxes, orig_labels)):
            iou = compute_iou_single(gt_box, p_box)
            if iou >= iou_thresh and iou > best_orig_iou:
                best_orig_iou = iou
                best_orig_idx = int(p_i)
                best_orig_label = int(p_label)

        best_roi_idx, best_roi_iou, best_roi_label = -1, 0.0, -1
        for p_i, (p_box, p_label) in enumerate(zip(roi_boxes, roi_labels)):
            iou = compute_iou_single(gt_box, p_box)
            if iou >= iou_thresh and iou > best_roi_iou:
                best_roi_iou = iou
                best_roi_idx = int(p_i)
                best_roi_label = int(p_label)

        if best_orig_idx >= 0 and best_roi_idx >= 0:
            if best_orig_label != gt_label and best_roi_label == gt_label:
                corrections.append({
                    "gt_idx": gt_i,
                    "gt_label_idx": gt_label,
                    "gt_label_name": thing_classes[gt_label] if gt_label < len(thing_classes) else str(gt_label),
                    "orig_label_idx": best_orig_label,
                    "orig_label_name": thing_classes[best_orig_label] if best_orig_label < len(thing_classes) else str(best_orig_label),
                    "roi_label_idx": best_roi_label,
                    "roi_label_name": thing_classes[best_roi_label] if best_roi_label < len(thing_classes) else str(best_roi_label),
                    "orig_iou": round(float(best_orig_iou), 3),
                    "roi_iou": round(float(best_roi_iou), 3),
                    "orig_score": round(float(orig_scores[best_orig_idx]), 4),
                    "roi_score": round(float(roi_scores[best_roi_idx]), 4),
                })
    return corrections


def compute_per_image_metrics(gt_boxes_norm, gt_labels, pred_boxes_norm, pred_labels, pred_scores,
                               target_class_indices, iou_thresh=0.5):
    result = {"recall": 0.0, "ap": 0.0, "n_gt": 0, "n_matched": 0}
    gt_boxes = torch.tensor(gt_boxes_norm).float()
    gt_lbls = torch.tensor(gt_labels).long()
    pred_boxes = torch.tensor(pred_boxes_norm).float()
    pred_lbls = torch.tensor(pred_labels).long()
    pred_sc = torch.tensor(pred_scores).float()

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return result

    gt_target_mask = torch.isin(gt_lbls, torch.tensor(target_class_indices))
    if not gt_target_mask.any():
        return result

    gt_boxes_target = gt_boxes[gt_target_mask]
    gt_lbls_target = gt_lbls[gt_target_mask]
    result["n_gt"] = int(len(gt_boxes_target))

    score_order = torch.argsort(pred_sc, descending=True)
    used_pred = set()
    matched = 0
    for gt_i in range(len(gt_boxes_target)):
        gt_cls = int(gt_lbls_target[gt_i])
        gt_box = gt_boxes_target[gt_i: gt_i + 1]
        for p_i in range(len(pred_boxes)):
            pi = int(score_order[p_i])
            if pi in used_pred:
                continue
            if int(pred_lbls[pi]) != gt_cls:
                continue
            iou = compute_iou(gt_box, pred_boxes[pi: pi + 1])[0, 0]
            if iou >= iou_thresh:
                matched += 1
                used_pred.add(pi)
                break

    result["n_matched"] = matched
    result["recall"] = float(matched) / float(result["n_gt"]) if result["n_gt"] > 0 else 0.0
    n_pred_target = int(torch.isin(pred_lbls, torch.tensor(target_class_indices)).sum())
    if n_pred_target > 0:
        result["ap"] = float(matched) / float(n_pred_target)
    return result


def main():
    parser = argparse.ArgumentParser(description="Re-filter and re-visualize ROI comparison results")
    parser.add_argument("--input-json", required=True, help="per_image_results.json 路径")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-file", default="configs/speaq_actiongenome_minimal.yaml")
    parser.add_argument("--dataset-name", default="AG_val")
    parser.add_argument("--score-thresh", type=float, default=0.2)
    parser.add_argument("--nms-thresh", type=float, default=0.25, help="GIoU threshold for post-filter NMS")
    parser.add_argument("--top-k-vis", type=int, default=50)
    parser.add_argument("--top-k-relations", type=int, default=20,
                        help="可视化中展示的关系数（默认 20）")
    parser.add_argument("--rel-score-thresh", type=float, default=0.2,
                        help="关系展示的分数阈值（默认 0.2）")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    print(f"Loading {args.input_json}...")
    with open(args.input_json) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} images")

    # Load metadata for visualization
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    if cfg.is_frozen():
        cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    register_datasets(cfg)
    metadata = MetadataCatalog.get(args.dataset_name)
    thing_classes = list(metadata.thing_classes)

    small_class_indices = [thing_classes.index(n) for n in SMALL_CLASS_NAMES if n in thing_classes]
    triplet_limit_indices = [thing_classes.index(n) for n in TRIPLET_LIMIT_CLASS_NAMES if n in thing_classes]
    cross_nms_groups = [[thing_classes.index(n) for n in group if n in thing_classes]
                        for group in CROSS_NMS_CLASS_GROUPS]
    cross_nms_groups = [g for g in cross_nms_groups if len(g) > 1]
    print(f"Target small classes (metrics): {SMALL_CLASS_NAMES} -> indices: {small_class_indices}")
    print(f"Triplet limit classes: {TRIPLET_LIMIT_CLASS_NAMES} -> indices: {triplet_limit_indices}")
    print(f"Cross-NMS groups: {cross_nms_groups}")
    print(f"Score threshold: {args.score_thresh}, GIoU NMS threshold: {args.nms_thresh}")
    print(f"Relation score threshold: {args.rel_score_thresh}")

    # Re-filter and re-compute metrics
    print("Re-filtering predictions and re-computing metrics...")
    updated = []
    for r in all_results:
        r["pred_boxes_orig"] = np.array(r["pred_boxes_orig"])
        r["pred_labels_orig"] = np.array(r["pred_labels_orig"])
        r["pred_scores_orig"] = np.array(r["pred_scores_orig"])
        r["pred_boxes_roi"] = np.array(r["pred_boxes_roi"])
        r["pred_labels_roi"] = np.array(r["pred_labels_roi"])
        r["pred_scores_roi"] = np.array(r["pred_scores_roi"])

        # Score filtering
        pb_orig, pl_orig, ps_orig, keep_orig = filter_preds(
            r["pred_boxes_orig"], r["pred_labels_orig"], r["pred_scores_orig"], args.score_thresh)
        pb_roi, pl_roi, ps_roi, keep_roi = filter_preds(
            r["pred_boxes_roi"], r["pred_labels_roi"], r["pred_scores_roi"], args.score_thresh)

        # GIoU NMS (per-class) + cross-class NMS + remap relations
        # --- orig ---
        if len(ps_orig) > 0:
            keep1 = batchedNmsCpuGiou(pb_orig, ps_orig, pl_orig, args.nms_thresh)
            pb_orig, pl_orig, ps_orig = pb_orig[keep1], pl_orig[keep1], ps_orig[keep1]
        else:
            keep1 = np.arange(len(ps_orig), dtype=np.int64)
        if len(ps_orig) > 0 and cross_nms_groups:
            keep2 = crossClassNmsGiou(pb_orig, ps_orig, pl_orig, cross_nms_groups, args.nms_thresh)
            keep1 = keep1[keep2]
            pb_orig, pl_orig, ps_orig = pb_orig[keep2], pl_orig[keep2], ps_orig[keep2]
        keep_orig_nms = np.zeros(len(r["pred_scores_orig"]), dtype=bool)
        orig_score_kept = np.where(keep_orig)[0]
        for k in keep1:
            keep_orig_nms[orig_score_kept[k]] = True

        # --- roi ---
        if len(ps_roi) > 0:
            keep1r = batchedNmsCpuGiou(pb_roi, ps_roi, pl_roi, args.nms_thresh)
            pb_roi, pl_roi, ps_roi = pb_roi[keep1r], pl_roi[keep1r], ps_roi[keep1r]
        else:
            keep1r = np.arange(len(ps_roi), dtype=np.int64)
        if len(ps_roi) > 0 and cross_nms_groups:
            keep2r = crossClassNmsGiou(pb_roi, ps_roi, pl_roi, cross_nms_groups, args.nms_thresh)
            keep1r = keep1r[keep2r]
            pb_roi, pl_roi, ps_roi = pb_roi[keep2r], pl_roi[keep2r], ps_roi[keep2r]
        keep_roi_nms = np.zeros(len(r["pred_scores_roi"]), dtype=bool)
        roi_score_kept = np.where(keep_roi)[0]
        for k in keep1r:
            keep_roi_nms[roi_score_kept[k]] = True

        # Remap relations through score+NMS pipeline
        rel_orig = r.get("pred_relations_orig", [])
        rel_scores_orig = r.get("pred_rel_scores_orig", None)
        if len(rel_orig) > 0:
            rel_orig_arr, rel_scores_orig_arr = _remap_relations(
                np.asarray(rel_orig)[:, :2],  # pair_idxs
                np.asarray(rel_orig)[:, 2],   # labels
                np.asarray(rel_scores_orig) if rel_scores_orig is not None else None,
                keep_orig_nms,
            )
        else:
            rel_orig_arr = np.zeros((0, 3), dtype=np.int64)
            rel_scores_orig_arr = None

        rel_roi = r.get("pred_relations_roi", [])
        rel_scores_roi = r.get("pred_rel_scores_roi", None)
        if len(rel_roi) > 0:
            rel_roi_arr, rel_scores_roi_arr = _remap_relations(
                np.asarray(rel_roi)[:, :2],
                np.asarray(rel_roi)[:, 2],
                np.asarray(rel_scores_roi) if rel_scores_roi is not None else None,
                keep_roi_nms,
            )
        else:
            rel_roi_arr = np.zeros((0, 3), dtype=np.int64)
            rel_scores_roi_arr = None

        # 每小目标类别限制 triplet 数量 ≤ num_persons (含 food/sandwich)
        person_class_idx = 0  # AG 中 person 固定为 index 0
        rel_orig_arr, rel_scores_orig_arr = applyPerClassTripletLimit(
            rel_orig_arr, rel_scores_orig_arr, pl_orig, triplet_limit_indices, person_class_idx)
        rel_roi_arr, rel_scores_roi_arr = applyPerClassTripletLimit(
            rel_roi_arr, rel_scores_roi_arr, pl_roi, triplet_limit_indices, person_class_idx)

        # Re-compute metrics
        gt_boxes = np.array(r["gt_boxes_norm"])
        gt_labels = np.array(r["gt_labels"])

        m_orig = compute_per_image_metrics(gt_boxes, gt_labels, pb_orig, pl_orig, ps_orig, small_class_indices)
        m_roi = compute_per_image_metrics(gt_boxes, gt_labels, pb_roi, pl_roi, ps_roi, small_class_indices)

        r["n_matched_orig"] = m_orig["n_matched"]
        r["n_matched_roi"] = m_roi["n_matched"]
        r["recall_orig"] = float(m_orig["recall"])
        r["recall_roi"] = float(m_roi["recall"])
        r["ap_orig"] = float(m_orig["ap"])
        r["ap_roi"] = float(m_roi["ap"])
        r["roi_better"] = m_roi["recall"] > m_orig["recall"]
        r["combined_score"] = float(m_roi["recall"] + m_roi["ap"])

        # 类别纠错检测：original 类别错误，ROI 类别正确
        class_corrections = findClassCorrections(
            gt_boxes, gt_labels, pb_orig, pl_orig, ps_orig,
            pb_roi, pl_roi, ps_roi, small_class_indices, thing_classes)
        r["class_corrections"] = class_corrections

        # Store filtered data for visualization — only boxes in triplets
        pb_orig_vis, pl_orig_vis, ps_orig_vis, rel_orig_vis = filterBoxesByRelations(
            pb_orig, pl_orig, ps_orig, rel_orig_arr)
        pb_roi_vis, pl_roi_vis, ps_roi_vis, rel_roi_vis = filterBoxesByRelations(
            pb_roi, pl_roi, ps_roi, rel_roi_arr)

        r["pred_boxes_orig_filt"] = pb_orig_vis.tolist() if hasattr(pb_orig_vis, 'tolist') else list(pb_orig_vis)
        r["pred_labels_orig_filt"] = pl_orig_vis.tolist() if hasattr(pl_orig_vis, 'tolist') else list(pl_orig_vis)
        r["pred_scores_orig_filt"] = ps_orig_vis.tolist() if hasattr(ps_orig_vis, 'tolist') else list(ps_orig_vis)
        r["pred_boxes_roi_filt"] = pb_roi_vis.tolist() if hasattr(pb_roi_vis, 'tolist') else list(pb_roi_vis)
        r["pred_labels_roi_filt"] = pl_roi_vis.tolist() if hasattr(pl_roi_vis, 'tolist') else list(pl_roi_vis)
        r["pred_scores_roi_filt"] = ps_roi_vis.tolist() if hasattr(ps_roi_vis, 'tolist') else list(ps_roi_vis)

        # Store remapped relations (already remapped for filtered boxes)
        r["pred_relations_orig_filt"] = rel_orig_vis.tolist() if hasattr(rel_orig_vis, 'tolist') else rel_orig_vis
        r["pred_rel_scores_orig_filt"] = (rel_scores_orig_arr.tolist() if rel_scores_orig_arr is not None and hasattr(rel_scores_orig_arr, 'tolist') else (list(rel_scores_orig_arr) if rel_scores_orig_arr is not None else None))
        r["pred_relations_roi_filt"] = rel_roi_vis.tolist() if hasattr(rel_roi_vis, 'tolist') else rel_roi_vis
        r["pred_rel_scores_roi_filt"] = (rel_scores_roi_arr.tolist() if rel_scores_roi_arr is not None and hasattr(rel_scores_roi_arr, 'tolist') else (list(rel_scores_roi_arr) if rel_scores_roi_arr is not None else None))

        for key in ["gt_boxes_norm", "gt_labels", "pred_boxes_orig", "pred_labels_orig",
                     "pred_scores_orig", "pred_boxes_roi", "pred_labels_roi", "pred_scores_roi"]:
            if key in r and hasattr(r[key], 'tolist'):
                r[key] = r[key].tolist()

        updated.append(r)

    all_results = updated

    os.makedirs(args.output_dir, exist_ok=True)

    has_small = [r for r in all_results if r["n_gt_small"] > 0]
    print(f"Frames with small-object GT: {len(has_small)} / {len(all_results)}")

    roi_better_results = [r for r in has_small if r["roi_better"]]
    print(f"Frames where ROI is better than original: {len(roi_better_results)}")

    roi_correct_orig_wrong = [r for r in has_small if r["recall_roi"] > 0 and r["recall_orig"] == 0]
    print(f"Frames where ROI correct but original wrong: {len(roi_correct_orig_wrong)}")

    # 类别纠错统计
    class_correction_results = [r for r in all_results if len(r.get("class_corrections", [])) > 0]
    total_corrections = sum(len(r.get("class_corrections", [])) for r in all_results)
    print(f"Class correction cases: {total_corrections} corrections in {len(class_correction_results)} frames")
    if total_corrections > 0:
        # 按纠错后类别统计
        from collections import Counter
        correction_pairs = Counter()
        for r in all_results:
            for cc in r.get("class_corrections", []):
                pair = f"{cc['orig_label_name']}→{cc['roi_label_name']}"
                correction_pairs[pair] += 1
        print("  Correction type breakdown:")
        for pair, cnt in correction_pairs.most_common():
            print(f"    {pair}: {cnt}")

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
    print(f"\nTop {top_k} results:")

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

    # Save comparison visualizations with relations (triplets)
    print(f"\nSaving top {top_k} comparison visualizations with relations...")
    vis_dir = os.path.join(args.output_dir, "comparison_vis")
    os.makedirs(os.path.join(vis_dir, "original"), exist_ok=True)
    os.makedirs(os.path.join(vis_dir, "roi_refine"), exist_ok=True)

    vis_orig = SceneGraphVisualizer(metadata, output_dir=os.path.join(vis_dir, "original"))
    vis_roi = SceneGraphVisualizer(metadata, output_dir=os.path.join(vis_dir, "roi_refine"))

    for i, r in enumerate(top_results):
        file_name = r["file_name"]
        if not file_name or not os.path.exists(file_name):
            print(f"  [{i+1}/{top_k}] skip: image not found")
            continue

        h, w = r["image_size"]

        def denorm(boxes, w, h):
            b = boxes.copy()
            b[:, [0, 2]] *= float(w)
            b[:, [1, 3]] *= float(h)
            return b

        stem = os.path.splitext(os.path.basename(file_name))[0]
        vis_name = f"{i+1:04d}_score{r['combined_score']:.3f}_{stem}"

        # ---- Original prediction ----
        boxes_orig = np.array(r.get("pred_boxes_orig_filt", r["pred_boxes_orig"]))
        labels_orig = np.array(r.get("pred_labels_orig_filt", r["pred_labels_orig"]))
        scores_orig = np.array(r.get("pred_scores_orig_filt", r["pred_scores_orig"]))
        rel_orig = np.array(r.get("pred_relations_orig_filt", []))
        rel_scores_orig = np.array(r.get("pred_rel_scores_orig_filt", [])) if r.get("pred_rel_scores_orig_filt") is not None else None

        if len(rel_orig.shape) != 2 or rel_orig.shape[1] < 3:
            rel_orig = np.zeros((0, 3), dtype=np.int64)

        if len(boxes_orig) > 0:
            vis_orig.visualize_scene_graph(
                file_name, denorm(boxes_orig, w, h), labels_orig,
                rel_orig,
                scores=scores_orig, rel_scores=rel_scores_orig,
                output_name=vis_name, top_k_relations=args.top_k_relations,
                score_threshold=args.rel_score_thresh,
            )

        # ---- ROI refined prediction ----
        boxes_roi = np.array(r.get("pred_boxes_roi_filt", r["pred_boxes_roi"]))
        labels_roi = np.array(r.get("pred_labels_roi_filt", r["pred_labels_roi"]))
        scores_roi = np.array(r.get("pred_scores_roi_filt", r["pred_scores_roi"]))
        rel_roi = np.array(r.get("pred_relations_roi_filt", []))
        rel_scores_roi = np.array(r.get("pred_rel_scores_roi_filt", [])) if r.get("pred_rel_scores_roi_filt") is not None else None

        if len(rel_roi.shape) != 2 or rel_roi.shape[1] < 3:
            rel_roi = np.zeros((0, 3), dtype=np.int64)

        if len(boxes_roi) > 0:
            vis_roi.visualize_scene_graph(
                file_name, denorm(boxes_roi, w, h), labels_roi,
                rel_roi,
                scores=scores_roi, rel_scores=rel_scores_roi,
                output_name=vis_name, top_k_relations=args.top_k_relations,
                score_threshold=args.rel_score_thresh,
            )

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{top_k}] visualizations saved...")

    print(f"\n===== Summary =====")
    print(f"Total images processed: {len(all_results)}")
    print(f"Images with small-object GT: {len(has_small)}")
    print(f"ROI correct, original wrong: {len(roi_correct_orig_wrong)}")

    avg_recall_orig = np.mean([r["recall_orig"] for r in has_small]) if has_small else 0
    avg_recall_roi = np.mean([r["recall_roi"] for r in has_small]) if has_small else 0
    print(f"Avg recall (original) on small-object frames: {avg_recall_orig:.4f}")
    print(f"Avg recall (ROI refined) on small-object frames: {avg_recall_roi:.4f}")

    avg_boxes_orig = np.mean([len(r.get("pred_boxes_orig_filt", r["pred_boxes_orig"])) for r in all_results])
    avg_boxes_roi = np.mean([len(r.get("pred_boxes_roi_filt", r["pred_boxes_roi"])) for r in all_results])
    print(f"Avg boxes per image (orig, filtered): {avg_boxes_orig:.1f}")
    print(f"Avg boxes per image (roi, filtered): {avg_boxes_roi:.1f}")

    print(f"\nComparison visualizations saved to: {vis_dir}")


if __name__ == "__main__":
    main()
