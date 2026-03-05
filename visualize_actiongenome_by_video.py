#!/usr/bin/env python3
"""
ActionGenome 按视频/帧保存推理可视化: OUTPUT_DIR/<video_id>/<frame_name>.png
用于 run_actiongenome_train_eval.sh 的第三步，也可单独运行。
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.engine import JointTransformerTrainer
from SpeaQ.data.dataset_mapper import DetrDatasetMapper
from SpeaQ.data.tools.utils import register_datasets
from SpeaQ.modeling import Detr  # noqa: F401

from visualization.visualizer import SceneGraphVisualizer


def _safe_dirname(name):
    return re.sub(r"[^\w\-.]", "_", str(name))


def _extract_relations(output):
    rel_pair_idxs = output.get("rel_pair_idxs")
    rel_labels = output.get("pred_rel_labels")
    rel_scores = output.get("pred_rel_scores")
    instances = output.get("instances")
    if rel_pair_idxs is None and instances is not None:
        rel_pair_idxs = getattr(instances, "_rel_pair_idxs", None)
    if rel_labels is None and instances is not None:
        rel_labels = getattr(instances, "_pred_rel_labels", None)
    if rel_scores is None and instances is not None:
        rel_scores = getattr(instances, "_pred_rel_scores", None)
    if rel_pair_idxs is None or rel_labels is None:
        return np.zeros((0, 3), dtype=np.int64), None
    rel_pair_idxs = rel_pair_idxs.cpu().numpy() if torch.is_tensor(rel_pair_idxs) else rel_pair_idxs
    rel_labels = rel_labels.cpu().numpy() if torch.is_tensor(rel_labels) else rel_labels
    if rel_pair_idxs.shape[0] != rel_labels.shape[0]:
        return np.zeros((0, 3), dtype=np.int64), None
    relations = np.concatenate([rel_pair_idxs, rel_labels.reshape(-1, 1)], axis=1)
    rel_scores_arr = None
    if rel_scores is not None:
        if torch.is_tensor(rel_scores):
            rel_scores = rel_scores.detach()
        if len(rel_scores.shape) == 2:
            rel_scores_arr = rel_scores.max(dim=1).values.cpu().numpy()
        else:
            rel_scores_arr = rel_scores.cpu().numpy()
    return relations, rel_scores_arr


def _classwise_min_iou_nms(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, thresh: float) -> np.ndarray:
    """
    Class-wise NMS using minIoU = inter / min(area_a, area_b).
    This suppresses near-duplicate detections where one box largely contains another.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    keep_all = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.size == 0:
            continue
        order = cls_idx[np.argsort(scores[cls_idx])[::-1]]
        cls_keep = []
        while order.size > 0:
            i = int(order[0])
            cls_keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]

            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            area_i = max((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]), 1e-6)
            area_rest = np.maximum((boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1]), 1e-6)
            min_area = np.minimum(area_i, area_rest)
            miniou = inter / min_area

            order = rest[miniou <= thresh]
        keep_all.extend(cls_keep)

    # Keep deterministic order by score desc after suppression.
    keep_all = np.array(keep_all, dtype=np.int64)
    keep_all = keep_all[np.argsort(scores[keep_all])[::-1]]
    return keep_all


def build_cfg(args):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    if cfg.is_frozen():
        cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if getattr(args, "model_weights", None):
        cfg.MODEL.WEIGHTS = args.model_weights
    cfg.freeze()
    register_datasets(cfg)
    if not hasattr(args, "num_gpus"):
        args.num_gpus = 1
    default_setup(cfg, args)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="ActionGenome inference vis: output_dir/<video_id>/<frame>.png")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--output-dir", required=True, help="Root dir for vis (e.g. OUTPUT_DIR/vis)")
    parser.add_argument("--dataset-name", default="AG_val")
    parser.add_argument("--num-images", type=int, default=-1, help="-1 = all images")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--rel-score-thresh", type=float, default=0.2)
    parser.add_argument("--box-score-thresh", type=float, default=0.0)
    # force-keep-person 已废弃，保留参数仅为兼容旧命令行
    parser.add_argument("--force-keep-person", type=int, default=0, help="(deprecated) kept for backward compatibility")
    parser.add_argument("--classwise-miniou-thresh", type=float, default=0.9)
    parser.add_argument("--debug-person-scores", type=int, default=0, help="1=print person score stats")
    parser.add_argument("--person-score-scale", type=float, default=1.0, help=">1.0 to boost person scores only (for vis)")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = build_cfg(args)
    model = JointTransformerTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    mapper = DetrDatasetMapper(cfg, False)
    data_loader = build_detection_test_loader(cfg, args.dataset_name, mapper=mapper)
    metadata = MetadataCatalog.get(args.dataset_name)

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            for input_per_image, output_per_image in zip(batch, outputs):
                if args.num_images >= 0 and count >= args.num_images:
                    break
                instances = output_per_image.get("instances")
                if instances is None:
                    continue
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                labels = instances.pred_classes.cpu().numpy()
                scores = instances.scores.cpu().numpy() if hasattr(instances, "scores") else None
                relations, rel_scores = _extract_relations(output_per_image)

                # Class-wise minIoU-NMS before visualization to suppress same-label duplicates,
                # including "large box covering small box" cases.
                if scores is not None and len(scores) > 0:
                    nms_keep = _classwise_min_iou_nms(
                        boxes=boxes,
                        scores=scores,
                        labels=labels,
                        thresh=float(args.classwise_miniou_thresh),
                    )
                    if len(nms_keep) < len(scores):
                        keep_set = {int(i) for i in nms_keep.tolist()}
                        remap = {int(old): int(new) for new, old in enumerate(nms_keep.tolist())}
                        boxes = boxes[nms_keep]
                        labels = labels[nms_keep]
                        scores = scores[nms_keep]
                        if relations is not None and relations.shape[0] > 0:
                            new_rels, new_scores = [], []
                            for ri, (s, o, p) in enumerate(relations):
                                s_i, o_i = int(s), int(o)
                                if s_i in keep_set and o_i in keep_set:
                                    new_rels.append([remap[s_i], remap[o_i], int(p)])
                                    if rel_scores is not None and ri < len(rel_scores):
                                        new_scores.append(float(rel_scores[ri]))
                            relations = np.array(new_rels, dtype=np.int64) if new_rels else np.zeros((0, 3), dtype=np.int64)
                            rel_scores = np.array(new_scores, dtype=np.float32) if (rel_scores is not None and len(new_scores) > 0) else None

                box_thresh = args.box_score_thresh
                if scores is not None:
                    # 仅在可视化阶段对 person 类别进行 score 缩放，不影响评测逻辑
                    if args.person_score_scale != 1.0 and hasattr(metadata, "thing_classes") and "person" in metadata.thing_classes:
                        person_class_idx = metadata.thing_classes.index("person")
                        person_mask = labels == person_class_idx
                        if person_mask.any():
                            orig = scores[person_mask].copy()
                            scores[person_mask] = np.clip(scores[person_mask] * args.person_score_scale, 0.0, 1.0)
                            if args.debug_person_scores:
                                print(
                                    f"[VIS_DEBUG] image={os.path.basename(str(input_per_image.get('file_name', 'unknown')))} "
                                    f"person_scores_before_min={orig.min():.6f} max={orig.max():.6f} "
                                    f"after_min={scores[person_mask].min():.6f} max={scores[person_mask].max():.6f}"
                                )
                    keep = scores >= box_thresh
                    if keep.sum() == 0:
                        keep[int(scores.argmax())] = True
                    boxes = boxes[keep]
                    labels = labels[keep]
                    scores = scores[keep]
                    if relations is not None and relations.shape[0] > 0:
                        kept_raw_indices = np.where(keep)[0]
                        kept_indices = {int(i) for i in kept_raw_indices}
                        remap = {int(raw_idx): new_idx for new_idx, raw_idx in enumerate(kept_raw_indices)}
                        new_rels, new_scores = [], []
                        for i, (s, o, p) in enumerate(relations):
                            if int(s) in kept_indices and int(o) in kept_indices:
                                new_idx_s = remap[int(s)]
                                new_idx_o = remap[int(o)]
                                new_rels.append([new_idx_s, new_idx_o, int(p)])
                                if rel_scores is not None:
                                    new_scores.append(float(rel_scores[i]))
                        relations = np.array(new_rels, dtype=np.int64) if new_rels else np.zeros((0, 3), dtype=np.int64)
                        rel_scores = np.array(new_scores, dtype=np.float32) if new_scores and rel_scores is not None else None

                video_id = input_per_image.get("video_id", "unknown")
                frame_id = input_per_image.get("frame_id") or os.path.basename(input_per_image.get("file_name", ""))
                file_name = input_per_image.get("file_name")
                if not file_name:
                    continue

                video_safe = _safe_dirname(video_id)
                frame_safe = _safe_dirname(os.path.basename(str(frame_id)))
                if not frame_safe:
                    frame_safe = f"frame_{count}"
                subdir = os.path.join(args.output_dir, video_safe)
                os.makedirs(subdir, exist_ok=True)
                output_name = os.path.splitext(frame_safe)[0]

                visualizer = SceneGraphVisualizer(metadata, output_dir=subdir)
                visualizer.visualize_scene_graph(
                    file_name,
                    boxes,
                    labels,
                    relations,
                    scores=scores,
                    rel_scores=rel_scores,
                    output_name=output_name,
                    top_k_relations=args.top_k,
                    score_threshold=args.rel_score_thresh,
                )
                count += 1
            if args.num_images >= 0 and count >= args.num_images:
                break
    print(f"Saved {count} visualizations under {args.output_dir}")


if __name__ == "__main__":
    main()
