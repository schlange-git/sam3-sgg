#!/usr/bin/env python3
"""
Alignment check utility:
- Uses current Detectron2/SpeaQ config and mapper pipeline
- Visualizes GT and prediction for N images
- Dumps per-image metadata (original size, transformed size, scale factors, counts)
to help diagnose resize/crop mapping issues.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import default_setup

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.data.dataset_mapper import DetrDatasetMapper
from SpeaQ.data.tools.utils import register_datasets
from SpeaQ.engine import JointTransformerTrainer
from SpeaQ.modeling import Detr  # noqa: F401
from visualization.visualizer import SceneGraphVisualizer


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


def _extract_relations(output: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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

    relations = np.concatenate([rel_pair_idxs, rel_labels.reshape(-1, 1)], axis=1).astype(np.int64)
    rel_scores_arr = None
    if rel_scores is not None:
        if torch.is_tensor(rel_scores):
            rel_scores = rel_scores.detach()
        if len(rel_scores.shape) == 2:
            rel_scores_arr = rel_scores.max(dim=1).values.cpu().numpy()
        else:
            rel_scores_arr = rel_scores.cpu().numpy() if torch.is_tensor(rel_scores) else rel_scores
    return relations, rel_scores_arr


def _to_numpy_relations(relations_any: Any) -> np.ndarray:
    if relations_any is None:
        return np.zeros((0, 3), dtype=np.int64)
    relations = relations_any.cpu().numpy() if torch.is_tensor(relations_any) else relations_any
    if len(relations.shape) != 2:
        return np.zeros((0, 3), dtype=np.int64)
    if relations.shape[1] >= 3:
        return relations[:, :3].astype(np.int64)
    return np.zeros((0, 3), dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Visualize GT+Pred and dump alignment metadata")
    parser.add_argument("--config-file", required=True, help="Config path")
    parser.add_argument("--model-weights", required=True, help="Checkpoint for prediction visualization")
    parser.add_argument("--output-dir", required=True, help="Output root dir")
    parser.add_argument("--dataset-name", default="AG_val", help="Dataset name")
    parser.add_argument("--num-images", type=int, default=100, help="Number of images to visualize")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K relations to draw")
    parser.add_argument("--rel-score-thresh", type=float, default=0.2, help="Relation score threshold")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = build_cfg(args)

    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0  # deterministic order, easier debugging
    cfg.freeze()

    mapper = DetrDatasetMapper(cfg, False)
    data_loader = build_detection_test_loader(cfg, args.dataset_name, mapper=mapper)
    metadata = MetadataCatalog.get(args.dataset_name)

    gt_dir = os.path.join(args.output_dir, "gt")
    pred_dir = os.path.join(args.output_dir, "pred")
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "meta.jsonl")

    gt_vis = SceneGraphVisualizer(metadata, output_dir=gt_dir)
    pred_vis = SceneGraphVisualizer(metadata, output_dir=pred_dir)

    model = JointTransformerTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    count = 0
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        with torch.no_grad():
            for batch in data_loader:
                outputs = model(batch)
                for input_per_image, output_per_image in zip(batch, outputs):
                    if count >= args.num_images:
                        break

                    file_name = input_per_image.get("file_name")
                    if not file_name or not os.path.exists(file_name):
                        continue

                    # Original size from disk
                    with Image.open(file_name) as img:
                        orig_w, orig_h = img.size

                    # GT instances are in transformed image space from mapper
                    gt_instances = input_per_image.get("instances")
                    if gt_instances is None or not hasattr(gt_instances, "gt_boxes"):
                        continue
                    tfm_h, tfm_w = gt_instances.image_size
                    scale_x = float(orig_w) / float(tfm_w)
                    scale_y = float(orig_h) / float(tfm_h)

                    gt_boxes_t = gt_instances.gt_boxes.tensor.cpu().numpy()
                    gt_boxes = gt_boxes_t.copy()
                    gt_boxes[:, [0, 2]] *= scale_x
                    gt_boxes[:, [1, 3]] *= scale_y
                    gt_boxes[:, [0, 2]] = np.clip(gt_boxes[:, [0, 2]], 0, orig_w)
                    gt_boxes[:, [1, 3]] = np.clip(gt_boxes[:, [1, 3]], 0, orig_h)
                    gt_labels = gt_instances.gt_classes.cpu().numpy()
                    gt_relations = _to_numpy_relations(input_per_image.get("relations"))

                    pred_instances = output_per_image.get("instances")
                    if pred_instances is None:
                        continue
                    pred_boxes = pred_instances.pred_boxes.tensor.cpu().numpy()
                    pred_labels = pred_instances.pred_classes.cpu().numpy()
                    pred_scores = pred_instances.scores.cpu().numpy() if hasattr(pred_instances, "scores") else None
                    pred_relations, pred_rel_scores = _extract_relations(output_per_image)

                    stem = os.path.splitext(os.path.basename(file_name))[0]
                    vis_name = f"{count:04d}_{stem}"

                    gt_vis.visualize_scene_graph(
                        file_name,
                        gt_boxes,
                        gt_labels,
                        gt_relations,
                        scores=None,
                        rel_scores=None,
                        output_name=vis_name,
                        top_k_relations=1000,
                        score_threshold=0.0,
                    )
                    pred_vis.visualize_scene_graph(
                        file_name,
                        pred_boxes,
                        pred_labels,
                        pred_relations,
                        scores=pred_scores,
                        rel_scores=pred_rel_scores,
                        output_name=vis_name,
                        top_k_relations=args.top_k,
                        score_threshold=args.rel_score_thresh,
                    )

                    meta = {
                        "index": count,
                        "file_name": file_name,
                        "image_stem": stem,
                        "orig_size_wh": [orig_w, orig_h],
                        "mapped_size_wh": [tfm_w, tfm_h],
                        "scale_xy": [scale_x, scale_y],
                        "gt_num_boxes": int(len(gt_boxes)),
                        "gt_num_relations": int(len(gt_relations)),
                        "pred_num_boxes": int(len(pred_boxes)),
                        "pred_num_relations": int(len(pred_relations)),
                        "gt_vis_path": os.path.join("gt", f"{vis_name}.png"),
                        "pred_vis_path": os.path.join("pred", f"{vis_name}.png"),
                    }
                    f_meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    count += 1

                if count >= args.num_images:
                    break

    print(f"[done] saved {count} images")
    print(f"[done] gt vis: {gt_dir}")
    print(f"[done] pred vis: {pred_dir}")
    print(f"[done] metadata: {meta_path}")


if __name__ == "__main__":
    main()

