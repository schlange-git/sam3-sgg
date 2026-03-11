import argparse
import os
import logging
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Set up paths similar to train_iterative_model.py
# This ensures SpeaQ imports work correctly
# Add current directory first to ensure we load the correct modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 项目根目录
sys.path.insert(0, current_dir)  # 当前目录（项目根目录）
sys.path.insert(0, project_root)  # 项目根目录
# Also add parent for legacy repo access
sys.path.append(os.path.abspath(os.path.join(project_root, '../../')))  # 父目录

# Add legacy SpeaQ repo path for visualization module (lower priority)
# 尝试查找可能的 legacy repo 路径
LEGACY_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../SpeaQ'))
if os.path.exists(LEGACY_REPO):
    if LEGACY_REPO not in sys.path:
        sys.path.append(LEGACY_REPO)

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.engine import JointTransformerTrainer
from SpeaQ.data.dataset_mapper import DetrDatasetMapper
from SpeaQ.data.tools.utils import register_datasets
# Import modeling to trigger META_ARCH registration
from SpeaQ.modeling import Detr  # noqa: F401

from visualization.visualizer import SceneGraphVisualizer


logger = logging.getLogger("speaq.visualize")


def build_cfg(args) -> Any:
    cfg = get_cfg()
    # Add all configs before merging - this ensures all keys exist
    # Same order as train_iterative_model.py
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    
    # Use base config file if the provided one is a saved config with incompatible keys
    config_file = args.config_file
    if 'output_pipeline' in config_file or 'pretrained_eval' in config_file or 'finetune' in config_file:
        # This is a saved config, use base config instead and extract key values via opts
        logger.info(f"Detected saved config file, using base configs/speaq.yaml instead")
        config_file = 'configs/speaq.yaml'
    
    # Merge from file (same as train_iterative_model.py)
    # Ensure cfg is not frozen before merge
    if cfg.is_frozen():
        cfg.defrost()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.freeze()
    register_datasets(cfg)
    # Create a minimal args object for default_setup if needed
    if not hasattr(args, 'num_gpus'):
        args.num_gpus = 1
    if not hasattr(args, 'num_machines'):
        args.num_machines = 1
    if not hasattr(args, 'machine_rank'):
        args.machine_rank = 0
    default_setup(cfg, args)
    return cfg


def _extract_relations(output: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    rel_pair_idxs = output.get("rel_pair_idxs")
    rel_labels = output.get("pred_rel_labels")
    rel_scores = output.get("pred_rel_scores")

    # Fallback to instances fields if not present at top level
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
            rel_scores_arr = rel_scores.cpu().numpy() if torch.is_tensor(rel_scores) else rel_scores

    return relations, rel_scores_arr


def visualize_gt(cfg, output_dir: str, dataset_name: str, num_images: int):
    """Visualize ground truth boxes and relations"""
    os.makedirs(output_dir, exist_ok=True)
    # Set NUM_WORKERS=0 to ensure deterministic batch order
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    mapper = DetrDatasetMapper(cfg, False)
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    metadata = MetadataCatalog.get(dataset_name)
    visualizer = SceneGraphVisualizer(metadata, output_dir=output_dir)

    count = 0
    for batch in data_loader:
        for input_per_image in batch:
            if count >= num_images:
                break

            # Extract GT boxes and labels
            instances = input_per_image.get("instances")
            if instances is None or not hasattr(instances, "gt_boxes"):
                continue

            # Get original image dimensions
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            if height is None or width is None:
                # Fallback: read from file
                from PIL import Image
                file_name = input_per_image.get("file_name")
                if file_name:
                    img = Image.open(file_name)
                    width, height = img.size
                else:
                    logger.warning(f"Could not determine image size for count={count}, skipping")
                    continue

            # Convert boxes from transformed coordinates back to original image coordinates
            # The boxes in instances are in transformed image space, need to convert back
            # Get transformed image size
            transformed_h, transformed_w = instances.image_size

            # Calculate scale factors
            scale_x = width / transformed_w
            scale_y = height / transformed_h

            # Scale boxes from transformed space to original space
            boxes_tensor = instances.gt_boxes.tensor.cpu().numpy()
            boxes = boxes_tensor.copy()
            boxes[:, [0, 2]] *= scale_x  # x coordinates
            boxes[:, [1, 3]] *= scale_y  # y coordinates

            # Clip boxes to image boundaries
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

            labels = instances.gt_classes.cpu().numpy()

            # Extract GT relations
            # Relations format in dataset: [subject_idx, object_idx, predicate] (predicate is 0-indexed)
            # From visual_genome.py: relations = np.column_stack((objects, predicates))
            # where objects is [subject_idx, object_idx] and predicates is [predicate]
            relations = input_per_image.get("relations")
            if relations is not None:
                if torch.is_tensor(relations):
                    relations = relations.cpu().numpy()
                # Ensure relations are in [subject_idx, object_idx, predicate] format
                if len(relations.shape) == 2:
                    if relations.shape[1] == 2:
                        # If only [subject_idx, object_idx], add predicate column (default to 0)
                        relations = np.column_stack([relations, np.zeros(len(relations), dtype=np.int64)])
                    elif relations.shape[1] >= 3:
                        # Take first 3 columns: [subject_idx, object_idx, predicate]
                        relations = relations[:, :3].astype(np.int64)
                    else:
                        relations = np.zeros((0, 3), dtype=np.int64)
                else:
                    relations = np.zeros((0, 3), dtype=np.int64)
            else:
                relations = np.zeros((0, 3), dtype=np.int64)

            file_name = input_per_image.get("file_name")
            image_id = input_per_image.get("image_id")
            if image_id is None:
                # Try to extract from file_name
                if file_name:
                    basename = os.path.basename(file_name)
                    try:
                        image_id = int(os.path.splitext(basename)[0])
                    except ValueError:
                        image_id = count
                else:
                    image_id = count
            vis_name = f"img_{image_id}"

            visualizer.visualize_scene_graph(
                file_name,
                boxes,
                labels,
                relations,
                scores=None,  # GT doesn't have scores
                rel_scores=None,
                output_name=vis_name,
                top_k_relations=1000,  # Show all GT relations
                score_threshold=0.0,
            )
            count += 1
        if count >= num_images:
            break
    logger.info("Saved GT visualizations to %s", output_dir)


def visualize(cfg, model, output_dir: str, dataset_name: str, num_images: int, top_k: int, rel_thresh: float):
    os.makedirs(output_dir, exist_ok=True)
    # Set NUM_WORKERS=0 to ensure deterministic batch order
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    mapper = DetrDatasetMapper(cfg, False)
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    metadata = MetadataCatalog.get(dataset_name)
    visualizer = SceneGraphVisualizer(metadata, output_dir=output_dir)

    model.eval()
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch)
            for input_per_image, output_per_image in zip(batch, outputs):
                if count >= num_images:
                    break
                instances = output_per_image.get("instances")
                if instances is None:
                    continue

                # Boxes are already postprocessed by detector_postprocess in model forward
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                labels = instances.pred_classes.cpu().numpy()
                scores = instances.scores.cpu().numpy() if hasattr(instances, "scores") else None

                relations, rel_scores = _extract_relations(output_per_image)

                # ------------------------------------------------------------------
                # 过滤低分数的检测框，避免大量 score≈0.0 的框干扰可视化
                # ------------------------------------------------------------------
                box_score_thresh = rel_thresh  # 复用关系阈值，保持一个控制参数
                if scores is not None:
                    keep = scores >= box_score_thresh
                    if keep.sum() == 0:
                        # 如果所有框都被过滤，则保留最高分的一个，避免空图
                            max_idx = int(scores.argmax())
                            keep[max_idx] = True
                    boxes = boxes[keep]
                    labels = labels[keep]
                    scores = scores[keep]

                    # 同步过滤关系：只保留主体和客体都仍然存在的关系
                    if relations is not None and relations.shape[0] > 0:
                        kept_indices = np.where(keep)[0]
                        index_map = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(kept_indices)}
                        new_relations = []
                        new_rel_scores = [] if rel_scores is not None else None
                        for i, (sub_idx, obj_idx, pred) in enumerate(relations):
                            sub_idx = int(sub_idx)
                            obj_idx = int(obj_idx)
                            if sub_idx in index_map and obj_idx in index_map:
                                new_relations.append(
                                    [index_map[sub_idx], index_map[obj_idx], int(pred)]
                                )
                                if new_rel_scores is not None:
                                    new_rel_scores.append(float(rel_scores[i]))
                        if new_relations:
                            relations = np.asarray(new_relations, dtype=np.int64)
                            if new_rel_scores is not None:
                                rel_scores = np.asarray(new_rel_scores, dtype=np.float32)
                        else:
                            relations = np.zeros((0, 3), dtype=np.int64)
                            rel_scores = None

                file_name = input_per_image.get("file_name")
                image_id = input_per_image.get("image_id")
                if image_id is None:
                    # Try to extract from file_name
                    if file_name:
                        basename = os.path.basename(file_name)
                        try:
                            image_id = int(os.path.splitext(basename)[0])
                        except ValueError:
                            image_id = count
                    else:
                        image_id = count
                vis_name = f"img_{image_id}"

                visualizer.visualize_scene_graph(
                    file_name,
                    boxes,
                    labels,
                    relations,
                    scores=scores,
                    rel_scores=rel_scores,
                    output_name=vis_name,
                    top_k_relations=top_k,
                    score_threshold=rel_thresh,
                )
                count += 1
            if count >= num_images:
                break
    logger.info("Saved visualizations to %s", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize detection and relations")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--model-weights", default=None, help="Path to model weights (not needed for --gt-only)")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualizations")
    parser.add_argument("--dataset-name", default=None, help="Dataset name to visualize (default: first train)")
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to visualize")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K relations to show")
    parser.add_argument("--rel-score-thresh", type=float, default=0.3, help="Relation score threshold")
    parser.add_argument("--gt-only", action="store_true", help="Visualize ground truth only (no model inference)")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_cfg(args)
    dataset_name = args.dataset_name or cfg.DATASETS.TRAIN[0]
    
    if args.gt_only:
        # Visualize GT only
        visualize_gt(cfg, args.output_dir, dataset_name, args.num_images)
    else:
        # Visualize model predictions
        if args.model_weights is None:
            raise ValueError("--model-weights is required when not using --gt-only")
        model = JointTransformerTrainer.build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        visualize(cfg, model, args.output_dir, dataset_name, args.num_images, args.top_k, args.rel_score_thresh)


if __name__ == "__main__":
    main()
