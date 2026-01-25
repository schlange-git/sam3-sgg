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
project_root = os.path.dirname(current_dir)  # /home/shi/abschluss/speaq
sys.path.insert(0, current_dir)  # /home/shi/abschluss/speaq/SpeaQ (current repo)
sys.path.insert(0, project_root)  # /home/shi/abschluss/speaq
# Also add parent for legacy repo access
sys.path.append(os.path.abspath(os.path.join(project_root, '../../')))  # /home/shi/abschluss

# Add legacy SpeaQ repo path for visualization module (lower priority)
# /home/shi/abschluss/speaq/SpeaQ -> ../../ -> /home/shi/abschluss -> SpeaQ -> /home/shi/abschluss/SpeaQ
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


def visualize(cfg, model, output_dir: str, dataset_name: str, num_images: int, top_k: int, rel_thresh: float):
    os.makedirs(output_dir, exist_ok=True)
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

                boxes = instances.pred_boxes.tensor.cpu().numpy()
                labels = instances.pred_classes.cpu().numpy()
                scores = instances.scores.cpu().numpy() if hasattr(instances, "scores") else None

                relations, rel_scores = _extract_relations(output_per_image)

                file_name = input_per_image.get("file_name")
                image_id = input_per_image.get("image_id", count)
                vis_name = f"img_{image_id:06d}"

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
    parser.add_argument("--model-weights", required=True, help="Path to model weights")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualizations")
    parser.add_argument("--dataset-name", default=None, help="Dataset name to visualize (default: first train)")
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to visualize")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K relations to show")
    parser.add_argument("--rel-score-thresh", type=float, default=0.3, help="Relation score threshold")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_cfg(args)
    dataset_name = args.dataset_name or cfg.DATASETS.TRAIN[0]
    model = JointTransformerTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    visualize(cfg, model, args.output_dir, dataset_name, args.num_images, args.top_k, args.rel_score_thresh)


if __name__ == "__main__":
    main()
