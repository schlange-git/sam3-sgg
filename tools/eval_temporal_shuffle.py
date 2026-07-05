#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys

from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data.build import trivial_batch_collator
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.engine import default_setup

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.data.dataset_mapper import DetrDatasetMapper
from SpeaQ.data.tools.utils import register_datasets
from SpeaQ.engine import JointTransformerTrainer
from SpeaQ.evaluation import SceneGraphEvaluator, scenegraph_inference_on_dataset
from SpeaQ.modeling import Detr  # noqa: F401


def buildCfg(args):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.OUTPUT_DIR = args.output_dir
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
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


def buildShuffledTestLoader(cfg, dataset_name, seed):
    dataset_dicts = list(DatasetCatalog.get(dataset_name))
    rng = random.Random(seed)
    rng.shuffle(dataset_dicts)
    dataset = DatasetFromList(dataset_dicts, copy=False)
    dataset = MapDataset(dataset, DetrDatasetMapper(cfg, False))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator,
    )


def buildEvaluator(cfg, dataset_name):
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_shuffle")
    if cfg.MODEL.ROI_REFINE.ENABLED and cfg.MODEL.ROI_REFINE.EVAL_DUAL:
        return {
            "override": SceneGraphEvaluator(
                dataset_name, cfg, True, os.path.join(output_folder, "override")
            ),
            "raw": SceneGraphEvaluator(
                dataset_name, cfg, True, os.path.join(output_folder, "raw")
            ),
        }
    return SceneGraphEvaluator(dataset_name, cfg, True, output_folder)


def main():
    parser = argparse.ArgumentParser(description="Temporal memory shuffle-order evaluation")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="AG_val")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--reset-temporal-each-batch",
        action="store_true",
        help="Reset temporal memory before every eval batch; used as a no-memory shuffle control.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = buildCfg(args)

    data_loader = buildShuffledTestLoader(cfg, args.dataset_name, args.seed)
    model = JointTransformerTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    if args.reset_temporal_each_batch:
        original_forward = model.forward

        def forwardWithTemporalReset(inputs):
            detr = getattr(model, "detr", None)
            if hasattr(detr, "reset_temporal_memory"):
                detr.reset_temporal_memory()
            return original_forward(inputs)

        model.forward = forwardWithTemporalReset
        print("[SHUFFLE_EVAL] reset_temporal_each_batch=True; temporal memory is cleared before every batch.", flush=True)

    evaluator = buildEvaluator(cfg, args.dataset_name)
    results = scenegraph_inference_on_dataset(cfg, model, data_loader, evaluator)
    if hasattr(model, "detr") and hasattr(model.detr, "reset_temporal_memory"):
        model.detr.reset_temporal_memory()

    out_path = os.path.join(args.output_dir, "shuffle_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("===== Shuffle eval results =====")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
