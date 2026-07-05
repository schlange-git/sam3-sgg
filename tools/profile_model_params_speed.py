#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import torch

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_setup

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.data.dataset_mapper import DetrDatasetMapper
from SpeaQ.data.tools.utils import register_datasets
from SpeaQ.engine import JointTransformerTrainer
from SpeaQ.modeling import Detr  # noqa: F401


def buildCfg(args):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATALOADER.NUM_WORKERS = 0
    if args.disable_temporal:
        cfg.MODEL.TEMPORAL.ENABLED = False
        cfg.MODEL.TEMPORAL.EVAL_ENABLED = False
    if args.roi_eval_dual is not None:
        cfg.MODEL.ROI_REFINE.EVAL_DUAL = bool(args.roi_eval_dual)
    cfg.freeze()
    register_datasets(cfg)
    if not hasattr(args, "num_gpus"):
        args.num_gpus = 1
    default_setup(cfg, args)
    return cfg


def countParams(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone = 0
    if hasattr(model, "detr") and hasattr(model.detr, "backbone"):
        backbone = sum(p.numel() for p in model.detr.backbone.parameters())
    return total, trainable, backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", default="AG_val")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--disable-temporal", action="store_true")
    parser.add_argument("--roi-eval-dual", type=int, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = buildCfg(args)
    model = JointTransformerTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    total, trainable, backbone = countParams(model)
    mapper = DetrDatasetMapper(cfg, False)
    data_loader = build_detection_test_loader(cfg, args.dataset_name, mapper=mapper)
    times = []
    n = 0
    with torch.no_grad():
        for batch in data_loader:
            if n >= args.warmup + args.iters:
                break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if n >= args.warmup:
                times.append(dt)
            n += 1
    arr = torch.tensor(times, dtype=torch.float64)
    ms = float(arr.mean().item() * 1000.0) if len(times) else None
    result = {
        "config_file": args.config_file,
        "model_weights": args.model_weights,
        "total_params": int(total),
        "trainable_params": int(trainable),
        "backbone_params": int(backbone),
        "non_backbone_params": int(total - backbone),
        "benchmark_images": len(times),
        "warmup_images": args.warmup,
        "bs": 1,
        "latency_ms_mean": ms,
        "fps": (1000.0 / ms) if ms else None,
        "latency_ms_median": float(arr.median().item() * 1000.0) if len(times) else None,
        "latency_ms_p90": float(torch.quantile(arr, 0.9).item() * 1000.0) if len(times) else None,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
