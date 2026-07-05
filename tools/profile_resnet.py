#!/usr/bin/env python3
"""统计 ResNet-101 baseline 参数量 + bs=1 推理耗时"""
import argparse
import json
import os
import sys
import time
import numpy as np

project_root = '/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching'
sys.path.insert(0, project_root)

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_setup

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.data.dataset_mapper import DetrDatasetMapper
from SpeaQ.data.tools.utils import register_datasets
from SpeaQ.engine import JointTransformerTrainer
from SpeaQ.modeling import Detr  # noqa: F401


def build_model(cfg):
    model = JointTransformerTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    model.to(cfg.MODEL.DEVICE)
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = 0
    backbone_trainable = 0
    for name, p in model.named_parameters():
        if 'backbone' in name.lower() or 'res' in name.lower() or 'sam3' in name.lower():
            backbone_params += p.numel()
            if p.requires_grad:
                backbone_trainable += p.numel()
    return {
        'total': total,
        'trainable': trainable,
        'backbone': backbone_params,
        'backbone_trainable': backbone_trainable,
        'non_backbone': total - backbone_params,
    }


def profile_inference(model, data_loader, warmup=8, benchmark=80):
    model.eval()
    device = next(model.parameters()).device
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            if count >= warmup:
                break
            for i in range(len(batch)):
                if 'instances' in batch[i]:
                    batch[i]['instances'] = batch[i]['instances'].to(device)
                if 'image' in batch[i]:
                    batch[i]['image'] = batch[i]['image'].to(device)
            _ = model(batch)
            count += 1
            if count == 1:
                print('  [warmup] first batch done')

    if device.type == 'cuda':
        torch.cuda.synchronize()

    latencies = []
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            if count >= benchmark:
                break
            for i in range(len(batch)):
                if 'instances' in batch[i]:
                    batch[i]['instances'] = batch[i]['instances'].to(device)
                if 'image' in batch[i]:
                    batch[i]['image'] = batch[i]['image'].to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            count += 1

    latencies = np.array(latencies)
    return {
        'benchmark_images': int(len(latencies)),
        'warmup_images': warmup,
        'bs': 1,
        'latency_ms_mean': float(np.mean(latencies)),
        'fps': float(1000.0 / np.mean(latencies)),
        'latency_ms_median': float(np.median(latencies)),
        'latency_ms_p90': float(np.percentile(latencies, 90)),
        'latency_ms_std': float(np.std(latencies)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--model-weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-name', default='AG_val')
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--benchmark', type=int, default=80)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    print(f'Building config from {args.config_file}...')
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    if cfg.is_frozen():
        cfg.defrost()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.freeze()
    register_datasets(cfg)

    if not hasattr(args, 'num_gpus'):
        args.num_gpus = 1
    if not hasattr(args, 'num_machines'):
        args.num_machines = 1
    if not hasattr(args, 'machine_rank'):
        args.machine_rank = 0
    default_setup(cfg, args)

    print('Building model...')
    model = build_model(cfg)

    print('Counting parameters...')
    params = count_params(model)
    print(f'  Total params: {params["total"]:,}')
    print(f'  Trainable params: {params["trainable"]:,}')
    print(f'  Backbone params: {params["backbone"]:,}')
    print(f'  Non-backbone params: {params["non_backbone"]:,}')

    print('Setting up data loader for profiling...')
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    mapper = DetrDatasetMapper(cfg, False)
    data_loader = build_detection_test_loader(cfg, args.dataset_name, mapper=mapper)

    print(f'Profiling inference (warmup={args.warmup}, benchmark={args.benchmark})...')
    inf_results = profile_inference(model, data_loader, warmup=args.warmup, benchmark=args.benchmark)

    print(f'  Mean latency: {inf_results["latency_ms_mean"]:.2f} ms')
    print(f'  Median latency: {inf_results["latency_ms_median"]:.2f} ms')
    print(f'  FPS: {inf_results["fps"]:.2f}')

    output = {
        'config_file': args.config_file,
        'model_weights': args.model_weights,
        'params': params,
        'inference': inf_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
