#!/usr/bin/env python3
import argparse
import json
import os
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
    if not hasattr(args, 'num_gpus'):
        args.num_gpus = 1
    if not hasattr(args, 'num_machines'):
        args.num_machines = 1
    if not hasattr(args, 'machine_rank'):
        args.machine_rank = 0
    default_setup(cfg, args)
    return cfg


def buildOrderedLoader(cfg, dataset_name):
    records = list(DatasetCatalog.get(dataset_name))
    records.sort(key=lambda rec: (str(rec['video_id']), int(rec['frame_idx'])))
    dataset = DatasetFromList(records, copy=False)
    dataset = MapDataset(dataset, DetrDatasetMapper(cfg, False))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator,
    )


def buildEvaluator(cfg, dataset_name):
    output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference_ordered_full')
    if cfg.MODEL.ROI_REFINE.ENABLED and cfg.MODEL.ROI_REFINE.EVAL_DUAL:
        return {
            'override': SceneGraphEvaluator(dataset_name, cfg, True, os.path.join(output_folder, 'override')),
            'raw': SceneGraphEvaluator(dataset_name, cfg, True, os.path.join(output_folder, 'raw')),
        }
    return SceneGraphEvaluator(dataset_name, cfg, True, output_folder)


def resetTemporal(model):
    detr = getattr(model, 'detr', None)
    if hasattr(detr, 'reset_temporal_memory'):
        detr.reset_temporal_memory()


def main():
    parser = argparse.ArgumentParser(description='Ordered temporal full eval only')
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--model-weights', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset-name', default='AG_val')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = buildCfg(args)
    gate_csv = os.path.join(args.output_dir, 'eval_gate_strength.csv')
    os.environ['TRIPLET_EVAL_GATE_CSV'] = gate_csv
    if os.path.exists(gate_csv):
        os.remove(gate_csv)

    model = JointTransformerTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    resetTemporal(model)

    data_loader = buildOrderedLoader(cfg, args.dataset_name)
    evaluator = buildEvaluator(cfg, args.dataset_name)
    results = scenegraph_inference_on_dataset(cfg, model, data_loader, evaluator)

    out_path = os.path.join(args.output_dir, 'ordered_full_eval_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('===== Ordered full eval results =====')
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f'Saved ordered full results to: {out_path}')
    print(f'Saved gate CSV to: {gate_csv}')


if __name__ == '__main__':
    main()
