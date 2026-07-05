#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from collections import OrderedDict, defaultdict

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


def buildLoader(cfg, records):
    dataset = DatasetFromList(records, copy=False)
    dataset = MapDataset(dataset, DetrDatasetMapper(cfg, False))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator,
    )


def buildEvaluator(cfg, dataset_name, output_folder):
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


def resetTemporal(model):
    detr = getattr(model, "detr", None)
    if hasattr(detr, "reset_temporal_memory"):
        detr.reset_temporal_memory()


def compactResult(results):
    compact = OrderedDict()
    for key, value in results.items():
        if isinstance(value, (int, float)):
            compact[key] = float(value)
    return compact


def writeSingleVideoLogHeader(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Single-video temporal evaluation results\n")
        f.write("# Each video is evaluated with temporal memory reset before its first frame.\n")
        f.write("# Frames are kept in ascending frame_idx order.\n")


def appendSingleVideoResult(path, video_id, num_frames, frame_start, frame_end, results):
    payload = OrderedDict()
    payload["video_id"] = video_id
    payload["num_frames"] = int(num_frames)
    payload["frame_start"] = int(frame_start)
    payload["frame_end"] = int(frame_end)
    payload["results"] = compactResult(results)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def writeVideoSummaryCsv(path, rows):
    if not rows:
        return
    keys = OrderedDict()
    for row in rows:
        for key in row.keys():
            keys[key] = True
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(keys.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Ordered temporal eval with gate CSV and per-video results")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="AG_val")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-full-eval", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = buildCfg(args)
    records = list(DatasetCatalog.get(args.dataset_name))
    records.sort(key=lambda rec: (str(rec["video_id"]), int(rec["frame_idx"])))

    gate_csv = os.path.join(args.output_dir, "eval_gate_strength.csv")
    os.environ["TRIPLET_EVAL_GATE_CSV"] = gate_csv
    if os.path.exists(gate_csv):
        os.remove(gate_csv)

    model = JointTransformerTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)

    if not args.skip_full_eval:
        resetTemporal(model)
        full_loader = buildLoader(cfg, records)
        full_evaluator = buildEvaluator(cfg, args.dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference_ordered_full"))
        full_results = scenegraph_inference_on_dataset(cfg, model, full_loader, full_evaluator)
        full_path = os.path.join(args.output_dir, "ordered_full_eval_results.json")
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        print("===== Ordered full eval results =====")
        print(json.dumps(full_results, indent=2, ensure_ascii=False))
        print(f"Saved ordered full results to: {full_path}")

    by_video = defaultdict(list)
    for rec in records:
        by_video[str(rec["video_id"])].append(rec)

    single_log = os.path.join(args.output_dir, "single_video_result.log")
    summary_csv = os.path.join(args.output_dir, "single_video_result_summary.csv")
    writeSingleVideoLogHeader(single_log)
    summary_rows = []

    for idx, (video_id, video_records) in enumerate(sorted(by_video.items()), 1):
        resetTemporal(model)
        loader = buildLoader(cfg, video_records)
        safe_video_id = video_id.replace("/", "_").replace(".", "_")
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_single_video", f"{idx:04d}_{safe_video_id}")
        evaluator = buildEvaluator(cfg, args.dataset_name, output_folder)
        results = scenegraph_inference_on_dataset(cfg, model, loader, evaluator)
        frame_idxs = [int(r["frame_idx"]) for r in video_records]
        appendSingleVideoResult(single_log, video_id, len(video_records), min(frame_idxs), max(frame_idxs), results)

        row = OrderedDict()
        row["video_id"] = video_id
        row["num_frames"] = int(len(video_records))
        row["frame_start"] = int(min(frame_idxs))
        row["frame_end"] = int(max(frame_idxs))
        row.update(compactResult(results))
        summary_rows.append(row)
        writeVideoSummaryCsv(summary_csv, summary_rows)
        print(f"[single-video] {idx}/{len(by_video)} {video_id}: frames={len(video_records)}", flush=True)

    print(f"Saved gate CSV to: {gate_csv}")
    print(f"Saved single-video log to: {single_log}")
    print(f"Saved single-video summary CSV to: {summary_csv}")


if __name__ == "__main__":
    main()
