#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import default_setup
from detectron2.structures import Boxes, pairwise_iou

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.data.dataset_mapper import DetrDatasetMapper
from SpeaQ.data.tools.utils import register_datasets
from SpeaQ.engine import JointTransformerTrainer
from SpeaQ.modeling import Detr  # noqa: F401


def safeName(value: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(value))


def buildCfg(args):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.MODEL.ROI_REFINE.ENABLED = True
    cfg.MODEL.ROI_REFINE.EVAL_DUAL = True
    cfg.MODEL.ROI_REFINE.APPLY_TO = "all"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    register_datasets(cfg)
    if not hasattr(args, "num_gpus"):
        args.num_gpus = 1
    default_setup(cfg, args)
    return cfg


def extractInstances(output):
    return output["instances"]


def boxArea(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def matchGtToPred(gt_box: torch.Tensor, target_label: int, instances, iou_thresh: float):
    if not hasattr(instances, "pred_boxes") or len(instances.pred_boxes) == 0:
        return None
    pred_boxes = instances.pred_boxes.tensor.detach().cpu()
    pred_classes = instances.pred_classes.detach().cpu()
    pred_scores = instances.scores.detach().cpu() if hasattr(instances, "scores") else torch.ones(len(pred_boxes))
    ious = pairwise_iou(Boxes(gt_box.view(1, 4)), Boxes(pred_boxes))[0]
    overlap_ids = torch.where(ious >= iou_thresh)[0]
    if overlap_ids.numel() == 0:
        return None
    same = overlap_ids[pred_classes[overlap_ids] == int(target_label)]
    if same.numel() > 0:
        best = same[torch.argmax(ious[same])]
        return {
            "status": "correct",
            "idx": int(best),
            "iou": float(ious[best]),
            "label": int(pred_classes[best]),
            "score": float(pred_scores[best]),
            "box": pred_boxes[best].numpy(),
        }
    best = overlap_ids[torch.argmax(ious[overlap_ids])]
    return {
        "status": "wrong_class",
        "idx": int(best),
        "iou": float(ious[best]),
        "label": int(pred_classes[best]),
        "score": float(pred_scores[best]),
        "box": pred_boxes[best].numpy(),
    }


def imageDetectionStats(instances, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, iou_thresh: float):
    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0, 0
    pred_boxes = instances.pred_boxes.tensor.detach().cpu() if hasattr(instances, "pred_boxes") else torch.zeros((0, 4))
    pred_labels = instances.pred_classes.detach().cpu() if hasattr(instances, "pred_classes") else torch.zeros((0,), dtype=torch.long)
    pred_scores = instances.scores.detach().cpu() if hasattr(instances, "scores") else torch.zeros((0,))
    order = torch.argsort(pred_scores, descending=True)
    matched = set()
    tp = 0
    fp = 0
    if len(pred_boxes) > 0:
        ious = pairwise_iou(Boxes(gt_boxes), Boxes(pred_boxes))
        for pi in order.tolist():
            same_gt = torch.where(gt_labels == pred_labels[pi])[0]
            if same_gt.numel() == 0:
                fp += 1
                continue
            gt_ious = ious[same_gt, pi]
            best_local = int(torch.argmax(gt_ious))
            best_gt = int(same_gt[best_local])
            best_iou = float(gt_ious[best_local])
            if best_iou >= iou_thresh and best_gt not in matched:
                tp += 1
                matched.add(best_gt)
            else:
                fp += 1
    recall = tp / max(1, int(len(gt_boxes)))
    precision = tp / max(1, tp + fp)
    return precision, recall, tp, int(len(gt_boxes))


def getGt(input_per_image):
    inst = input_per_image["instances"]
    return inst.gt_boxes.tensor.detach().cpu(), inst.gt_classes.detach().cpu()


def drawPanel(draw, title, image, gt_box, roi_match, raw_match, class_names, target_label, target_name):
    canvas = image.copy().convert("RGB")
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
        small_font = None
    d.rectangle([gt_box[0], gt_box[1], gt_box[2], gt_box[3]], outline=(255, 215, 0), width=4)
    d.text((max(0, gt_box[0]), max(0, gt_box[1] - 22)), f"GT {target_name}", fill=(255, 215, 0), font=small_font)
    for match, color, prefix in [(roi_match, (0, 220, 0), "ROI"), (raw_match, (255, 0, 0), "Origin")]:
        if match is None:
            continue
        box = match["box"]
        label = class_names[match["label"]] if match["label"] < len(class_names) else str(match["label"])
        d.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=3)
        d.text((max(0, box[0]), min(canvas.height - 20, max(0, box[3] + 3))),
               f"{prefix}: {label} {match[score]:.3f} IoU {match[iou]:.2f}", fill=color, font=small_font)
    d.rectangle([0, 0, canvas.width, 30], fill=(0, 0, 0))
    d.text((8, 5), title, fill=(255, 255, 255), font=font)
    return canvas


def saveCase(case_dir, input_per_image, target, roi_match, raw_match, class_names, metrics):
    os.makedirs(case_dir, exist_ok=True)
    img = Image.open(input_per_image["file_name"]).convert("RGB")
    gt_box = target["gt_box"]
    panel = drawPanel(None, f"{target[target_name]} | ROI correct vs Origin wrong", img, gt_box, roi_match, raw_match, class_names, target["target_label"], target["target_name"])
    panel.save(os.path.join(case_dir, "roi_vs_origin.png"))
    shutil.copy2(input_per_image["file_name"], os.path.join(case_dir, "source_image" + os.path.splitext(input_per_image["file_name"])[1]))
    with open(os.path.join(case_dir, "metadata.json"), "w") as f:
        json.dump({**target, **metrics, "gt_box": [float(x) for x in gt_box.tolist()], "roi_match": serializeMatch(roi_match), "origin_match": serializeMatch(raw_match)}, f, indent=2)


def serializeMatch(match):
    if match is None:
        return None
    out = dict(match)
    out["box"] = [float(x) for x in match["box"].tolist()]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="AG_val")
    parser.add_argument("--num-images", type=int, default=-1)
    parser.add_argument("--max-keep", type=int, default=40)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--small-area-ratio", type=float, default=0.001)
    parser.add_argument("--target-labels", default="doorknob,light,dish,sandwich,medicine")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = buildCfg(args)
    metadata = MetadataCatalog.get(args.dataset_name)
    class_names = list(metadata.thing_classes)
    target_names = [x.strip() for x in args.target_labels.split(",") if x.strip()]
    target_ids = {class_names.index(name): name for name in target_names}
    assert len(target_ids) == len(target_names), f"target label missing: {target_names} vs {class_names}"

    model = JointTransformerTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    mapper = DetrDatasetMapper(cfg, False)
    data_loader = build_detection_test_loader(cfg, args.dataset_name, mapper=mapper)

    os.makedirs(args.output_dir, exist_ok=True)
    all_dir = os.path.join(args.output_dir, "all_candidates")
    top_dir = os.path.join(args.output_dir, "top_ranked")
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(top_dir, exist_ok=True)
    rows = []
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            filtered_batch = []
            for input_per_image in batch:
                if args.num_images >= 0 and count >= args.num_images:
                    break
                count += 1
                gt_boxes, gt_labels = getGt(input_per_image)
                if len(gt_boxes) == 0:
                    continue
                img_area = float(input_per_image["height"] * input_per_image["width"])
                has_target_small = False
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    label_id = int(gt_label)
                    if label_id not in target_ids:
                        continue
                    area_ratio = boxArea(gt_box.numpy()) / max(1.0, img_area)
                    if area_ratio < args.small_area_ratio:
                        has_target_small = True
                        break
                if has_target_small:
                    filtered_batch.append(input_per_image)
            if not filtered_batch:
                if args.num_images >= 0 and count >= args.num_images:
                    break
                continue
            outputs = model(filtered_batch)
            assert isinstance(outputs, dict) and outputs.get("__roi_dual__"), "ROI dual output is required"
            for input_per_image, roi_output, raw_output in zip(filtered_batch, outputs["override"], outputs["raw"]):
                gt_boxes, gt_labels = getGt(input_per_image)
                img_area = float(input_per_image["height"] * input_per_image["width"])
                roi_inst = extractInstances(roi_output)
                raw_inst = extractInstances(raw_output)
                roi_p, roi_r, roi_tp, roi_gt_n = imageDetectionStats(roi_inst, gt_boxes, gt_labels, args.iou_thresh)
                raw_p, raw_r, raw_tp, _ = imageDetectionStats(raw_inst, gt_boxes, gt_labels, args.iou_thresh)
                for gi, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    label_id = int(gt_label)
                    if label_id not in target_ids:
                        continue
                    area_ratio = boxArea(gt_box.numpy()) / max(1.0, img_area)
                    if area_ratio >= args.small_area_ratio:
                        continue
                    roi_match = matchGtToPred(gt_box, label_id, roi_inst, args.iou_thresh)
                    raw_match = matchGtToPred(gt_box, label_id, raw_inst, args.iou_thresh)
                    if roi_match is None or roi_match["status"] != "correct":
                        continue
                    if raw_match is None or raw_match["status"] != "wrong_class":
                        continue
                    score = roi_p + roi_r + max(0.0, roi_match["score"] - raw_match["score"]) + 0.2 * roi_match["iou"]
                    video_id = input_per_image.get("video_id", "unknown")
                    frame_id = input_per_image.get("frame_id", os.path.basename(input_per_image.get("file_name", "frame")))
                    case_name = f"{len(rows):04d}_{score:.4f}_{target_ids[label_id]}_{safeName(video_id)}_{safeName(frame_id)}"
                    metrics = {
                        "rank_score": float(score), "roi_precision": float(roi_p), "roi_recall": float(roi_r),
                        "raw_precision": float(raw_p), "raw_recall": float(raw_r), "area_ratio": float(area_ratio),
                        "video_id": str(video_id), "frame_id": str(frame_id), "image_id": int(input_per_image.get("image_id", -1)),
                        "file_name": str(input_per_image.get("file_name", "")), "gt_index": int(gi),
                    }
                    target = {"target_label": label_id, "target_name": target_ids[label_id], "gt_box": gt_box.numpy()}
                    saveCase(os.path.join(all_dir, case_name), input_per_image, target, roi_match, raw_match, class_names, metrics)
                    row = {k: v for k, v in metrics.items() if k != "gt_box"}
                    row.update({
                        "case_name": case_name, "target": target_ids[label_id],
                        "roi_label": class_names[roi_match["label"]], "roi_score": roi_match["score"], "roi_iou": roi_match["iou"],
                        "origin_label": class_names[raw_match["label"]], "origin_score": raw_match["score"], "origin_iou": raw_match["iou"],
                    })
                    rows.append(row)
            if args.num_images >= 0 and count >= args.num_images:
                break

    rows.sort(key=lambda x: x["rank_score"], reverse=True)
    for rank, row in enumerate(rows[:args.max_keep], start=1):
        src = os.path.join(all_dir, row["case_name"])
        dst = os.path.join(top_dir, f"rank_{rank:02d}_" + row["case_name"])
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    csv_path = os.path.join(args.output_dir, "ranked_cases.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "processed_images": count,
        "num_candidates": len(rows),
        "target_labels": target_names,
        "small_area_ratio": args.small_area_ratio,
        "iou_thresh": args.iou_thresh,
        "top_dir": top_dir,
        "csv": csv_path,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
