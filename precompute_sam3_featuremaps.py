import sys
import os
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import argparse
from typing import List

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.structures import ImageList

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.data.tools import register_datasets
from SpeaQ.modeling.backbone.sam3_backbone import Sam3MaskedBackbone


def build_cfg(config_file: str, opts: List[str]):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute SAM3 feature maps.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", default="data/featuremaps")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = build_cfg(args.config_file, args.opts)
    os.makedirs(args.output_dir, exist_ok=True)

    register_datasets(cfg)
    dataset_name = cfg.DATASETS.TRAIN[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)
    limit = args.limit
    if limit <= 0:
        limit = cfg.DATASETS.VISUAL_GENOME.OVERFIT_NUM_IMAGES
    if limit and limit > 0:
        dataset_dicts = dataset_dicts[: limit]

    # Build SAM3 backbone with precomputed mode disabled.
    cfg = cfg.clone()
    cfg.defrost()
    cfg.MODEL.DEVICE = cfg.MODEL.SAM3.DEVICE
    cfg.MODEL.SAM3.USE_PRECOMPUTED = False
    cfg.freeze()
    backbone = Sam3MaskedBackbone(cfg)
    backbone.eval()

    for record in dataset_dicts:
        image_id = record.get("image_id")
        if image_id is None:
            continue
        out_path = os.path.join(args.output_dir, f"{image_id}.pt")
        if os.path.isfile(out_path):
            continue

        image = utils.read_image(record["file_name"], format=cfg.INPUT.FORMAT)
        image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        images = ImageList.from_tensors([image_tensor.to(cfg.MODEL.SAM3.DEVICE)])

        with torch.no_grad():
            features = backbone(images)["sam3"]
        feat = features.tensors[0].to("cpu")
        payload = {
            "image_id": image_id,
            "image_size": (record.get("height"), record.get("width")),
            "feature": feat,
            "feature_stride": backbone.feature_stride,
        }
        torch.save(payload, out_path)


if __name__ == "__main__":
    main()
