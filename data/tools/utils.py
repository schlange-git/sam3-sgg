import os
import sys

from ..datasets import VisualGenomeTrainData
from detectron2.data.datasets import register_coco_instances

def register_datasets(cfg):
    if cfg.DATASETS.TYPE == 'VISUAL GENOME':
        # Only register splits that are actually used to avoid unnecessary data loading
        splits_to_register = set()
        # Add splits from TRAIN
        for dataset_name in cfg.DATASETS.TRAIN:
            if dataset_name.startswith('VG_'):
                splits_to_register.add(dataset_name.split('_')[1])  # Extract 'train', 'val', or 'test'
        # Add splits from TEST
        for dataset_name in cfg.DATASETS.TEST:
            if dataset_name.startswith('VG_'):
                splits_to_register.add(dataset_name.split('_')[1])
        # Always include 'test' so eval hooks won't fail
        splits_to_register.add('test')
        # Fallback: if no splits found, register all (for backward compatibility)
        if not splits_to_register:
            splits_to_register = {'train', 'val', 'test'}
        # Register only needed splits
        for split in splits_to_register:
            dataset_instance = VisualGenomeTrainData(cfg, split=split)
        