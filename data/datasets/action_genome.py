import os
import json
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def _safe_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def _parse_bbox_xyxy(bbox):
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x", "y", "w", "h")):
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            return [float(x), float(y), float(x + w), float(y + h)]
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            return [float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])]
        return None
    if isinstance(bbox, np.ndarray):
        bbox = bbox.tolist()
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]
        return [x1, y1, x1 + max(0.0, x2), y1 + max(0.0, y2)]
    return None


class ActionGenomeTrainData:
    """
    Minimal Action Genome dataset adaptor for SpeaQ training.
    - Keeps single-frame relation prediction.
    - Adds video_id/frame_id fields for future temporal memory.
    """

    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.ann_dir = cfg.DATASETS.ACTION_GENOME.ANNOTATIONS
        self.frames_root = cfg.DATASETS.ACTION_GENOME.FRAMES
        self.num_videos_train = cfg.DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN
        self.num_videos_val = cfg.DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL
        self.val_set_randomized = cfg.DATASETS.ACTION_GENOME.VAL_SET_RANDOMIZED
        self.seed = cfg.DATASETS.VISUAL_GENOME.OVERFIT_SEED

        self.object_anno, self.person_anno, self.frame_list = self._load_annotations()
        self.video_to_frames = self._group_frames_by_video(self.frame_list)
        self.split_videos = self._build_video_split()

        self.class_names, self.predicate_names = self._collect_vocab()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.predicate_to_idx = {name: idx for idx, name in enumerate(self.predicate_names)}

        self.dataset_dicts = self._build_dataset_dicts()
        self.register_dataset()
        self.statistics = self.get_statistics()

    def _load_annotations(self):
        with open(os.path.join(self.ann_dir, "object_bbox_and_relationship.pkl"), "rb") as f:
            object_anno = pickle.load(f)
        with open(os.path.join(self.ann_dir, "person_bbox.pkl"), "rb") as f:
            person_anno = pickle.load(f)
        frame_list = []
        with open(os.path.join(self.ann_dir, "frame_list.txt"), "r") as f:
            for frame in f:
                frame_list.append(frame.strip())
        return object_anno, person_anno, frame_list

    def _group_frames_by_video(self, frame_list):
        video_to_frames = defaultdict(list)
        for frame_key in frame_list:
            if "/" not in frame_key:
                continue
            video, frame_name = frame_key.split("/", 1)
            video_to_frames[video].append(frame_name)
        return video_to_frames

    def _build_video_split(self):
        videos = sorted(self.video_to_frames.keys())
        if self.val_set_randomized:
            rng = random.Random(self.seed)
            rng.shuffle(videos)

        n_val = max(0, min(len(videos), int(self.num_videos_val)))
        if self.split == "train":
            selected = videos[:-n_val] if n_val > 0 else videos
            if self.num_videos_train > 0:
                selected = selected[: self.num_videos_train]
        elif self.split in ("val", "test"):
            selected = videos[-n_val:] if n_val > 0 else videos
        else:
            selected = videos
        return set(selected)

    def _extract_rel_labels(self, obj):
        rels = []
        for key in ("attention_relationship", "spatial_relationship", "contacting_relationship", "relationships"):
            for item in _safe_list(obj.get(key)):
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                rels.append(str(item))
        return rels

    def _load_class_file(self, filename):
        """Load lines from a class file under ann_dir; strip whitespace, skip empty."""
        path = os.path.join(self.ann_dir, filename)
        if not os.path.isfile(path):
            return None
        names = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)
        return names

    def _collect_vocab(self):
        """
        Use AG_ANNOTATIONS class files as the single source of truth for object and
        predicate vocab. Indices 0..K-1 are real classes; +1 is used as no-object /
        no-relation (background) in the model and is not listed in these files.
        """
        class_names = self._load_class_file("object_classes.txt")
        predicate_names = self._load_class_file("relationship_classes.txt")
        if class_names is None or predicate_names is None:
            raise FileNotFoundError(
                "Action Genome vocab must be loaded from annotations. "
                "Ensure both object_classes.txt and relationship_classes.txt exist under "
                f"DATASETS.ACTION_GENOME.ANNOTATIONS={self.ann_dir}"
            )
        return class_names, predicate_names

    def _get_frame_path(self, video, frame_name):
        path = os.path.join(self.frames_root, video, frame_name)
        if os.path.isfile(path):
            return path
        stem, _ = os.path.splitext(frame_name)
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(self.frames_root, video, stem + ext)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _extract_person_boxes(self, person_item):
        # Standard AG format: {'bbox': np.ndarray([...])}
        bbox_data = person_item.get("bbox") if isinstance(person_item, dict) else None
        if bbox_data is None:
            return []
        if isinstance(bbox_data, np.ndarray):
            if bbox_data.ndim == 1 and bbox_data.shape[0] == 4:
                return [_parse_bbox_xyxy(bbox_data)]
            if bbox_data.ndim == 2 and bbox_data.shape[1] == 4:
                return [_parse_bbox_xyxy(row) for row in bbox_data]
        if isinstance(bbox_data, (list, tuple)):
            if len(bbox_data) == 4 and all(isinstance(v, (int, float)) for v in bbox_data):
                return [_parse_bbox_xyxy(bbox_data)]
            boxes = []
            for item in bbox_data:
                box = _parse_bbox_xyxy(item)
                if box is not None:
                    boxes.append(box)
            return boxes
        return []

    def _build_dataset_dicts(self):
        dataset_dicts = []
        image_id = 0
        for frame_key in self.frame_list:
            if "/" not in frame_key:
                continue
            video, frame_name = frame_key.split("/", 1)
            if video not in self.split_videos:
                continue

            frame_path = self._get_frame_path(video, frame_name)
            if frame_path is None:
                continue

            try:
                with Image.open(frame_path) as img:
                    width, height = img.size
            except Exception:
                continue

            annotations = []
            relations = []

            # Person instances first; AG relations are mainly person-object.
            person_boxes = self._extract_person_boxes(self.person_anno.get(frame_key, {}))
            for pb in person_boxes:
                if pb is None:
                    continue
                annotations.append(
                    {
                        "bbox": pb,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": self.class_to_idx["person"],
                        "attribute": np.zeros((1,), dtype=np.int64),
                    }
                )

            objects = self.object_anno.get(frame_key, [])
            for obj in objects:
                bbox = _parse_bbox_xyxy(obj.get("bbox"))
                if bbox is None:
                    continue
                cls_name = str(obj.get("class", "object"))
                if cls_name not in self.class_to_idx:
                    continue
                obj_idx = len(annotations)
                annotations.append(
                    {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": self.class_to_idx[cls_name],
                        "attribute": np.zeros((1,), dtype=np.int64),
                    }
                )
                # Link first person to current object for each rel label.
                if len(person_boxes) > 0:
                    for rel_name in self._extract_rel_labels(obj):
                        rel_id = self.predicate_to_idx.get(rel_name, None)
                        if rel_id is not None:
                            relations.append([0, obj_idx, rel_id])

            if len(annotations) == 0:
                continue

            record = {
                "file_name": frame_path,
                "image_id": image_id,
                "height": height,
                "width": width,
                "video_id": video,
                "frame_id": frame_name,
                "annotations": annotations,
                # Use int64 so that downstream tensors are torch.long and match DETR expectations
                "relations": np.array(relations, dtype=np.int64) if len(relations) > 0 else np.zeros((0, 3), dtype=np.int64),
            }
            dataset_dicts.append(record)
            image_id += 1
        return dataset_dicts

    def register_dataset(self):
        dataset_name = f"AG_{self.split}"
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
        DatasetCatalog.register(dataset_name, lambda: self.dataset_dicts)
        MetadataCatalog.get(dataset_name).set(
            thing_classes=self.class_names,
            predicate_classes=self.predicate_names,
            attribute_classes=[],
        )

    def get_statistics(self, eps=1e-3):
        num_object_classes = len(self.class_names) + 1
        num_relation_classes = len(self.predicate_names) + 1

        fg_matrix = np.zeros((num_object_classes, num_object_classes, num_relation_classes), dtype=np.int64)
        fg_rel_count = np.zeros((num_relation_classes,), dtype=np.int64)
        bg_matrix = np.ones((num_object_classes, num_object_classes), dtype=np.int64)

        for data in self.dataset_dicts:
            rels = data["relations"]
            anns = data["annotations"]
            if len(anns) == 0:
                continue
            gt_classes = np.array([a["category_id"] for a in anns], dtype=np.int64)
            for s_idx, o_idx, rel_id in rels:
                if s_idx >= len(gt_classes) or o_idx >= len(gt_classes):
                    continue
                s_cls = gt_classes[s_idx]
                o_cls = gt_classes[o_idx]
                fg_matrix[s_cls, o_cls, rel_id] += 1
                fg_rel_count[rel_id] += 1

        fg_matrix[:, :, -1] = bg_matrix
        denom = np.maximum(fg_matrix.sum(2)[:, :, None], 1)
        pred_dist = np.log(fg_matrix / denom + eps)

        result = {
            "fg_matrix": torch.from_numpy(fg_matrix),
            "pred_dist": torch.from_numpy(pred_dist).float(),
            "fg_rel_count": torch.from_numpy(fg_rel_count).float() + 1.0,
            "obj_classes": self.class_names + ["__background__"],
            "rel_classes": self.predicate_names + ["__background__"],
            "att_classes": [],
        }
        MetadataCatalog.get(f"AG_{self.split}").set(statistics=result)
        return result

