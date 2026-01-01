"""
Visual Genome 150 Dataset Loader for PredCls
参照: https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
使用 VG-SGG-with-attri.h5 格式（VG150 专用格式）
"""
import json
import os
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("Warning: h5py not available. Please install: pip install h5py")


# VG150 的 50 个 predicate（从 1 开始，0 是 background）
VG150_PREDICATES = [
    "__background__",  # 0
    "above", "across", "against", "along", "and",
    "at", "attached to", "behind", "belonging to", "between",
    "carrying", "covered in", "covering", "eating", "flying in",
    "for", "from", "hanging from", "has", "holding",
    "in", "in front of", "inside", "intersect", "laying on",
    "looking at", "lying on", "made of", "near", "of",
    "on", "on back of", "over", "painted on", "parked on",
    "part of", "playing", "riding", "says", "sitting on",
    "standing on", "to", "under", "using", "walking in",
    "walking on", "watching", "wearing", "wears", "with",
]


class VG150Dataset(Dataset):
    """
    Visual Genome 150 Dataset for PredCls task
    
    参照 Scene-Graph-Benchmark.pytorch 的数据格式
    使用 VG-SGG-with-attri.h5 和 VG-SGG-dicts-with-attri.json
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 1008,
        max_objects: int = 50,
        max_relations: int = 200,
    ):
        """
        Args:
            data_root: Path to Visual Genome dataset root
                Should contain:
                - VG-SGG-dicts-with-attri.json
                - VG-SGG-with-attri.h5
                - image_data.json
                - images/ (or images2/)
            split: "train" or "val"
            image_size: Target image size for SAM3
            max_objects: Maximum number of objects per image (filter out if exceeded)
            max_relations: Maximum number of relations per image (filter out if exceeded)
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required. Please install: pip install h5py")
        
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.max_objects = max_objects
        self.max_relations = max_relations
        
        # Load predicate vocabulary
        dicts_path = os.path.join(data_root, "VG-SGG-dicts-with-attri.json")
        if os.path.exists(dicts_path):
            with open(dicts_path, 'r') as f:
                dicts = json.load(f)
            self.predicate_to_idx = dicts.get("predicate_to_idx", {})
            self.idx_to_predicate = dicts.get("idx_to_predicate", {})
            # Convert string keys to int for idx_to_predicate
            self.idx_to_predicate = {int(k): v for k, v in self.idx_to_predicate.items()}
        else:
            # Fallback: use hardcoded VG150 predicates
            print(f"Warning: {dicts_path} not found, using hardcoded VG150 predicates")
            self.predicate_to_idx = {pred: idx for idx, pred in enumerate(VG150_PREDICATES)}
            self.idx_to_predicate = {idx: pred for idx, pred in enumerate(VG150_PREDICATES)}
        
        # Add background if not present
        if "__background__" not in self.predicate_to_idx:
            self.predicate_to_idx["__background__"] = 0
            self.idx_to_predicate[0] = "__background__"
        
        self.num_predicates = len(self.predicate_to_idx)
        print(f"Loaded {self.num_predicates} predicates (including background)")
        
        # Load h5 file
        h5_path = os.path.join(data_root, "VG-SGG-with-attri.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"VG-SGG-with-attri.h5 not found in {data_root}")
        
        print(f"Loading h5 file from {h5_path}...")
        self.h5_file = h5py.File(h5_path, 'r')
        
        # Debug: print all keys (limit to first 50 to avoid too much output)
        all_keys = list(self.h5_file.keys())
        print(f"Debug: Found {len(all_keys)} keys in h5 file")
        if len(all_keys) <= 50:
            print(f"Debug: All keys: {all_keys}")
        else:
            print(f"Debug: First 50 keys: {all_keys[:50]}")
        
        # Load image data for image paths
        image_data_path = os.path.join(data_root, "image_data.json")
        if os.path.exists(image_data_path):
            with open(image_data_path, 'r') as f:
                self.image_data = json.load(f)
            # Build image_id -> image_info mapping
            self.image_id_to_info = {}
            for img_data in self.image_data:
                img_id = img_data.get("image_id", img_data.get("id"))
                self.image_id_to_info[img_id] = img_data
        else:
            print(f"Warning: {image_data_path} not found, will try to infer image paths from h5")
            self.image_id_to_info = {}
        
        # Extract data from h5 file
        # Common keys in VG-SGG h5 files (Scene-Graph-Benchmark.pytorch format):
        # - 'img_to_first_box', 'img_to_last_box': image to box mapping
        # - 'img_to_first_rel', 'img_to_last_rel': image to relation mapping
        # - 'boxes_512' or 'boxes': bounding boxes
        # - 'labels_512' or 'labels': object labels
        # - 'relationships': relationships
        
        print("Available keys in h5 file:", list(self.h5_file.keys())[:20])  # Print first 20 keys
        
        # Get image indices - try different key names
        img_to_first_box_key = None
        for key in ['img_to_first_box', 'img_to_first_obj', 'img_to_first_roi']:
            if key in self.h5_file:
                img_to_first_box_key = key
                break
        
        if img_to_first_box_key:
            self.img_to_first_box = np.array(self.h5_file[img_to_first_box_key][:])
            last_box_key = img_to_first_box_key.replace('first', 'last')
            if last_box_key in self.h5_file:
                self.img_to_last_box = np.array(self.h5_file[last_box_key][:])
            else:
                # Calculate from first_box
                self.img_to_last_box = self.img_to_first_box.copy()
        else:
            # Try to infer from relationships
            if 'img_to_first_rel' in self.h5_file:
                num_images = len(self.h5_file['img_to_first_rel'])
            else:
                # Last resort: try to get from other keys
                num_images = 0
                for key in self.h5_file.keys():
                    if 'img_to' in key.lower():
                        num_images = max(num_images, len(self.h5_file[key]))
            if num_images == 0:
                raise KeyError("Cannot determine number of images from h5 file")
            print(f"Warning: Using fallback for img_to_first_box, num_images={num_images}")
            self.img_to_first_box = np.zeros(num_images, dtype=np.int32)
            self.img_to_last_box = np.zeros(num_images, dtype=np.int32)
        
        # Get relationships mapping
        if 'img_to_first_rel' in self.h5_file:
            self.img_to_first_rel = np.array(self.h5_file['img_to_first_rel'][:])
            self.img_to_last_rel = np.array(self.h5_file['img_to_last_rel'][:])
        else:
            num_images = len(self.img_to_first_box)
            self.img_to_first_rel = np.zeros(num_images, dtype=np.int32)
            self.img_to_last_rel = np.zeros(num_images, dtype=np.int32)
        
        # Get boxes - try different key names (Scene-Graph-Benchmark.pytorch uses boxes_512)
        boxes_key = None
        for key in ['boxes_512', 'boxes', 'rois_512', 'rois', 'box_rois', 'box_roi']:
            if key in self.h5_file:
                boxes_key = key
                break
        
        if boxes_key:
            boxes_data = self.h5_file[boxes_key]
            self.boxes = np.array(boxes_data[:])
            print(f"Loaded boxes from '{boxes_key}': shape={self.boxes.shape}, dtype={self.boxes.dtype}")
            # Convert to float32 if needed
            if self.boxes.dtype != np.float32:
                self.boxes = self.boxes.astype(np.float32)
        else:
            available_keys = list(self.h5_file.keys())
            raise KeyError(f"Boxes not found in h5 file. Available keys: {available_keys[:50]}")
        
        # Get labels - try different key names (Scene-Graph-Benchmark.pytorch uses labels_512)
        labels_key = None
        for key in ['labels_512', 'labels', 'obj_labels', 'roi_labels', 'box_labels', 'roi_label']:
            if key in self.h5_file:
                labels_key = key
                break
        
        if labels_key:
            labels_data = self.h5_file[labels_key]
            self.labels = np.array(labels_data[:])
            print(f"Loaded labels from '{labels_key}': shape={self.labels.shape}, dtype={self.labels.dtype}")
            # Flatten if needed (labels might be [N, 1] instead of [N])
            if len(self.labels.shape) > 1 and self.labels.shape[1] == 1:
                self.labels = self.labels.flatten()
                print(f"Flattened labels to shape: {self.labels.shape}")
            # Convert to int64 if needed
            if self.labels.dtype != np.int64:
                self.labels = self.labels.astype(np.int64)
        else:
            available_keys = list(self.h5_file.keys())
            raise KeyError(f"Labels not found in h5 file. Available keys: {available_keys[:50]}")
        
        # Get relationships - try different key names
        rels_key = None
        for key in ['relationships', 'rels', 'relations', 'rel_rois', 'rel_pairs', 'rel_roi']:
            if key in self.h5_file:
                rels_key = key
                break
        
        if rels_key:
            rels_data = self.h5_file[rels_key]
            self.relationships = np.array(rels_data[:])
            print(f"Loaded relationships from '{rels_key}': shape={self.relationships.shape}, dtype={self.relationships.dtype}")
            
            # Check if predicates are in a separate key
            if 'predicates' in self.h5_file:
                self.predicates = np.array(self.h5_file['predicates'][:])
                print(f"Loaded predicates from 'predicates': shape={self.predicates.shape}, dtype={self.predicates.dtype}")
                
                # Combine relationships and predicates
                if self.relationships.shape[1] == 2:
                    # relationships is [R, 2] (sub_idx, obj_idx)
                    preds_flat = self.predicates.flatten() if len(self.predicates.shape) > 1 else self.predicates
                    if len(preds_flat) == len(self.relationships):
                        # Combine: [R, 2] + [R] -> [R, 3]
                        self.relationships = np.column_stack([
                            self.relationships[:, 0],  # sub_idx
                            self.relationships[:, 1],  # obj_idx
                            preds_flat  # pred_idx
                        ])
                        print(f"Combined relationships and predicates: shape={self.relationships.shape}")
                    else:
                        raise ValueError(f"Cannot combine: relationships shape {self.relationships.shape}, predicates shape {self.predicates.shape}, preds_flat length {len(preds_flat)}")
                else:
                    print(f"Relationships already has {self.relationships.shape[1]} columns, using as is")
            else:
                # If relationships already has 3 columns, use as is
                if self.relationships.shape[1] == 3:
                    print("Relationships already has 3 columns, using as is")
                else:
                    raise ValueError(f"Relationships has {self.relationships.shape[1]} columns but 'predicates' key not found. Cannot determine predicate indices.")
            
            # Ensure it's int64
            if self.relationships.dtype != np.int64:
                self.relationships = self.relationships.astype(np.int64)
        else:
            available_keys = list(self.h5_file.keys())
            raise KeyError(f"Relationships not found in h5 file. Available keys: {available_keys[:50]}")
        
        # Get image IDs (if available)
        if 'image_id' in self.h5_file:
            self.image_ids = self.h5_file['image_id'][:]
        else:
            # Generate from indices
            self.image_ids = np.arange(len(self.img_to_first_box))
        
        # Filter valid images
        self.valid_indices = []
        num_total = len(self.img_to_first_box)
        print(f"Debug: Total images in h5: {num_total}")
        print(f"Debug: Boxes shape: {self.boxes.shape}, Labels shape: {self.labels.shape}, Relationships shape: {self.relationships.shape}")
        
        for i in range(num_total):
            first_box = int(self.img_to_first_box[i])
            last_box = int(self.img_to_last_box[i])
            num_objs = last_box - first_box + 1 if last_box >= first_box else 0
            
            first_rel = int(self.img_to_first_rel[i])
            last_rel = int(self.img_to_last_rel[i])
            num_rels = last_rel - first_rel + 1 if last_rel >= first_rel else 0
            
            # Filter based on constraints
            if num_objs >= 2 and num_objs <= max_objects and num_rels <= max_relations:
                self.valid_indices.append(i)
        
        print(f"Loaded {len(self.valid_indices)} valid images for {split} split (out of {num_total} total)")
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
        # Find image directory
        self.image_dir = None
        for img_dir_name in ['images', 'images2']:
            img_dir = os.path.join(data_root, img_dir_name)
            if os.path.exists(img_dir):
                self.image_dir = img_dir
                break
        
        if self.image_dir is None:
            print(f"Warning: Could not find images/ or images2/ directory in {data_root}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        img_idx = self.valid_indices[idx]
        img_id = int(self.image_ids[img_idx])
        
        # Get boxes for this image
        first_box = int(self.img_to_first_box[img_idx])
        last_box = int(self.img_to_last_box[img_idx])
        
        if last_box < first_box:
            # No boxes for this image
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_obj_labels = []
        else:
            img_boxes = self.boxes[first_box:last_box+1]  # [G, 4] (x1, y1, x2, y2) in pixels
            img_labels = self.labels[first_box:last_box+1]  # [G] object class labels
            
            # Get image info for dimensions
            if img_id in self.image_id_to_info:
                img_info = self.image_id_to_info[img_id]
                orig_w = img_info.get("width", 1000)
                orig_h = img_info.get("height", 1000)
            else:
                # Try to load image to get dimensions
                orig_w, orig_h = 1000, 1000  # Default, will be updated when loading image
            
            # Normalize boxes to [0, 1]
            # Note: boxes in h5 are in pixel coordinates
            # We need to normalize them, but we need image dimensions first
            # For now, we'll load the image to get dimensions
            
            # Load image
            img_path = self._get_image_path(img_id)
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                orig_w, orig_h = image.size
            else:
                # Fallback: create dummy image
                image = Image.new('RGB', (orig_w, orig_h), color='white')
            
            # Normalize boxes
            gt_boxes = []
            for box in img_boxes:
                x1, y1, x2, y2 = box
                x1_norm = x1 / orig_w
                y1_norm = y1 / orig_h
                x2_norm = x2 / orig_w
                y2_norm = y2 / orig_h
                gt_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
            
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)  # [G, 4]
            gt_obj_labels = [int(l) for l in img_labels]  # List of object class IDs
        
        # Get relationships for this image
        first_rel = int(self.img_to_first_rel[img_idx])
        last_rel = int(self.img_to_last_rel[img_idx])
        
        gt_rels = []
        if last_rel >= first_rel:
            img_rels = self.relationships[first_rel:last_rel+1]  # [R, 3] (sub_idx, obj_idx, pred_idx)
            
            for rel in img_rels:
                sub_idx, obj_idx, pred_idx = int(rel[0]), int(rel[1]), int(rel[2])
                
                # Convert from global box indices to local object indices
                # sub_idx and obj_idx are indices in the global boxes array
                # We need to map them to local indices (0, 1, ..., G-1)
                if last_box >= first_box:
                    sub_local = sub_idx - first_box
                    obj_local = obj_idx - first_box
                    
                    # Check if indices are valid
                    if 0 <= sub_local < len(gt_boxes) and 0 <= obj_local < len(gt_boxes) and sub_local != obj_local:
                        # pred_idx in h5 is 1-indexed (1-50), we keep it as is (0 is background)
                        # But we need to ensure it's in our vocabulary
                        if pred_idx in self.idx_to_predicate or pred_idx == 0:
                            gt_rels.append((sub_local, obj_local, pred_idx))
        
        gt_rels = torch.tensor(gt_rels, dtype=torch.long) if len(gt_rels) > 0 else torch.zeros((0, 3), dtype=torch.long)
        
        # Transform image
        image_tensor = self.transform(image)
        
        return {
            "image": image_tensor,
            "image_pil": image,  # Keep PIL for SAM3
            "gt_boxes": gt_boxes,  # [G, 4] normalized xyxy
            "gt_obj_labels": gt_obj_labels,  # List[int] (object class IDs)
            "gt_rels": gt_rels,  # [R, 3] (s_idx, o_idx, pred_idx)
            "image_id": img_id,
        }
    
    def _get_image_path(self, img_id: int) -> Optional[str]:
        """Get image path from image ID"""
        # Try to get from image_data.json
        if img_id in self.image_id_to_info:
            img_info = self.image_id_to_info[img_id]
            img_path = img_info.get("url", img_info.get("file_name", ""))
            if img_path:
                # Try different possible paths
                possible_paths = [
                    os.path.join(self.data_root, "images", img_path),
                    os.path.join(self.data_root, "images2", img_path),
                    os.path.join(self.data_root, img_path),
                    img_path,
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        return path
        
        # Fallback: try common naming patterns
        if self.image_dir:
            for ext in ['.jpg', '.png', '.jpeg']:
                for pattern in [f"{img_id}{ext}", f"{img_id:06d}{ext}"]:
                    path = os.path.join(self.image_dir, pattern)
                    if os.path.exists(path):
                        return path
        
        return None
    
    def get_predicate_vocab(self):
        """Return predicate vocabulary"""
        return {
            "predicate_to_idx": self.predicate_to_idx,
            "idx_to_predicate": self.idx_to_predicate,
            "num_predicates": self.num_predicates,
        }
    
    def __del__(self):
        """Close h5 file when dataset is deleted"""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass