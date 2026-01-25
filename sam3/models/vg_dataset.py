"""
Visual Genome Dataset Loader for SGG (Scene Graph Generation)
"""
import json
import os
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class VGDataset(Dataset):
    """Visual Genome Dataset for PredCls task"""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_objects: int = 50,
        max_relations: int = 200,
        image_size: int = 1008,
    ):
        """
        Args:
            data_root: Path to Visual Genome dataset root
            split: "train" or "val" (for now we use all data)
            max_objects: Maximum number of objects per image
            max_relations: Maximum number of relations per image
            image_size: Target image size for SAM3
        """
        self.data_root = data_root
        self.split = split
        self.max_objects = max_objects
        self.max_relations = max_relations
        self.image_size = image_size
        
        # Load annotations
        self.relationships_path = os.path.join(data_root, "relationships_v1_2.json", "relationships.json")
        self.objects_path = os.path.join(data_root, "objects_v1_2.json", "objects.json")
        self.image_data_path = os.path.join(data_root, "image_data.json")
        
        print(f"Loading Visual Genome dataset from {data_root}...")
        
        # Load all data
        with open(self.relationships_path, 'r') as f:
            self.relationships_data = json.load(f)
        
        with open(self.objects_path, 'r') as f:
            self.objects_data = json.load(f)
        
        with open(self.image_data_path, 'r') as f:
            self.image_data = json.load(f)
        
        # Create image_id to data mapping
        self.image_id_to_rels = {item['image_id']: item for item in self.relationships_data}
        self.image_id_to_objs = {item['image_id']: item for item in self.objects_data}
        self.image_id_to_info = {item['image_id']: item for item in self.image_data}
        
        # Build predicate vocabulary
        self.predicate_vocab = self._build_predicate_vocab()
        self.num_predicates = len(self.predicate_vocab)
        print(f"Found {self.num_predicates} unique predicates")
        
        # Filter valid images (must have both objects and relationships)
        self.valid_image_ids = [
            img_id for img_id in self.image_id_to_info.keys()
            if img_id in self.image_id_to_objs and img_id in self.image_id_to_rels
            and len(self.image_id_to_objs[img_id]['objects']) > 0
            and len(self.image_id_to_rels[img_id]['relationships']) > 0
        ]
        
        print(f"Found {len(self.valid_image_ids)} valid images")
        
        # Image transforms
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _build_predicate_vocab(self, min_freq: int = 1, max_predicates: int = 500) -> Dict[str, int]:
        """Build predicate vocabulary from all relationships
        
        Args:
            min_freq: Minimum frequency for a predicate to be included
            max_predicates: Maximum number of predicates (excluding background)
        """
        # Count predicate frequencies
        pred_counts = {}
        for item in self.relationships_data:
            for rel in item.get('relationships', []):
                pred = rel.get('predicate', '').strip().upper()
                if pred:
                    pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        # Filter by frequency and get top N
        filtered_preds = [
            pred for pred, count in pred_counts.items()
            if count >= min_freq
        ]
        # Sort by frequency (descending) then alphabetically
        filtered_preds = sorted(
            filtered_preds,
            key=lambda p: (-pred_counts[p], p)
        )[:max_predicates]
        
        # Add background class at index 0
        vocab = {'__background__': 0}
        for i, pred in enumerate(filtered_preds, start=1):
            vocab[pred] = i
        
        print(f"Built predicate vocab: {len(vocab)-1} predicates (filtered from {len(pred_counts)} unique)")
        return vocab
    
    def _get_image_path(self, image_id: int) -> str:
        """Get image file path"""
        info = self.image_id_to_info[image_id]
        # Try VG_100K first, then VG_100K_2
        for folder in ['VG_100K', 'VG_100K_2']:
            path = os.path.join(self.data_root, 'images', folder, f"{image_id}.jpg")
            if os.path.exists(path):
                return path
            path = os.path.join(self.data_root, 'images2', folder, f"{image_id}.jpg")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image {image_id}.jpg not found")
    
    def __len__(self) -> int:
        return len(self.valid_image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        image_id = self.valid_image_ids[idx]
        
        # Load image
        image_path = self._get_image_path(image_id)
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        image_tensor = self.transform(image)
        
        # Get objects and relationships
        objs_data = self.image_id_to_objs[image_id]['objects']
        rels_data = self.image_id_to_rels[image_id]['relationships']
        
        # Build object_id to index mapping
        obj_id_to_idx = {}
        gt_boxes = []
        gt_obj_labels = []
        
        for i, obj in enumerate(objs_data):
            obj_id = obj['object_id']
            obj_id_to_idx[obj_id] = i
            
            # Convert (x, y, w, h) to (x1, y1, x2, y2)
            x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            
            # Normalize to [0, 1] based on original image size
            x1_norm = x1 / original_size[0]
            y1_norm = y1 / original_size[1]
            x2_norm = x2 / original_size[0]
            y2_norm = y2 / original_size[1]
            
            gt_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
            # For PredCls, we don't need object labels, but we keep them for compatibility
            gt_obj_labels.append(0)  # Dummy label
        
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        gt_obj_labels = torch.tensor(gt_obj_labels, dtype=torch.long)
        
        # Build relationships: (subject_idx, object_idx, predicate_id)
        gt_rels = []
        for rel in rels_data:
            sub_id = rel['subject']['object_id']
            obj_id = rel['object']['object_id']
            predicate = rel.get('predicate', '').strip().upper()
            
            if sub_id in obj_id_to_idx and obj_id in obj_id_to_idx:
                sub_idx = obj_id_to_idx[sub_id]
                obj_idx = obj_id_to_idx[obj_id]
                
                if sub_idx != obj_idx:  # Skip self-relations
                    pred_id = self.predicate_vocab.get(predicate, 0)  # 0 is background
                    if pred_id > 0:  # Only add non-background relations
                        gt_rels.append([sub_idx, obj_idx, pred_id])
        
        if len(gt_rels) == 0:
            # If no valid relations, create a dummy one
            gt_rels = [[0, 0, 0]] if len(gt_boxes) >= 2 else []
        
        gt_rels = torch.tensor(gt_rels, dtype=torch.long)
        
        # Limit number of objects and relations
        if len(gt_boxes) > self.max_objects:
            gt_boxes = gt_boxes[:self.max_objects]
            gt_obj_labels = gt_obj_labels[:self.max_objects]
            # Filter relations that reference removed objects
            mask = (gt_rels[:, 0] < self.max_objects) & (gt_rels[:, 1] < self.max_objects)
            gt_rels = gt_rels[mask]
        
        if len(gt_rels) > self.max_relations:
            gt_rels = gt_rels[:self.max_relations]
        
        return {
            'image': image_tensor,
            'image_id': image_id,
            'gt_boxes': gt_boxes,  # [G, 4] normalized xyxy
            'gt_obj_labels': gt_obj_labels,  # [G] (dummy for PredCls)
            'gt_rels': gt_rels,  # [R, 3] (s_idx, o_idx, pred_id)
            'original_size': torch.tensor(original_size, dtype=torch.long),  # [W, H]
        }
    
    def get_predicate_vocab(self) -> Dict[str, int]:
        """Get predicate vocabulary"""
        return self.predicate_vocab.copy()

