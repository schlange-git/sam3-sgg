"""
VG150 Reader for offline preprocessing
基于现有的 vg150_dataset.py，提供简化的接口用于离线缓存生成
完全复用官方demo的box转换方式和索引查找方式
"""
from dataclasses import dataclass
from typing import Iterator, Optional, Dict, List
import os
import json
import numpy as np
from PIL import Image
import h5py

from sgg.datasets.vg150_dataset import VG150Dataset


@dataclass
class VGSample:
    """Single VG150 sample for preprocessing"""
    image_id: int
    image_path: str
    width: int
    height: int
    boxes_xyxy: np.ndarray        # [G,4] float32 in pixel coords (x1,y1,x2,y2)
    obj_labels: np.ndarray        # [G] int64 (1..150)
    rels: np.ndarray              # [R,3] int64 (s_local, o_local, pred_id in 1..K)


class VG150Reader:
    """
    Simplified reader for VG150 data, based on existing VG150Dataset
    Used for offline preprocessing only
    """
    
    def __init__(self, data_root: str, split: str = "train"):
        """
        Args:
            data_root: Path to VG150 dataset root
            split: "train" or "val"
        """
        # Reuse existing dataset implementation
        self.dataset = VG150Dataset(
            data_root=data_root,
            split=split,
            image_size=1008,  # Not used for preprocessing, but required
            max_objects=50,
            max_relations=200,
        )
        self.data_root = data_root
        self.split = split
        
        # 直接打开h5文件，使用官方demo的转换方式
        h5_path = os.path.join(data_root, "VG-SGG-with-attri.h5")
        if not os.path.exists(h5_path):
            h5_path = os.path.join(data_root, "VG-SGG.h5")
        self.h5_file = h5py.File(h5_path, 'r')
        
        # 确定使用boxes_1024还是boxes_512
        if 'boxes_1024' in self.h5_file:
            self.box_key = 'boxes_1024'
            self.USE_BOX_SIZE = 1024
        elif 'boxes_512' in self.h5_file:
            self.box_key = 'boxes_512'
            self.USE_BOX_SIZE = 512
        else:
            raise ValueError("Cannot find boxes_1024 or boxes_512 in h5 file")
        
        # 完全复用官方demo的方式：从image_data.json加载image_info
        # 这样可以直接通过image_id查找对应的索引
        image_data_path = os.path.join(data_root, 'image_data.json')
        if os.path.exists(image_data_path):
            image_data = json.load(open(image_data_path))
            # 过滤掉corrupted images（与官方demo一致）
            corrupted_ims = ['1592', '1722', '4616', '4617']
            self.image_info = []
            for item in image_data:
                if str(item.get('image_id', '')) not in corrupted_ims:
                    self.image_info.append(item)
            # 建立image_id到索引的映射（与官方demo一致）
            self.image_id_to_idx = {}
            for i, img_info in enumerate(self.image_info):
                img_id = img_info.get('image_id')
                if img_id is not None:
                    self.image_id_to_idx[img_id] = i
        else:
            self.image_info = []
            self.image_id_to_idx = {}
            print(f"Warning: image_data.json not found at {image_data_path}, falling back to dataset-based lookup")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def iter_samples(self) -> Iterator[VGSample]:
        """
        Iterate over all samples, yielding VGSample objects
        完全复用官方demo的box转换方式
        现在使用与get_sample_by_image_id相同的索引映射方式，确保一致性
        """
        # 完全复用官方demo的方式：遍历image_info中的所有image_id
        # 这样确保与get_sample_by_image_id使用相同的索引映射
        for img_info in self.image_info:
            image_id = img_info.get('image_id')
            if image_id is None:
                continue
            
            # 使用get_sample_by_image_id获取样本（确保一致性）
            sample = self.get_sample_by_image_id(image_id)
            if sample is not None:
                yield sample
    
    def get_sample_by_image_id(self, image_id: int) -> Optional[VGSample]:
        """
        通过image_id直接获取样本（完全复用官方demo的方式）
        直接从image_info中查找image_id对应的索引，然后从h5文件读取
        """
        # 完全复用官方demo的方式：从image_info中查找image_id对应的索引
        if image_id not in self.image_id_to_idx:
            # image_id不存在
            return None
        
        img_idx = self.image_id_to_idx[image_id]
        img_info = self.image_info[img_idx]
        
        # 验证image_id是否匹配
        if img_info.get('image_id') != image_id:
            print(f"Warning: Image ID mismatch! Expected {image_id}, got {img_info.get('image_id')}")
            return None
        
        # 获取图像尺寸
        width = img_info.get('width')
        height = img_info.get('height')
        if width is None or height is None:
            print(f"Warning: Image {image_id} missing width/height in image_info")
            return None
        
        # 完全复用官方demo的box转换方式
        ith_s = int(self.h5_file['img_to_first_box'][img_idx])
        ith_e = int(self.h5_file['img_to_last_box'][img_idx])
        
        if ith_e >= ith_s:
            # 完全复用官方demo：从boxes_1024转换
            wrong_boxes = self.h5_file[self.box_key][ith_s : ith_e+1].copy()
            
            # 转换为x1,y1,x2,y2格式（仍在USE_BOX_SIZE坐标系中）
            wrong_boxes = wrong_boxes.astype(np.float32)
            wrong_boxes[:, :2] = wrong_boxes[:, :2] - wrong_boxes[:, 2:] / 2
            wrong_boxes[:, 2:] = wrong_boxes[:, :2] + wrong_boxes[:, 2:]
            
            # 转换回原始图像尺寸（完全复用官方demo）
            boxes_xyxy = wrong_boxes.astype(np.float32) / self.USE_BOX_SIZE * max(height, width)
            
            # Clamp to image boundaries
            boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, width)
            boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, height)
            boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, width)
            boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, height)
            
            # Ensure valid boxes (x2 > x1, y2 > y1)
            for i in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[i]
                if x2 <= x1:
                    boxes_xyxy[i, 2] = x1 + 1.0
                if y2 <= y1:
                    boxes_xyxy[i, 3] = y1 + 1.0
        else:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        
        # 获取labels和relationships
        if ith_e >= ith_s:
            obj_labels = self.h5_file['labels'][ith_s : ith_e+1].astype(np.int64)
            if len(obj_labels.shape) > 1:
                obj_labels = obj_labels.flatten()
        else:
            obj_labels = np.zeros((0,), dtype=np.int64)
        
        # 获取relationships（完全复用vg150_dataset的逻辑）
        first_rel = int(self.h5_file['img_to_first_rel'][img_idx])
        last_rel = int(self.h5_file['img_to_last_rel'][img_idx])
        if last_rel >= first_rel:
            # 读取relationships和predicates（与vg150_dataset一致）
            rels = self.h5_file['relationships'][first_rel:last_rel+1].astype(np.int64)
            
            # 检查是否需要合并predicates
            if rels.shape[1] == 2 and 'predicates' in self.h5_file:
                # relationships是[R, 2]，需要合并predicates
                preds = self.h5_file['predicates'][first_rel:last_rel+1].astype(np.int64)
                preds_flat = preds.flatten() if len(preds.shape) > 1 else preds
                # 合并：[R, 2] + [R] -> [R, 3]
                rels = np.column_stack([
                    rels[:, 0],  # sub_idx
                    rels[:, 1],  # obj_idx
                    preds_flat   # pred_idx
                ])
            elif rels.shape[1] != 3:
                # 如果既不是2列也不是3列，报错
                raise ValueError(f"Unexpected relationships shape: {rels.shape}")
            
            # 转换为local索引
            rels_local = rels.copy()
            for i in range(len(rels_local)):
                if len(rels_local[i]) >= 2:
                    rels_local[i, 0] = rels_local[i, 0] - ith_s  # sub_idx -> local
                    rels_local[i, 1] = rels_local[i, 1] - ith_s  # obj_idx -> local
        else:
            rels_local = np.zeros((0, 3), dtype=np.int64)
        
        # Get image path（完全复用官方demo的方式）
        image_path = None
        possible_paths = [
            os.path.join(self.data_root, "images", "VG_100K", f"{image_id}.jpg"),
            os.path.join(self.data_root, "images2", "VG_100K_2", f"{image_id}.jpg"),
            os.path.join(self.data_root, "images", "VG_100K", f"{image_id:06d}.jpg"),
            os.path.join(self.data_root, "images2", "VG_100K_2", f"{image_id:06d}.jpg"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if image_path is None:
            print(f"Warning: Image {image_id} not found in any directory")
            return None
        
        return VGSample(
            image_id=image_id,
            image_path=image_path,
            width=width,
            height=height,
            boxes_xyxy=boxes_xyxy,
            obj_labels=obj_labels,
            rels=rels_local,
        )
    
    def get_image_pil(self, image_id: int) -> Optional[Image.Image]:
        """Get PIL image by image_id"""
        sample = self.get_sample_by_image_id(image_id)
        if sample is not None:
            return Image.open(sample.image_path).convert("RGB")
        return None
    
    def close(self) -> None:
        """Close underlying dataset resources"""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
        if hasattr(self.dataset, 'h5_file') and self.dataset.h5_file is not None:
            try:
                self.dataset.h5_file.close()
            except Exception:
                pass

