"""
SAM3 Mask Generator Adapter
统一接口：predict(image_pil, boxes_xyxy_pixel) -> masks_bool[G,Hm,Wm], obj_emb[G,D] or None
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from PIL import Image

from sgg.models.frozen_sam3_gt import FrozenSAM3GT


@dataclass
class SAM3Output:
    masks: np.ndarray                 # [G, Hm, Wm] bool
    obj_emb: Optional[np.ndarray]     # [G, D] float32 or None


class BaseSAM3MaskGenerator:
    """Base class for SAM3 mask generators"""
    
    def __init__(self, mask_size: int = 256, device: str = "cuda"):
        self.mask_size = int(mask_size)
        self.device = device
    
    def predict(self, image: Image.Image, boxes_xyxy: np.ndarray) -> SAM3Output:
        """
        Generate masks for GT boxes
        
        Args:
            image: PIL Image
            boxes_xyxy: [G,4] float32 in original pixel coords (x1,y1,x2,y2)
            
        Returns:
            SAM3Output with masks [G, Hm, Wm] bool and optional embeddings
        """
        raise NotImplementedError


class RealSAM3MaskGenerator(BaseSAM3MaskGenerator):
    """
    Real SAM3 implementation using FrozenSAM3GT
    Generates masks from GT boxes using box prompts
    """
    
    def __init__(self, mask_size: int = 256, device: str = "cuda"):
        super().__init__(mask_size, device)
        print("Loading SAM3 model for mask generation...")
        self.sam3 = FrozenSAM3GT(device=device)
        print("SAM3 model loaded")
    
    def predict(self, image: Image.Image, boxes_xyxy: np.ndarray) -> SAM3Output:
        """
        Generate masks using real SAM3
        """
        G = boxes_xyxy.shape[0]
        if G == 0:
            return SAM3Output(
                masks=np.zeros((0, self.mask_size, self.mask_size), dtype=bool),
                obj_emb=None
            )
        
        W, H = image.size
        
        # Normalize boxes to [0, 1] for SAM3
        boxes_norm = boxes_xyxy.copy().astype(np.float32)
        boxes_norm[:, 0] /= W  # x1
        boxes_norm[:, 1] /= H  # y1
        boxes_norm[:, 2] /= W  # x2
        boxes_norm[:, 3] /= H  # y2
        
        boxes_tensor = torch.from_numpy(boxes_norm).to(self.device)
        
        # Get embeddings and masks from SAM3
        # Note: FrozenSAM3GT.forward_batch_boxes returns embeddings
        # We need to also get masks - let's check the implementation
        embs = self.sam3.forward_batch_boxes(image, boxes_tensor)  # [G, 256]
        
        # For now, we'll generate masks from boxes (rectangular)
        # TODO: If SAM3 provides masks directly, use them
        masks = np.zeros((G, self.mask_size, self.mask_size), dtype=bool)
        
        # Scale boxes to mask size
        sx = self.mask_size / max(W, 1)
        sy = self.mask_size / max(H, 1)
        
        for i in range(G):
            x1, y1, x2, y2 = boxes_xyxy[i]
            x1m = int(max(0, min(self.mask_size - 1, round(x1 * sx))))
            x2m = int(max(0, min(self.mask_size - 1, round(x2 * sx))))
            y1m = int(max(0, min(self.mask_size - 1, round(y1 * sy))))
            y2m = int(max(0, min(self.mask_size - 1, round(y2 * sy))))
            
            if x2m > x1m and y2m > y1m:
                masks[i, y1m:y2m, x1m:x2m] = True
        
        # Convert embeddings to numpy
        obj_emb = embs.cpu().numpy().astype(np.float32) if embs is not None else None
        
        return SAM3Output(masks=masks, obj_emb=obj_emb)


class DummySAM3MaskGenerator(BaseSAM3MaskGenerator):
    """
    Dummy generator: creates rectangular masks from GT boxes
    Useful for testing without SAM3
    """
    
    def predict(self, image: Image.Image, boxes_xyxy: np.ndarray) -> SAM3Output:
        W, H = image.size
        G = boxes_xyxy.shape[0]
        Hm = Wm = self.mask_size
        
        # Scale boxes to mask grid
        sx = Wm / max(W, 1)
        sy = Hm / max(H, 1)
        
        masks = np.zeros((G, Hm, Wm), dtype=bool)
        for i in range(G):
            x1, y1, x2, y2 = boxes_xyxy[i]
            x1m = int(max(0, min(Wm - 1, round(x1 * sx))))
            x2m = int(max(0, min(Wm - 1, round(x2 * sx))))
            y1m = int(max(0, min(Hm - 1, round(y1 * sy))))
            y2m = int(max(0, min(Hm - 1, round(y2 * sy))))
            
            if x2m > x1m and y2m > y1m:
                masks[i, y1m:y2m, x1m:x2m] = True
        
        return SAM3Output(masks=masks, obj_emb=None)

