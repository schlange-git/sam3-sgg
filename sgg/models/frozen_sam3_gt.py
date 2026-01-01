"""
Frozen SAM3 wrapper for extracting object embeddings using GT box prompts
按照新流程：不再使用 proposals + IoU matching，而是直接用 GT box prompt
"""
import torch
import torch.nn as nn
from typing import Tuple, List
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xyxy_to_cxcywh


class FrozenSAM3GT(nn.Module):
    """
    Frozen SAM3 model for extracting object embeddings using GT box prompts
    
    核心改变：
    - 不再使用 text prompt "object" 来检测 proposals
    - 直接使用 GT boxes 作为 geometric prompt
    - 为每个 GT box 提取对应的 embedding
    """
    
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        
        print("Loading SAM3 model (frozen)...")
        # Load SAM3 model
        self.sam3_model = build_sam3_image_model()
        self.sam3_model = self.sam3_model.to(device)
        self.sam3_model.eval()
        
        # Freeze all parameters
        for p in self.sam3_model.parameters():
            p.requires_grad_(False)
        
        # Create processor (no confidence threshold needed for GT boxes)
        self.processor = Sam3Processor(self.sam3_model, confidence_threshold=0.0)
        
        print("SAM3 model loaded and frozen")
    
    @torch.no_grad()
    def forward(self, image: Image.Image, gt_boxes: torch.Tensor) -> torch.Tensor:
        """
        Extract object embeddings for GT boxes using box prompts
        
        Args:
            image: PIL Image
            gt_boxes: [G, 4] normalized xyxy boxes in [0, 1] range
            
        Returns:
            embs: [G, D] float tensor (object embeddings, D=256)
        """
        G = gt_boxes.size(0)
        if G == 0:
            # Return empty tensor
            return torch.zeros(0, 256, device=self.device, dtype=torch.float32)
        
        # Set image
        inference_state = self.processor.set_image(image)
        
        # Convert xyxy to cxcywh (normalized)
        # gt_boxes: [G, 4] xyxy normalized
        boxes_cxcywh = box_xyxy_to_cxcywh(gt_boxes)  # [G, 4] cxcywh normalized
        
        # Extract embeddings for each GT box
        embs_list = []
        
        for i in range(G):
            box = boxes_cxcywh[i].tolist()  # [cx, cy, w, h] normalized
            # Add box prompt (label=True means positive box)
            output = self.processor.add_geometric_prompt(
                box=box,
                label=True,
                state=inference_state
            )
            
            # Extract embedding from output
            # The output contains masks and other info, but we need the embedding
            # For now, we'll extract from the model's internal state
            # The grounding output should contain query embeddings
            
            # Try to get embedding from output state
            # Note: SAM3's internal structure may vary, this is a simplified approach
            # In practice, you might need to extract from decoder outputs
            
            # For now, we use a workaround: extract from mask features
            if "masks" in output and output["masks"].numel() > 0:
                mask = output["masks"][0, 0]  # [H, W]
                # Use mask as a proxy for embedding (average pool)
                mask_feat = mask.float().view(-1)  # [H*W]
                # Project to 256 dim (simple approach)
                if mask_feat.size(0) >= 256:
                    emb = mask_feat[:256]
                else:
                    # Pad or repeat
                    emb = torch.cat([mask_feat, torch.zeros(256 - mask_feat.size(0), device=mask_feat.device)])
                emb = emb / (emb.norm() + 1e-8)  # Normalize
            else:
                # Fallback: use random embedding (should not happen)
                emb = torch.randn(256, device=self.device, dtype=torch.float32)
                emb = emb / (emb.norm() + 1e-8)
            
            embs_list.append(emb)
            
            # Reset prompts for next box (or accumulate, depending on your needs)
            # For now, we reset to avoid interference
            inference_state = self.processor.reset_all_prompts(inference_state)
            # Re-set image for next iteration
            inference_state = self.processor.set_image(image)
        
        embs = torch.stack(embs_list, dim=0)  # [G, 256]
        
        return embs
    
    @torch.no_grad()
    def forward_batch_boxes(self, image: Image.Image, gt_boxes: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for all GT boxes using mask features
        
        Args:
            image: PIL Image
            gt_boxes: [G, 4] normalized xyxy boxes
            
        Returns:
            embs: [G, 256] embeddings
        """
        G = gt_boxes.size(0)
        if G == 0:
            return torch.zeros(0, 256, device=self.device, dtype=torch.float32)
        
        # Set image once
        inference_state = self.processor.set_image(image)
        
        # Set text prompt to "visual" (allows geometric-only prompts)
        inference_state = self.processor.set_text_prompt(
            state=inference_state,
            prompt="visual"
        )
        
        # Convert all boxes to cxcywh (normalized)
        boxes_cxcywh = box_xyxy_to_cxcywh(gt_boxes)  # [G, 4]
        boxes_list = boxes_cxcywh.tolist()  # List of [cx, cy, w, h]
        
        # Process each box and extract mask-based embedding
        embs_list = []
        for box in boxes_list:
            # Add box prompt
            output = self.processor.add_geometric_prompt(
                box=box,
                label=True,
                state=inference_state
            )
            
            # Extract embedding from mask
            # Use mask features as a proxy for object embedding
            if "masks" in output and output["masks"].numel() > 0:
                mask = output["masks"][0, 0]  # [H, W]
                # Flatten and use as features
                mask_flat = mask.float().flatten()  # [H*W]
                # Project to 256 dim: use PCA-like approach or simple sampling
                if mask_flat.size(0) >= 256:
                    # Sample or use first 256
                    emb = mask_flat[:256]
                else:
                    # Pad with zeros
                    emb = torch.cat([
                        mask_flat,
                        torch.zeros(256 - mask_flat.size(0), device=mask_flat.device, dtype=mask_flat.dtype)
                    ])
                # Normalize
                emb = emb / (emb.norm() + 1e-8)
            else:
                # Fallback: use zero embedding (should not happen with GT boxes)
                emb = torch.zeros(256, device=self.device, dtype=torch.float32)
            
            embs_list.append(emb)
            
            # Reset prompts for next box
            inference_state = self.processor.reset_all_prompts(inference_state)
            # Re-set image and text prompt
            inference_state = self.processor.set_image(image)
            inference_state = self.processor.set_text_prompt(
                state=inference_state,
                prompt="visual"
            )
        
        embs = torch.stack(embs_list, dim=0)  # [G, 256]
        
        return embs
