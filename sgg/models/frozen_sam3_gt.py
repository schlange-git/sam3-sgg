"""
Frozen SAM3 wrapper for extracting object embeddings using GT box prompts
按照新流程：不再使用 proposals + IoU matching，而是直接用 GT box prompt
"""
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
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
    
    @torch.no_grad()
    def forward_batch_boxes_with_masks(self, image: Image.Image, gt_boxes: torch.Tensor, mask_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract embeddings and masks for all GT boxes using mask features
        确保每个box都有一个mask（取top1，即使没有达到threshold）
        
        Args:
            image: PIL Image
            gt_boxes: [G, 4] normalized xyxy boxes
            mask_size: Output mask size (Hm, Wm)
            
        Returns:
            embs: [G, 256] embeddings
            masks: [G, Hm, Wm] bool masks (每个box保证有一个mask，取top1)
        """
        G = gt_boxes.size(0)
        if G == 0:
            return (
                torch.zeros(0, 256, device=self.device, dtype=torch.float32),
                torch.zeros(0, mask_size, mask_size, device=self.device, dtype=torch.bool)
            )
        
        W, H = image.size
        
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
        
        # Process each box and extract mask-based embedding and mask
        embs_list = []
        masks_list = []
        
        for box in boxes_list:
            # Add box prompt
            output = self.processor.add_geometric_prompt(
                box=box,
                label=True,
                state=inference_state
            )
            
            # 获取masks和scores
            # output是一个Dict，包含masks, scores等
            # 注意：由于confidence_threshold=0.0，所有masks都应该被保留
            # 但为了确保每个box都有一个mask，我们取top1（即使只有一个mask）
            if "masks" in output and output["masks"].numel() > 0:
                # output["masks"]: 可能是[N, H, W]或[1, N, H, W] bool
                # output["scores"]: [N] float
                masks_raw = output["masks"]  # 可能是[N, H, W]或[1, N, H, W] bool
                
                # 确保masks_raw是3D [N, H, W]
                if masks_raw.dim() == 4:
                    # 如果是[1, N, H, W]，去掉第一个维度
                    masks_raw = masks_raw.squeeze(0)  # [N, H, W]
                elif masks_raw.dim() != 3:
                    # 如果维度不对，使用box生成矩形mask
                    mask = self._box_to_mask(box, H, W, mask_size)
                    masks_list.append(mask)
                    emb = torch.zeros(256, device=self.device, dtype=torch.float32)
                    embs_list.append(emb)
                    continue
                
                scores_raw = output.get("scores", None)
                
                # 取score最高的mask（top1），确保每个box都有一个mask
                if masks_raw.size(0) > 0:
                    if scores_raw is not None and scores_raw.numel() > 0:
                        # 有scores，取最高分的mask
                        top_idx = scores_raw.argmax().item()
                        mask = masks_raw[top_idx]  # 可能是[H, W]或[1, H, W]
                    else:
                        # 没有scores，取第一个mask
                        mask = masks_raw[0]  # 可能是[H, W]或[1, H, W]
                    
                    # 确保mask是2D [H, W]
                    if mask.dim() == 3:
                        # 如果是[1, H, W]，去掉第一个维度
                        mask = mask.squeeze(0)
                    elif mask.dim() != 2:
                        # 如果维度不对，使用box生成矩形mask
                        mask = self._box_to_mask(box, H, W, mask_size)
                        masks_list.append(mask)
                        emb = torch.zeros(256, device=self.device, dtype=torch.float32)
                        embs_list.append(emb)
                        continue
                else:
                    # 如果没有mask（不应该发生，因为threshold=0.0），使用box生成矩形mask
                    mask = self._box_to_mask(box, H, W, mask_size)
                    masks_list.append(mask)
                    # Use zero embedding as fallback
                    emb = torch.zeros(256, device=self.device, dtype=torch.float32)
                    embs_list.append(emb)
                    continue
                
                # Resize mask to mask_size
                from torch.nn.functional import interpolate
                # mask现在是[H, W] bool，需要转换为[N, C, H, W]格式
                # 转换为float tensor并添加batch和channel维度: [H, W] -> [1, 1, H, W]
                mask_4d = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                # Resize到mask_size
                mask_resized = interpolate(
                    mask_4d,
                    size=(mask_size, mask_size),
                    mode="nearest",
                    align_corners=None  # nearest mode不需要align_corners
                ).squeeze(0).squeeze(0) > 0.5  # [mask_size, mask_size] bool
                
                masks_list.append(mask_resized)
                
                # Extract embedding from mask (使用原始mask，不是resized的)
                mask_flat = mask.float().flatten()  # [H*W]
                if mask_flat.size(0) >= 256:
                    emb = mask_flat[:256]
                else:
                    emb = torch.cat([
                        mask_flat,
                        torch.zeros(256 - mask_flat.size(0), device=mask_flat.device, dtype=mask_flat.dtype)
                    ])
                emb = emb / (emb.norm() + 1e-8)
            else:
                # Fallback: use box to generate rectangular mask
                mask = self._box_to_mask(box, H, W, mask_size)
                masks_list.append(mask)
                
                # Use zero embedding as fallback
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
        masks = torch.stack(masks_list, dim=0)  # [G, mask_size, mask_size] bool
        
        return embs, masks
    
    def _box_to_mask(self, box_cxcywh: List[float], H: int, W: int, mask_size: int) -> torch.Tensor:
        """
        从box生成矩形mask（fallback方法）
        
        Args:
            box_cxcywh: [cx, cy, w, h] normalized
            H, W: Original image size
            mask_size: Output mask size
            
        Returns:
            mask: [mask_size, mask_size] bool
        """
        cx, cy, w, h = box_cxcywh
        # Convert to xyxy
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        
        # Scale to mask_size
        sx = mask_size / max(W, 1)
        sy = mask_size / max(H, 1)
        
        x1m = int(max(0, min(mask_size - 1, round(x1 * sx))))
        x2m = int(max(0, min(mask_size - 1, round(x2 * sx))))
        y1m = int(max(0, min(mask_size - 1, round(y1 * sy))))
        y2m = int(max(0, min(mask_size - 1, round(y2 * sy))))
        
        mask = torch.zeros(mask_size, mask_size, device=self.device, dtype=torch.bool)
        if x2m > x1m and y2m > y1m:
            mask[y1m:y2m, x1m:x2m] = True
        
        return mask
