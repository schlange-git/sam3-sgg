"""
Frozen SAM3 wrapper for extracting proposals
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class FrozenSAM3(nn.Module):
    """Frozen SAM3 model for extracting mask proposals and embeddings"""
    
    def __init__(self, confidence_threshold: float = 0.02, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        print("Loading SAM3 model...")
        # Load SAM3 model
        self.sam3_model = build_sam3_image_model()
        self.sam3_model = self.sam3_model.to(device)
        self.sam3_model.eval()
        
        # Freeze all parameters
        for p in self.sam3_model.parameters():
            p.requires_grad_(False)
        
        # Create processor
        self.processor = Sam3Processor(self.sam3_model, confidence_threshold=confidence_threshold)
        
        print("SAM3 model loaded and frozen")
    
    @torch.no_grad()
    def forward(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract proposals from image
        
        Args:
            image: PIL Image
            
        Returns:
            masks: [N, H, W] bool tensor
            embs: [N, D] float tensor (mask embeddings)
            scores: [N] float tensor (confidence scores)
        """
        # Set image
        inference_state = self.processor.set_image(image)
        
        # Use a generic text prompt to get all objects
        # We use "object" as a general prompt
        output = self.processor.set_text_prompt(state=inference_state, prompt="object")
        
        masks = output["masks"]  # [N, 1, H, W] or [N, H, W]
        boxes = output["boxes"]  # [N, 4]
        scores = output["scores"]  # [N]
        
        # Squeeze mask dimensions if needed
        if masks.dim() == 4:
            masks = masks.squeeze(1)  # [N, H, W]
        
        # Try to extract embeddings from model's internal state
        # The model outputs queries in the grounding forward pass
        # We'll use the backbone output features as embeddings
        # For now, we use a simple approach: average pool mask features
        # In practice, you might want to extract query tokens from the decoder
        
        N = masks.size(0)
        D = 256  # Typical embedding dimension
        
        # Use mask features as embeddings (simple approach)
        # Convert masks to float and use them as features
        # In a more sophisticated approach, you'd extract query embeddings
        # from the transformer decoder outputs
        if N == 0:
            # Return empty tensors if no objects detected
            embs = torch.zeros(0, D, device=masks.device, dtype=torch.float32)
        else:
            mask_features = masks.float()  # [N, H, W]
            # Average pool to get embeddings
            H, W = mask_features.shape[1:]
            embs = mask_features.view(N, -1)  # [N, H*W]
            # Project to D dimensions using a simple linear layer or use PCA-like approach
            # For now, we'll use a simple approach: take first D dimensions or use mean pooling
            if embs.size(1) >= D:
                embs = embs[:, :D]
            else:
                # Pad or repeat if needed
                padding = torch.zeros(N, D - embs.size(1), device=embs.device, dtype=embs.dtype)
                embs = torch.cat([embs, padding], dim=1)
            
            # Normalize embeddings
            embs = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
        
        # Convert masks to bool
        masks = masks > 0.5
        
        return masks, embs, scores
    
    @torch.no_grad()
    def extract_embeddings(self, image: Image.Image, boxes: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings for specific boxes
        
        Args:
            image: PIL Image
            boxes: [N, 4] normalized xyxy boxes
            
        Returns:
            embs: [N, D] embeddings
        """
        # This is a placeholder - in practice you'd extract embeddings
        # from the model's internal representations
        N = boxes.size(0)
        D = 256
        return torch.randn(N, D, device=boxes.device, dtype=torch.float32)

