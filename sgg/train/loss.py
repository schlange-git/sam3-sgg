"""
Masked Cross Entropy Loss
"""
import torch
import torch.nn as nn


class MaskedCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with masking support
    Only computes loss on valid pairs (mask==True and label>=0)
    """
    
    def __init__(self, num_classes: int, bg_weight: float = 0.2):
        """
        Args:
            num_classes: Number of predicate classes (including background)
            bg_weight: Weight for background class (0)
        """
        super().__init__()
        w = torch.ones(num_classes, dtype=torch.float32)
        w[0] = bg_weight  # Background weight
        self.register_buffer("weight", w)
        self.ce = nn.CrossEntropyLoss(weight=self.weight, reduction="none")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, C] logits
            labels: [N] labels (can contain -1 for padding)
            mask: [N] bool mask (True for valid pairs)
            
        Returns:
            Scalar loss
        """
        # Compute loss for all positions
        loss = self.ce(logits, labels.clamp_min(0))  # [N]
        
        # Apply mask: only valid pairs with label >= 0
        valid_mask = mask & (labels >= 0)
        loss = loss[valid_mask]
        
        if loss.numel() == 0:
            return logits.sum() * 0.0  # Return zero loss if no valid pairs
        
        return loss.mean()

