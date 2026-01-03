"""
Relation Head for Geometry-only Features
Only uses geometric features (box_geom + mask_geom), no embeddings
"""
import torch
import torch.nn as nn


class RelationHeadMLP(nn.Module):
    """
    MLP-based relation head using only geometric features
    Input: geom_feat [P, Gd] where Gd = 11 (6 box + 5 mask)
    Output: logits [P, num_classes]
    """
    
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.1) -> None:
        """
        Args:
            in_dim: Input dimension (geom_dim, typically 11)
            num_classes: Number of predicate classes (including background)
            hidden: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] geometric features
            
        Returns:
            logits: [N, num_classes]
        """
        return self.net(x)

