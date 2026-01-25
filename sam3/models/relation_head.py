"""
Relation Head for Scene Graph Generation
"""
import torch
import torch.nn as nn


class RelationHead(nn.Module):
    """MLP-based relation head for predicate classification"""
    
    def __init__(
        self,
        emb_dim: int = 256,
        geom_dim: int = 6,
        num_predicates: int = 50,
        hidden: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            emb_dim: Dimension of object embeddings
            geom_dim: Dimension of geometric features
            num_predicates: Number of predicate classes (including background)
            hidden: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        in_dim = emb_dim * 2 + geom_dim  # Two object embeddings + geometric features
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_predicates),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [P, in_dim] pair features
            
        Returns:
            logits: [P, num_predicates]
        """
        return self.net(z)

