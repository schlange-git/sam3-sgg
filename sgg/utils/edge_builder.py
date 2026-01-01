"""
Build edges and sample pairs for SGG training
按照新流程：使用 edge list 而不是 label matrix
"""
import torch
import random
from typing import List, Tuple, Set


def build_edges(gt_rels: torch.Tensor) -> Tuple[List[Tuple[int, int, int]], Set[Tuple[int, int]]]:
    """
    Build positive edges from GT relations
    
    Args:
        gt_rels: [R, 3] tensor with (s_idx, o_idx, pred_id) where pred_id in [1..K]
        
    Returns:
        pos_edges: List of (s, o, p) tuples
        pos_pairs: Set of (s, o) tuples (for negative sampling)
    """
    pos_edges = []
    pos_pairs = set()
    
    for rel in gt_rels:
        s, o, p = rel[0].item(), rel[1].item(), rel[2].item()
        if s != o and p > 0:  # Skip background (p=0) and self-loops
            pos_edges.append((s, o, p))
            pos_pairs.add((s, o))
    
    return pos_edges, pos_pairs


def sample_neg_pairs(
    num_objs: int,
    pos_pairs: Set[Tuple[int, int]],
    neg_ratio: int = 3,
    max_negs: int = None,
) -> List[Tuple[int, int, int]]:
    """
    Sample negative pairs (background relations)
    
    Args:
        num_objs: Number of objects
        pos_pairs: Set of positive (s, o) pairs
        neg_ratio: Ratio of negatives to positives
        max_negs: Maximum number of negatives (default: 50 if no positives)
        
    Returns:
        neg_edges: List of (s, o, 0) tuples (0 = background)
    """
    # Generate all possible ordered pairs
    all_pairs = [(i, j) for i in range(num_objs) for j in range(num_objs) if i != j]
    
    # Filter out positive pairs
    neg_cands = [pair for pair in all_pairs if pair not in pos_pairs]
    
    # Determine number of negatives
    if len(pos_pairs) > 0:
        n_neg = int(len(pos_pairs) * neg_ratio)
    else:
        n_neg = min(len(neg_cands), max_negs if max_negs is not None else 50)
    
    if max_negs is not None:
        n_neg = min(n_neg, max_negs)
    
    # Sample negatives
    if len(neg_cands) > 0:
        neg_pairs = random.sample(neg_cands, min(n_neg, len(neg_cands)))
    else:
        neg_pairs = []
    
    # Create negative edges with background label (0)
    neg_edges = [(i, j, 0) for (i, j) in neg_pairs]
    
    return neg_edges


def build_pair_features(
    embs: torch.Tensor,
    boxes: torch.Tensor,
    edges: List[Tuple[int, int, int]],
    geom_feat_fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build pair features and labels from edges
    
    Args:
        embs: [G, D] object embeddings
        boxes: [G, 4] normalized xyxy boxes
        edges: List of (s, o, p) tuples
        geom_feat_fn: Function to compute geometric features (box_geom_feat)
        
    Returns:
        feats: [P, 2*D + 6] pair features
        labels: [P] predicate labels
    """
    feats = []
    labels = []
    
    for s, o, p in edges:
        # Concatenate embeddings and geometric features
        geom = geom_feat_fn(boxes[s], boxes[o])  # [6]
        z = torch.cat([embs[s], embs[o], geom], dim=0)  # [2*D + 6]
        feats.append(z)
        labels.append(p)
    
    if len(feats) == 0:
        # Return empty tensors
        D = embs.size(1)
        return torch.zeros(0, 2 * D + 6, device=embs.device, dtype=embs.dtype), \
               torch.zeros(0, dtype=torch.long, device=embs.device)
    
    feats = torch.stack(feats, dim=0)  # [P, 2*D + 6]
    labels = torch.tensor(labels, dtype=torch.long, device=embs.device)  # [P]
    
    return feats, labels
