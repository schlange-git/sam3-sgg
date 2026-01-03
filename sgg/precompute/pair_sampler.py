"""
Pair Sampler: Build positive/negative pairs with fixed P
"""
from dataclasses import dataclass
import random
from typing import List, Set, Tuple
import numpy as np


@dataclass
class PairSample:
    pair_idx: np.ndarray     # [P,2] int64
    pair_label: np.ndarray   # [P] int64, 0=bg, >0 predicate, -1=pad
    pair_mask: np.ndarray    # [P] uint8, 1 valid, 0 pad


def sample_pairs_fixed_P(
    num_obj: int,
    rels: np.ndarray,          # [R,3] (s,o,pred) local indices, pred>0
    P: int = 128,
    neg_ratio: int = 3,
    seed: int = 0,
) -> PairSample:
    """
    Sample pairs with fixed size P
    
    Args:
        num_obj: Number of objects
        rels: [R,3] (s_local, o_local, pred_id) where pred_id > 0
        P: Fixed number of pairs
        neg_ratio: Ratio of negatives to positives
        seed: Random seed
        
    Returns:
        PairSample with fixed size P
    """
    rng = random.Random(seed)
    
    if num_obj < 2:
        pair_idx = np.full((P, 2), -1, dtype=np.int64)
        pair_label = np.full((P,), -1, dtype=np.int64)
        pair_mask = np.zeros((P,), dtype=np.uint8)
        return PairSample(pair_idx, pair_label, pair_mask)
    
    # Build positive edges: keep all (allow duplicates for multi-predicate)
    pos_edges: List[Tuple[int, int, int]] = []
    pos_pairs_set: Set[Tuple[int, int]] = set()
    
    for rel in rels:
        if len(rel) < 3:
            continue
        s, o, p = int(rel[0]), int(rel[1]), int(rel[2])
        if s == o:
            continue
        if s < 0 or s >= num_obj or o < 0 or o >= num_obj:
            continue
        if p <= 0:
            continue
        pos_edges.append((s, o, p))
        pos_pairs_set.add((s, o))
    
    # All ordered pairs (excluding self-loops)
    all_pairs = [(i, j) for i in range(num_obj) for j in range(num_obj) if i != j]
    neg_cands = [pair for pair in all_pairs if pair not in pos_pairs_set]
    
    # Decide how many negatives
    num_pos = len(pos_edges)
    if num_pos > P:
        # Too many positives: randomly downsample positives, no negatives
        chosen = rng.sample(pos_edges, P)
        pair_idx = np.array([[s, o] for (s, o, _) in chosen], dtype=np.int64)
        pair_label = np.array([p for (_, _, p) in chosen], dtype=np.int64)
        pair_mask = np.ones((P,), dtype=np.uint8)
        return PairSample(pair_idx, pair_label, pair_mask)
    
    num_neg = min(len(neg_cands), max(0, num_pos * neg_ratio))
    neg_pairs = rng.sample(neg_cands, num_neg) if len(neg_cands) > num_neg else neg_cands
    neg_edges = [(i, j, 0) for (i, j) in neg_pairs]  # 0 = background
    
    edges = pos_edges + neg_edges
    # Pad or subsample to P (keep all positives)
    if len(edges) > P:
        # Keep all pos, sample remaining from negatives
        pos_keep = pos_edges
        remain = P - len(pos_keep)
        if remain <= 0:
            chosen = rng.sample(pos_keep, P)
        else:
            chosen = pos_keep + rng.sample(neg_edges, min(remain, len(neg_edges)))
        edges = chosen
    
    # Pad to P if needed
    pair_idx = np.full((P, 2), -1, dtype=np.int64)
    pair_label = np.full((P,), -1, dtype=np.int64)
    pair_mask = np.zeros((P,), dtype=np.uint8)
    
    for k, (s, o, p) in enumerate(edges):
        if k >= P:
            break
        pair_idx[k, 0] = s
        pair_idx[k, 1] = o
        pair_label[k] = p
        pair_mask[k] = 1
    
    return PairSample(pair_idx, pair_label, pair_mask)

