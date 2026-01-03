"""
Geometry Feature Extractor: Compute box_geom + mask_geom
"""
from dataclasses import dataclass
import numpy as np


def _safe_log(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Safe logarithm"""
    return np.log(np.maximum(x, eps))


def box_iou_xyxy(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute IoU between boxes
    
    Args:
        a: [P,4] xyxy
        b: [P,4] xyxy
    Returns:
        [P] IoU values
    """
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 2], b[:, 2])
    y2 = np.minimum(a[:, 3], b[:, 3])
    
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h
    
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a + area_b - inter
    return inter / (union + eps)


def box_geom_features_xyxy(box_s: np.ndarray, box_o: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute box geometric features
    
    Args:
        box_s, box_o: [P,4] xyxy in pixel coords
    Returns:
        [P,6] -> dx,dy,dw,dh,da,iou
    """
    xs1, ys1, xs2, ys2 = box_s[:, 0], box_s[:, 1], box_s[:, 2], box_s[:, 3]
    xo1, yo1, xo2, yo2 = box_o[:, 0], box_o[:, 1], box_o[:, 2], box_o[:, 3]
    
    # Centers
    cxs = (xs1 + xs2) * 0.5
    cys = (ys1 + ys2) * 0.5
    cxo = (xo1 + xo2) * 0.5
    cyo = (yo1 + yo2) * 0.5
    
    # Widths and heights (ensure minimum size to avoid extreme ratios)
    ws = np.maximum(xs2 - xs1, eps)
    hs = np.maximum(ys2 - ys1, eps)
    wo = np.maximum(xo2 - xo1, eps)
    ho = np.maximum(yo2 - yo1, eps)
    
    # Additional safety: if boxes are extremely large, normalize them
    # This handles cases where boxes are in pixel coordinates but values are huge
    max_box_dim = max(ws.max(), hs.max(), wo.max(), ho.max()) if len(ws) > 0 else 1.0
    if max_box_dim > 10000:  # Likely unnormalized pixel coordinates
        # Normalize by image size (assume max dimension ~1000-2000 pixels)
        norm_factor = max_box_dim / 1000.0
        ws = ws / norm_factor
        hs = hs / norm_factor
        wo = wo / norm_factor
        ho = ho / norm_factor
        cxs = cxs / norm_factor
        cys = cys / norm_factor
        cxo = cxo / norm_factor
        cyo = cyo / norm_factor
    
    # Relative position (normalized by object size)
    # Use safe division to avoid NaN/Inf
    dx = np.divide(cxo - cxs, wo, out=np.zeros_like(cxo), where=(wo > eps))
    dy = np.divide(cyo - cys, ho, out=np.zeros_like(cyo), where=(ho > eps))
    dw = _safe_log(ws / wo, eps)
    dh = _safe_log(hs / ho, eps)
    da = _safe_log((ws * hs) / (wo * ho), eps)
    iou = box_iou_xyxy(box_s, box_o, eps)
    
    feat = np.stack([dx, dy, dw, dh, da, iou], axis=1).astype(np.float32)
    
    # Clip extreme values to prevent numerical overflow
    # dx, dy: typically in range [-10, 10] for reasonable object positions
    # dw, dh, da: log ratios, typically in range [-5, 5]
    # iou: in range [0, 1]
    feat[:, 0] = np.clip(feat[:, 0], -10.0, 10.0)  # dx
    feat[:, 1] = np.clip(feat[:, 1], -10.0, 10.0)  # dy
    feat[:, 2] = np.clip(feat[:, 2], -5.0, 5.0)   # dw
    feat[:, 3] = np.clip(feat[:, 3], -5.0, 5.0)   # dh
    feat[:, 4] = np.clip(feat[:, 4], -5.0, 5.0)   # da
    feat[:, 5] = np.clip(feat[:, 5], 0.0, 1.0)    # iou
    
    # Replace any remaining NaN/Inf with 0
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feat


def mask_pair_geom_features(
    masks_i: np.ndarray,   # [P,H,W] bool
    masks_j: np.ndarray,   # [P,H,W] bool
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute mask geometric features
    
    Returns [P,5]:
      mask_iou, inter_over_min, contain_i_in_j, contain_j_in_i, rel_area_log
    """
    mi = masks_i.astype(np.bool_)
    mj = masks_j.astype(np.bool_)
    
    # Flatten masks and compute areas
    Ai = mi.reshape(mi.shape[0], -1).sum(axis=1).astype(np.float32)
    Aj = mj.reshape(mj.shape[0], -1).sum(axis=1).astype(np.float32)
    
    # Intersection and union
    inter = np.logical_and(mi, mj).reshape(mi.shape[0], -1).sum(axis=1).astype(np.float32)
    union = np.logical_or(mi, mj).reshape(mi.shape[0], -1).sum(axis=1).astype(np.float32)
    
    # Safe division to avoid NaN/Inf
    mask_iou = np.divide(inter, union + eps, out=np.zeros_like(inter), where=(union + eps > 0))
    inter_over_min = np.divide(inter, np.minimum(Ai, Aj) + eps, out=np.zeros_like(inter), where=(np.minimum(Ai, Aj) + eps > 0))
    contain_i_in_j = np.divide(inter, Ai + eps, out=np.zeros_like(inter), where=(Ai + eps > 0))
    contain_j_in_i = np.divide(inter, Aj + eps, out=np.zeros_like(inter), where=(Aj + eps > 0))
    rel_area_log = _safe_log((Ai + eps) / (Aj + eps), eps).astype(np.float32)
    
    feat = np.stack([mask_iou, inter_over_min, contain_i_in_j, contain_j_in_i, rel_area_log], axis=1).astype(np.float32)
    
    # Clip to reasonable ranges
    # mask_iou, inter_over_min, contain_*: in range [0, 1]
    # rel_area_log: log ratio, typically in range [-5, 5]
    feat[:, 0] = np.clip(feat[:, 0], 0.0, 1.0)   # mask_iou
    feat[:, 1] = np.clip(feat[:, 1], 0.0, 1.0)   # inter_over_min
    feat[:, 2] = np.clip(feat[:, 2], 0.0, 1.0)   # contain_i_in_j
    feat[:, 3] = np.clip(feat[:, 3], 0.0, 1.0)   # contain_j_in_i
    feat[:, 4] = np.clip(feat[:, 4], -5.0, 5.0)  # rel_area_log
    
    # Replace any remaining NaN/Inf with 0
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feat


@dataclass
class GeomOutput:
    geom_feat: np.ndarray    # [P, Gd] float32
    geom_dim: int


def build_pair_geom(
    boxes_xyxy: np.ndarray,     # [G,4] float32 pixel
    masks: np.ndarray,          # [G,Hm,Wm] bool
    pair_idx: np.ndarray,       # [P,2] int64, -1 padded
    pair_mask: np.ndarray       # [P] uint8
) -> GeomOutput:
    """
    Build geometric features for pairs (box + mask)
    
    Args:
        boxes_xyxy: [G,4] boxes in pixel coordinates
        masks: [G,Hm,Wm] boolean masks
        pair_idx: [P,2] pair indices, -1 for padding
        pair_mask: [P] 1 for valid, 0 for padding
        
    Returns:
        GeomOutput with [P, 11] features (6 box + 5 mask)
    """
    P = pair_idx.shape[0]
    valid = pair_mask.astype(bool)
    geom = np.zeros((P, 11), dtype=np.float32)  # 6 box + 5 mask
    
    if valid.sum() == 0:
        return GeomOutput(geom_feat=geom, geom_dim=geom.shape[1])
    
    idx = np.where(valid)[0]
    pi = pair_idx[idx, 0]
    pj = pair_idx[idx, 1]
    
    # Box features
    box_s = boxes_xyxy[pi]
    box_o = boxes_xyxy[pj]
    geom_box = box_geom_features_xyxy(box_s, box_o)  # [V,6]
    
    # Mask features
    masks_i = masks[pi]
    masks_j = masks[pj]
    geom_mask = mask_pair_geom_features(masks_i, masks_j)  # [V,5]
    
    # Concatenate
    geom[idx] = np.concatenate([geom_box, geom_mask], axis=1)
    
    # Final safety check: replace any NaN/Inf with 0
    geom = np.nan_to_num(geom, nan=0.0, posinf=0.0, neginf=0.0)
    
    return GeomOutput(geom_feat=geom, geom_dim=geom.shape[1])

