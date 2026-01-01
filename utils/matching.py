"""
Matching utilities for SAM3 proposals and GT boxes
"""
import torch
from typing import Optional


def mask_to_box(mask: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Convert mask to bounding box
    
    Args:
        mask: [H, W] bool tensor
        
    Returns:
        box: [4] xyxy tensor (normalized) or None if empty
    """
    if mask.dim() > 2:
        mask = mask.squeeze()
    
    ys, xs = torch.where(mask > 0.5)
    if ys.numel() == 0:
        return None
    
    H, W = mask.shape
    x1, x2 = xs.min().float() / W, xs.max().float() / W
    y1, y2 = ys.min().float() / H, ys.max().float() / H
    
    return torch.stack([x1, y1, x2, y2], dim=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: [N, 4] xyxy
        boxes2: [M, 4] xyxy
        
    Returns:
        iou: [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    inter = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter_area = inter[:, :, 0] * inter[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter_area
    iou = inter_area / union.clamp(min=1e-6)
    
    return iou


def single_box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two single boxes"""
    return box_iou(box1.unsqueeze(0), box2.unsqueeze(0))[0, 0]


def match_by_iou(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thr: float = 0.5
) -> torch.Tensor:
    """
    Match predictions to GT boxes using greedy IoU matching
    
    Args:
        pred_boxes: [N, 4] xyxy
        gt_boxes: [G, 4] xyxy
        iou_thr: IoU threshold
        
    Returns:
        gt_to_pred: [G] LongTensor, -1 if unmatched
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return torch.full((len(gt_boxes),), -1, dtype=torch.long, device=gt_boxes.device)
    
    iou = box_iou(gt_boxes, pred_boxes)  # [G, N]
    gt_to_pred = torch.full((gt_boxes.size(0),), -1, dtype=torch.long, device=gt_boxes.device)
    
    # Greedy matching: for each GT, pick best pred not taken
    taken = set()
    for g in range(gt_boxes.size(0)):
        vals, idxs = torch.sort(iou[g], descending=True)
        for v, n in zip(vals.tolist(), idxs.tolist()):
            if v < iou_thr:
                break
            if n not in taken:
                gt_to_pred[g] = n
                taken.add(n)
                break
    
    return gt_to_pred

