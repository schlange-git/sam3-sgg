"""
Geometry features for object pairs
"""
import torch


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two boxes (xyxy format)
    
    Args:
        box1: [4] xyxy
        box2: [4] xyxy
        
    Returns:
        iou: scalar
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    x1_i = torch.max(x1_1, x1_2)
    y1_i = torch.max(y1_1, y1_2)
    x2_i = torch.min(x2_1, x2_2)
    y2_i = torch.min(y2_1, y2_2)
    
    inter_area = torch.clamp(x2_i - x1_i, min=0) * torch.clamp(y2_i - y1_i, min=0)
    
    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou


def box_geom_feat(box_s: torch.Tensor, box_o: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute geometric features between two boxes
    
    Args:
        box_s: [4] xyxy normalized (subject box)
        box_o: [4] xyxy normalized (object box)
        eps: small epsilon for numerical stability
        
    Returns:
        feat: [6] geometric features [dx, dy, dw, dh, da, iou]
    """
    x1s, y1s, x2s, y2s = box_s
    x1o, y1o, x2o, y2o = box_o
    
    # Centers
    cxs = (x1s + x2s) * 0.5
    cys = (y1s + y2s) * 0.5
    cxo = (x1o + x2o) * 0.5
    cyo = (y1o + y2o) * 0.5
    
    # Widths and heights
    ws = (x2s - x1s).clamp(min=eps)
    hs = (y2s - y1s).clamp(min=eps)
    wo = (x2o - x1o).clamp(min=eps)
    ho = (y2o - y1o).clamp(min=eps)
    
    # Relative position (normalized by object size)
    dx = (cxo - cxs) / torch.max(ws, torch.tensor(eps, device=ws.device))
    dy = (cyo - cys) / torch.max(hs, torch.tensor(eps, device=hs.device))
    
    # Relative size (log ratios)
    dw = torch.log((wo + eps) / (ws + eps))
    dh = torch.log((ho + eps) / (hs + eps))
    
    # Area ratio (log)
    da = torch.log(((wo * ho) + eps) / ((ws * hs) + eps))
    
    # IoU
    iou = box_iou(box_s, box_o)
    
    return torch.stack([dx, dy, dw, dh, da, iou], dim=0)  # [6]
