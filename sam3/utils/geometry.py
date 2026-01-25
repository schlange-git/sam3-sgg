"""
Geometry features for object pairs
"""
import torch


def box_geom_feat(box_s: torch.Tensor, box_o: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute geometric features between two boxes
    
    Args:
        box_s: [4] xyxy (subject box)
        box_o: [4] xyxy (object box)
        eps: small epsilon for numerical stability
        
    Returns:
        feat: [6] geometric features
    """
    x1s, y1s, x2s, y2s = box_s
    x1o, y1o, x2o, y2o = box_o
    
    # Centers
    cxs, cys = (x1s + x2s) * 0.5, (y1s + y2s) * 0.5
    cxo, cyo = (x1o + x2o) * 0.5, (y1o + y2o) * 0.5
    
    # Widths and heights
    ws = (x2s - x1s).clamp(min=eps)
    hs = (y2s - y1s).clamp(min=eps)
    wo = (x2o - x1o).clamp(min=eps)
    ho = (y2o - y1o).clamp(min=eps)
    
    # Relative position
    dx = (cxs - cxo) / wo
    dy = (cys - cyo) / ho
    
    # Relative size
    dw = torch.log(ws / wo)
    dh = torch.log(hs / ho)
    
    # Area ratio
    da = torch.log((ws * hs) / (wo * ho))
    
    # IoU
    from .matching import single_box_iou
    iou = single_box_iou(box_s, box_o)
    
    return torch.stack([dx, dy, dw, dh, da, iou], dim=0)  # [6]

