"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        print ("Box1", boxes1, boxes1[:, :2][boxes1[:, 2:] < boxes1[:, :2]], boxes1[:, 2:][boxes1[:, 2:] < boxes1[:, :2]], (boxes1[:, 2:] < boxes1[:, :2]).nonzero(), boxes1.max(), boxes1.min(), boxes1.isnan().nonzero())
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        print ("Box2", boxes2, boxes2[:, :2][boxes2[:, 2:] < boxes2[:, :2]], boxes2[:, 2:][boxes2[:, 2:] < boxes2[:, :2]], (boxes2[:, 2:] < boxes2[:, :2]).nonzero(), boxes2.max(), boxes2.min(), boxes2.isnan().nonzero())
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def box_iou_xyxy(boxes1, boxes2):
    """Compute IoU between two sets of boxes in xyxy format.
    Returns a [N, M] pairwise matrix.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / union


def complete_box_iou(boxes1, boxes2):
    """
    CIoU (Complete IoU) from https://arxiv.org/abs/1911.08287
    CIoU = IoU - (rho^2(b, b_gt) / c^2) - alpha * v
    where:
        v = (4 / pi^2) * (arctan(w_gt/h_gt) - arctan(w/h))^2
        alpha = v / (1 - IoU + v)  (detached gradient)
    Boxes in xyxy format. Returns [N, M] pairwise matrix.
    """
    iou, _ = box_iou(boxes1, boxes2)  # reuse box_iou that returns (iou, union)

    cx1 = (boxes1[:, 0] + boxes1[:, 2]) / 2  # (N,)
    cy1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
    cx2 = (boxes2[:, 0] + boxes2[:, 2]) / 2  # (M,)
    cy2 = (boxes2[:, 1] + boxes2[:, 3]) / 2

    # rho^2: squared distance between centers
    rho2 = (cx1[:, None] - cx2[None, :]) ** 2 + (cy1[:, None] - cy2[None, :]) ** 2

    # c^2: squared diagonal of smallest enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    c2 = wh[:, :, 0] ** 2 + wh[:, :, 1] ** 2 + 1e-7

    # v: consistency of aspect ratio
    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]
    arctan_diff = torch.atan(w1[:, None] / (h1[:, None] + 1e-7)) - torch.atan(w2[None, :] / (h2[None, :] + 1e-7))
    v = (4.0 / (torch.pi ** 2)) * (arctan_diff ** 2)
    # alpha = v / (1 - IoU + v), with detached gradient for alpha
    alpha = v / (1 - iou + v + 1e-7)
    alpha = alpha.detach()

    ciou = iou - rho2 / c2 - alpha * v
    return ciou


def efficient_box_iou(boxes1, boxes2):
    """
    EIoU (Efficient IoU) from https://arxiv.org/abs/2101.08158
    EIoU = IoU - rho^2(b, b_gt)/c^2 - rho^2(w, w_gt)/c_w^2 - rho^2(h, h_gt)/c_h^2
    where:
        c_w, c_h are width/height of the smallest enclosing box
    Boxes in xyxy format. Returns [N, M] pairwise matrix.
    """
    iou, _ = box_iou(boxes1, boxes2)

    cx1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
    cy1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
    cx2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
    cy2 = (boxes2[:, 1] + boxes2[:, 3]) / 2

    # Center distance penalty
    rho2_center = (cx1[:, None] - cx2[None, :]) ** 2 + (cy1[:, None] - cy2[None, :]) ** 2

    # Smallest enclosing box diagonal
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    c2 = wh[:, :, 0] ** 2 + wh[:, :, 1] ** 2 + 1e-7

    # Width and height difference penalty
    w1 = boxes1[:, 2] - boxes1[:, 0]
    h1 = boxes1[:, 3] - boxes1[:, 1]
    w2 = boxes2[:, 2] - boxes2[:, 0]
    h2 = boxes2[:, 3] - boxes2[:, 1]

    rho2_w = (w1[:, None] - w2[None, :]) ** 2
    rho2_h = (h1[:, None] - h2[None, :]) ** 2
    c_w2 = wh[:, :, 0] ** 2 + 1e-7
    c_h2 = wh[:, :, 1] ** 2 + 1e-7

    eiou = iou - rho2_center / c2 - rho2_w / c_w2 - rho2_h / c_h2
    return eiou