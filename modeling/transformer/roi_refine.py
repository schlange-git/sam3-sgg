import torch
from torch import nn
from torchvision.ops import roi_align


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)


class ROIRefineHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        pool_size: int,
        stride: int,
        small_area_thresh: float,
        detach_boxes: bool,
        use_gate: bool,
        apply_to: str,
        class_names: list = None,
    ):
        super().__init__()
        assert hidden_dim > 0, "ROIRefineHead hidden_dim must be positive."
        assert pool_size > 0, "ROIRefineHead pool_size must be positive."
        assert stride > 0, "ROIRefineHead stride must be positive."
        assert apply_to in ("small_only", "all"), (
            f"Unsupported MODEL.ROI_REFINE.APPLY_TO={apply_to}; expected small_only or all."
        )
        self.hidden_dim = int(hidden_dim)
        self.pool_size = int(pool_size)
        self.stride = int(stride)
        self.spatial_scale = 1.0 / float(stride)
        self.small_area_thresh = float(small_area_thresh)
        self.detach_boxes = bool(detach_boxes)
        self.apply_to = str(apply_to)
        self.class_names = class_names
        self.only_roi_cls = False
        self._last_gate_stats = None

        in_dim = self.hidden_dim * self.pool_size * self.pool_size
        self.roi_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.gate = None
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid(),
            )

    def _build_rois(
        self, boxes_cxcywh: torch.Tensor, image_h: int, image_w: int
    ):
        if self.detach_boxes:
            boxes_cxcywh = boxes_cxcywh.detach()
        assert boxes_cxcywh.dim() == 3 and boxes_cxcywh.shape[-1] == 4, (
            f"Expected boxes [B,N,4], got {tuple(boxes_cxcywh.shape)}."
        )
        b, n, _ = boxes_cxcywh.shape
        area = boxes_cxcywh[..., 2].clamp(min=0) * boxes_cxcywh[..., 3].clamp(min=0)
        if self.apply_to == "small_only":
            keep_mask = area < self.small_area_thresh
        else:
            keep_mask = torch.ones_like(area, dtype=torch.bool)

        boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh).clamp(0.0, 1.0)
        boxes_xyxy_pixel = boxes_xyxy.clone()
        boxes_xyxy_pixel[..., [0, 2]] *= float(image_w)
        boxes_xyxy_pixel[..., [1, 3]] *= float(image_h)

        batch_idx = torch.arange(b, device=boxes_cxcywh.device)[:, None].expand(b, n)
        flat_keep = keep_mask.reshape(-1)
        flat_indices = torch.nonzero(flat_keep, as_tuple=False).squeeze(1)
        if flat_indices.numel() == 0:
            return boxes_cxcywh.new_zeros((0, 5)), keep_mask, flat_indices

        flat_boxes = boxes_xyxy_pixel.reshape(b * n, 4)[flat_indices]
        flat_batch = batch_idx.reshape(b * n)[flat_indices].to(flat_boxes.dtype).unsqueeze(1)
        rois = torch.cat((flat_batch, flat_boxes), dim=1)
        return rois, keep_mask, flat_indices

    def forward(
        self,
        embeddings: torch.Tensor,
        boxes_cxcywh: torch.Tensor,
        feature: torch.Tensor,
        image_h: int,
        image_w: int,
        labels: torch.Tensor = None,
    ):
        assert embeddings.dim() == 3, f"Expected embeddings [B,N,C], got {tuple(embeddings.shape)}."
        assert feature.dim() == 4, f"Expected feature [B,C,H,W], got {tuple(feature.shape)}."
        b, n, c = embeddings.shape
        assert c == self.hidden_dim, (
            f"ROIRefineHead hidden_dim mismatch: embeddings C={c}, expected {self.hidden_dim}."
        )
        assert feature.shape[0] == b, (
            f"ROIRefineHead batch mismatch: embeddings B={b}, feature B={feature.shape[0]}."
        )
        assert feature.shape[1] == self.hidden_dim, (
            f"ROIRefineHead feature channel mismatch: feature C={feature.shape[1]}, expected {self.hidden_dim}."
        )
        assert image_h % self.stride == 0 and image_w % self.stride == 0, (
            f"Image size ({image_h},{image_w}) must be divisible by ROI stride {self.stride}."
        )
        assert feature.shape[-2] == image_h // self.stride, (
            f"Feature height {feature.shape[-2]} does not match native stride{self.stride} for image_h={image_h}."
        )
        assert feature.shape[-1] == image_w // self.stride, (
            f"Feature width {feature.shape[-1]} does not match native stride{self.stride} for image_w={image_w}."
        )

        rois, keep_mask, flat_indices = self._build_rois(boxes_cxcywh, image_h, image_w)
        if flat_indices.numel() == 0:
            self._last_gate_stats = {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            return embeddings, keep_mask

        roi_feat = roi_align(
            feature,
            rois,
            output_size=(self.pool_size, self.pool_size),
            spatial_scale=self.spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )
        roi_emb = self.roi_proj(roi_feat.flatten(1)).contiguous()

        flat_embeddings = embeddings.reshape(b * n, c)
        selected_embeddings = flat_embeddings[flat_indices]
        if self.gate is not None:
            gate_values = self.gate(torch.cat((selected_embeddings, roi_emb), dim=-1))
            gate_detached = gate_values.detach().float()
            stats = {
                "count": int(gate_detached.numel()),
                "mean": float(gate_detached.mean().item()),
                "std": float(gate_detached.std(unbiased=False).item()),
                "min": float(gate_detached.min().item()),
                "max": float(gate_detached.max().item()),
            }
            # TODO: align labels with ROI subset (flat_indices vs combined_labels mismatch)
            # if labels is not None:
            #     labels = labels[flat_indices]
            if labels is not None and labels.numel() == gate_detached.numel():
                for cls_id in labels.unique().tolist():
                    mask = labels == cls_id
                    if mask.sum() > 0:
                        gv = gate_detached[mask]
                        cname = self.class_names[int(cls_id)] if self.class_names and int(cls_id) < len(self.class_names) else f"cls_{int(cls_id)}"
                        stats[f"{cname}_mean"] = float(gv.mean().item())
                        stats[f"{cname}_count"] = int(mask.sum().item())
            self._last_gate_stats = stats
            residual = gate_values * roi_emb
        else:
            self._last_gate_stats = None
            residual = roi_emb

        refined_flat = flat_embeddings.clone()
        if self.gate is not None:
            # Convex fusion: gate=0 keeps the query feature, gate=1 uses the ROI feature.
            refined_flat[flat_indices] = (1.0 - gate_values) * selected_embeddings + gate_values * roi_emb
        elif self.only_roi_cls:
            # Replace embedding with roi projection entirely (no original embedding contribution).
            refined_flat[flat_indices] = residual
        else:
            # Default no-gate fallback: residual addition.
            refined_flat[flat_indices] = refined_flat[flat_indices] + residual
        return refined_flat.reshape(b, n, c).contiguous(), keep_mask
