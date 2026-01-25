import logging
import math
import os
import sys
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import v2

from detectron2.structures import ImageList

from ..transformer.util.utils import NestedTensor


class Sam3MaskedBackbone(nn.Module):
    """
    SAM3 backbone wrapper that returns a single feature map as NestedTensor.
    This is a minimal integration without explicit FPN fusion.
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.sam3_device = torch.device(cfg.MODEL.SAM3.DEVICE)
        self.feature_dim = cfg.MODEL.SAM3.FEATURE_DIM
        self.channel_repeat = cfg.MODEL.SAM3.CHANNEL_REPEAT
        self.image_size = cfg.MODEL.SAM3.IMAGE_SIZE
        self.freeze = cfg.MODEL.SAM3.FREEZE
        self.checkpoint_path = cfg.MODEL.SAM3.CHECKPOINT_PATH
        self.use_precomputed = cfg.MODEL.SAM3.USE_PRECOMPUTED
        self.featuremap_dir = cfg.MODEL.SAM3.FEATUREMAP_DIR
        self.feature_stride = 16
        self.target_stride = getattr(cfg.MODEL.SAM3, "TARGET_STRIDE", 0)

        # If using precomputed features, verify directory exists
        if self.use_precomputed:
            if not os.path.isdir(self.featuremap_dir):
                raise FileNotFoundError(
                    f"SAM3 precomputed feature directory does not exist: {self.featuremap_dir}. "
                    f"Please run precomputation first or set MODEL.SAM3.USE_PRECOMPUTED=False"
                )

        # Ensure sam3 is importable from repo local path
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sam3_path = os.path.join(repo_root, "sam3")
        if sam3_path not in sys.path:
            sys.path.insert(0, sam3_path)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        self.sam3_model = None
        self.transform = None
        if not self.use_precomputed:
            try:
                from sam3.model_builder import build_sam3_image_model
            except Exception as exc:  # pragma: no cover - handled at runtime
                raise ImportError(
                    "sam3 is not available on PYTHONPATH or dependencies missing. "
                    "Ensure SpeaQ/sam3 is present and required packages are installed."
                ) from exc
            logging.getLogger("detectron2").info("Loading SAM3 image backbone...")
            self.sam3_model = build_sam3_image_model(
                device=str(self.sam3_device),
                checkpoint_path=self.checkpoint_path if self.checkpoint_path else None,
            ).to(self.sam3_device)
            self.sam3_model.eval()

            if self.freeze:
                for p in self.sam3_model.parameters():
                    p.requires_grad_(False)

            # SAM3 image transform (aligns with Sam3Processor)
            self.transform = v2.Compose(
                [
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(size=(self.image_size, self.image_size)),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        # 1x1 projection to feature_dim (lazy init)
        self._proj_layer = None

        # exposed for Joiner
        self.num_channels = self.feature_dim
        self.feature_strides = [self.feature_stride]

    def _build_proj(self, in_channels: int) -> None:
        if self._proj_layer is not None:
            return
        if in_channels == self.feature_dim:
            self._proj_layer = nn.Identity()
        else:
            self._proj_layer = nn.Conv2d(in_channels, self.feature_dim, kernel_size=1, bias=False)
            with torch.no_grad():
                self._proj_layer.weight.zero_()
                copy_channels = min(in_channels, self.feature_dim)
                for ch in range(copy_channels):
                    self._proj_layer.weight[ch, ch, 0, 0] = 1.0
        self._proj_layer.to(self.device)
        self._proj_layer.eval()
        for p in self._proj_layer.parameters():
            p.requires_grad_(False)

    def _normalize_batch(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W] (uint8 or float)
        processed = []
        for img in images:
            if isinstance(img, torch.Tensor) and img.device.type != "cpu":
                img = img.cpu()
            img = v2.functional.to_image(img).to(self.sam3_device)
            img = self.transform(img)
            processed.append(img)
        return torch.stack(processed, dim=0)

    def _select_feature(self, backbone_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        candidate = None
        if "backbone_fpn" in backbone_out:
            fpn_feats = backbone_out["backbone_fpn"]
            if isinstance(fpn_feats, (list, tuple)) and len(fpn_feats) > 0:
                candidate = fpn_feats[-1]
        if candidate is None and "vision_features" in backbone_out:
            candidate = backbone_out["vision_features"]
        if candidate is None:
            raise RuntimeError("SAM3 backbone returned no usable features")

        # ensure [B, C, H, W]
        if candidate.dim() == 3:
            # assume [B, C, HW] with square spatial size
            b, c, hw = candidate.shape
            h = int(math.sqrt(hw))
            if h * h != hw:
                raise ValueError(f"Cannot reshape SAM3 features with HW={hw}")
            candidate = candidate.view(b, c, h, h)
        elif candidate.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected SAM3 feature shape: {candidate.shape}")
        return candidate

    def forward(self, images: ImageList) -> Dict[str, NestedTensor]:
        if self.use_precomputed:
            # Must use precomputed features - no fallback allowed
            if self.sam3_model is not None:
                raise RuntimeError(
                    "SAM3 model is loaded but USE_PRECOMPUTED=True. "
                    "This should not happen - precomputed mode should not load SAM3 model."
                )
            return {"sam3": self._load_precomputed(images)}

        # Normalize images to SAM3 expected format
        image_batch = self._normalize_batch(images.tensor)

        # Ensure SAM3 model device matches input device
        if next(self.sam3_model.parameters()).device != self.sam3_device:
            self.sam3_model.to(self.sam3_device)
        model_device = next(self.sam3_model.parameters()).device
        if image_batch.device != model_device:
            image_batch = image_batch.to(model_device)

        with torch.no_grad():
            backbone_out = self.sam3_model.backbone.forward_image(image_batch)

        feat = self._select_feature(backbone_out)
        if self.channel_repeat > 1:
            # Repeat channels to increase feature width deterministically.
            feat = feat.repeat(1, self.channel_repeat, 1, 1)
        if feat.device != self.device:
            feat = feat.to(self.device)
        in_channels = int(feat.shape[1])
        self._build_proj(in_channels)

        if isinstance(self._proj_layer, nn.Identity):
            proj_feat = feat
        else:
            proj_feat = self._proj_layer(feat)

        if proj_feat.device != self.device:
            proj_feat = proj_feat.to(self.device)

        # Estimate stride based on input size and feature map size
        h_in = image_batch.shape[2]
        h_feat = proj_feat.shape[2]
        self.feature_stride = max(1, int(round(h_in / float(h_feat))))
        # Downsample to target stride if requested (helps reduce encoder memory)
        if self.target_stride and self.feature_stride < self.target_stride:
            factor = self.target_stride // self.feature_stride
            if factor > 1:
                proj_feat = F.avg_pool2d(proj_feat, kernel_size=factor, stride=factor)
                self.feature_stride = self.feature_stride * factor
        self.feature_strides = [self.feature_stride]

        masks = self._mask_out_padding([proj_feat.shape], images.image_sizes, proj_feat.device)
        nested = NestedTensor(proj_feat, masks[0])

        return {"sam3": nested}

    def _load_precomputed(self, images: ImageList) -> NestedTensor:
        image_ids = getattr(images, "image_ids", None)
        if not image_ids:
            raise ValueError(
                "Precomputed SAM3 features require image_ids in ImageList. "
                "Cannot proceed without precomputed features when USE_PRECOMPUTED=True."
            )
        feats = []
        feature_stride = None
        for image_id in image_ids:
            feature_path = os.path.join(self.featuremap_dir, f"{image_id}.pt")
            if not os.path.isfile(feature_path):
                raise FileNotFoundError(
                    f"Required SAM3 precomputed feature file missing: {feature_path}\n"
                    f"Please run precomputation first. Cannot proceed without .pt file when USE_PRECOMPUTED=True."
                )
            payload = torch.load(feature_path, map_location=self.device)
            feat = payload["feature"]
            if feat.dim() == 3:
                feat = feat.unsqueeze(0)
            if feature_stride is None:
                feature_stride = int(payload.get("feature_stride", self.feature_stride))
            feats.append(feat)
        proj_feat = torch.cat(feats, dim=0)
        if proj_feat.shape[1] != self.feature_dim:
            raise ValueError(
                f"Precomputed feature dim {proj_feat.shape[1]} != expected {self.feature_dim}"
            )
        self.feature_stride = feature_stride or self.feature_stride
        # Downsample precomputed features to target_stride to reduce memory
        if self.target_stride and self.feature_stride < self.target_stride:
            factor = self.target_stride // self.feature_stride
            if factor > 1:
                proj_feat = F.avg_pool2d(proj_feat, kernel_size=factor, stride=factor)
                self.feature_stride = self.feature_stride * factor
        self.feature_strides = [self.feature_stride]
        masks = self._mask_out_padding([proj_feat.shape], images.image_sizes, proj_feat.device)
        return NestedTensor(proj_feat, masks[0])

    def _mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        for idx, shape in enumerate(feature_shapes):
            n, _, h, w = shape
            masks_per_feature_level = torch.ones((n, h, w), dtype=torch.bool, device=device)
            for img_idx, (img_h, img_w) in enumerate(image_sizes):
                stride = self.feature_strides[idx]
                masks_per_feature_level[
                    img_idx,
                    : int(math.ceil(float(img_h) / stride)),
                    : int(math.ceil(float(img_w) / stride)),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks
