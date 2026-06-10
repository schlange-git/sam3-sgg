import gc
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import v2

from detectron2.structures import ImageList

from ..transformer.util.utils import NestedTensor


class Sam3MaskedBackbone(nn.Module):
    """
    SAM3 backbone wrapper that returns feature maps as NestedTensor.
    Supports both single-scale and FPN-style multi-scale features.
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
        self.use_fpn = getattr(cfg.MODEL.SAM3, "USE_FPN", False)
        self.fpn_strides = getattr(cfg.MODEL.SAM3, "FPN_STRIDES", [4, 8, 16, 32])
        # Prefer native SAM3 multi-scale outputs when available.
        self.use_backbone_fpn = getattr(cfg.MODEL.SAM3, "USE_BACKBONE_FPN", True)
        # Multi-scale merge strategy: "last" | "sum" | "concat"
        self.multiscale_merge = str(
            getattr(cfg.MODEL.SAM3, "MULTISCALE_MERGE", "last")
        ).lower()
        if self.multiscale_merge not in ("last", "sum", "concat"):
            raise ValueError(
                f"Unsupported MODEL.SAM3.MULTISCALE_MERGE={self.multiscale_merge}, "
                "expected one of: last, sum, concat"
            )
        self.use_patch_merge = getattr(cfg.MODEL.SAM3, "USE_PATCH_MERGE", False)
        self.patch_merge_init_noise_std = float(
            getattr(cfg.MODEL.SAM3, "PATCH_MERGE_INIT_NOISE_STD", 0.0)
        )
        self._patch_merge_logged = False
        self._last_aux_features = {}  # cached intermediate features for ROI_REFINE

        # If using precomputed features, verify directory exists
        if self.use_precomputed:
            if not os.path.isdir(self.featuremap_dir):
                raise FileNotFoundError(
                    f"SAM3 precomputed feature directory does not exist: {self.featuremap_dir}. "
                    f"Please run precomputation first or set MODEL.SAM3.USE_PRECOMPUTED=False"
                )

        # Ensure sam3 is importable from repo local path
        # sam3_backbone.py is at: SpeaQ/modeling/backbone/sam3_backbone.py
        # So we need to go up 3 levels to reach project root
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up: backbone -> modeling -> SpeaQ -> project_root
        repo_root = os.path.abspath(os.path.join(current_file_dir, "..", "..", ".."))
        sam3_path = os.path.join(repo_root, "sam3")
        if sam3_path not in sys.path:
            sys.path.insert(0, sam3_path)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # Find BPE file path manually (pkg_resources may fail if sam3 is not installed as package)
        bpe_path = None
        bpe_candidates = [
            os.path.join(repo_root, "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
            os.path.join(repo_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
            os.path.join(sam3_path, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"),
            os.path.join(sam3_path, "assets", "bpe_simple_vocab_16e6.txt.gz"),
        ]
        for candidate in bpe_candidates:
            if os.path.exists(candidate):
                bpe_path = candidate
                logging.getLogger("detectron2").info(f"Found BPE file at: {bpe_path}")
                break
        
        # If not found, try pkg_resources as fallback
        if bpe_path is None:
            try:
                import pkg_resources
                bpe_path = pkg_resources.resource_filename(
                    "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
                )
                if os.path.exists(bpe_path):
                    logging.getLogger("detectron2").info(f"Found BPE file via pkg_resources: {bpe_path}")
            except Exception as e:
                logging.getLogger("detectron2").debug(f"pkg_resources failed: {e}")
                # If pkg_resources also fails, try to find it relative to sam3 module
                try:
                    import sam3
                    if hasattr(sam3, '__file__') and sam3.__file__:
                        sam3_dir = os.path.dirname(os.path.dirname(sam3.__file__))
                        bpe_path = os.path.join(sam3_dir, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
                        if os.path.exists(bpe_path):
                            logging.getLogger("detectron2").info(f"Found BPE file via sam3 module: {bpe_path}")
                        else:
                            bpe_path = None
                except Exception:
                    pass
        
        if bpe_path is None or not os.path.exists(bpe_path):
            logging.getLogger("detectron2").warning(
                f"BPE file not found at expected locations. Tried: {bpe_candidates}. "
                "Will let build_sam3_image_model handle it (may fail or download)."
            )
            # Set to None to let build_sam3_image_model handle it
            bpe_path = None

        # SAM3 权重固定路径（相对项目根）
        sam3_ckpt = os.path.join(repo_root, "sam3", "weights", "sam3.pt")
        checkpoint_path = sam3_ckpt if os.path.isfile(sam3_ckpt) else None
        if checkpoint_path is None:
            logging.getLogger("detectron2").info(
                f"SAM3 checkpoint not found at {sam3_ckpt}, will try HuggingFace."
            )

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
            # 无 barrier 的可中断加载：
            # - 避免分布式 barrier 卡死（Ctrl+C 难中断）
            # - 可选按 rank 延迟错峰，降低同时加载峰值内存
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
            stagger_sec = int(os.environ.get("SAM3_LOAD_STAGGER_SEC", "8"))
            if world_size > 1 and stagger_sec > 0:
                delay = rank * stagger_sec
                logging.getLogger("detectron2").info(
                    "SAM3 load stagger enabled: rank %s/%s sleeps %ss before loading.",
                    rank, world_size, delay
                )
                time.sleep(delay)
            try:
                logging.getLogger("detectron2").info(
                    "Loading SAM3 image backbone (rank %s/%s, on CPU first to save memory)...", rank, world_size
                )
                self.sam3_model = build_sam3_image_model(
                    device="cpu",
                    checkpoint_path=checkpoint_path,
                    load_from_HF=(checkpoint_path is None),
                    bpe_path=bpe_path,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.sam3_model = self.sam3_model.to(self.sam3_device)
                if self.freeze:
                    self.sam3_model.eval()
                else:
                    self.sam3_model.train()
            except Exception as e:
                logging.getLogger("detectron2").error(
                    "SAM3 load failed on rank %s/%s: %s. Shutting down.", rank, world_size, e
                )
                raise RuntimeError(f"SAM3 image backbone load failed on rank {rank}/{world_size}: {e}") from e

            if self.sam3_model is None:
                raise RuntimeError(
                    f"SAM3 image backbone was not loaded on rank {rank}/{world_size}. Aborting."
                )
            logging.getLogger("detectron2").info(
                "SAM3 image backbone loaded on rank %s/%s.", rank, world_size
            )

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
        # Per-level projection layers for native SAM3 multi-scale features
        self._level_proj_layers = nn.ModuleDict()
        # Optional fusion layer for concat multi-scale merge
        self._merge_proj_layer: Optional[nn.Module] = None
        # Optional patch-merge projection layer (pixel-unshuffle → 1×1 conv)
        self._patch_merge_proj: Optional[nn.Module] = None
        self._patch_merge_factor: Optional[int] = None
        self._patch_merge_init_weight: Optional[torch.Tensor] = None

        # FPN layers for multi-scale features (if enabled)
        # Note: FPN layer creation is deferred until forward() when actual feature_stride is known
        self.fpn_layers = None
        self._fpn_layers_initialized = False

        # exposed for Joiner
        self.num_channels = self.feature_dim
        if self.use_fpn:
            self.feature_strides = self.fpn_strides
        else:
            self.feature_strides = [self.feature_stride]

    def _initialize_fpn_layers(self) -> None:
        """Initialize FPN layers based on actual feature_stride"""
        if self._fpn_layers_initialized:
            return
        self.fpn_layers = nn.ModuleDict()
        # Build FPN layers for each stride level
        for stride in self.fpn_strides:
            if stride == self.feature_stride:
                # Identity - no layer needed
                self.fpn_layers[f'fpn_{stride}'] = nn.Identity()
            elif stride > self.feature_stride:
                # Downsampling layers
                factor = stride // self.feature_stride
                self.fpn_layers[f'fpn_{stride}'] = nn.Conv2d(
                    self.feature_dim, self.feature_dim, 
                    kernel_size=3, stride=factor, 
                    padding=1, bias=False
                )
            else:
                # Upsampling layers
                factor = self.feature_stride // stride
                self.fpn_layers[f'fpn_{stride}'] = nn.Sequential(
                    nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False),
                    nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)
                )
        # Initialize FPN layers
        for layer in self.fpn_layers.values():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Sequential):
                for m in layer:
                    if isinstance(m, nn.Conv2d):
                        nn.init.xavier_uniform_(m.weight)
            # Identity layers don't need initialization
        # Move FPN layers to device
        for layer in self.fpn_layers.values():
            layer.to(self.device)
            if self.freeze:
                for p in layer.parameters():
                    p.requires_grad_(False)
        self._fpn_layers_initialized = True

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
        if self.freeze:
            self._proj_layer.eval()
        else:
            self._proj_layer.train()
        for p in self._proj_layer.parameters():
            p.requires_grad_(not self.freeze)

    def _build_level_proj(self, level_key: str, in_channels: int) -> nn.Module:
        if level_key in self._level_proj_layers:
            return self._level_proj_layers[level_key]
        if in_channels == self.feature_dim:
            layer = nn.Identity()
        else:
            layer = nn.Conv2d(in_channels, self.feature_dim, kernel_size=1, bias=False)
            with torch.no_grad():
                layer.weight.zero_()
                copy_channels = min(in_channels, self.feature_dim)
                for ch in range(copy_channels):
                    layer.weight[ch, ch, 0, 0] = 1.0
        layer.to(self.device)
        if self.freeze:
            layer.eval()
        else:
            layer.train()
        for p in layer.parameters():
            p.requires_grad_(not self.freeze)
        self._level_proj_layers[level_key] = layer
        return self._level_proj_layers[level_key]

    def _build_merge_proj(self, in_channels: int) -> nn.Module:
        if self._merge_proj_layer is not None:
            return self._merge_proj_layer
        if in_channels == self.feature_dim:
            layer = nn.Identity()
        else:
            layer = nn.Conv2d(in_channels, self.feature_dim, kernel_size=1, bias=False)
            with torch.no_grad():
                layer.weight.zero_()
                copy_channels = min(in_channels, self.feature_dim)
                for ch in range(copy_channels):
                    layer.weight[ch, ch, 0, 0] = 1.0
        layer.to(self.device)
        if self.freeze:
            layer.eval()
        else:
            layer.train()
        for p in layer.parameters():
            p.requires_grad_(not self.freeze)
        self._merge_proj_layer = layer
        return self._merge_proj_layer

    def _build_patch_merge_proj(self, factor: int) -> nn.Module:
        """Build and initialize the 1×1 conv that follows pixel_unshuffle.

        After pixel_unshuffle with factor f, the feature has C * f² channels.
        This conv reduces back to C channels.

        Weight initialization mimics avg_pool behavior: each output channel
        receives equal contribution (1/f²) from all f² sub-pixel input channels.
        This ensures training starts identically to avg_pool2d, then gradually
        learns to exploit sub-pixel structure.
        """
        if self._patch_merge_proj is not None:
            return self._patch_merge_proj
        in_channels = self.feature_dim * factor * factor
        layer = nn.Conv2d(in_channels, self.feature_dim, kernel_size=1, bias=False)
        with torch.no_grad():
            layer.weight.zero_()
            for c in range(self.feature_dim):
                for dy in range(factor):
                    for dx in range(factor):
                        src_c = c + (dy * factor + dx) * self.feature_dim
                        layer.weight[c, src_c, 0, 0] = 1.0 / (factor * factor)
            if self.patch_merge_init_noise_std > 0.0:
                avgpool_weight = layer.weight.detach().clone()
                layer.weight.add_(
                    torch.randn_like(layer.weight) * self.patch_merge_init_noise_std
                )
                deviation = (layer.weight - avgpool_weight).abs().max().item()
                assert deviation > 0.0, (
                    "PATCH_MERGE_INIT_NOISE_STD>0 but patch-merge init weight unchanged"
                )
                logging.getLogger("detectron2").info(
                    "X-SAM patch-merge: injected init noise std=%.4g, max|delta|=%.4g vs avg-pool",
                    self.patch_merge_init_noise_std,
                    deviation,
                )
        self._patch_merge_factor = int(factor)
        self._patch_merge_init_weight = layer.weight.detach().clone()
        layer.to(self.device)
        self._patch_merge_init_weight = self._patch_merge_init_weight.to(self.device)
        # Always trainable: the X-SAM 1x1 conv learns sub-pixel fusion weights
        # even when SAM3 backbone is frozen.
        layer.train()
        for p in layer.parameters():
            p.requires_grad_(True)
        self._patch_merge_proj = layer
        return self._patch_merge_proj

    def get_patch_merge_stats(self):
        """Return X-SAM patch-merge deviation from avg-pool initialization."""
        if self._patch_merge_proj is None or self._patch_merge_init_weight is None:
            return None
        weight = self._patch_merge_proj.weight.detach().float()
        init = self._patch_merge_init_weight.to(weight.device).float()
        delta = weight - init
        return {
            "factor": int(self._patch_merge_factor or 0),
            "weight_mean": float(weight.mean().item()),
            "weight_std": float(weight.std(unbiased=False).item()),
            "delta_mean_abs": float(delta.abs().mean().item()),
            "delta_max_abs": float(delta.abs().max().item()),
            "delta_norm": float(delta.norm().item()),
            "init_norm": float(init.norm().item()),
        }

    def set_freeze(self, freeze: bool) -> None:
        """
        更新 SAM3 backbone 的冻结状态。
        切换 sam3_model、_proj_layer、多尺度层 的 train/eval 与 requires_grad，
        使 forward 中 torch.set_grad_enabled(not self.freeze) 与新状态一致。
        """
        self.freeze = freeze
        if self.sam3_model is not None:
            if freeze:
                self.sam3_model.eval()
            else:
                self.sam3_model.train()
            for p in self.sam3_model.parameters():
                p.requires_grad_(not freeze)
        if self._proj_layer is not None:
            if freeze:
                self._proj_layer.eval()
            else:
                self._proj_layer.train()
            for p in self._proj_layer.parameters():
                p.requires_grad_(not freeze)
        for layer in self._level_proj_layers.values():
            if freeze:
                layer.eval()
            else:
                layer.train()
            for p in layer.parameters():
                p.requires_grad_(not freeze)
        if self._merge_proj_layer is not None:
            if freeze:
                self._merge_proj_layer.eval()
            else:
                self._merge_proj_layer.train()
            for p in self._merge_proj_layer.parameters():
                p.requires_grad_(not freeze)
        if self._patch_merge_proj is not None:
            # X-SAM 1x1 conv is always trainable (learns sub-pixel fusion)
            self._patch_merge_proj.train()
            for p in self._patch_merge_proj.parameters():
                p.requires_grad_(True)
        if self.fpn_layers is not None:
            for layer in self.fpn_layers.values():
                if freeze:
                    layer.eval()
                else:
                    layer.train()
                for p in layer.parameters():
                    p.requires_grad_(not freeze)

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

    def get_last_aux_features(self):
        """Return cached intermediate feature maps for ROI_REFINE.
        Returns dict mapping stride -> NestedTensor."""
        assert self._last_aux_features, (
            "get_last_aux_features() called before forward(). "
            "ROI_REFINE requires SAM3 backbone to run forward() first."
        )
        return self._last_aux_features

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
        return self._to_4d_feature(candidate)

    def _to_4d_feature(self, candidate: torch.Tensor) -> torch.Tensor:
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

    def _extract_native_multiscale(
        self, backbone_out: Dict[str, torch.Tensor], image_h: int
    ) -> Optional[List[Tuple[int, torch.Tensor]]]:
        fpn_feats = backbone_out.get("backbone_fpn", None)
        if not (isinstance(fpn_feats, (list, tuple)) and len(fpn_feats) > 0):
            return None
        features: List[torch.Tensor] = []
        strides: List[int] = []
        for idx, feat in enumerate(fpn_feats):
            feat = self._to_4d_feature(feat)
            if self.channel_repeat > 1:
                feat = feat.repeat(1, self.channel_repeat, 1, 1)
            if feat.device != self.device:
                feat = feat.to(self.device)
            layer = self._build_level_proj(f"native_{idx}", int(feat.shape[1]))
            proj_feat = feat if isinstance(layer, nn.Identity) else layer(feat)
            stride = max(1, int(round(float(image_h) / float(proj_feat.shape[2]))))
            features.append(proj_feat)
            strides.append(stride)
        if len(features) <= 1:
            return None
        return sorted(zip(strides, features), key=lambda x: x[0])

    def _build_nested_from_multiscale(
        self, ordered_feats: List[Tuple[int, torch.Tensor]], images: ImageList
    ) -> Dict[str, NestedTensor]:
        feature_shapes = [feat.shape for _, feat in ordered_feats]
        self.feature_strides = [stride for stride, _ in ordered_feats]
        masks = self._mask_out_padding(feature_shapes, images.image_sizes, self.device)

        if self.multiscale_merge == "last":
            nested_features: Dict[str, NestedTensor] = {}
            for idx, (stride, feat) in enumerate(ordered_feats):
                nested_features[f"fpn_{stride}"] = NestedTensor(feat, masks[idx])
            return nested_features

        target_h = ordered_feats[-1][1].shape[-2]
        target_w = ordered_feats[-1][1].shape[-1]
        resized_feats = []
        resized_masks = []
        for idx, (_, feat) in enumerate(ordered_feats):
            if feat.shape[-2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
                m = F.interpolate(
                    masks[idx][None].float(),
                    size=(target_h, target_w),
                    mode="nearest",
                )[0].to(torch.bool)
            else:
                m = masks[idx]
            resized_feats.append(feat)
            resized_masks.append(m)

        if self.multiscale_merge == "sum":
            merged = resized_feats[0]
            for feat in resized_feats[1:]:
                merged = merged + feat
        else:  # concat
            concat_feat = torch.cat(resized_feats, dim=1)
            merge_proj = self._build_merge_proj(int(concat_feat.shape[1]))
            merged = concat_feat if isinstance(merge_proj, nn.Identity) else merge_proj(concat_feat)

        merged_mask = resized_masks[0]
        for m in resized_masks[1:]:
            merged_mask = merged_mask | m
        merged_stride = ordered_feats[-1][0]
        self.feature_strides = [merged_stride]
        return {f"fpn_merged_{self.multiscale_merge}": NestedTensor(merged, merged_mask)}

    def forward(self, images: ImageList) -> Dict[str, NestedTensor]:
        if not self._patch_merge_logged and not self.use_precomputed:
            logger = logging.getLogger("detectron2")
            method = (
                f"pixel_unshuffle + 1×1 conv (factor={self.target_stride // max(self.feature_stride, 1)})"
                if self.use_patch_merge
                else "avg_pool2d"
            )
            logger.info(
                "[SAM3 PatchMerge] Config check — USE_PATCH_MERGE=%s, TARGET_STRIDE=%d, "
                "downsample method: %s",
                self.use_patch_merge, self.target_stride, method,
            )
            self._patch_merge_logged = True

        if self.use_precomputed:
            # Must use precomputed features - no fallback allowed
            if self.sam3_model is not None:
                raise RuntimeError(
                    "SAM3 model is loaded but USE_PRECOMPUTED=True. "
                    "This should not happen - precomputed mode should not load SAM3 model."
                )
            precomputed_result = self._load_precomputed(images)
            # If FPN is enabled, _load_precomputed returns a dict, otherwise returns NestedTensor
            if self.use_fpn:
                return precomputed_result  # Already a dict
            else:
                return {"sam3": precomputed_result}  # Wrap single NestedTensor in dict

        # Normalize images to SAM3 expected format
        image_batch = self._normalize_batch(images.tensor)

        # Ensure SAM3 model device matches input device
        if next(self.sam3_model.parameters()).device != self.sam3_device:
            self.sam3_model.to(self.sam3_device)
        model_device = next(self.sam3_model.parameters()).device
        if image_batch.device != model_device:
            image_batch = image_batch.to(model_device)

        with torch.set_grad_enabled(not self.freeze):
            backbone_out = self.sam3_model.backbone.forward_image(image_batch)

        if self.use_fpn and self.use_backbone_fpn:
            ordered_native = self._extract_native_multiscale(backbone_out, image_batch.shape[2])
            if ordered_native is not None:
                result = self._build_nested_from_multiscale(ordered_native, images)
                self._last_aux_features = result  # also use FPN features as aux
                return result

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
        _orig_stride = self.feature_stride  # saved for assertion below
        # Cache intermediate features for ROI_REFINE (before any downsampling)
        masks = self._mask_out_padding([proj_feat.shape], images.image_sizes, proj_feat.device)
        self._last_aux_features = {self.feature_stride: NestedTensor(proj_feat, masks[0])}
        # Downsample to target stride if requested (helps reduce encoder memory)
        if self.target_stride and self.feature_stride < self.target_stride:
            factor = self.target_stride // self.feature_stride
            if factor > 1:
                is_power_of_two = (factor & (factor - 1)) == 0
                if self.use_patch_merge and is_power_of_two:
                    proj_feat = F.pixel_unshuffle(proj_feat, downscale_factor=factor)
                    self._build_patch_merge_proj(factor)
                    proj_feat = self._patch_merge_proj(proj_feat)
                    if not self._patch_merge_logged:
                        logger = logging.getLogger("detectron2")
                        logger.info(
                            "[SAM3 PatchMerge] ENABLED — pixel_unshuffle(factor=%d) + 1x1 conv "
                            "(%d→%d) replacing avg_pool2d. "
                            "stride: %d → %d. "
                            "1x1 conv initialized to mimic avg_pool (equal sub-pixel weights).",
                            factor, self.feature_dim * factor * factor, self.feature_dim,
                            self.feature_stride, self.feature_stride * factor,
                        )
                        self._patch_merge_logged = True
                elif self.use_patch_merge:
                    assert False, (
                        f"[SAM3 PatchMerge] FAILED — factor={factor} is not a power of 2. "
                        f"USE_PATCH_MERGE=True requires power-of-2 downsample factor. "
                        f"feature_stride={self.feature_stride}, target_stride={self.target_stride}"
                    )
                else:
                    proj_feat = F.avg_pool2d(proj_feat, kernel_size=factor, stride=factor, ceil_mode=False)
                self.feature_stride = self.feature_stride * factor

        # Assert: if USE_PATCH_MERGE=True, patch merge MUST have been used.
        # This catches config mistakes (e.g. non-power-of-2 factor) that silently
        # fall back to avg_pool2d.
        if self.use_patch_merge and self.target_stride and _orig_stride < self.target_stride:
            factor = self.target_stride // _orig_stride
            if factor > 1:
                assert self._patch_merge_proj is not None, (
                    f"[SAM3 PatchMerge] ASSERTION FAILED — USE_PATCH_MERGE=True but "
                    f"_patch_merge_proj was NOT built. "
                    f"factor={factor} (target_stride={self.target_stride} / orig_stride={_orig_stride}). "
                    f"is_power_of_two={(factor & (factor - 1)) == 0}. "
                    "If factor is not a power of 2, set USE_PATCH_MERGE=False or adjust TARGET_STRIDE."
                )

        # Assert patch merge integrity
        if self.use_patch_merge and self._patch_merge_proj is not None:
            assert proj_feat.shape[1] == self.feature_dim, (
                f"[SAM3 PatchMerge] Output channels mismatch: "
                f"expected {self.feature_dim}, got {proj_feat.shape[1]}. "
                "The 1×1 conv projection is NOT reducing channels correctly."
            )

        if self.use_fpn:
            # Initialize FPN layers on first forward (now we know actual feature_stride)
            if not self._fpn_layers_initialized:
                self._initialize_fpn_layers()
            
            # Generate multi-scale features using FPN
            fpn_feats = []
            for stride in self.fpn_strides:
                if stride == self.feature_stride:
                    fpn_feat = proj_feat
                else:
                    fpn_feat = self.fpn_layers[f'fpn_{stride}'](proj_feat)
                fpn_feats.append(fpn_feat)
            
            ordered = list(zip(self.fpn_strides, fpn_feats))
            return self._build_nested_from_multiscale(ordered, images)
        else:
            # Single-scale output (original behavior)
            self.feature_strides = [self.feature_stride]
            masks = self._mask_out_padding([proj_feat.shape], images.image_sizes, proj_feat.device)
            nested = NestedTensor(proj_feat, masks[0])
            return {"sam3": nested}

    def _load_precomputed(self, images: ImageList):
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
            del payload
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
                is_power_of_two = (factor & (factor - 1)) == 0
                if self.use_patch_merge and is_power_of_two:
                    proj_feat = F.pixel_unshuffle(proj_feat, downscale_factor=factor)
                    self._build_patch_merge_proj(factor)
                    proj_feat = self._patch_merge_proj(proj_feat)
                else:
                    proj_feat = F.avg_pool2d(proj_feat, kernel_size=factor, stride=factor)
                self.feature_stride = self.feature_stride * factor

        if self.use_fpn:
            # Initialize FPN layers on first forward (now we know actual feature_stride)
            if not self._fpn_layers_initialized:
                self._initialize_fpn_layers()
            
            # Generate multi-scale features using FPN
            fpn_feats = []
            for stride in self.fpn_strides:
                if stride == self.feature_stride:
                    fpn_feat = proj_feat
                else:
                    fpn_feat = self.fpn_layers[f'fpn_{stride}'](proj_feat)
                fpn_feats.append(fpn_feat)
            
            ordered = list(zip(self.fpn_strides, fpn_feats))
            return self._build_nested_from_multiscale(ordered, images)
        else:
            # Single-scale output (original behavior)
            self.feature_strides = [self.feature_stride]
            masks = self._mask_out_padding([proj_feat.shape], images.image_sizes, proj_feat.device)
            return NestedTensor(proj_feat, masks[0])

    def set_freeze(self, freeze: bool) -> None:
        """
        Dynamically switch SAM3 freeze state during training.
        """
        self.freeze = bool(freeze)

        if self.sam3_model is not None:
            if self.freeze:
                self.sam3_model.eval()
            else:
                self.sam3_model.train()
            for p in self.sam3_model.parameters():
                p.requires_grad_(not self.freeze)

        if self._proj_layer is not None:
            if self.freeze:
                self._proj_layer.eval()
            else:
                self._proj_layer.train()
            for p in self._proj_layer.parameters():
                p.requires_grad_(not self.freeze)

        if self._merge_proj_layer is not None:
            if self.freeze:
                self._merge_proj_layer.eval()
            else:
                self._merge_proj_layer.train()
            for p in self._merge_proj_layer.parameters():
                p.requires_grad_(not self.freeze)

        if self._patch_merge_proj is not None:
            # X-SAM 1x1 conv is always trainable
            self._patch_merge_proj.train()
            for p in self._patch_merge_proj.parameters():
                p.requires_grad_(True)

        for layer in self._level_proj_layers.values():
            if self.freeze:
                layer.eval()
            else:
                layer.train()
            for p in layer.parameters():
                p.requires_grad_(not self.freeze)

        if self.fpn_layers is not None:
            for layer in self.fpn_layers.values():
                if self.freeze:
                    layer.eval()
                else:
                    layer.train()
                for p in layer.parameters():
                    p.requires_grad_(not self.freeze)

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
