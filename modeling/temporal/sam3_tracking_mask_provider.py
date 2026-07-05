"""
Sam3TrackingMaskProvider — frozen SAM3 image-level mask as per-frame feature.

Design
-------
- Wraps a frozen SAM3 image model + Sam3Processor (same path as FrozenSAM3).
- Per (video_id, frame_idx) cache keyed by integer frame_idx.
  In overfit mode the same frames repeat, so cache hit >= 99% after warm-up.
- Produces [K, Hf, Wf] binary masks (one per detected object) and per-mask scores.
- Caller is responsible for projection (1x1 conv) and gating.
"""
import logging
import sys
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from PIL import Image


class _Sam3ImageMaskRunner:
    """Lightweight frozen SAM3 image model runner for mask extraction."""

    def __init__(self, cfg):
        self.logger = logging.getLogger("detectron2")
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

        repo_root = None
        cur = os.path.abspath(os.path.dirname(__file__))
        for _ in range(8):
            if os.path.isfile(os.path.join(cur, "sam3", "weights", "sam3.pt")):
                repo_root = cur
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        assert repo_root is not None, (
            "[Sam3TrackingMaskProvider] Could not find sam3/weights/sam3.pt "
            "from provider path."
        )
        sam3_path = os.path.join(repo_root, "sam3")
        for path_entry in [sam3_path, repo_root]:
            if path_entry not in sys.path:
                sys.path.insert(0, path_entry)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.logger.info("[Sam3TrackingMaskProvider] Loading SAM3 image model ...")
        sam3_model = build_sam3_image_model(
            checkpoint_path=os.path.join(repo_root, "sam3", "weights", "sam3.pt"),
            device=str(self.device),
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        sam3_model = sam3_model.to(self.device)
        sam3_model.eval()
        for p in sam3_model.parameters():
            p.requires_grad_(False)

        self.processor = Sam3Processor(
            sam3_model,
            resolution=int(cfg.MODEL.SAM3.IMAGE_SIZE),
            device=str(self.device),
            confidence_threshold=0.0,
        )
        self.logger.info("[Sam3TrackingMaskProvider] SAM3 image model ready.")

    @torch.inference_mode()
    def run(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_tensor: FloatTensor [3, H, W] in [0, 255].
        Returns:
            masks: FloatTensor [K, H, W] binary masks, K may be 0.
            scores: FloatTensor [K]
        """
        img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        img_np = img_np.astype("uint8")
        pil_img = Image.fromarray(img_np)

        state = self.processor.set_image(pil_img)
        prompt = str(self.cfg.MODEL.SAM3.TRACKING_MASK_TEXT_PROMPT)
        output = self.processor.set_text_prompt(prompt, state)

        boxes = output.get("boxes", torch.empty(0, 4, device="cpu"))
        scores = output.get("scores", torch.empty(0, device="cpu"))
        masks = output.get("masks", torch.empty(0, 1, 1, device="cpu"))

        # score filter
        thresh = float(getattr(self.cfg.MODEL.SAM3, "TRACKING_MASK_SCORE_THRESH", 0.0))
        if thresh > 0 and scores.numel() > 0:
            keep = scores > thresh
            boxes = boxes[keep]
            scores = scores[keep]
            masks = masks[keep]

        # top-k
        topk = int(getattr(self.cfg.MODEL.SAM3, "TRACKING_MASK_TOPK", 8))
        if masks.shape[0] > topk and topk > 0:
            _, idx = torch.topk(scores, k=min(topk, scores.shape[0]))
            boxes = boxes[idx]
            scores = scores[idx]
            masks = masks[idx]

        K = masks.shape[0]
        if K == 0:
            H, W = image_tensor.shape[-2:]
            return torch.zeros(0, H, W, device="cpu"), torch.zeros(0, device="cpu")

        if masks.dim() == 4:
            masks = masks.squeeze(1)  # [K, H, W]
        # resize to image resolution
        H, W = image_tensor.shape[-2:]
        if masks.shape[-2:] != (H, W):
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1).float(), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1)

        return masks.cpu().float(), scores.cpu().float()


class Sam3TrackingMaskProvider:
    """
    Per-frame SAM3 mask provider with (video_id, frame_idx) cache.

    Thread-unsafe but fine for DDP single-process-per-rank training.
    """

    def __init__(self, cfg, device="cuda"):
        self.logger = logging.getLogger("detectron2")
        self.cfg = cfg
        self.device = torch.device(device)
        self.enabled = bool(getattr(cfg.MODEL.SAM3, "USE_TRACKING_MASK", False))
        self._runner: Optional[_Sam3ImageMaskRunner] = None
        self._cache: OrderedDict = OrderedDict()
        self._cache_max = int(getattr(cfg.MODEL.SAM3, "TRACKING_MASK_CACHE_MAX_FRAMES", 256))
        assert self._cache_max > 0, "TRACKING_MASK_CACHE_MAX_FRAMES must be positive."
        self._hit_count = 0
        self._miss_count = 0
        self._logged = False

        if self.enabled:
            self._runner = _Sam3ImageMaskRunner(cfg)
            self.logger.info(
                "[Sam3TrackingMaskProvider] ENABLED text_prompt=%s topk=%s score_thresh=%s",
                getattr(cfg.MODEL.SAM3, "TRACKING_MASK_TEXT_PROMPT", "object"),
                getattr(cfg.MODEL.SAM3, "TRACKING_MASK_TOPK", 8),
                getattr(cfg.MODEL.SAM3, "TRACKING_MASK_SCORE_THRESH", 0.0),
            )

    def get_masks_for_batch(
        self,
        images_tensor: torch.Tensor,
        video_ids: List[str],
        frame_idxs: List[int],
    ) -> Dict:
        """
        Args:
            images_tensor: FloatTensor [B, 3, H, W] in [0, 255].
            video_ids: list of B strings.
            frame_idxs: list of B ints.
        Returns:
            dict: "masks_sum": FloatTensor [B, H, W] spatial mask sum per image;
                  "masks_raw": list of [K_i, H, W] FloatTensor per image;
                  "scores_raw": list of [K_i] FloatTensor per image;
                  "hit_rate": float.
        """
        if not self.enabled:
            B = images_tensor.shape[0]
            H, W = images_tensor.shape[-2:]
            return {
                "masks_sum": torch.zeros(B, H, W, device=images_tensor.device),
                "masks_raw": [torch.zeros(0, H, W) for _ in range(B)],
                "scores_raw": [torch.zeros(0) for _ in range(B)],
                "hit_rate": 0.0,
            }

        B = images_tensor.shape[0]
        assert B == len(video_ids) == len(frame_idxs)

        masks_sum = []
        masks_raw = []
        scores_raw = []

        for b in range(B):
            key = f"{video_ids[b]}_{frame_idxs[b]}"
            if key in self._cache:
                raw_masks, raw_scores = self._cache.pop(key)
                self._hit_count += 1
                self._cache[key] = (raw_masks, raw_scores)
            else:
                self._miss_count += 1
                img = images_tensor[b].cpu()  # [3, H, W] on CPU for PIL
                raw_masks, raw_scores = self._runner.run(img)
                raw_masks = raw_masks.cpu()
                raw_scores = raw_scores.cpu()
                self._cache[key] = (raw_masks, raw_scores)
                if len(self._cache) > self._cache_max:
                    self._cache.popitem(last=False)

            masks_raw.append(raw_masks)
            scores_raw.append(raw_scores)

            if raw_masks.shape[0] == 0:
                H, W = images_tensor.shape[-2:]
                spatial = torch.zeros(H, W, device=images_tensor.device)
            else:
                spatial = raw_masks.to(images_tensor.device).sum(0).clamp(0.0, 1.0)
            masks_sum.append(spatial)

        masks_sum_t = torch.stack(masks_sum, 0)  # [B, H, W]
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0.0

        if not self._logged and total >= 10:
            self.logger.info(
                "[Sam3TrackingMaskProvider] hit_rate=%.4f (%d/%d) after %d frames",
                hit_rate, self._hit_count, total, total
            )
            self._logged = True

        total_masks = sum(m.shape[0] for m in masks_raw)
        assert total_masks > 0, (
            f"[Sam3TrackingMaskProvider] ASSERTION FAILED: total_masks={total_masks} "
            f"(hit={self._hit_count}, miss={self._miss_count}). "
            "SAM3 image model returned zero masks for all batched frames."
        )

        return {
            "masks_sum": masks_sum_t,
            "masks_raw": masks_raw,
            "scores_raw": scores_raw,
            "hit_rate": hit_rate,
        }
