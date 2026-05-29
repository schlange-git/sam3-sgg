from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MemoryState:
    queries: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    ages: torch.Tensor
    miss: torch.Tensor
    valid_mask: torch.Tensor


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _pairwise_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=1e-6) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=1e-6))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=1e-6) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=1e-6))
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


class ObjectMemoryBank(nn.Module):
    def __init__(self, d_model: int, cfg):
        super().__init__()
        tcfg = cfg.MODEL.TEMPORAL
        self.num_slots = int(getattr(tcfg, "NUM_MEMORY_SLOTS", 32))
        self.topk = int(getattr(tcfg, "MEMORY_TOPK", 16))
        self.score_thresh = float(getattr(tcfg, "MEMORY_SCORE_THRESH", 0.5))
        self.iou_thresh = float(getattr(tcfg, "MEMORY_IOU_THRESH", 0.5))
        self.max_miss = int(getattr(tcfg, "MEMORY_MAX_MISS", 2))
        self.detach_memory = bool(getattr(tcfg, "DETACH_MEMORY", True))
        self.d_model = int(d_model)
        self.ema_momentum = float(getattr(tcfg, "EMA_MOMENTUM", 0.9))
        self.match_iou_thresh = float(getattr(tcfg, "MEMORY_MATCH_IOU_THRESH", self.iou_thresh))
        self.match_require_class = bool(getattr(tcfg, "MEMORY_MATCH_REQUIRE_CLASS", True))
        self.store_gt_boxes = bool(getattr(tcfg, "MEMORY_STORE_GT_BOXES", True))

    def init_empty(self, device: torch.device) -> MemoryState:
        return MemoryState(
            queries=torch.zeros((self.num_slots, self.d_model), device=device),
            boxes=torch.zeros((self.num_slots, 4), device=device),
            scores=torch.zeros((self.num_slots,), device=device),
            labels=torch.full((self.num_slots,), -1, dtype=torch.long, device=device),
            ages=torch.zeros((self.num_slots,), dtype=torch.long, device=device),
            miss=torch.zeros((self.num_slots,), dtype=torch.long, device=device),
            valid_mask=torch.zeros((self.num_slots,), dtype=torch.bool, device=device),
        )

    def get_memory_queries(self, state: Optional[MemoryState]) -> Optional[torch.Tensor]:
        """Return memory queries weighted by confidence scores."""
        if state is None:
            return None
        out = state.queries.new_zeros((self.num_slots, self.d_model))
        valid_ids = torch.where(state.valid_mask)[0]
        if valid_ids.numel() == 0:
            return out
        scores = state.scores[valid_ids]
        order = valid_ids[torch.argsort(scores, descending=True)]
        k = min(self.num_slots, order.numel())
        # Score-weighted: low-confidence memories fade out
        w = state.scores[order[:k]] / scores.max().clamp(min=1e-6)
        out[:k] = state.queries[order[:k]] * w[:k].unsqueeze(-1)
        return out

    def _select_candidates(self, hs_obj: torch.Tensor, pred_logits: torch.Tensor, pred_boxes: torch.Tensor):
        probs = pred_logits.softmax(dim=-1)
        cls_probs = probs[..., :-1]
        scores, labels = cls_probs.max(dim=-1)
        keep = scores >= self.score_thresh
        if int(keep.sum().item()) == 0:
            return None
        queries = hs_obj[keep]
        boxes = pred_boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if scores.numel() > self.topk:
            topk_idx = torch.topk(scores, k=self.topk, dim=0).indices
            queries = queries[topk_idx]
            boxes = boxes[topk_idx]
            scores = scores[topk_idx]
            labels = labels[topk_idx]
        return queries, boxes, scores, labels

    def _select_matched_gt_candidates(
        self,
        hs_obj: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
    ):
        if (
            hs_obj is None or pred_logits is None or pred_boxes is None or
            target_boxes is None or target_labels is None or target_boxes.numel() == 0
        ):
            return None
        probs = pred_logits.softmax(dim=-1)
        cls_probs = probs[..., :-1]
        scores, labels = cls_probs.max(dim=-1)
        pred_xyxy = _cxcywh_to_xyxy(pred_boxes)
        target_xyxy = _cxcywh_to_xyxy(target_boxes)
        ious = _pairwise_iou_xyxy(pred_xyxy, target_xyxy)
        if self.match_require_class:
            class_match = labels[:, None] == target_labels.long()[None, :]
            ious = ious.masked_fill(~class_match, -1.0)
        best_iou, best_tgt = ious.max(dim=1)
        keep = best_iou >= self.match_iou_thresh
        if int(keep.sum().item()) == 0:
            return None

        keep_idx = torch.where(keep)[0]
        matched_targets = best_tgt[keep_idx]
        queries = hs_obj[keep_idx]
        boxes = target_boxes[matched_targets] if self.store_gt_boxes else pred_boxes[keep_idx]
        scores = (scores[keep_idx] * best_iou[keep_idx].clamp(min=0.0)).clamp(max=1.0)
        labels = target_labels[matched_targets].long()

        # Keep only the strongest prediction for each GT target to avoid filling memory
        # with duplicate predictions for the same object/relation.
        selected = []
        for tgt_idx in torch.unique(matched_targets):
            same = torch.where(matched_targets == tgt_idx)[0]
            best_local = same[torch.argmax(scores[same])]
            selected.append(best_local)
        selected = torch.stack(selected) if len(selected) > 0 else keep_idx.new_zeros((0,))
        if selected.numel() == 0:
            return None
        queries = queries[selected]
        boxes = boxes[selected]
        scores = scores[selected]
        labels = labels[selected]

        if scores.numel() > self.topk:
            topk_idx = torch.topk(scores, k=self.topk, dim=0).indices
            queries = queries[topk_idx]
            boxes = boxes[topk_idx]
            scores = scores[topk_idx]
            labels = labels[topk_idx]
        return queries, boxes, scores, labels

    def _expire_unmatched(self, state: MemoryState, matched_slots: torch.Tensor):
        valid_ids = torch.where(state.valid_mask)[0]
        for slot_idx in valid_ids.tolist():
            if matched_slots[slot_idx]:
                continue
            state.miss[slot_idx] += 1
            if state.miss[slot_idx] > self.max_miss:
                state.valid_mask[slot_idx] = False
                state.scores[slot_idx] = 0.0
                state.labels[slot_idx] = -1
                state.ages[slot_idx] = 0
                state.miss[slot_idx] = 0
                state.queries[slot_idx].zero_()
                state.boxes[slot_idx].zero_()

    def update(self, state: MemoryState, hs_obj: torch.Tensor, pred_logits: torch.Tensor, pred_boxes: torch.Tensor) -> MemoryState:
        if hs_obj is None or pred_logits is None or pred_boxes is None:
            return state
        selected = self._select_candidates(hs_obj, pred_logits, pred_boxes)
        return self._update_from_selected(state, selected)

    def update_matched_gt(
        self,
        state: MemoryState,
        hs_obj: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> MemoryState:
        selected = self._select_matched_gt_candidates(
            hs_obj, pred_logits, pred_boxes, target_boxes, target_labels
        )
        return self._update_from_selected(state, selected)

    def _update_from_selected(self, state: MemoryState, selected) -> MemoryState:
        matched_slots = torch.zeros((self.num_slots,), dtype=torch.bool, device=state.queries.device)
        if selected is None:
            self._expire_unmatched(state, matched_slots)
            return state

        cand_queries, cand_boxes, cand_scores, cand_labels = selected
        if self.detach_memory:
            cand_queries = cand_queries.detach()
            cand_boxes = cand_boxes.detach()
            cand_scores = cand_scores.detach()
            cand_labels = cand_labels.detach()

        cand_boxes_xyxy = _cxcywh_to_xyxy(cand_boxes)

        for q, b, s, l, bxyxy in zip(cand_queries, cand_boxes, cand_scores, cand_labels, cand_boxes_xyxy):
            valid_ids = torch.where(state.valid_mask & (state.labels == l))[0]
            matched_slot = None
            if valid_ids.numel() > 0:
                mem_xyxy = _cxcywh_to_xyxy(state.boxes[valid_ids])
                ious = _pairwise_iou_xyxy(bxyxy.unsqueeze(0), mem_xyxy)[0]
                best_iou, best_idx = torch.max(ious, dim=0)
                if float(best_iou.item()) >= self.iou_thresh:
                    matched_slot = int(valid_ids[int(best_idx.item())].item())

            if matched_slot is None:
                empty_ids = torch.where(~state.valid_mask)[0]
                if empty_ids.numel() > 0:
                    matched_slot = int(empty_ids[0].item())
                else:
                    # Replace the weakest slot when full.
                    replace_idx = int(torch.argmin(state.scores - 0.01 * state.ages.float()).item())
                    matched_slot = replace_idx

            if state.valid_mask[matched_slot]:
                # EMA update for existing slot: smooth transition
                state.queries[matched_slot] = (
                    self.ema_momentum * state.queries[matched_slot] +
                    (1.0 - self.ema_momentum) * q
                )
            else:
                # New slot: direct assignment
                state.queries[matched_slot] = q
            state.boxes[matched_slot] = b
            state.scores[matched_slot] = s
            state.labels[matched_slot] = l
            state.valid_mask[matched_slot] = True
            state.ages[matched_slot] += 1
            state.miss[matched_slot] = 0
            matched_slots[matched_slot] = True

        self._expire_unmatched(state, matched_slots)
        return state


