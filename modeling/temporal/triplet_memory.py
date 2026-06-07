"""
Temporal Triplet Memory Module (temporal_v3).

Stores triplet-level (subject, predicate, object) memories across video frames.
Uses pseudo-identity (sub_label, pred_label, obj_label) for slot matching,
gated cross-attention injection into object and relation queries,
and a GT-to-Prediction curriculum for training stability.

Ref: abschluss-paper/triplet_memory_feasibility2.md
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Geometry utilities
# ──────────────────────────────────────────────

def box_xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box.unbind(-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)


def make_union_box(sub_box: torch.Tensor, obj_box: torch.Tensor) -> torch.Tensor:
    x1 = torch.minimum(sub_box[..., 0], obj_box[..., 0])
    y1 = torch.minimum(sub_box[..., 1], obj_box[..., 1])
    x2 = torch.maximum(sub_box[..., 2], obj_box[..., 2])
    y2 = torch.maximum(sub_box[..., 3], obj_box[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def relative_geometry(sub_box: torch.Tensor, obj_box: torch.Tensor) -> torch.Tensor:
    """
    sub_box, obj_box: [..., 4] normalized xyxy.
    Returns: [..., 8] with dx, dy, dw, dh, area_ratio, sub_union_ratio, obj_union_ratio, center_dist.
    """
    s = box_xyxy_to_cxcywh(sub_box)
    o = box_xyxy_to_cxcywh(obj_box)
    union = make_union_box(sub_box, obj_box)
    u = box_xyxy_to_cxcywh(union)

    dx = o[..., 0] - s[..., 0]
    dy = o[..., 1] - s[..., 1]
    dw = torch.log(o[..., 2] / s[..., 2].clamp(min=1e-6))
    dh = torch.log(o[..., 3] / s[..., 3].clamp(min=1e-6))

    s_area = s[..., 2] * s[..., 3]
    o_area = o[..., 2] * o[..., 3]
    u_area = u[..., 2] * u[..., 3]

    area_ratio = torch.log(o_area / s_area.clamp(min=1e-6))
    sub_union_ratio = s_area / u_area.clamp(min=1e-6)
    obj_union_ratio = o_area / u_area.clamp(min=1e-6)
    center_dist = torch.sqrt(dx * dx + dy * dy)

    return torch.stack([dx, dy, dw, dh, area_ratio, sub_union_ratio, obj_union_ratio, center_dist], dim=-1)


def box_iou_1v1(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """IoU between two single boxes (xyxy). Returns scalar float."""
    x1 = max(box1[0].item(), box2[0].item())
    y1 = max(box1[1].item(), box2[1].item())
    x2 = min(box1[2].item(), box2[2].item())
    y2 = min(box1[3].item(), box2[3].item())
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(1e-6, (box1[2].item() - box1[0].item()) * (box1[3].item() - box1[1].item()))
    a2 = max(1e-6, (box2[2].item() - box2[0].item()) * (box2[3].item() - box2[1].item()))
    return inter / (a1 + a2 - inter + 1e-6)


# ──────────────────────────────────────────────
# TripletMemorySlot
# ──────────────────────────────────────────────

@dataclass
class TripletMemorySlot:
    valid: bool = True
    signature: tuple = ()      # (sub_label, pred_label, obj_label)
    feat: torch.Tensor = None  # [mem_dim]
    sub_box: torch.Tensor = None   # [4] xyxy
    obj_box: torch.Tensor = None   # [4] xyxy
    union_box: torch.Tensor = None # [4] xyxy
    score: float = 0.0
    sub_score: float = 0.0
    obj_score: float = 0.0
    pred_score: float = 0.0
    frame_idx: int = 0
    miss: int = 0
    age: int = 0


# ──────────────────────────────────────────────
# TemporalDeltaEmbedding
# ──────────────────────────────────────────────

class TemporalDeltaEmbedding(nn.Module):
    """Bucketized frame-delta embedding, inspired by SAM3 maskmem_tpos_enc."""

    def __init__(self, mem_dim: int = 128, num_buckets: int = 7):
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, mem_dim)

    def bucketize(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta: [M] int tensor of frame gaps.
        Buckets: 0, 1, 2, 3, 4-7, 8-15, 16+
        """
        bucket = torch.zeros_like(delta)
        bucket = torch.where(delta == 1, torch.ones_like(bucket), bucket)
        bucket = torch.where(delta == 2, torch.full_like(bucket, 2), bucket)
        bucket = torch.where(delta == 3, torch.full_like(bucket, 3), bucket)
        bucket = torch.where((delta >= 4) & (delta <= 7), torch.full_like(bucket, 4), bucket)
        bucket = torch.where((delta >= 8) & (delta <= 15), torch.full_like(bucket, 5), bucket)
        bucket = torch.where(delta >= 16, torch.full_like(bucket, 6), bucket)
        return bucket.long().clamp(0, self.num_buckets - 1)

    def forward(self, memory_feat: torch.Tensor, current_frame_idx: int, memory_frame_idx: torch.Tensor) -> torch.Tensor:
        """Add temporal delta embedding to memory features."""
        delta = torch.full_like(memory_frame_idx, current_frame_idx) - memory_frame_idx
        bucket = self.bucketize(delta)
        return memory_feat + self.emb(bucket)


# ──────────────────────────────────────────────
# TripletMemoryEncoder
# ──────────────────────────────────────────────

class TripletMemoryEncoder(nn.Module):
    """Encodes a triplet candidate into a compact memory feature."""

    def __init__(self, d_model: int = 256, num_rel_classes: int = 26, mem_dim: int = 128):
        super().__init__()
        self.rel_query_proj = nn.Linear(d_model, mem_dim)
        self.box_proj = nn.Sequential(
            nn.Linear(12, 64),  # sub_box + obj_box + union_box
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.geom_proj = nn.Sequential(
            nn.Linear(8, 64),  # relative_geometry
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.pred_proj = nn.Sequential(
            nn.Linear(num_rel_classes, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        in_dim = mem_dim + 64 + 64 + 32  # was +mem_dim(obj_query), removed (unsafe index)
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, mem_dim),
            nn.LayerNorm(mem_dim),
            nn.GELU(),
            nn.Linear(mem_dim, mem_dim),
        )

    def forward(self, rel_query: torch.Tensor,
                sub_box: torch.Tensor, obj_box: torch.Tensor,
                pred_prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_query: [N, d_model]
            sub_box: [N, 4] xyxy normalized
            obj_box: [N, 4] xyxy normalized
            pred_prob: [N, num_rel_classes] detached
        Returns: [N, mem_dim]
        """
        union_box = make_union_box(sub_box, obj_box)
        geom = relative_geometry(sub_box, obj_box)

        box_feat = self.box_proj(torch.cat([sub_box, obj_box, union_box], dim=-1))
        geom_feat = self.geom_proj(geom)
        pred_feat = self.pred_proj(pred_prob.detach())
        rel_feat = self.rel_query_proj(rel_query)

        feat = torch.cat([rel_feat, box_feat, geom_feat, pred_feat], dim=-1)
        return self.fusion(feat)


# ──────────────────────────────────────────────
# TripletMemoryBank
# ──────────────────────────────────────────────

class TripletMemoryBank:
    """Per-video memory bank of triplet slots."""

    def __init__(self, memory_size: int = 32, mem_dim: int = 128,
                 max_miss: int = 2, ema_momentum: float = 0.9,
                 match_iou_thresh: float = 0.3):
        self.memory_size = memory_size
        self.mem_dim = mem_dim
        self.max_miss = max_miss
        self.ema_momentum = ema_momentum
        self.match_iou_thresh = match_iou_thresh
        self.slots: List[TripletMemorySlot] = []

    def clear(self):
        self.slots = []

    def get_valid_slots(self) -> List[TripletMemorySlot]:
        return [s for s in self.slots if s.valid and s.feat is not None]

    def get_memory(self, device: torch.device, current_frame_idx: int = None,
                   delta_t_emb: Optional[TemporalDeltaEmbedding] = None
                   ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            memory_feats: [M, mem_dim] or None
            memory_mask: [M] bool, True=padded (all False here)
        """
        valid = self.get_valid_slots()
        if len(valid) == 0:
            return None, None
        feats = torch.stack([s.feat.to(device) for s in valid], dim=0)
        frame_ids = torch.tensor([s.frame_idx for s in valid], device=device)
        if delta_t_emb is not None and current_frame_idx is not None:
            feats = delta_t_emb(feats, current_frame_idx, frame_ids)
        # Score-weighted memory
        scores = torch.tensor([s.score for s in valid], device=device).clamp(0.0, 1.0)
        feats = feats * scores[:, None]
        mask = torch.zeros(feats.shape[0], dtype=torch.bool, device=device)
        return feats, mask

    def find_match(self, cand: dict) -> Optional[int]:
        """Find best matching slot for candidate by signature + IoU."""
        best_id = None
        best_score = -1.0
        for i, slot in enumerate(self.slots):
            if not slot.valid:
                continue
            if slot.signature != cand["signature"]:
                continue
            sub_iou = box_iou_1v1(slot.sub_box, cand["sub_box"])
            obj_iou = box_iou_1v1(slot.obj_box, cand["obj_box"])
            union_iou = box_iou_1v1(slot.union_box, cand["union_box"])
            pair_iou = (sub_iou + obj_iou + union_iou) / 3.0
            if pair_iou > self.match_iou_thresh and pair_iou > best_score:
                best_score = pair_iou
                best_id = i
        return best_id

    def ema_update(self, slot_id: int, cand: dict, frame_idx: int):
        slot = self.slots[slot_id]
        m = self.ema_momentum
        slot.feat = m * slot.feat + (1.0 - m) * cand["feat"].detach().cpu()
        slot.sub_box = m * slot.sub_box + (1.0 - m) * cand["sub_box"].detach().cpu()
        slot.obj_box = m * slot.obj_box + (1.0 - m) * cand["obj_box"].detach().cpu()
        slot.union_box = m * slot.union_box + (1.0 - m) * cand["union_box"].detach().cpu()
        slot.score = max(slot.score, float(cand["score"]))
        slot.sub_score = float(cand.get("sub_score", 0))
        slot.obj_score = float(cand.get("obj_score", 0))
        slot.pred_score = float(cand.get("pred_score", 0))
        slot.frame_idx = int(frame_idx)
        slot.miss = 0
        slot.age += 1

    def insert_or_replace(self, cand: dict, frame_idx: int) -> int:
        slot = TripletMemorySlot(
            valid=True,
            signature=cand["signature"],
            feat=cand["feat"].detach().cpu(),
            sub_box=cand["sub_box"].detach().cpu(),
            obj_box=cand["obj_box"].detach().cpu(),
            union_box=cand["union_box"].detach().cpu(),
            score=float(cand["score"]),
            sub_score=float(cand.get("sub_score", 0)),
            obj_score=float(cand.get("obj_score", 0)),
            pred_score=float(cand.get("pred_score", 0)),
            frame_idx=int(frame_idx),
            miss=0,
            age=0,
        )
        if len(self.slots) < self.memory_size:
            self.slots.append(slot)
            return len(self.slots) - 1
        # Replace weakest
        weakest_id = min(
            range(len(self.slots)),
            key=lambda i: self.slots[i].score - 0.05 * self.slots[i].miss
        )
        self.slots[weakest_id] = slot
        return weakest_id

    def update(self, candidates: List[dict], frame_idx: int):
        matched_ids = set()
        for cand in candidates:
            match_id = self.find_match(cand)
            if match_id is not None:
                self.ema_update(match_id, cand, frame_idx)
                matched_ids.add(match_id)
            else:
                new_id = self.insert_or_replace(cand, frame_idx)
                matched_ids.add(new_id)
        self.expire_unmatched(matched_ids)

    def expire_unmatched(self, matched_ids: set):
        for i, slot in enumerate(self.slots):
            if not slot.valid:
                continue
            if i not in matched_ids:
                slot.miss += 1
            if slot.miss > self.max_miss:
                slot.valid = False


# ──────────────────────────────────────────────
# TemporalTripletInjector
# ──────────────────────────────────────────────

class TemporalTripletInjector(nn.Module):
    """Gated cross-attention injection of triplet memory into queries."""

    def __init__(self, d_model: int = 256, mem_dim: int = 128, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model={d_model} must be divisible by nhead={nhead}"
        self.query_proj = nn.Linear(d_model, mem_dim)
        self.mem_proj = nn.Linear(mem_dim, mem_dim)
        self.attn = nn.MultiheadAttention(embed_dim=mem_dim, num_heads=nhead,
                                          dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(mem_dim, d_model)

    def forward(self, queries: torch.Tensor,
                memory: Optional[torch.Tensor],
                memory_mask: Optional[torch.Tensor] = None,
                gate: float = 0.0) -> torch.Tensor:
        """
        queries: [B, N, D]
        memory: [B, M, mem_dim] or None
        gate: scalar in [0, 1]
        Returns: [B, N, D]
        """
        if memory is None or gate <= 0.0:
            return queries
        B, N, D = queries.shape
        # Reshape batch*memory for MHA
        M = memory.shape[1]
        if memory_mask is None:
            memory_mask = torch.zeros(B, M, dtype=torch.bool, device=queries.device)
        q = self.query_proj(queries)  # [B, N, mem_dim]
        kv = self.mem_proj(memory)    # [B, M, mem_dim]
        out, _ = self.attn(query=q, key=kv, value=kv, key_padding_mask=memory_mask, need_weights=False)
        out = self.out_proj(out)
        return queries + gate * out


# ──────────────────────────────────────────────
# TripletMemoryManager
# ──────────────────────────────────────────────

class TripletMemoryManager:
    """Manages per-video memory banks and the encoder/injector/delta_emb modules."""

    def __init__(self, cfg):
        self.cfg = cfg
        tcfg = cfg.MODEL.TEMPORAL
        mem_dim = int(getattr(tcfg, "TRIPLET_MEMORY_DIM", 128))
        mem_size = int(getattr(tcfg, "TRIPLET_MEMORY_SIZE", 32))
        max_miss = int(getattr(tcfg, "TRIPLET_MEMORY_MAX_MISS", 2))
        ema_m = float(getattr(tcfg, "UPDATE_EMA_MOMENTUM", 0.9))
        match_iou = float(getattr(tcfg, "MATCH_IOU_THRESH", 0.3))
        self.mem_dim = mem_dim
        self.banks: Dict[str, TripletMemoryBank] = {}
        self.last_frame_idx: Dict[str, int] = {}
        self._bank_kwargs = dict(memory_size=mem_size, mem_dim=mem_dim,
                                 max_miss=max_miss, ema_momentum=ema_m,
                                 match_iou_thresh=match_iou)
        self.delta_t_emb = TemporalDeltaEmbedding(mem_dim, int(getattr(tcfg, "MAX_DELTA_T_BUCKET", 7))) \
            if getattr(tcfg, "USE_DELTA_T_EMB", True) else None
        self.debug_memory = bool(getattr(tcfg, "DEBUG_MEMORY", False))

    def reset_video(self, video_id: str):
        if video_id in self.banks:
            self.banks[video_id].clear()

    def get_or_create_bank(self, video_id: str) -> TripletMemoryBank:
        if video_id not in self.banks:
            self.banks[video_id] = TripletMemoryBank(**self._bank_kwargs)
        return self.banks[video_id]

    def maybe_clear_on_video_jump(self, video_id: str, frame_idx: int):
        if video_id not in self.last_frame_idx:
            self.last_frame_idx[video_id] = frame_idx
            return
        last = self.last_frame_idx[video_id]
        if frame_idx < last:
            self.reset_video(video_id)
        self.last_frame_idx[video_id] = frame_idx

    def to(self, device):
        if self.delta_t_emb is not None:
            self.delta_t_emb = self.delta_t_emb.to(device)
        return self

    def get_batch_memory(self, video_ids: List[str], frame_idxs: List[int],
                         device: torch.device
                         ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns padded [B, M_max, mem_dim] memory and [B, M_max] bool mask."""
        assert len(video_ids) == len(frame_idxs), "video_ids and frame_idxs must match"
        mem_list: List[Optional[torch.Tensor]] = []
        mask_list: List[Optional[torch.Tensor]] = []
        for vid, fidx in zip(video_ids, frame_idxs):
            self.maybe_clear_on_video_jump(vid, fidx)
            bank = self.get_or_create_bank(vid)
            dt_emb = self.delta_t_emb.to(device) if self.delta_t_emb is not None else None
            mem, mask = bank.get_memory(device=device, current_frame_idx=fidx, delta_t_emb=dt_emb)
            mem_list.append(mem)
            mask_list.append(mask)
        return _pad_memory_batch(mem_list, mask_list, device)

    def update_batch(self, video_ids: List[str], frame_idxs: List[int],
                     batch_candidates: List[List[dict]]):
        for vid, fidx, cands in zip(video_ids, frame_idxs, batch_candidates):
            bank = self.get_or_create_bank(vid)
            bank.update(cands, fidx)


def _pad_memory_batch(mem_list: List[Optional[torch.Tensor]],
                      mask_list: List[Optional[torch.Tensor]],
                      device: torch.device
                      ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    valid = [m for m in mem_list if m is not None]
    if len(valid) == 0:
        return None, None
    B = len(mem_list)
    mem_dim = valid[0].shape[-1]
    max_m = max(m.shape[0] if m is not None else 0 for m in mem_list)
    mem_batch = torch.zeros(B, max_m, mem_dim, device=device)
    mask_batch = torch.ones(B, max_m, dtype=torch.bool, device=device)
    for b, mem in enumerate(mem_list):
        if mem is None:
            continue
        m = mem.shape[0]
        mem_batch[b, :m] = mem.to(device)
        mask_batch[b, :m] = False
    return mem_batch, mask_batch


# ──────────────────────────────────────────────
# Gate & curriculum helpers
# ──────────────────────────────────────────────

def get_temporal_gate(iteration: int, max_iter: int,
                      gate_max: float,
                      gate_zero_end_ratio: float = 0.10,
                      gate_warmup_end_ratio: float = 0.30) -> float:
    """Gate warmup: 0 for first 10% iters, linear to gate_max from 10-30%, then gate_max."""
    if max_iter <= 0:
        return gate_max
    r = float(iteration) / float(max_iter)
    if r < gate_zero_end_ratio:
        return 0.0
    if r < gate_warmup_end_ratio:
        return gate_max * (r - gate_zero_end_ratio) / (gate_warmup_end_ratio - gate_zero_end_ratio)
    return gate_max


def get_memory_update_mode(iteration: int, max_iter: int,
                           gt_end_ratio: float = 0.30,
                           mixed_end_ratio: float = 0.70) -> str:
    """Returns 'gt_aligned', 'mixed', or 'prediction'."""
    if max_iter <= 0:
        return "prediction"
    r = float(iteration) / float(max_iter)
    if r < gt_end_ratio:
        return "gt_aligned"
    if r < mixed_end_ratio:
        return "mixed"
    return "prediction"


def get_prediction_threshold(iteration: int, max_iter: int,
                             thresh_start: float = 0.15,
                             thresh_end: float = 0.05) -> float:
    """Linearly decay prediction quality threshold."""
    if max_iter <= 0:
        return thresh_end
    r = float(iteration) / float(max_iter)
    return thresh_start + (thresh_end - thresh_start) * min(r / 0.70, 1.0)
