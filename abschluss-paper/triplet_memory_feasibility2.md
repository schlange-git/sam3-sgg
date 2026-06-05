# Temporal Triplet Memory for SpeaQ：Cursor 落地方案

## 0. 方案目标

在现有 **SAM3 + SpeaQ + Action Genome** 框架中加入一个轻量的 **Temporal Triplet Memory Module**，用于在视频连续帧之间保留 `(subject, predicate, object)` 三元组级别的历史信息。

当前代码已经具备较好的接入基础：已有 `TemporalAggregator`、`ObjectMemoryBank`、`TemporalQueryInjector` 三类时序组件，且已有 memory bank、slot、EMA、miss 过期和 cross-attention 注入机制。
SpeaQ 侧本身以 subject/object/relation query 预测 triplet，评估侧也围绕 `(sub_label, pred_label, obj_label)` 构造三元组。
因此本方案不是重新设计 SGG pipeline，而是在 SpeaQ 原有三分支结构外部增加一个 **可开关、低侵入、query-level 的 triplet memory 模块**。

---

# 1. 核心设计结论

## 1.1 记忆对象

使用：

```text
Triplet Memory Item = subject + predicate + object 的联合记忆
```

而不是只存 relation，也不是显式 tracking ID。

每个 memory slot 对应一个短期伪身份：

```text
triplet_signature = (sub_label, pred_label, obj_label)
```

在 Action Genome 中，subject 基本是 `person`，所以实际接近：

```text
(person, predicate, object_class)
```

这个设计适合 AG，因为：

```text
1. AG 没有显式 instance ID；
2. 同一帧中出现两个完全相同 triplet 的概率较低；
3. 不需要额外 tracking 预处理；
4. 可以直接复用已有 triplet 构造逻辑。
```

---

## 1.2 注入位置

主方案不是只注入 relation，而是：

```text
Memory item：完整 triplet-level
Injection target：object branch + relation branch
Subject branch：默认关闭，但保留开关
```

原因：

```text
1. object 错误会直接影响 triplet recall；
2. predicate 更依赖时序上下文；
3. AG 中 subject 基本为 person，当前 subject recall 已较高，默认不干扰；
4. 如果 object injection 破坏 detection，可以只关闭 object injection，不需要重写代码。
```

---

## 1.3 两阶段实验策略

只保留两个阶段：

```text
Stage 1：Full Triplet Memory，默认 obj + relation 注入
Stage 2：Relation-only fallback，只在 Stage 1 不稳定或 detection 掉点时运行
```

不做复杂 ablation 矩阵。

---

# 2. 总体流程图

```text
Input frame I_t
    │
    ▼
SAM3 / ResNet Backbone
    │
    ▼
Transformer Encoder
    │
    ▼
IterativeRelationDecoder
    │
    ├── Subject Query hs_sub
    ├── Object Query  hs_obj
    └── Relation Query hs_rel
            │
            │ 读取当前 video_id 对应的历史 Triplet Memory
            ▼
TripletMemoryBank.get_memory(video_id, frame_idx)
            │
            ▼
TemporalTripletInjector
    │
    ├── Object Query Injection   可开关
    └── Relation Query Injection 可开关
            │
            ▼
Prediction Heads
    │
    ├── subject logits / boxes
    ├── object logits / boxes
    └── relation logits
            │
            ▼
构造当前帧预测 triplet candidates
            │
            ▼
TripletMemoryEncoder
            │
            ▼
TripletMemoryBank.update(video_id, candidates)
            │
            ▼
Loss / Evaluation
```

---

# 3. 推荐文件改动结构

## 3.1 新增文件

```text
modeling/temporal/triplet_memory.py
```

包含：

```text
TripletMemorySlot
TripletMemoryEncoder
TripletMemoryBank
TemporalTripletInjector
TripletMemoryManager
```

如果你希望文件更清晰，也可以拆成：

```text
modeling/temporal/triplet_memory_encoder.py
modeling/temporal/triplet_memory_bank.py
modeling/temporal/triplet_memory_injector.py
```

但给 Cursor 实现时，建议先放在一个文件里，减少 import 链路错误。

---

## 3.2 修改文件

```text
configs/defaults.py
```

新增 temporal triplet memory 配置。

```text
modeling/transformer/detr.py
```

或你当前 SpeaQ transformer 主体文件中输出 `hs_sub / hs_obj / hs_rel` 的位置。

```text
modeling/meta_arch/detr.py
```

负责：

```text
1. 从 batched_inputs 读取 video_id / frame_idx；
2. 调用 temporal memory；
3. 在 forward 结束后更新 memory。
```

```text
data/datasets/action_genome.py
```

确保每个样本返回：

```python
{
    "video_id": ...,
    "frame_idx": ...,
    "is_keyframe": ...,
}
```

如果已经有这些字段，只需要确认 forward 中能拿到。

---

# 4. 配置项设计

给 Cursor 的任务：在 `configs/defaults.py` 中加入如下配置。

```python
# Temporal Triplet Memory
_C.MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED = False

# memory type
_C.MODEL.TEMPORAL.TRIPLET_MEMORY_DIM = 128
_C.MODEL.TEMPORAL.TRIPLET_MEMORY_SIZE = 32
_C.MODEL.TEMPORAL.TRIPLET_MEMORY_TOPK_UPDATE = 16
_C.MODEL.TEMPORAL.TRIPLET_MEMORY_MAX_MISS = 2

# injection switches
_C.MODEL.TEMPORAL.INJECT_SUBJECT = False
_C.MODEL.TEMPORAL.INJECT_OBJECT = True
_C.MODEL.TEMPORAL.INJECT_RELATION = True

# gate
_C.MODEL.TEMPORAL.GATE_INIT = 0.0
_C.MODEL.TEMPORAL.GATE_MAX_SUBJECT = 0.0
_C.MODEL.TEMPORAL.GATE_MAX_OBJECT = 0.15
_C.MODEL.TEMPORAL.GATE_MAX_RELATION = 0.30
_C.MODEL.TEMPORAL.GATE_WARMUP_ITERS = 5000

# memory update
_C.MODEL.TEMPORAL.UPDATE_WITH_PREDICTION = True
_C.MODEL.TEMPORAL.UPDATE_SCORE_THRESH = 0.05
_C.MODEL.TEMPORAL.UPDATE_EMA_MOMENTUM = 0.9
_C.MODEL.TEMPORAL.MATCH_IOU_THRESH = 0.3

# temporal position encoding
_C.MODEL.TEMPORAL.USE_DELTA_T_EMB = True
_C.MODEL.TEMPORAL.MAX_DELTA_T_BUCKET = 6

# debug
_C.MODEL.TEMPORAL.DEBUG_MEMORY = False
```

---

# 5. 核心模块伪代码

## 5.1 TripletMemorySlot

```python
from dataclasses import dataclass
import torch


@dataclass
class TripletMemorySlot:
    valid: bool
    signature: tuple  # (sub_label, pred_label, obj_label)

    feat: torch.Tensor       # [mem_dim]
    sub_box: torch.Tensor    # [4]
    obj_box: torch.Tensor    # [4]
    union_box: torch.Tensor  # [4]

    score: float
    sub_score: float
    obj_score: float
    pred_score: float

    frame_idx: int
    miss: int
    age: int
```

说明：

```text
feat 是压缩后的 triplet memory feature。
signature 用于无 ID 情况下的短期伪身份匹配。
miss 用于视频内短时消散。
video 切换时整个 bank 清空。
```

---

## 5.2 相对几何编码函数

不要只依赖绝对 box 坐标，因为 AG 中相机可能移动。使用相对几何更稳。

```python
import torch


def box_xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box.unbind(-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)


def make_union_box(sub_box, obj_box):
    x1 = torch.minimum(sub_box[..., 0], obj_box[..., 0])
    y1 = torch.minimum(sub_box[..., 1], obj_box[..., 1])
    x2 = torch.maximum(sub_box[..., 2], obj_box[..., 2])
    y2 = torch.maximum(sub_box[..., 3], obj_box[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def relative_geometry(sub_box, obj_box):
    """
    sub_box, obj_box: [..., 4], normalized xyxy
    return: [..., 8]
    """
    s = box_xyxy_to_cxcywh(sub_box)
    o = box_xyxy_to_cxcywh(obj_box)

    dx = o[..., 0] - s[..., 0]
    dy = o[..., 1] - s[..., 1]
    dw = torch.log(o[..., 2] / s[..., 2])
    dh = torch.log(o[..., 3] / s[..., 3])

    s_area = s[..., 2] * s[..., 3]
    o_area = o[..., 2] * o[..., 3]
    area_ratio = torch.log(o_area / s_area.clamp(min=1e-6))

    union = make_union_box(sub_box, obj_box)
    u = box_xyxy_to_cxcywh(union)
    union_area = u[..., 2] * u[..., 3]

    s_union_ratio = s_area / union_area.clamp(min=1e-6)
    o_union_ratio = o_area / union_area.clamp(min=1e-6)
    center_dist = torch.sqrt(dx * dx + dy * dy)

    return torch.stack([
        dx, dy, dw, dh,
        area_ratio,
        s_union_ratio,
        o_union_ratio,
        center_dist,
    ], dim=-1)
```

---

## 5.3 TripletMemoryEncoder

第一版不要做 union ROI feature，避免改动 backbone feature sampling。先用 query + box + relative geometry + predicate distribution。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMemoryEncoder(nn.Module):
    def __init__(self, d_model=256, num_rel_classes=26, mem_dim=128):
        super().__init__()

        self.rel_query_proj = nn.Linear(d_model, mem_dim)
        self.obj_query_proj = nn.Linear(d_model, mem_dim)

        self.box_proj = nn.Sequential(
            nn.Linear(12, 64),  # sub_box + obj_box + union_box
            nn.LayerNorm(64),
            nn.GELU(),
        )

        self.geom_proj = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        self.pred_proj = nn.Sequential(
            nn.Linear(num_rel_classes, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )

        in_dim = mem_dim + mem_dim + 64 + 64 + 32

        self.fusion = nn.Sequential(
            nn.Linear(in_dim, mem_dim),
            nn.LayerNorm(mem_dim),
            nn.GELU(),
            nn.Linear(mem_dim, mem_dim),
        )

    def forward(
        self,
        rel_query,
        obj_query,
        sub_box,
        obj_box,
        pred_prob,
    ):
        """
        rel_query: [N, D]
        obj_query: [N, D]
        sub_box: [N, 4], normalized xyxy
        obj_box: [N, 4], normalized xyxy
        pred_prob: [N, C_rel], detached probability
        return: [N, mem_dim]
        """
        union_box = make_union_box(sub_box, obj_box)
        geom = relative_geometry(sub_box, obj_box)

        box_feat = self.box_proj(torch.cat([sub_box, obj_box, union_box], dim=-1))
        geom_feat = self.geom_proj(geom)
        pred_feat = self.pred_proj(pred_prob.detach())

        rel_feat = self.rel_query_proj(rel_query)
        obj_feat = self.obj_query_proj(obj_query)

        feat = torch.cat([
            rel_feat,
            obj_feat,
            box_feat,
            geom_feat,
            pred_feat,
        ], dim=-1)

        return self.fusion(feat)
```

注意：

```text
pred_prob.detach() 是为了避免 memory update 反向污染历史预测。
memory 本身只服务下一帧，不应该让历史 slot 参与反向传播。
```

---

## 5.4 Temporal Delta Embedding

```python
class TemporalDeltaEmbedding(nn.Module):
    def __init__(self, mem_dim=128, num_buckets=7):
        super().__init__()
        self.emb = nn.Embedding(num_buckets, mem_dim)

    def bucketize(self, delta):
        """
        delta: Tensor[int], frame gap
        bucket:
          0, 1, 2, 3, 4-7, 8-15, 16+
        """
        bucket = torch.zeros_like(delta)
        bucket = torch.where(delta <= 0, torch.zeros_like(bucket), bucket)
        bucket = torch.where(delta == 1, torch.ones_like(bucket) * 1, bucket)
        bucket = torch.where(delta == 2, torch.ones_like(bucket) * 2, bucket)
        bucket = torch.where(delta == 3, torch.ones_like(bucket) * 3, bucket)
        bucket = torch.where((delta >= 4) & (delta <= 7), torch.ones_like(bucket) * 4, bucket)
        bucket = torch.where((delta >= 8) & (delta <= 15), torch.ones_like(bucket) * 5, bucket)
        bucket = torch.where(delta >= 16, torch.ones_like(bucket) * 6, bucket)
        return bucket.long()

    def forward(self, memory_feat, current_frame_idx, memory_frame_idx):
        """
        memory_feat: [M, D]
        current_frame_idx: int
        memory_frame_idx: [M]
        """
        delta = current_frame_idx - memory_frame_idx
        bucket = self.bucketize(delta)
        return memory_feat + self.emb(bucket)
```

---

## 5.5 TripletMemoryBank

```python
class TripletMemoryBank:
    def __init__(
        self,
        memory_size=32,
        mem_dim=128,
        max_miss=2,
        ema_momentum=0.9,
        match_iou_thresh=0.3,
    ):
        self.memory_size = memory_size
        self.mem_dim = mem_dim
        self.max_miss = max_miss
        self.ema_momentum = ema_momentum
        self.match_iou_thresh = match_iou_thresh

        self.slots = []

    def clear(self):
        self.slots = []

    def get_valid_slots(self):
        return [s for s in self.slots if s.valid]

    def get_memory(self, device, current_frame_idx=None, delta_t_emb=None):
        valid = self.get_valid_slots()

        if len(valid) == 0:
            return None, None

        feats = torch.stack([s.feat.to(device) for s in valid], dim=0)
        frame_ids = torch.tensor([s.frame_idx for s in valid], device=device)

        if delta_t_emb is not None and current_frame_idx is not None:
            feats = delta_t_emb(feats, current_frame_idx, frame_ids)

        scores = torch.tensor([s.score for s in valid], device=device).clamp(min=0.0, max=1.0)
        feats = feats * scores[:, None]

        # return memory [M, D], mask [M]
        mask = torch.zeros(feats.shape[0], dtype=torch.bool, device=device)
        return feats, mask

    def update(self, candidates, frame_idx):
        """
        candidates: List[dict]
        each candidate:
          {
            "signature": (sub_label, pred_label, obj_label),
            "feat": Tensor[mem_dim],
            "sub_box": Tensor[4],
            "obj_box": Tensor[4],
            "union_box": Tensor[4],
            "score": float,
            "sub_score": float,
            "obj_score": float,
            "pred_score": float,
          }
        """
        matched_slot_ids = set()

        for cand in candidates:
            match_id = self.find_match(cand)

            if match_id is not None:
                self.ema_update(match_id, cand, frame_idx)
                matched_slot_ids.add(match_id)
            else:
                new_id = self.insert_or_replace(cand, frame_idx)
                matched_slot_ids.add(new_id)

        self.expire_unmatched(matched_slot_ids)

    def find_match(self, cand):
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

    def ema_update(self, slot_id, cand, frame_idx):
        slot = self.slots[slot_id]
        m = self.ema_momentum

        slot.feat = m * slot.feat + (1.0 - m) * cand["feat"].detach().cpu()
        slot.sub_box = m * slot.sub_box + (1.0 - m) * cand["sub_box"].detach().cpu()
        slot.obj_box = m * slot.obj_box + (1.0 - m) * cand["obj_box"].detach().cpu()
        slot.union_box = m * slot.union_box + (1.0 - m) * cand["union_box"].detach().cpu()

        slot.score = max(slot.score, float(cand["score"]))
        slot.sub_score = float(cand["sub_score"])
        slot.obj_score = float(cand["obj_score"])
        slot.pred_score = float(cand["pred_score"])

        slot.frame_idx = int(frame_idx)
        slot.miss = 0
        slot.age += 1

    def insert_or_replace(self, cand, frame_idx):
        slot = TripletMemorySlot(
            valid=True,
            signature=cand["signature"],
            feat=cand["feat"].detach().cpu(),
            sub_box=cand["sub_box"].detach().cpu(),
            obj_box=cand["obj_box"].detach().cpu(),
            union_box=cand["union_box"].detach().cpu(),
            score=float(cand["score"]),
            sub_score=float(cand["sub_score"]),
            obj_score=float(cand["obj_score"]),
            pred_score=float(cand["pred_score"]),
            frame_idx=int(frame_idx),
            miss=0,
            age=0,
        )

        if len(self.slots) < self.memory_size:
            self.slots.append(slot)
            return len(self.slots) - 1

        # replace weakest
        weakest_id = min(
            range(len(self.slots)),
            key=lambda i: self.slots[i].score - 0.05 * self.slots[i].miss
        )
        self.slots[weakest_id] = slot
        return weakest_id

    def expire_unmatched(self, matched_slot_ids):
        for i, slot in enumerate(self.slots):
            if not slot.valid:
                continue

            if i not in matched_slot_ids:
                slot.miss += 1

            if slot.miss > self.max_miss:
                slot.valid = False
```

`box_iou_1v1` 可直接复用已有 box_ops 里的 IoU 函数，或者写一个 1v1 简化版本。

---

## 5.6 TemporalTripletInjector

```python
class TemporalTripletInjector(nn.Module):
    def __init__(self, d_model=256, mem_dim=128, nhead=8, dropout=0.1):
        super().__init__()

        self.query_proj = nn.Linear(d_model, mem_dim)
        self.mem_proj = nn.Linear(mem_dim, mem_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=mem_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.out_proj = nn.Linear(mem_dim, d_model)

    def forward(self, queries, memory, memory_mask=None, gate=0.0):
        """
        queries: [B, N, D]
        memory: [B, M, mem_dim] or None
        memory_mask: [B, M], True means padded
        gate: float
        """
        if memory is None or gate <= 0:
            return queries

        q = self.query_proj(queries)
        kv = self.mem_proj(memory)

        out, attn = self.attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=memory_mask,
            need_weights=False,
        )

        out = self.out_proj(out)
        return queries + gate * out
```

---

## 5.7 Gate Warmup

```python
def get_temporal_gate(iteration, warmup_iters, gate_max):
    if warmup_iters <= 0:
        return gate_max

    ratio = min(1.0, float(iteration) / float(warmup_iters))
    return gate_max * ratio
```

推荐：

```text
object gate:   0 → 0.15
relation gate: 0 → 0.30
subject gate:  0
```

---

# 6. Forward 集成流程

## 6.1 推荐集成位置

不要改 encoder，不要改 backbone，不要改 matcher。

只在 decoder 输出 query 后、prediction head 前进行：

```text
hs_obj[-1] = TemporalTripletInjector(hs_obj[-1], memory)
hs_rel[-1] = TemporalTripletInjector(hs_rel[-1], memory)
```

如果你的代码中 `hs_obj` / `hs_rel` 是：

```python
hs_obj: [num_layers, B, N_obj, D]
hs_rel: [num_layers, B, N_rel, D]
```

则只改最后一层：

```python
hs_obj_temporal = hs_obj.clone()
hs_rel_temporal = hs_rel.clone()

hs_obj_temporal[-1] = inject_obj(hs_obj[-1], memory, gate_obj)
hs_rel_temporal[-1] = inject_rel(hs_rel[-1], memory, gate_rel)
```

然后后续 head 使用 `hs_obj_temporal` / `hs_rel_temporal`。

这样做有两个好处：

```text
1. aux loss 仍然大体保持原结构；
2. temporal 模块主要影响最终输出，不强行扰动所有 decoder 层。
```

---

## 6.2 Forward 伪代码

```python
def forward(self, batched_inputs):
    video_ids = [x.get("video_id", "default") for x in batched_inputs]
    frame_idxs = [x.get("frame_idx", 0) for x in batched_inputs]

    # 1. 原始 SpeaQ 前向
    features, pos = self.backbone_and_encoder(batched_inputs)

    hs_sub, hs_obj, hs_rel, memory_from_encoder = self.transformer(...)

    # 2. 读取历史 triplet memory
    if self.cfg.MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED:
        mem, mem_mask = self.triplet_memory_manager.get_batch_memory(
            video_ids=video_ids,
            frame_idxs=frame_idxs,
            device=hs_obj.device,
        )

        iter_now = self.iteration

        gate_obj = get_temporal_gate(
            iter_now,
            self.cfg.MODEL.TEMPORAL.GATE_WARMUP_ITERS,
            self.cfg.MODEL.TEMPORAL.GATE_MAX_OBJECT,
        )

        gate_rel = get_temporal_gate(
            iter_now,
            self.cfg.MODEL.TEMPORAL.GATE_WARMUP_ITERS,
            self.cfg.MODEL.TEMPORAL.GATE_MAX_RELATION,
        )

        hs_obj = hs_obj.clone()
        hs_rel = hs_rel.clone()

        if self.cfg.MODEL.TEMPORAL.INJECT_OBJECT:
            hs_obj[-1] = self.object_triplet_injector(
                queries=hs_obj[-1],
                memory=mem,
                memory_mask=mem_mask,
                gate=gate_obj,
            )

        if self.cfg.MODEL.TEMPORAL.INJECT_RELATION:
            hs_rel[-1] = self.relation_triplet_injector(
                queries=hs_rel[-1],
                memory=mem,
                memory_mask=mem_mask,
                gate=gate_rel,
            )

    # 3. 原始 prediction heads
    outputs = self.prediction_heads(hs_sub, hs_obj, hs_rel)

    # 4. loss 正常计算
    losses = self.compute_losses(outputs, targets)

    # 5. 使用当前预测更新 memory，给后续帧使用
    if self.cfg.MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED:
        with torch.no_grad():
            candidates = self.build_triplet_memory_candidates(
                outputs=outputs,
                hs_sub=hs_sub,
                hs_obj=hs_obj,
                hs_rel=hs_rel,
                batched_inputs=batched_inputs,
            )

            self.triplet_memory_manager.update_batch(
                video_ids=video_ids,
                frame_idxs=frame_idxs,
                candidates=candidates,
            )

    return losses, outputs
```

---

# 7. Triplet candidates 构造

## 7.1 需要使用已有 triplet 连接逻辑

不要重新发明 pair 构造。你当前流程中已经有：

```text
relation_logits
rel_pair_idx
pred_logits / pred_boxes
```

并且训练/评估会围绕 `rel_pair_idx` 构造 triplet。

Cursor 任务：

```text
找到当前代码中构造 triplet 或 rel_pair_idx 的位置。
复用同一套 subject/object/relation 索引。
不要新写一套 pair enumeration。
```

---

## 7.2 Candidate 构造伪代码

```python
def build_triplet_memory_candidates(
    outputs,
    hs_sub,
    hs_obj,
    hs_rel,
    batched_inputs,
):
    """
    return:
      batch_candidates: List[List[dict]]
    """
    batch_candidates = []

    pred_sub_logits = outputs["pred_sub_logits"]      # [B, N_sub, C_obj+1]
    pred_obj_logits = outputs["pred_obj_logits"]      # [B, N_obj, C_obj+1]
    pred_rel_logits = outputs["pred_rel_logits"]      # [B, N_rel, C_rel+1]

    pred_sub_boxes = outputs["pred_sub_boxes"]        # [B, N_sub, 4]
    pred_obj_boxes = outputs["pred_obj_boxes"]        # [B, N_obj, 4]

    rel_pair_idx = outputs["rel_pair_idx"]            # list or tensor, per image

    for b in range(len(batched_inputs)):
        candidates_b = []

        sub_prob = pred_sub_logits[b].softmax(-1)
        obj_prob = pred_obj_logits[b].softmax(-1)
        rel_prob = pred_rel_logits[b].softmax(-1)

        # remove background class if last dim is background
        sub_score, sub_label = sub_prob[..., :-1].max(-1)
        obj_score, obj_label = obj_prob[..., :-1].max(-1)
        pred_score, pred_label = rel_prob[..., :-1].max(-1)

        for r in range(rel_prob.shape[0]):
            s_idx, o_idx = get_pair_indices(rel_pair_idx, b, r)

            quality = (
                sub_score[s_idx]
                * obj_score[o_idx]
                * pred_score[r]
            )

            if quality < self.cfg.MODEL.TEMPORAL.UPDATE_SCORE_THRESH:
                continue

            signature = (
                int(sub_label[s_idx]),
                int(pred_label[r]),
                int(obj_label[o_idx]),
            )

            sub_box = pred_sub_boxes[b, s_idx]
            obj_box = pred_obj_boxes[b, o_idx]
            union_box = make_union_box(sub_box, obj_box)

            # relation query and object query from final decoder layer
            rel_query = hs_rel[-1, b, r]
            obj_query = hs_obj[-1, b, o_idx]

            pred_dist = rel_prob[r, :-1]

            mem_feat = self.triplet_memory_encoder(
                rel_query=rel_query[None],
                obj_query=obj_query[None],
                sub_box=sub_box[None],
                obj_box=obj_box[None],
                pred_prob=pred_dist[None],
            )[0]

            candidates_b.append({
                "signature": signature,
                "feat": mem_feat,
                "sub_box": sub_box,
                "obj_box": obj_box,
                "union_box": union_box,
                "score": float(quality),
                "sub_score": float(sub_score[s_idx]),
                "obj_score": float(obj_score[o_idx]),
                "pred_score": float(pred_score[r]),
            })

        # top-k by quality
        candidates_b = sorted(
            candidates_b,
            key=lambda x: x["score"],
            reverse=True,
        )[:self.cfg.MODEL.TEMPORAL.TRIPLET_MEMORY_TOPK_UPDATE]

        batch_candidates.append(candidates_b)

    return batch_candidates
```

---

# 8. TripletMemoryManager

用于管理多个 video 的 memory。

```python
class TripletMemoryManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.banks = {}
        self.last_frame_idx = {}

    def reset_video(self, video_id):
        if video_id in self.banks:
            self.banks[video_id].clear()

    def get_or_create_bank(self, video_id):
        if video_id not in self.banks:
            self.banks[video_id] = TripletMemoryBank(
                memory_size=self.cfg.MODEL.TEMPORAL.TRIPLET_MEMORY_SIZE,
                mem_dim=self.cfg.MODEL.TEMPORAL.TRIPLET_MEMORY_DIM,
                max_miss=self.cfg.MODEL.TEMPORAL.TRIPLET_MEMORY_MAX_MISS,
                ema_momentum=self.cfg.MODEL.TEMPORAL.UPDATE_EMA_MOMENTUM,
                match_iou_thresh=self.cfg.MODEL.TEMPORAL.MATCH_IOU_THRESH,
            )
        return self.banks[video_id]

    def maybe_clear_on_video_jump(self, video_id, frame_idx):
        """
        如果 dataloader 保证同一个 video 连续输入，这里只需要检测 frame_idx 是否倒退。
        """
        if video_id not in self.last_frame_idx:
            self.last_frame_idx[video_id] = frame_idx
            return

        last = self.last_frame_idx[video_id]

        if frame_idx < last:
            self.reset_video(video_id)

        self.last_frame_idx[video_id] = frame_idx

    def get_batch_memory(self, video_ids, frame_idxs, device):
        mem_list = []
        mask_list = []

        for video_id, frame_idx in zip(video_ids, frame_idxs):
            self.maybe_clear_on_video_jump(video_id, frame_idx)

            bank = self.get_or_create_bank(video_id)

            mem, mask = bank.get_memory(
                device=device,
                current_frame_idx=frame_idx,
                delta_t_emb=self.delta_t_emb if self.cfg.MODEL.TEMPORAL.USE_DELTA_T_EMB else None,
            )

            mem_list.append(mem)
            mask_list.append(mask)

        return pad_memory_batch(mem_list, mask_list, device)

    def update_batch(self, video_ids, frame_idxs, candidates):
        for video_id, frame_idx, cand in zip(video_ids, frame_idxs, candidates):
            bank = self.get_or_create_bank(video_id)
            bank.update(cand, frame_idx)
```

`pad_memory_batch`：

```python
def pad_memory_batch(mem_list, mask_list, device):
    """
    mem_list: List[Tensor[M_i, D] or None]
    return:
      mem_batch: [B, M_max, D] or None
      mask_batch: [B, M_max], True means padding
    """
    valid_mems = [m for m in mem_list if m is not None]

    if len(valid_mems) == 0:
        return None, None

    mem_dim = valid_mems[0].shape[-1]
    max_m = max(m.shape[0] if m is not None else 0 for m in mem_list)
    B = len(mem_list)

    mem_batch = torch.zeros(B, max_m, mem_dim, device=device)
    mask_batch = torch.ones(B, max_m, dtype=torch.bool, device=device)

    for b, mem in enumerate(mem_list):
        if mem is None:
            continue
        m = mem.shape[0]
        mem_batch[b, :m] = mem
        mask_batch[b, :m] = False

    return mem_batch, mask_batch
```

---

# 9. Cursor 两阶段 Plan

## Stage 1：Full Triplet Memory 主方案

### 目标

实现：

```text
Triplet memory item
+ Temporal delta embedding
+ Object branch injection
+ Relation branch injection
+ Prediction-based memory update
```

### 配置

```yaml
MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED: True
MODEL.TEMPORAL.INJECT_SUBJECT: False
MODEL.TEMPORAL.INJECT_OBJECT: True
MODEL.TEMPORAL.INJECT_RELATION: True
MODEL.TEMPORAL.TRIPLET_MEMORY_DIM: 128
MODEL.TEMPORAL.TRIPLET_MEMORY_SIZE: 32
MODEL.TEMPORAL.TRIPLET_MEMORY_TOPK_UPDATE: 16
MODEL.TEMPORAL.TRIPLET_MEMORY_MAX_MISS: 2
MODEL.TEMPORAL.GATE_MAX_OBJECT: 0.15
MODEL.TEMPORAL.GATE_MAX_RELATION: 0.30
MODEL.TEMPORAL.GATE_WARMUP_ITERS: 5000
MODEL.TEMPORAL.UPDATE_SCORE_THRESH: 0.05
MODEL.TEMPORAL.USE_DELTA_T_EMB: True
```

### Cursor 任务拆分

#### Task 1：新增配置

```text
修改 configs/defaults.py
添加 MODEL.TEMPORAL.TRIPLET_MEMORY_* 配置项
确保默认全部关闭，不影响 baseline
```

验收：

```text
默认配置运行 baseline 不报错，输出完全不经过 triplet memory。
```

---

#### Task 2：实现 triplet_memory.py

新增：

```text
modeling/temporal/triplet_memory.py
```

包含：

```text
TripletMemorySlot
TripletMemoryEncoder
TemporalDeltaEmbedding
TripletMemoryBank
TemporalTripletInjector
TripletMemoryManager
```

验收：

```text
可以单独 import；
输入 dummy query / box / pred_prob 能得到 [N, mem_dim]；
empty memory 时 injector 直接返回原 queries。
```

---

#### Task 3：数据字段检查

修改或检查：

```text
data/datasets/action_genome.py
```

确保 sample 中有：

```python
"video_id"
"frame_idx"
"is_keyframe"
```

验收：

```text
训练 dataloader 打印一个 batch，能看到 video_id 和 frame_idx。
```

---

#### Task 4：接入 forward

在 `modeling/meta_arch/detr.py` 或当前 SpeaQ 主 forward 中：

```text
1. 初始化 TripletMemoryManager；
2. decoder 输出 hs_sub/hs_obj/hs_rel 后读取 memory；
3. 对 hs_obj[-1] 和 hs_rel[-1] 做 injection；
4. prediction head 正常输出；
5. no_grad 构造 candidates；
6. 更新 memory。
```

验收：

```text
TRIPLET_MEMORY_ENABLED=False 时训练行为不变；
TRIPLET_MEMORY_ENABLED=True 时 forward 不报错；
memory size 从 0 逐步增加；
video 切换后 memory 清空或重新建 bank。
```

---

#### Task 5：Debug 日志

只在 `DEBUG_MEMORY=True` 时打印：

```text
video_id
frame_idx
num_memory_slots
num_update_candidates
mean_memory_score
gate_object
gate_relation
```

验收：

```text
每隔 100 iter 打印一次即可，不要每 iter 刷屏。
```

---

### Stage 1 训练判断

观察：

```text
1. loss 是否能正常下降；
2. object loss 是否明显异常升高；
3. relation loss 是否明显更稳定；
4. R@20/R@50 是否提升；
5. mR@20/mR@50 是否提升；
6. detection AP 是否大幅下降。
```

判定：

```text
若 R/mR 提升，且 detection 没有明显回退，Stage 1 成立。
若 relation 有提升但 detection 掉点，进入 Stage 2。
若完全不收敛，也进入 Stage 2。
```

---

## Stage 2：Relation-only Fallback

### 目标

保留完整 triplet memory item，但只注入 relation branch。

### 配置

```yaml
MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED: True
MODEL.TEMPORAL.INJECT_SUBJECT: False
MODEL.TEMPORAL.INJECT_OBJECT: False
MODEL.TEMPORAL.INJECT_RELATION: True
MODEL.TEMPORAL.GATE_MAX_RELATION: 0.25
MODEL.TEMPORAL.GATE_WARMUP_ITERS: 5000
```

### Cursor 任务

不需要新增代码，只改配置。

验收：

```text
hs_obj 不经过 temporal injector；
hs_rel 经过 temporal injector；
memory 仍然由完整 triplet candidates 更新。
```

### Stage 2 判断

```text
若 Stage 2 > baseline：
    说明 triplet memory 对 predicate 有效，但 object injection 干扰 detection。

若 Stage 2 ≈ baseline：
    说明当前抽帧稀疏和 AG 标注下，query-level memory 增益有限。

若 Stage 2 < baseline：
    说明 memory update 本身噪声较强，需要提高 UPDATE_SCORE_THRESH 或延长 gate warmup。
```

---

# 10. 必要的安全约束

## 10.1 不改 matcher

不要改：

```text
SpeaQHungarianMatcher
Quality-Aware Multi-Assignment
Groupwise Query Specialization
```

SpeaQ 的核心贡献本身就是 query specialization 和 quality-aware multi-assignment，改 matcher 会引入额外变量，难以判断 temporal module 是否有效。

---

## 10.2 不把 memory 反向传播到历史帧

所有写入 memory 的内容必须：

```python
.detach().cpu()
```

否则会产生跨 iter graph 问题，也会显存爆炸。

---

## 10.3 视频切换必须清空或隔离

必须按 `video_id` 管理 bank：

```text
不同 video 不共享 memory。
```

不能只靠 miss 自然消散，否则跨视频污染非常严重。

---

## 10.4 非关键帧不参与 loss

非关键帧可以：

```text
1. forward；
2. update memory；
3. 不参与 evaluation；
4. 不参与 supervised loss，除非有 GT。
```

如果当前 dataloader 已经把中间帧加入序列，但没有 GT，则 loss 端需要跳过这些帧。

---

# 11. 最终给 Cursor 的总指令

可以直接把下面这段给 Cursor：

```text
请在当前 SpeaQ + SAM3 + Action Genome 代码中实现一个 Temporal Triplet Memory 模块。

目标：
1. 不修改 backbone、encoder、matcher 和原始 SpeaQ triplet 构造逻辑。
2. 新增 query-level triplet memory，memory item 表示完整 (subject, predicate, object) 三元组。
3. 使用 pseudo triplet signature = (sub_label, pred_label, obj_label) 作为短期伪身份，不引入外部 tracking id。
4. memory 从上一帧或历史帧读取，通过 gated cross-attention 注入当前 object query 和 relation query。
5. subject injection 默认关闭，但保留配置项。
6. 当前帧预测结束后，使用高置信度 triplet candidates 更新 memory，所有 memory 写入 detach，不反向传播。
7. 按 video_id 管理 memory bank，不同视频不能共享 memory。
8. 默认所有新功能关闭，确保 baseline 行为不变。

新增文件：
- modeling/temporal/triplet_memory.py

新增类：
- TripletMemorySlot
- TripletMemoryEncoder
- TemporalDeltaEmbedding
- TripletMemoryBank
- TemporalTripletInjector
- TripletMemoryManager

修改文件：
- configs/defaults.py：添加 MODEL.TEMPORAL.TRIPLET_MEMORY_* 配置
- data/datasets/action_genome.py：确保 sample 返回 video_id / frame_idx / is_keyframe
- modeling/meta_arch/detr.py 或当前 SpeaQ 主 forward 文件：接入 memory get / inject / update
- 必要时修改 transformer 输出，使 forward 能拿到 hs_sub / hs_obj / hs_rel 的最后一层 query feature

Stage 1 默认配置：
- TRIPLET_MEMORY_ENABLED=True
- INJECT_SUBJECT=False
- INJECT_OBJECT=True
- INJECT_RELATION=True
- TRIPLET_MEMORY_DIM=128
- TRIPLET_MEMORY_SIZE=32
- GATE_MAX_OBJECT=0.15
- GATE_MAX_RELATION=0.30
- GATE_WARMUP_ITERS=5000
- UPDATE_SCORE_THRESH=0.05
- USE_DELTA_T_EMB=True

Stage 2 fallback 配置：
- INJECT_OBJECT=False
- INJECT_RELATION=True
- GATE_MAX_RELATION=0.25

验收标准：
1. TRIPLET_MEMORY_ENABLED=False 时 baseline 完全正常。
2. TRIPLET_MEMORY_ENABLED=True 时 forward / loss / eval 不报错。
3. memory slot 数量随视频帧推进正常增长和过期。
4. video_id 切换时不会污染 memory。
5. object/relation injection 可通过配置独立开启和关闭。
```

---

# 12. 最终建议

优先实现 **Stage 1：完整 triplet memory + obj/relation 注入**。
这符合你的研究动机，也不会强行拆成过多实验。代码层面通过 `INJECT_OBJECT`、`INJECT_RELATION` 控制风险；如果 Stage 1 不稳定，Stage 2 只需要改配置，不需要重写模块。





## 1. 记忆训练稳定性：必须考虑，而且建议加入 GT-to-Prediction curriculum

你的担心是对的。**如果训练一开始就用 prediction 更新 memory，memory bank 里很可能全是错误 triplet**，后面 cross-attention 反而会把错误历史信息注入当前 query，形成 confirmation bias。

所以建议把 memory update 分成三个阶段：

```text id="qafn1v"
早期：GT-aligned memory
中期：GT + prediction 混合 memory
后期：prediction memory
```

这里不是用当前帧 GT 泄漏当前帧答案，而是：

```text id="hoh06w"
处理第 t 帧时：
    读取的是 t-1 / t-2 历史帧 memory

处理完第 t 帧后：
    再用第 t 帧的 GT-aligned triplet 更新 memory
    给后续帧使用
```

所以不会出现“当前帧直接看当前帧 GT”的 label leakage。

---

# 1.1 推荐 memory update curriculum

## 阶段 A：GT-aligned memory warmup

训练前期只用 GT 对齐后的 triplet 更新 memory。

```text id="hn68ze"
iteration < 30% total_iters:
    update_source = "gt_aligned"
    prediction_update = False
```

所谓 `gt_aligned` 不是直接把 GT 文本标签塞进去，而是：

```text id="09ytxg"
使用 Hungarian matcher 得到 matched prediction query
使用 GT 的 triplet label / GT box / GT predicate one-hot 作为 memory identity 和几何信息
```

也就是说：

```text id="yq1zdi"
memory feature = matched prediction query feature
memory label  = GT triplet label
memory box    = GT sub/obj box
```

这样有两个好处：

1. memory feature 仍然来自模型 query 空间，不会和 inference 分布完全脱节；
2. memory identity 是干净的，不会一开始全是错误 predicate。

SpeaQ 本身就强调 GT 到 prediction 的 assignment 对训练信号质量非常关键，尤其是其 Quality-Aware Multi-Assignment 会基于 subject、object、predicate 的整体质量分配更多正样本，而不是盲目使用所有 prediction。 你的 memory update 也应该遵循类似思想：**早期不要相信低质量 prediction**。

---

## 阶段 B：GT + prediction 混合更新

中期逐渐引入预测 memory。

```text id="zg8jnt"
30% total_iters <= iteration < 70% total_iters:
    update_source = "mixed"
```

混合比例：

```text id="rs5yhe"
p_pred = linear_schedule(0 → 1)
p_gt   = 1 - p_pred
```

实现上可以很简单：

```python id="y3ydvk"
if random.random() < p_pred:
    candidates = build_pred_memory_candidates(...)
else:
    candidates = build_gt_aligned_memory_candidates(...)
```

或者更稳定一点：

```python id="qei10c"
gt_candidates = build_gt_aligned_memory_candidates(...)
pred_candidates = build_pred_memory_candidates(...)

candidates = gt_candidates + filter_high_conf_pred_candidates(pred_candidates)
```

第二种更推荐，因为 GT memory 仍然兜底，prediction 只作为补充。

---

## 阶段 C：prediction memory fine-tuning

训练后期切换到真实推理分布。

```text id="f43ph3"
iteration >= 70% total_iters:
    update_source = "prediction"
```

但仍然要加高置信度过滤：

```text id="mlekwe"
quality = sub_score * obj_score * pred_score
quality > threshold
```

推荐阈值：

```text id="7d2tmn"
UPDATE_SCORE_THRESH = 0.05  # 起步
如果 memory 噪声大，升到 0.1 或 0.15
```

你的已有方案文档里已经提到当前 `ObjectMemoryBank` 是按 score 筛选候选、再用 IoU 匹配、EMA 更新，并通过 miss 计数过期，这个机制可以直接迁移到 triplet memory。

---

# 1.2 Cursor 需要实现的配置项

建议新增：

```python id="vjuckh"
_C.MODEL.TEMPORAL.MEMORY_UPDATE_SCHEDULE = "gt_to_pred"
_C.MODEL.TEMPORAL.GT_UPDATE_END_RATIO = 0.3
_C.MODEL.TEMPORAL.MIXED_UPDATE_END_RATIO = 0.7
_C.MODEL.TEMPORAL.PRED_UPDATE_THRESH_START = 0.15
_C.MODEL.TEMPORAL.PRED_UPDATE_THRESH_END = 0.05
_C.MODEL.TEMPORAL.USE_GT_ALIGNED_MEMORY = True
```

---

# 1.3 伪代码

```python id="he9qxb"
def get_memory_update_mode(iteration, max_iter, cfg):
    r = iteration / max_iter

    if r < cfg.MODEL.TEMPORAL.GT_UPDATE_END_RATIO:
        return "gt_aligned"

    if r < cfg.MODEL.TEMPORAL.MIXED_UPDATE_END_RATIO:
        return "mixed"

    return "prediction"
```

```python id="ntx9ba"
def update_triplet_memory_after_forward(
    outputs,
    targets,
    hs_sub,
    hs_obj,
    hs_rel,
    matcher_indices,
    iteration,
    max_iter,
):
    mode = get_memory_update_mode(iteration, max_iter, cfg)

    if mode == "gt_aligned":
        candidates = build_gt_aligned_memory_candidates(
            outputs=outputs,
            targets=targets,
            hs_sub=hs_sub,
            hs_obj=hs_obj,
            hs_rel=hs_rel,
            matcher_indices=matcher_indices,
        )

    elif mode == "mixed":
        gt_candidates = build_gt_aligned_memory_candidates(...)
        pred_candidates = build_pred_memory_candidates(...)

        pred_candidates = filter_by_quality(
            pred_candidates,
            thresh=current_prediction_threshold(iteration, max_iter),
        )

        candidates = gt_candidates + pred_candidates

    else:
        candidates = build_pred_memory_candidates(...)
        candidates = filter_by_quality(
            candidates,
            thresh=current_prediction_threshold(iteration, max_iter),
        )

    memory_bank.update(candidates)
```

---

# 1.4 GT-aligned candidate 怎么构造？

核心思想：

```text id="a847km"
用 matched prediction query 作为 feature
用 GT triplet 作为 signature
用 GT box 作为 geometry
```

伪代码：

```python id="u1loi6"
def build_gt_aligned_memory_candidates(
    outputs,
    targets,
    hs_obj,
    hs_rel,
    matcher_indices,
):
    candidates = []

    for b, target in enumerate(targets):
        # target["relations"]: [N_rel, 3] = sub_gt_idx, obj_gt_idx, pred_label
        relations = target["relations"]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        # matcher_indices 需要能告诉我们：
        # gt object index -> predicted query index
        gt_to_pred_obj = build_gt_to_pred_map(matcher_indices[b])

        for rel in relations:
            gt_sub_idx = int(rel[0])
            gt_obj_idx = int(rel[1])
            gt_pred_label = int(rel[2])

            if gt_sub_idx not in gt_to_pred_obj:
                continue
            if gt_obj_idx not in gt_to_pred_obj:
                continue

            pred_sub_q = gt_to_pred_obj[gt_sub_idx]
            pred_obj_q = gt_to_pred_obj[gt_obj_idx]

            sub_label = int(gt_labels[gt_sub_idx])
            obj_label = int(gt_labels[gt_obj_idx])

            sub_box = gt_boxes[gt_sub_idx]
            obj_box = gt_boxes[gt_obj_idx]

            # relation query 这里有两种做法：
            # 1. 如果能找到 matched relation query，用 matched relation query
            # 2. 如果找不到，用 obj/sub query + pred embedding 构造 memory
            rel_query = find_matched_relation_query_or_fallback(
                hs_rel=hs_rel[-1, b],
                outputs=outputs,
                gt_sub_idx=gt_sub_idx,
                gt_obj_idx=gt_obj_idx,
                gt_pred_label=gt_pred_label,
            )

            obj_query = hs_obj[-1, b, pred_obj_q]

            pred_onehot = one_hot(gt_pred_label, num_rel_classes)

            mem_feat = triplet_memory_encoder(
                rel_query=rel_query[None],
                obj_query=obj_query[None],
                sub_box=sub_box[None],
                obj_box=obj_box[None],
                pred_prob=pred_onehot[None],
            )[0]

            candidates.append({
                "signature": (sub_label, gt_pred_label, obj_label),
                "feat": mem_feat,
                "sub_box": sub_box,
                "obj_box": obj_box,
                "union_box": make_union_box(sub_box, obj_box),
                "score": 1.0,
                "sub_score": 1.0,
                "obj_score": 1.0,
                "pred_score": 1.0,
                "source": "gt_aligned",
            })

    return candidates
```

如果 relation query 不好直接匹配，第一版可以简化：

```python id="w9vh2c"
rel_query = relation_label_embedding(gt_pred_label)
```

但我更推荐优先找 matched relation query，因为这样 memory 仍处在模型 decoder query 空间。

---

# 1.5 Gate 也要 warmup

即使用 GT-aligned memory，训练初期也不要让 memory 立刻强干预。

推荐：

```text id="r8zq5r"
前 10% iter:
    gate = 0，仅更新 memory，不注入

10% - 30% iter:
    gate 从 0 线性 warmup 到 gate_max

30% 之后:
    正常注入
```

伪代码：

```python id="b2qu57"
def get_gate(iteration, max_iter, gate_max):
    r = iteration / max_iter

    if r < 0.1:
        return 0.0

    if r < 0.3:
        return gate_max * (r - 0.1) / 0.2

    return gate_max
```

这样训练稳定性会明显高于“一开始 prediction memory + gate 开启”。

---

# 2. 相对几何编码是什么意思？

你可以把它理解成：

```text id="qvhqao"
不要告诉模型：subject 和 object 在图像的绝对哪个位置；
而是告诉模型：object 相对于 subject 在哪里、大小关系如何、两者重叠/距离如何。
```

因为 AG 视频里相机在动，绝对位置不稳定。例如：

```text id="f3rwzs"
第 t 帧：person 在图像左侧，cup 在 person 右侧
第 t+1 帧：相机移动后 person 和 cup 都到了图像中间
```

如果只记绝对坐标，memory 会觉得位置变化很大；
但如果记相对几何，仍然可以知道：

```text id="tl5hga"
cup 仍然在 person 右侧附近
cup 相对 person 的大小差不多
person-cup 这个 pair 仍然稳定
```

这就是相对几何的意义。

---

# 2.1 举例说明

假设 subject 是 person，object 是 cup。

person box：

```text id="w9cl2s"
sub_box = [0.20, 0.30, 0.60, 0.90]
```

cup box：

```text id="brp47u"
obj_box = [0.55, 0.45, 0.70, 0.65]
```

绝对 box 只能告诉模型：

```text id="617gyg"
person 在图像左中区域
cup 在图像右中区域
```

相对几何告诉模型：

```text id="rw35e7"
cup 的中心在 person 中心的右上方
cup 比 person 小很多
cup 和 person 的 union box 覆盖范围是多少
cup 和 person 距离多远
```

这些信息更接近 predicate 判断，例如：

```text id="2glqbf"
cup near person
person holding cup
cup in front of person
cup on table
```

SGG 里 triplet 的评估和训练本身就是围绕 subject/object box 与 predicate label 构造的，当前代码中的 `_triplet` 也会输出 `(sub_label, pred_label, obj_label)` 以及拼接的 subject/object boxes。 所以把 subject-object 几何关系显式编码进 memory，是和现有 triplet 定义一致的。

---

# 2.2 相对几何编码具体包含什么？

先把 box 从 `xyxy` 转成 `cxcywh`。

```text id="kky6va"
sub_box = [x1_s, y1_s, x2_s, y2_s]
obj_box = [x1_o, y1_o, x2_o, y2_o]
```

转成：

```text id="qupk0n"
sub = [cx_s, cy_s, w_s, h_s]
obj = [cx_o, cy_o, w_o, h_o]
```

然后计算：

```text id="12bgsp"
dx = cx_o - cx_s
dy = cy_o - cy_s
dw = log(w_o / w_s)
dh = log(h_o / h_s)
area_ratio = log(area_o / area_s)
center_dist = sqrt(dx^2 + dy^2)
```

这些就叫相对几何。

---

# 2.3 为什么不用绝对位置？

不是完全不用。建议输入里保留：

```text id="ayrqj8"
sub_box
obj_box
union_box
```

但不要只依赖它们。最终 memory encoder 输入应是：

```text id="md04b9"
absolute geometry:
    sub_box
    obj_box
    union_box

relative geometry:
    dx, dy, dw, dh, area_ratio, union_ratio, center_dist
```

也就是说：

```text id="o6owxq"
绝对 box：告诉模型它们大概在哪里
相对 geometry：告诉模型二者之间是什么空间结构
```

对于移动相机视频，相对几何通常比绝对坐标更稳定。

---

# 2.4 最简版本代码

```python id="q4gsp8"
def box_xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box.unbind(-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)


def relative_geometry(sub_box, obj_box):
    s = box_xyxy_to_cxcywh(sub_box)
    o = box_xyxy_to_cxcywh(obj_box)

    dx = o[..., 0] - s[..., 0]
    dy = o[..., 1] - s[..., 1]
    dw = torch.log(o[..., 2] / s[..., 2])
    dh = torch.log(o[..., 3] / s[..., 3])

    s_area = s[..., 2] * s[..., 3]
    o_area = o[..., 2] * o[..., 3]

    area_ratio = torch.log(o_area / s_area.clamp(min=1e-6))
    center_dist = torch.sqrt(dx * dx + dy * dy)

    return torch.stack([
        dx,
        dy,
        dw,
        dh,
        area_ratio,
        center_dist,
    ], dim=-1)
```

---

# 2.5 更完整版本

加入 union box 信息：

```python id="mdwf9y"
def make_union_box(sub_box, obj_box):
    x1 = torch.minimum(sub_box[..., 0], obj_box[..., 0])
    y1 = torch.minimum(sub_box[..., 1], obj_box[..., 1])
    x2 = torch.maximum(sub_box[..., 2], obj_box[..., 2])
    y2 = torch.maximum(sub_box[..., 3], obj_box[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


def relative_geometry_full(sub_box, obj_box):
    s = box_xyxy_to_cxcywh(sub_box)
    o = box_xyxy_to_cxcywh(obj_box)

    union_box = make_union_box(sub_box, obj_box)
    u = box_xyxy_to_cxcywh(union_box)

    dx = o[..., 0] - s[..., 0]
    dy = o[..., 1] - s[..., 1]
    dw = torch.log(o[..., 2] / s[..., 2])
    dh = torch.log(o[..., 3] / s[..., 3])

    s_area = s[..., 2] * s[..., 3]
    o_area = o[..., 2] * o[..., 3]
    u_area = u[..., 2] * u[..., 3]

    area_ratio = torch.log(o_area / s_area.clamp(min=1e-6))
    sub_union_ratio = s_area / u_area.clamp(min=1e-6)
    obj_union_ratio = o_area / u_area.clamp(min=1e-6)
    center_dist = torch.sqrt(dx * dx + dy * dy)

    return torch.stack([
        dx,
        dy,
        dw,
        dh,
        area_ratio,
        sub_union_ratio,
        obj_union_ratio,
        center_dist,
    ], dim=-1)
```

---

# 3. 更新后的最终方案

## 3.1 Memory update 策略

原来：

```text id="w5kg27"
prediction memory from the beginning
```

改成：

```text id="rj8euh"
GT-aligned → mixed → prediction
```

推荐配置：

```yaml id="rkl8lc"
MODEL.TEMPORAL.MEMORY_UPDATE_SCHEDULE: "gt_to_pred"
MODEL.TEMPORAL.GT_UPDATE_END_RATIO: 0.30
MODEL.TEMPORAL.MIXED_UPDATE_END_RATIO: 0.70
MODEL.TEMPORAL.GATE_ZERO_END_RATIO: 0.10
MODEL.TEMPORAL.GATE_WARMUP_END_RATIO: 0.30
MODEL.TEMPORAL.PRED_UPDATE_THRESH_START: 0.15
MODEL.TEMPORAL.PRED_UPDATE_THRESH_END: 0.05
```

---

## 3.2 Memory encoder 输入

原来：

```text id="muy03u"
rel_query + sub_box + obj_box + pred_prob
```

改成：

```text id="cifseg"
rel_query
obj_query
sub_box
obj_box
union_box
relative_geometry
predicate_distribution
```

其中：

```text id="q2uv52"
relative_geometry = [
    dx,
    dy,
    dw,
    dh,
    area_ratio,
    sub_union_ratio,
    obj_union_ratio,
    center_dist
]
```

---

## 3.3 Cursor 追加任务

你可以直接把下面这段给 Cursor：

```text id="nksixd"
请在 Temporal Triplet Memory 中加入训练稳定性 curriculum：

1. 新增配置：
   - MEMORY_UPDATE_SCHEDULE="gt_to_pred"
   - GT_UPDATE_END_RATIO=0.30
   - MIXED_UPDATE_END_RATIO=0.70
   - GATE_ZERO_END_RATIO=0.10
   - GATE_WARMUP_END_RATIO=0.30
   - PRED_UPDATE_THRESH_START=0.15
   - PRED_UPDATE_THRESH_END=0.05

2. memory update 分三阶段：
   - 前 30% iteration：只使用 GT-aligned memory update
   - 30%-70% iteration：GT-aligned + high-confidence prediction 混合更新
   - 70% 之后：只使用 high-confidence prediction 更新

3. GT-aligned memory 的构造方式：
   - 使用 matcher 得到 matched prediction query
   - 使用 GT triplet label 作为 signature
   - 使用 GT subject/object box 作为 geometry
   - 使用 GT predicate one-hot 作为 pred_prob
   - 所有写入 memory 的 tensor 都 detach

4. prediction memory 仍然需要 quality filtering：
   quality = sub_score * obj_score * pred_score
   quality > dynamic_threshold

5. gate 也需要 warmup：
   - 前 10% iteration gate=0，只更新 memory，不注入
   - 10%-30% iteration gate 从 0 线性增大到 gate_max
   - 30% 之后正常注入

6. 将 TripletMemoryEncoder 的 geometry 输入从 sub_box+obj_box 扩展为：
   - sub_box
   - obj_box
   - union_box
   - relative_geometry_full(sub_box, obj_box)

7. relative_geometry_full 包含：
   dx, dy, dw, dh, area_ratio, sub_union_ratio, obj_union_ratio, center_dist
```

---

## 4. 最终判断

你的两个想法都应该采纳：

```text id="izmmko"
1. 训练早期使用 GT-aligned triplet memory，更稳；
2. memory encoder 中加入相对几何编码，降低移动相机和抽帧稀疏带来的绝对坐标不稳定问题。
```

最终方案应改成：

```text id="xzu0u3"
Temporal Triplet Memory
+ GT-to-Prediction memory curriculum
+ gated injection warmup
+ relative subject-object geometry encoding
+ prediction-only inference
```

这样训练阶段更容易收敛，推理阶段又能回到真实 prediction memory 分布，不会依赖 AG 的 GT。
