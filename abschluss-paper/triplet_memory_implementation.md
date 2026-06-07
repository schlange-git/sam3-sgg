# Temporal Triplet Memory (temporal_v3) 落地方案与实现文档

> 分支: `temporal_v3`  
> 基 commit: `7f76e38` (temporal_memory_v2 final)  
> Stage 1 commit: `1552882`  
> Stage 2 commit: `211b700`

---

## 1. 设计目标

在 SAM3 + SpeaQ + Action Genome 框架中，增加 **Temporal Triplet Memory** 模块，在视频连续帧之间保留 `(subject, predicate, object)` 三元组级别的历史信息。

与已有 temporal_v2 (object_query_memory_v1) 的关系：

```
temporal_v2:
  ObjectMemoryBank  ──→  query_injector  ──→  subject/object query embedding 注入
  RelationMemoryBank ──→  relation_query_injector ──→  relation query embedding 注入
  （query embedding 级别，在 decoder 之前）

temporal_v3（本实现）:
  TripletMemoryBank ──→  TripletInjector ──→  decoder 输出层 hs_obj[-1] / hs_rel[-1] 注入
  （decoder feature 级别，在 decoder 之后、prediction head 之前）
```

两者的关系是**互补**而非替代：temporal_v2 在 encoder 侧影响 query initialization，temporal_v3 在 decoder 侧影响最终输出。

---

## 2. 总体架构

### 2.1 数据流

```
batched_inputs (含 video_id, frame_idx)
    │
    ├─→ meta_arch/detr.py: IterativeRelationDetr.forward()
    │       │
    │       ├─ 提取 video_ids / frame_idxs → 附加到 images (NestedTensor)
    │       │
    │       └─→ transformer/detr.py: DETR.forward()
    │               │
    │               ├─ Backbone (SAM3 frozen)
    │               ├─ temporal_v2 injection (query embedding level)
    │               ├─ Transformer Encoder + Decoder
    │               │       │
    │               │       └─ output dict: hs_subject_last, hs_object_last,
    │               │                       hs_relation_last, 各 prediction logits
    │               │
    │               ├─ ★ Triplet Memory Injection ★
    │               │   读取 memory → gate warmup → cross-attn 注入
    │               │   hs_object_last  ← TripletInjector_obj(mem, gate_obj)
    │               │   hs_relation_last ← TripletInjector_rel(mem, gate_rel)
    │               │
    │               ├─ Prediction heads (使用注入后的 query)
    │               ├─ Loss computation
    │               │
    │               └─ ★ Triplet Memory Update ★
    │                   构造 triplet candidates → match → EMA update → expire
    │
    └─→ losses / outputs
```

### 2.2 模块关系

```
TripletMemoryManager (per-model singleton)
    │
    ├─ dict[video_id → TripletMemoryBank]
    │       │
    │       ├─ List[TripletMemorySlot] (max 32)
    │       │     ├─ signature: (sub_label, pred_label, obj_label)
    │       │     ├─ feat: [mem_dim=128]
    │       │     ├─ sub_box, obj_box, union_box
    │       │     ├─ score, sub_score, obj_score, pred_score
    │       │     ├─ frame_idx, miss, age
    │       │
    │       ├─ find_match(cand) → slot_id or None
    │       ├─ ema_update(slot_id, cand)
    │       ├─ insert_or_replace(cand) → slot_id
    │       └─ expire_unmatched(matched_ids)
    │
    ├─ TemporalDeltaEmbedding (可学习 bucket embedding)
    │      bucket: 0, 1, 2, 3, 4-7, 8-15, 16+
    │
    ├─ get_batch_memory(video_ids, frame_idxs) → [B, M_max, D], [B, M_max]
    └─ update_batch(video_ids, frame_idxs, batch_candidates)


TripletMemoryEncoder
    │
    ├─ rel_query_proj:  Linear(d_model → mem_dim)
    ├─ obj_query_proj:  Linear(d_model → mem_dim)
    ├─ box_proj:        Linear(12 → 64)  # sub + obj + union box
    ├─ geom_proj:       Linear(8 → 64)   # relative_geometry
    ├─ pred_proj:       Linear(C_rel → 32)
    └─ fusion:          Linear(416 → mem_dim → mem_dim)

    输入: rel_query, obj_query, sub_box, obj_box, pred_prob
    输出: [N, mem_dim] 压缩记忆特征


TemporalTripletInjector (per branch: obj / rel)
    │
    ├─ query_proj:  Linear(d_model → mem_dim)
    ├─ mem_proj:    Linear(mem_dim → mem_dim)
    ├─ attn:        MultiheadAttention(mem_dim, nhead, batch_first)
    ├─ out_proj:    Linear(mem_dim → d_model)
    │
    forward(queries, memory, memory_mask, gate):
        if memory is None or gate <= 0: return queries
        q = query_proj(queries)
        kv = mem_proj(memory)
        out = attn(q, kv, kv, key_padding_mask=memory_mask)
        return queries + gate * out_proj(out)


Gate / Curriculum Helpers
    │
    ├─ get_temporal_gate(iter, max_iter, gate_max, zero_ratio, warmup_ratio) → float
    │     0%──10%: gate=0
    │     10%──30%: gate = gate_max * (r-0.1)/0.2
    │     30%+:     gate = gate_max
    │
    ├─ get_memory_update_mode(iter, max_iter, gt_ratio, mixed_ratio) → str
    │     0%──30%: "gt_aligned"     (未实现, 已预留)
    │     30%──70%: "mixed"         (未实现, 已预留)
    │     70%+:     "prediction"
    │
    └─ get_prediction_threshold(iter, max_iter, start, end) → float
          线性衰减: start → end
```

---

## 3. 文件改动详表

| 文件 | 改动类型 | 内容 |
|------|---------|------|
| `configs/defaults.py` | 新增 | 22 个 `TRIPLET_MEMORY_*` 配置项 |
| `modeling/temporal/triplet_memory.py` | **新建** | 全部核心模块 (497 行) |
| `data/datasets/action_genome.py` | 修改 | `frame_idx`, `is_keyframe` 字段 |
| `modeling/meta_arch/detr.py` | 修改 | 传递 video_ids/frame_idxs |
| `modeling/transformer/detr.py` | 修改 | init 构建、forward 注入、forward 更新 |
| `tools/overfit_triplet_v3_16000.sh` | **新建** | 单卡过拟合训练脚本 |

---

## 4. 配置项说明

### 4.1 核心开关

```python
# 总开关
_C.MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED = False

# 注入目标
_C.MODEL.TEMPORAL.INJECT_SUBJECT = False   # subject 默认关闭
_C.MODEL.TEMPORAL.INJECT_OBJECT = True     # Stage 1 开启
_C.MODEL.TEMPORAL.INJECT_RELATION = True   # Stage 1 开启
```

### 4.2 Memory 结构

```python
_C.MODEL.TEMPORAL.TRIPLET_MEMORY_DIM = 128        # 记忆特征维度
_C.MODEL.TEMPORAL.TRIPLET_MEMORY_SIZE = 32        # 每个 video 最多 slot 数
_C.MODEL.TEMPORAL.TRIPLET_MEMORY_TOPK_UPDATE = 16 # 每帧最多更新 k 个候选
_C.MODEL.TEMPORAL.TRIPLET_MEMORY_MAX_MISS = 2     # 允许最大 miss 次数
```

### 4.3 Gate

```python
_C.MODEL.TEMPORAL.GATE_MAX_OBJECT = 0.15      # obj 分支最大注入强度
_C.MODEL.TEMPORAL.GATE_MAX_RELATION = 0.30    # rel 分支最大注入强度
_C.MODEL.TEMPORAL.GATE_ZERO_END_RATIO = 0.10  # gate=0 持续到 10% iters
_C.MODEL.TEMPORAL.GATE_WARMUP_END_RATIO = 0.30 # gate 线性增长到 30% iters
```

### 4.4 Memory Update Curriculum（已预留，当前仅 prediction 模式生效）

```python
_C.MODEL.TEMPORAL.MEMORY_UPDATE_SCHEDULE = "gt_to_pred"  # gt_aligned / mixed / prediction
_C.MODEL.TEMPORAL.GT_UPDATE_END_RATIO = 0.30
_C.MODEL.TEMPORAL.MIXED_UPDATE_END_RATIO = 0.70
_C.MODEL.TEMPORAL.USE_GT_ALIGNED_MEMORY = True
_C.MODEL.TEMPORAL.PRED_UPDATE_THRESH_START = 0.15
_C.MODEL.TEMPORAL.PRED_UPDATE_THRESH_END = 0.05

_C.MODEL.TEMPORAL.UPDATE_SCORE_THRESH = 0.05     # prediction 更新质量阈值
_C.MODEL.TEMPORAL.UPDATE_EMA_MOMENTUM = 0.9      # EMA 动量
_C.MODEL.TEMPORAL.MATCH_IOU_THRESH = 0.3         # slot 匹配 IoU 阈值
```

### 4.5 时序编码 & 调试

```python
_C.MODEL.TEMPORAL.USE_DELTA_T_EMB = True         # 帧差 bucket embedding
_C.MODEL.TEMPORAL.MAX_DELTA_T_BUCKET = 7         # bucket 数量
_C.MODEL.TEMPORAL.DEBUG_MEMORY = False           # 每 100 iter 打印 memory 统计
```

### 4.6 Stage 差异

| 配置项 | Stage 1 (Full) | Stage 2 (Relation-only) |
|--------|---------------|------------------------|
| `INJECT_OBJECT` | True | **False** |
| `INJECT_RELATION` | True | True |
| `GATE_MAX_OBJECT` | 0.15 | (ignored) |
| `GATE_MAX_RELATION` | 0.30 | **0.25** |

---

## 5. 核心伪代码

### 5.1 TripletMemorySlot

```
@dataclass TripletMemorySlot:
    valid: bool = True
    signature: tuple              # (sub_label: int, pred_label: int, obj_label: int)
    feat: Tensor[mem_dim]         # 压缩记忆特征 (CPU)
    sub_box: Tensor[4]            # subject box, xyxy, normalized (CPU)
    obj_box: Tensor[4]            # object box, xyxy, normalized (CPU)
    union_box: Tensor[4]          # union box, xyxy, normalized (CPU)
    score: float                  # quality = sub_score × obj_score × pred_score
    sub_score: float              # subject detection score
    obj_score: float              # object detection score
    pred_score: float             # predicate prediction score
    frame_idx: int                # 来源帧序号
    miss: int                     # 连续未匹配次数
    age: int                      # 该 slot 被创建的年龄
```

### 5.2 辅助几何函数

```
function box_xyxy_to_cxcywh(box: Tensor[..., 4]) → Tensor[..., 4]:
    x1, y1, x2, y2 = box.unbind(-1)
    w = (x2 - x1).clamp_min(1e-6)
    h = (y2 - y1).clamp_min(1e-6)
    return stack([x1 + 0.5*w, y1 + 0.5*h, w, h], -1)

function make_union_box(sub_box, obj_box: Tensor[..., 4]) → Tensor[..., 4]:
    return stack([
        min(sub_box[...,0], obj_box[...,0]),  # x1
        min(sub_box[...,1], obj_box[...,1]),  # y1
        max(sub_box[...,2], obj_box[...,2]),  # x2
        max(sub_box[...,3], obj_box[...,3]),  # y2
    ], -1)

function relative_geometry(sub_box, obj_box: Tensor[4]) → Tensor[8]:
    s = box_xyxy_to_cxcywh(sub_box)        # [cx_s, cy_s, w_s, h_s]
    o = box_xyxy_to_cxcywh(obj_box)        # [cx_o, cy_o, w_o, h_o]
    u = box_xyxy_to_cxcywh(make_union_box(sub_box, obj_box))

    dx = o.cx - s.cx
    dy = o.cy - s.cy
    dw = log(o.w / s.w)
    dh = log(o.h / s.h)

    s_area = s.w * s.h
    o_area = o.w * o.h
    u_area = u.w * u.h

    area_ratio = log(o_area / s_area)
    sub_union_ratio = s_area / u_area
    obj_union_ratio = o_area / u_area
    center_dist = sqrt(dx² + dy²)

    return [dx, dy, dw, dh, area_ratio, sub_union_ratio, obj_union_ratio, center_dist]

function box_iou_1v1(box1, box2: Tensor[4]) → float:
    # 两个单 box 的 IoU, 纯 Python float 运算, 用于 slot matching
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    a1 = max(1e-6, (box1[2]-box1[0]) * (box1[3]-box1[1]))
    a2 = max(1e-6, (box2[2]-box2[0]) * (box2[3]-box2[1]))
    return inter / (a1 + a2 - inter + 1e-6)
```

### 5.3 TemporalDeltaEmbedding

```
class TemporalDeltaEmbedding(nn.Module):
    init(self, mem_dim=128, num_buckets=7):
        self.emb = Embedding(num_buckets, mem_dim)

    function bucketize(delta: Tensor[int]) → Tensor[int]:
        # delta = current_frame_idx - slot.frame_idx
        # Bucket 映射: 0→0, 1→1, 2→2, 3→3, [4,7]→4, [8,15]→5, [16,∞)→6
        bucket = zeros_like(delta)
        bucket[delta==1] = 1
        bucket[delta==2] = 2
        bucket[delta==3] = 3
        bucket[(delta>=4) & (delta<=7)] = 4
        bucket[(delta>=8) & (delta<=15)] = 5
        bucket[delta>=16] = 6
        return bucket.clamp(0, num_buckets-1)

    forward(memory_feat, current_frame_idx, memory_frame_idx):
        delta = current_frame_idx - memory_frame_idx
        bucket = bucketize(delta)
        return memory_feat + emb(bucket)
```

### 5.4 TripletMemoryEncoder

```
class TripletMemoryEncoder(nn.Module):
    init(self, d_model=256, num_rel_classes=26, mem_dim=128):
        self.rel_query_proj = Linear(d_model, mem_dim)
        self.obj_query_proj = Linear(d_model, mem_dim)
        self.box_proj = Sequential(
            Linear(12, 64), LayerNorm(64), GELU()
        )
        self.geom_proj = Sequential(
            Linear(8, 64), LayerNorm(64), GELU()
        )
        self.pred_proj = Sequential(
            Linear(num_rel_classes, 32), LayerNorm(32), GELU()
        )
        # 总输入: mem_dim(rel) + mem_dim(obj) + 64(box) + 64(geom) + 32(pred)
        self.fusion = Sequential(
            Linear(mem_dim+mem_dim+64+64+32, mem_dim),
            LayerNorm(mem_dim),
            GELU(),
            Linear(mem_dim, mem_dim),
        )

    forward(rel_query, obj_query, sub_box, obj_box, pred_prob):
        # 所有输入: [N, *_dim] 或 [N, 4] 或 [N, C_rel]
        union_box = make_union_box(sub_box, obj_box)
        geom = relative_geometry(sub_box, obj_box)

        box_feat = box_proj(cat([sub_box, obj_box, union_box], -1))   # [N, 64]
        geom_feat = geom_proj(geom)                                     # [N, 64]
        pred_feat = pred_proj(pred_prob.detach())                       # [N, 32]
        rel_feat = rel_query_proj(rel_query)                            # [N, 128]
        obj_feat = obj_query_proj(obj_query)                            # [N, 128]

        feat = cat([rel_feat, obj_feat, box_feat, geom_feat, pred_feat], -1)
        return fusion(feat)  # [N, 128]
```

### 5.5 TripletMemoryBank

```
class TripletMemoryBank:
    init(self, memory_size=32, mem_dim=128, max_miss=2, ema_momentum=0.9, match_iou_thresh=0.3):
        self.slots: List[TripletMemorySlot] = []   # 最多 memory_size 个

    function clear():
        self.slots = []

    function get_valid_slots() → List[TripletMemorySlot]:
        return [s for s in slots if s.valid and s.feat is not None]

    function get_memory(device, current_frame_idx, delta_t_emb) → (Tensor|None, Tensor|None):
        valid = get_valid_slots()
        if len(valid) == 0:
            return None, None

        feats = stack([s.feat for s in valid])          # [M, mem_dim]
        frame_ids = tensor([s.frame_idx for s in valid])

        if delta_t_emb is not None and current_frame_idx is not None:
            feats = delta_t_emb(feats, current_frame_idx, frame_ids)

        # Score-weighted: 低质量记忆被抑制
        scores = tensor([s.score for s in valid]).clamp(0, 1)
        feats = feats * scores[:, None]

        mask = zeros(M, dtype=bool)   # 无 padding
        return feats.to(device), mask.to(device)

    function find_match(cand: dict) → int | None:
        """
        cand: {"signature", "feat", "sub_box", "obj_box", "union_box",
               "score", "sub_score", "obj_score", "pred_score"}
        返回最佳匹配 slot index, 无匹配返回 None
        """
        best_id, best_score = None, -1.0
        for i, slot in enumerate(slots):
            if not slot.valid: continue
            if slot.signature != cand["signature"]: continue

            sub_iou = box_iou_1v1(slot.sub_box, cand["sub_box"])
            obj_iou = box_iou_1v1(slot.obj_box, cand["obj_box"])
            union_iou = box_iou_1v1(slot.union_box, cand["union_box"])

            pair_iou = (sub_iou + obj_iou + union_iou) / 3.0
            if pair_iou > match_iou_thresh and pair_iou > best_score:
                best_score = pair_iou
                best_id = i
        return best_id

    function ema_update(slot_id, cand, frame_idx):
        slot = slots[slot_id]
        m = ema_momentum
        slot.feat = m × slot.feat + (1-m) × cand["feat"].detach().cpu()
        slot.sub_box = m × slot.sub_box + (1-m) × cand["sub_box"].detach().cpu()
        slot.obj_box = m × slot.obj_box + (1-m) × cand["obj_box"].detach().cpu()
        slot.union_box = m × slot.union_box + (1-m) × cand["union_box"].detach().cpu()
        slot.score = max(slot.score, float(cand["score"]))
        slot.sub_score = float(cand["sub_score"])
        slot.obj_score = float(cand["obj_score"])
        slot.pred_score = float(cand["pred_score"])
        slot.frame_idx = frame_idx
        slot.miss = 0
        slot.age += 1

    function insert_or_replace(cand, frame_idx) → int:
        new_slot = TripletMemorySlot(
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
            frame_idx=frame_idx,
            miss=0, age=0,
        )
        if len(slots) < memory_size:
            slots.append(new_slot)
            return len(slots) - 1
        # FIFO + 质量淘汰: 替换 score - 0.05 × miss 最低的
        weakest = argmin_i(slots[i].score - 0.05 × slots[i].miss)
        slots[weakest] = new_slot
        return weakest

    function update(candidates: List[dict], frame_idx):
        matched_ids = set()
        for cand in candidates:
            match_id = find_match(cand)
            if match_id is not None:
                ema_update(match_id, cand, frame_idx)
                matched_ids.add(match_id)
            else:
                new_id = insert_or_replace(cand, frame_idx)
                matched_ids.add(new_id)
        expire_unmatched(matched_ids)

    function expire_unmatched(matched_ids: set):
        for i, slot in enumerate(slots):
            if not slot.valid: continue
            if i not in matched_ids:
                slot.miss += 1
            if slot.miss > max_miss:
                slot.valid = False
```

### 5.6 TemporalTripletInjector

```
class TemporalTripletInjector(nn.Module):
    init(self, d_model=256, mem_dim=128, nhead=8, dropout=0.1):
        self.query_proj = Linear(d_model, mem_dim)
        self.mem_proj = Linear(mem_dim, mem_dim)
        self.attn = MultiheadAttention(mem_dim, nhead, dropout, batch_first=True)
        self.out_proj = Linear(mem_dim, d_model)

    forward(queries, memory, memory_mask, gate):
        """
        queries:    [B, N_query, D]   hs_obj[-1] 或 hs_rel[-1]
        memory:     [B, M, mem_dim]   来自 TripletMemoryManager.get_batch_memory()
        memory_mask: [B, M]           True=padded
        gate:       float             get_temporal_gate() 返回值
        """
        if memory is None or gate <= 0:
            return queries   # no-op

        B, N, D = queries.shape
        M = memory.shape[1]

        q = query_proj(queries)     # [B, N, mem_dim]
        kv = mem_proj(memory)       # [B, M, mem_dim]

        # 如果 memory_mask 为 None 则全 False (全有效)
        if memory_mask is None:
            memory_mask = zeros(B, M, dtype=bool, device=queries.device)

        out, _ = attn(query=q, key=kv, value=kv,
                       key_padding_mask=memory_mask, need_weights=False)
        out = out_proj(out)         # [B, N, D]
        return queries + gate × out  # 残差连接
```

### 5.7 TripletMemoryManager

```
class TripletMemoryManager:
    init(self, cfg):
        tcfg = cfg.MODEL.TEMPORAL
        mem_dim = tcfg.TRIPLET_MEMORY_DIM
        mem_size = tcfg.TRIPLET_MEMORY_SIZE
        max_miss = tcfg.TRIPLET_MEMORY_MAX_MISS
        ema_m = tcfg.UPDATE_EMA_MOMENTUM
        match_iou = tcfg.MATCH_IOU_THRESH

        self.banks: dict[str → TripletMemoryBank] = {}
        self.last_frame_idx: dict[str → int] = {}
        self.delta_t_emb = TemporalDeltaEmbedding(mem_dim, tcfg.MAX_DELTA_T_BUCKET) \
                           if tcfg.USE_DELTA_T_EMB else None
        self.debug_memory = tcfg.DEBUG_MEMORY

    function reset_video(video_id):
        if video_id in banks:
            banks[video_id].clear()

    function get_or_create_bank(video_id) → TripletMemoryBank:
        if video_id not in banks:
            banks[video_id] = TripletMemoryBank(
                memory_size, mem_dim, max_miss, ema_m, match_iou)
        return banks[video_id]

    function maybe_clear_on_video_jump(video_id, frame_idx):
        # 如果 frame_idx 比上一帧小(跳回), 说明换视频了, 清空该 video 记忆
        if video_id not in last_frame_idx:
            last_frame_idx[video_id] = frame_idx
            return
        last = last_frame_idx[video_id]
        if frame_idx < last:
            reset_video(video_id)
        last_frame_idx[video_id] = frame_idx

    function get_batch_memory(video_ids, frame_idxs, device) → (Tensor|None, Tensor|None):
        mem_list, mask_list = [], []
        for vid, fidx in zip(video_ids, frame_idxs):
            maybe_clear_on_video_jump(vid, fidx)
            bank = get_or_create_bank(vid)
            mem, mask = bank.get_memory(device, current_frame_idx=fidx,
                                        delta_t_emb=delta_t_emb)
            mem_list.append(mem)
            mask_list.append(mask)
        return _pad_memory_batch(mem_list, mask_list, device)

    function update_batch(video_ids, frame_idxs, batch_candidates):
        # batch_candidates: List[List[dict]], B × [N_cand_i × dict]
        for vid, fidx, cands in zip(video_ids, frame_idxs, batch_candidates):
            bank = get_or_create_bank(vid)
            bank.update(cands, fidx)
```

### 5.8 pad_memory_batch

```
function _pad_memory_batch(mem_list, mask_list, device) → (Tensor|None, Tensor|None):
    valid_mems = [m for m in mem_list if m is not None]
    if len(valid_mems) == 0: return None, None

    B = len(mem_list)
    D = valid_mems[0].shape[-1]
    M_max = max(m.shape[0] if m is not None else 0 for m in mem_list)

    mem_batch = zeros(B, M_max, D, device=device)
    mask_batch = ones(B, M_max, dtype=bool, device=device)

    for b, mem in enumerate(mem_list):
        if mem is None: continue
        m = mem.shape[0]
        mem_batch[b, :m] = mem.to(device)
        mask_batch[b, :m] = False   # False = valid (匹配 PyTorch MHA 语义)

    return mem_batch, mask_batch
```

### 5.9 Gate Warmup

```
function get_temporal_gate(iteration, max_iter, gate_max,
                            gate_zero_end_ratio=0.10,
                            gate_warmup_end_ratio=0.30) → float:
    """
    训练稳定性: gate 从 0 开始, 逐步增大到 gate_max
    [0%──10%]: gate = 0          ← 纯预测, 不注入
    [10%──30%]: gate 线性增长     ← 逐步引入 temporal 信息
    [30%+]: gate = gate_max      ← 全量注入
    """
    if max_iter <= 0: return gate_max
    r = iteration / max_iter
    if r < gate_zero_end_ratio: return 0.0
    if r < gate_warmup_end_ratio:
        return gate_max × (r - gate_zero_end_ratio) / (gate_warmup_end_ratio - gate_zero_end_ratio)
    return gate_max
```

### 5.10 Transformer Forward 中的 Triplet Memory 注入

```
# ─── 位置: modeling/transformer/detr.py, DETR.forward() ───

# 在 decoder 输出后, hs_*_last 提取前注入

if self.triplet_memory_enabled and self.triplet_memory_manager is not None \
   and (self.training or self.temporal_eval):
    video_ids = getattr(samples, "video_ids", None)
    frame_idxs = getattr(samples, "frame_idxs", None)

    if video_ids is not None and frame_idxs is not None:
        # ── Step 1: 读取当前 batch 各 sample 的历史 triplet memory ──
        mem, mem_mask = self.triplet_memory_manager.get_batch_memory(
            video_ids=video_ids,
            frame_idxs=frame_idxs,
            device=src.device,
        )

        self.triplet_iter_counter += 1

        # ── Step 2: 计算当前 gate 值 ──
        gate_obj = get_temporal_gate(
            self.triplet_iter_counter, 80000,   # ← 当前硬编码, 应改为从 config 读取
            0.15, 0.10, 0.30)
        gate_rel = get_temporal_gate(
            self.triplet_iter_counter, 80000,
            0.30, 0.10, 0.30)

        # ── Step 3: 注入 object query ──
        if mem is not None and self.triplet_injector_obj is not None \
           and output.get("hs_object_last") is not None:
            obj_q = output["hs_object_last"]          # [B, N_obj, D]
            B_obj = obj_q.shape[0]

            # 确保 memory 维度匹配: [M, D] → [B, M, D]
            mem_exp = mem if mem.dim() == 3 else mem.unsqueeze(0).expand(B_obj, -1, -1)
            mask_exp = mem_mask if (mem_mask is not None and mem_mask.dim() == 2) \
                       else (mem_mask.unsqueeze(0).expand(B_obj, -1) if mem_mask is not None else None)

            output["hs_object_last"] = self.triplet_injector_obj(
                obj_q, mem_exp, memory_mask=mask_exp, gate=gate_obj)

        # ── Step 4: 注入 relation query ──
        if mem is not None and self.triplet_injector_rel is not None \
           and output.get("hs_relation_last") is not None:
            rel_q = output["hs_relation_last"]        # [B, N_rel, D]
            B_rel = rel_q.shape[0]
            mem_exp = mem if mem.dim() == 3 else mem.unsqueeze(0).expand(B_rel, -1, -1)
            mask_exp = mem_mask if (mem_mask is not None and mem_mask.dim() == 2) \
                       else (mem_mask.unsqueeze(0).expand(B_rel, -1) if mem_mask is not None else None)

            output["hs_relation_last"] = self.triplet_injector_rel(
                rel_q, mem_exp, memory_mask=mask_exp, gate=gate_rel)

# ─── 后续: 正常提取 hs_*_last, 组装 out dict ───
out[hs_subject_last] = output.get(hs_subject_last)
out[hs_object_last] = output.get(hs_object_last)
out[hs_relation_last] = output.get(hs_relation_last)
```

### 5.11 Transformer Forward 中的 Triplet Memory 更新

```
# ─── 位置: modeling/transformer/detr.py, DETR.forward(), return out 之前 ───

if self.triplet_memory_enabled and self.triplet_memory_manager is not None \
   and (self.training or self.temporal_eval):
    video_ids = getattr(samples, "video_ids", None)
    frame_idxs = getattr(samples, "frame_idxs", None)

    if video_ids is not None and frame_idxs is not None \
       and out.get("hs_relation_last") is not None:

        with torch.no_grad():          # ★ 关键: 不反向传播到历史帧
            # 提取当前帧预测
            rel_logits = out["relation_logits"]               # [B, N_rel, C_rel+1]
            rel_sub_logits = out["relation_subject_logits"]   # [B, N_sub, C_obj+1]
            rel_obj_logits = out["relation_object_logits"]    # [B, N_obj, C_obj+1]
            rel_sub_boxes = out["relation_subject_boxes"]     # [B, N_sub, 4]
            rel_obj_boxes = out["relation_object_boxes"]      # [B, N_obj, 4]
            hs_rel = out["hs_relation_last"]                  # [B, N_rel, D]
            hs_obj = out["hs_object_last"]                    # [B, N_obj, D]

            batch_candidates = []   # List[List[dict]]

            for b in range(bs):
                # ── 获取 softmax 概率 ──
                sub_prob = softmax(rel_sub_logits[b], dim=-1)   # [N_sub, C_obj+1]
                obj_prob = softmax(rel_obj_logits[b], dim=-1)   # [N_obj, C_obj+1]
                rel_prob = softmax(rel_logits[b], dim=-1)       # [N_rel, C_rel+1]

                # ── 取最大值(去掉 background 最后一维) ──
                sub_score, sub_label = sub_prob[..., :-1].max(-1)
                obj_score, obj_label = obj_prob[..., :-1].max(-1)
                pred_score, pred_label = rel_prob[..., :-1].max(-1)

                # ── 计算 triplet quality ──
                quality = sub_score × obj_score × pred_score   # [N_rel]

                # ── Top-K 候选(按 quality 排序) ──
                topk = min(16, len(pred_score))
                _, topk_idx = quality.topk(min(topk, quality.shape[0]))

                cands = []
                threshold = 0.05

                for r in topk_idx:
                    q = quality[r]
                    if q < threshold: continue   # 低质量过滤

                    # 构造伪身份签名
                    signature = (int(sub_label[r]),
                                 int(pred_label[r]),
                                 int(obj_label[r]))

                    sub_bx = rel_sub_boxes[b, r]   # [4] xyxy normalized
                    obj_bx = rel_obj_boxes[b, r]   # [4] xyxy normalized
                    union_bx = make_union_box(sub_bx, obj_bx)

                    # 取对应的 decoder feature
                    rel_query = hs_rel[b, r]         # [D]
                    obj_query = hs_obj[b, r]         # [D]  (← 简化: 用 r 索引)

                    pred_dist = rel_prob[r, :-1]     # [C_rel]

                    # 编码为压缩记忆
                    mem_feat = self.triplet_encoder(
                        rel_query=rel_query.unsqueeze(0),
                        obj_query=obj_query.unsqueeze(0),
                        sub_box=sub_bx.unsqueeze(0),
                        obj_box=obj_bx.unsqueeze(0),
                        pred_prob=pred_dist.unsqueeze(0),
                    )[0]  # [mem_dim]

                    cands.append({
                        "signature": signature,
                        "feat": mem_feat,
                        "sub_box": sub_bx,
                        "obj_box": obj_bx,
                        "union_box": union_bx,
                        "score": float(q),
                        "sub_score": float(sub_score[r]),
                        "obj_score": float(obj_score[r]),
                        "pred_score": float(pred_score[r]),
                    })

                batch_candidates.append(cands)

            # ── 批量更新 memory ──
            self.triplet_memory_manager.update_batch(
                video_ids=video_ids,
                frame_idxs=frame_idxs,
                batch_candidates=batch_candidates,
            )
```

### 5.12 DETR __init__ 中的模块构建

```
# ─── 位置: modeling/transformer/detr.py, IterativeRelationTransformer.__init__() ───

# 初始化成员变量 (紧随 relation_memory_bank = None 之后)
self.triplet_memory_enabled = getattr(cfg.MODEL.TEMPORAL, "TRIPLET_MEMORY_ENABLED", False)
self.triplet_memory_manager = None
self.triplet_encoder = None
self.triplet_injector_obj = None
self.triplet_injector_rel = None
self.triplet_iter_counter = 0

# 构建模块 (紧随 temporal 其他模块构建之后)
if self.triplet_memory_enabled:
    tcfg = cfg.MODEL.TEMPORAL
    mem_dim = tcfg.TRIPLET_MEMORY_DIM
    num_rel_cls = cfg.MODEL.DETR.NUM_RELATION_CLASSES

    # ── Manager: 管理多个 video 的 memory bank ──
    self.triplet_memory_manager = TripletMemoryManager(cfg)

    # ── Encoder: triplet → compact feature ──
    self.triplet_encoder = TripletMemoryEncoder(
        d_model=transformer.d_model,
        num_rel_classes=num_rel_cls,
        mem_dim=mem_dim,
    )

    # ── Assert: 至少一个注入目标开启 ──
    assert tcfg.INJECT_OBJECT or tcfg.INJECT_RELATION, \
        "TRIPLET_MEMORY_ENABLED=True but INJECT_OBJECT=False and INJECT_RELATION=False!"

    # ── Injector: 只构建开启的分支 ──
    if tcfg.INJECT_OBJECT:
        self.triplet_injector_obj = TemporalTripletInjector(
            d_model=transformer.d_model, mem_dim=mem_dim,
            nhead=cfg.MODEL.DETR.NHEADS, dropout=cfg.MODEL.DETR.DROPOUT)
    if tcfg.INJECT_RELATION:
        self.triplet_injector_rel = TemporalTripletInjector(
            d_model=transformer.d_model, mem_dim=mem_dim,
            nhead=cfg.MODEL.DETR.NHEADS, dropout=cfg.MODEL.DETR.DROPOUT)
```

### 5.13 Meta-Arch 中传递 video 元数据

```
# ─── 位置: modeling/meta_arch/detr.py, IterativeRelationDetr.forward() ───

# images = self.preprocess_image(batched_inputs)
# 在调用 self.detr(images, targets=targets) 之前:

video_ids = [x.get("video_id", "__novid__") for x in batched_inputs]
frame_idxs = [x.get("frame_idx", 0) for x in batched_inputs]
images.video_ids = video_ids
images.frame_idxs = frame_idxs
output = self.detr(images, targets=targets)
```

### 5.14 数据集字段

```
# ─── 位置: data/datasets/action_genome.py, ActionGenomeTrainData ───

# 在构造 record dict 时新增:

record = {
    ...
    "video_id": video,                           # str, 如 "001YG.mp4"
    "frame_id": frame_name,                      # str, 如 "000001.png"
    "frame_idx": int(extract_digits(frame_name)), # int, 从文件名提取, 如 1
    "is_keyframe": True,                          # bool
    ...
}
```

---

## 6. 训练脚本配置

### 6.1 Stage 1: Full Triplet Memory

```bash
MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED True
MODEL.TEMPORAL.INJECT_OBJECT True
MODEL.TEMPORAL.INJECT_RELATION True
MODEL.TEMPORAL.GATE_MAX_OBJECT 0.15
MODEL.TEMPORAL.GATE_MAX_RELATION 0.30
```

其他关键配置:
- SAM3 ENABLED=True, FREEZE=True, USE_PATCH_MERGE=False
- ROI_REFINE ENABLED=False
- TEMPORAL v2 ENABLED=True (与 triplet memory 共存)
- AG_TEMPORAL ENABLED=True, between_keyframes mode
- 预训练权重: model_0099999.pth (从 VG detection 预训练)

### 6.2 Stage 2: Relation-only Fallback

```bash
MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED True
MODEL.TEMPORAL.INJECT_OBJECT False      # ← 仅这里不同
MODEL.TEMPORAL.INJECT_RELATION True
MODEL.TEMPORAL.GATE_MAX_RELATION 0.25   # ← 略低
```

---

## 7. 已知限制与 TODO

### 7.1 当前实现的限制

| 限制 | 说明 | 优先级 |
|------|------|--------|
| gate_max_iter 硬编码 | `get_temporal_gate()` 调用时 max_iter=80000 写死, 未从 SOLVER.MAX_ITER 读取 | 高 |
| config 参数未透传 | topk_update 硬编码 16, threshold 硬编码 0.05, 未读取 TRIPLET_MEMORY_TOPK_UPDATE / UPDATE_SCORE_THRESH | 高 |
| DEBUG_MEMORY 日志不生效 | forward 中 `tcfg` 变量不可用 (只在 __init__ 中存在) | 中 |
| GT-aligned memory 未实现 | MEMORY_UPDATE_SCHEDULE=gt_to_pred 的配置已预留, 但 forward 中仅支持 prediction 模式 | 中 |
| obj_query 索引简化 | update 中直接用 `hs_obj[b, r]` 匹配 relation index r, 未经 rel_pair_idx 映射 | 中 |
| 单卡限制 | 未测试多卡 DDP 下 triplet memory 行为 | 低 |

### 7.2 后续优化方向

1. **GT-aligned memory**: 利用 Hungarian matcher indices 将 GT triplet 映射到 prediction query, 构造高质量 memory candidates
2. **混合更新策略**: gt_aligned (0-30%) → mixed (30-70%) → prediction (70-100%)
3. **obj_query 正确映射**: 通过 rel_pair_idx 而非直接索引, 保证 obj_query 对应到正确的 object query
4. **gate warmup 读取真实 max_iter**: 从 self.triplet_iter_counter 改为读取 trainer.max_iter
5. **多卡支持**: 确保 DDP 下各 rank 的 memory 一致 (需考虑 broadcast)
