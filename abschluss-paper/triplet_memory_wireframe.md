# Temporal Triplet Memory v3 — 架构精讲

## 1. 线框总览

```
batched_inputs (list of dict)
 [{video_id:"001YG.mp4", frame_idx:123, image:Tensor, ...}]
  │
  ▼
meta_arch/detr.py :: IterativeRelationDetr.forward()
  video_ids = batch["video_id"], frame_idxs = batch["frame_idx"]
  images.video_ids = video_ids   ◄── 附加到 NestedTensor
  images.frame_idxs = frame_idxs
  │
  ▼
transformer/detr.py :: DETR.forward(samples, targets)
  │
  ├─ SAM3 Backbone (frozen, avg_pool2d)
  ├─ Encoder + Decoder
  │    output dict:
  │      hs_subject_last, hs_object_last, hs_relation_last  [B,N,D]
  │      relation_logits, relation_subject_logits, ...
  │      relation_boxes, ...
  │
  ├─╔══════ TRIPLET MEMORY INJECTION ══════╗
  │ ║                                       ║
  │ ║  samples.video_ids ─┐                ║
  │ ║  samples.frame_idxs ┤                ║
  │ ║                      ▼                ║
  │ ║  TripletMemoryManager                 ║
  │ ║   .get_batch_memory(vids, fidxs)      ║
  │ ║     │ per-video Bank                  ║
  │ ║     │  valid slots → delta_t_emb      ║
  │ ║     │  → score × feat                 ║
  │ ║     ▼  memory: [B, M_max, 128]        ║
  │ ║                                       ║
  │ ║  gate_warmup(iter) → gate_obj, gate_rel║
  │ ║                                       ║
  │ ║  TemporalTripletInjector(obj)         ║
  │ ║   Q=hs_object_last, K/V=memory        ║
  │ ║   → hs_obj += gate_obj × MHA(Q,K,V)   ║
  │ ║                                       ║
  │ ║  TemporalTripletInjector(rel)         ║
  │ ║   Q=hs_relation_last, K/V=memory      ║
  │ ║   → hs_rel += gate_rel × MHA(Q,K,V)   ║
  │ ╚═══════════════════════════════════════╝
  │
  ├─ Prediction Heads (使用注入后的 hs_*_last)
  ├─ Loss Computation
  │
  └─╔══════ TRIPLET MEMORY UPDATE (no_grad) ═╗
    ║  for each sample b in batch:            ║
    ║    quality = sub_s × obj_s × pred_s     ║
    ║    topk candidates (quality > 0.05)     ║
    ║                                        ║
    ║    TripletMemoryEncoder                 ║
    ║     in:  rel_q, obj_q, sub/obj/union_box║
    ║          relative_geom[8], pred_dist[26]║
    ║     out: mem_feat[128]                  ║
    ║                                        ║
    ║    TripletMemoryBank.update(cands, fidx)║
    ║     find_match(signature + IoU)         ║
    ║       → ema_update OR insert_or_replace║
    ║     expire_unmatched(miss > 2)          ║
    ╚══════════════════════════════════════════╝
```

---

## 2. frame_idx 的来源与传播链

**起源：磁盘文件名**

```
dataset/frames/001YG.mp4/000123.png
                   │         │
                   ▼         ▼
               video_id    frame_name
              "001YG.mp4"  "000123.png"
```

**解析：action_genome.py**
```python
video, frame_name = frame_key.split("/", 1)
frame_idx = int(''.join(filter(str.isdigit, frame_name)))  # "000123" → 123
record["video_id"] = video
record["frame_idx"] = frame_idx
```

**完整传播链：**

```
磁盘文件
 → action_genome.py: record["video_id"], record["frame_idx"]
   → DatasetCatalog: dict 持久化
     → DataLoader: __getitem__
       → batched_inputs[i]["video_id"], batched_inputs[i]["frame_idx"]
         → meta_arch/detr.py: 提取并附加到 images (NestedTensor)
           → images.video_ids, images.frame_idxs
             → transformer/detr.py: getattr(samples, "video_ids")
               → TripletMemoryManager.get_batch_memory(vids, fidxs)
                 → Per-video Bank:
                   1. maybe_clear_on_video_jump: fidx倒退 → 清空bank
                   2. get_or_create_bank: 查找/创建该video的bank
                   3. Bank.get_memory(current_frame_idx=fidx):
                      → delta = fidx - slot.frame_idx
                      → TemporalDeltaEmbedding(delta)
                      → bucket(delta) → emb[bucket] → feats + emb
                      → feats × slot.score  (score-weighted)
```

**frame_idx 的三重作用：**

| 作用 | 机制 | 代码位置 |
|------|------|---------|
| 视频顺序检测 | `fidx < last_frame_idx[vid]` → 清空 bank | `maybe_clear_on_video_jump` |
| 时序距离编码 | `delta = current_fidx - slot.frame_idx` → bucket embedding | `TemporalDeltaEmbedding` |
| Slot 过期标记 | 每个 slot 存 `frame_idx`，miss 计数基于帧差 | `TripletMemorySlot.frame_idx` |

---

## 3. signature 的构建与匹配原理

### 3.1 signature 构建

```python
signature = (sub_label: int, pred_label: int, obj_label: int)
```

**构建时机：** memory update 阶段，从当前帧预测结果中提取

```python
sub_label  = argmax(softmax(rel_sub_logits[b, r])[:-1])  # 去掉 bg 类
obj_label  = argmax(softmax(rel_obj_logits[b, r])[:-1])
pred_label = argmax(softmax(rel_logits[b, r])[:-1])
signature  = (sub_label, pred_label, obj_label)
```

### 3.2 为什么用 signature 而不是 tracking ID？

- Action Genome 没有显式 instance ID
- 同一帧中两个完全相同 `(person, holding, cup)` triplet 概率极低
- 不需要额外 tracking 预处理
- 短期伪身份足以区分不同交互

### 3.3 匹配逻辑

`TripletMemoryBank.find_match(cand)`

```
1. 遍历所有 valid slot
2. 跳过 signature != cand.signature 的 slot   ← 快速过滤
3. 对匹配的 slot 计算 3-box 平均 IoU:
     sub_iou   = box_iou(slot.sub_box, cand.sub_box)
     obj_iou   = box_iou(slot.obj_box, cand.obj_box)
     union_iou = box_iou(slot.union_box, cand.union_box)
     pair_iou  = (sub_iou + obj_iou + union_iou) / 3.0
4. pair_iou > match_thresh(0.3) 且最高分 → 匹配成功
```

**为什么同时用 signature 和 IoU？**
- **signature** 保证是同一种三元组（person-holding-cup 不匹配 person-eating-sandwich）
- **IoU** 保证空间位置一致（同一组 person 和 cup 应在相近位置）

---

## 4. query_idx (r) 的索引逻辑

### 4.1 背景

Decoder 输出三个独立 query 序列：

```
hs_sub: [B, N_sub, D]    N_sub = 100
hs_obj: [B, N_obj, D]    N_obj = 100
hs_rel: [B, N_rel, D]    N_rel = 100
```

每个 relation query r 预测一个 triplet：

```
rel_logits[b, r, :]          → predicate 分布
rel_sub_logits[b, r, :]      → subject class 分布
rel_obj_logits[b, r, :]      → object class 分布
rel_sub_boxes[b, r, :]       → subject box
rel_obj_boxes[b, r, :]       → object box
```

### 4.2 当前实现的简化

```python
for r in topk_idx:           # r 是 relation query 索引
    rel_query = hs_rel[b, r]       # [256] — 正确: 用 r 索引
    obj_query = hs_obj[b, r]       # [256] — 简化: 假设 obj[r] 对应 rel[r]
    sub_bx = rel_sub_boxes[b, r]   # [4]
    obj_bx = rel_obj_boxes[b, r]   # [4]
```

### 4.3 正确方式 (TODO)

```python
# rel_pair_idx: [(sub_query_idx, obj_query_idx) for each relation query]
sub_q_idx, obj_q_idx = rel_pair_idx[b][r]
obj_query = hs_obj[b, obj_q_idx]
sub_bx = pred_boxes_sub[b, sub_q_idx]
obj_bx = pred_boxes_obj[b, obj_q_idx]
```

当前简化在 overfit 场景下偏差可控，留作后续优化方向。

---

## 5. Memory 生命周期与 EMA 更新

```
新 triplet → insert_or_replace → slot 创建 (miss=0, age=0)
    │
    ▼
下一帧: find_match → 匹配成功 → ema_update
    │                               │
    │                  feat = 0.9 × old + 0.1 × new
    │                  score = max(slot.score, new_score)
    │                  miss = 0, age++
    │
    ├── 持续匹配 ──→ slot 年龄增长，feat 越来越平滑
    │
    └── 未匹配 ──→ expire_unmatched
                     miss++
                     miss > 2 → slot.valid = False (淘汰)
```

**为什么 EMA=0.9？**
- 每帧预测有噪声，EMA 提供时间平滑
- 0.9 高动量 = 老记忆主导，新帧微调
- `score = max(old, new)` 保留历史最高置信度

**为什么 miss > 2 即淘汰？**
- 物体消失后 triplet 不再出现
- 连续 3 帧不匹配 = 该交互已结束
- 避免过期记忆污染 cross-attention

---

## 6. Gate Warmup 时序

```
gate
 │
 │                              ┌──────────── gate = gate_max (全量注入)
 │                             ╱
 │                           ╱
 │                         ╱  (线性增长)
 │   ────────────────────
 │   gate = 0 (不注入)
 │
 └────┼──────────┼─────────┼─────────→ iter/max_iter
     0%         10%       30%      100%
```

| 阶段 | iter 比例 | gate | 含义 |
|------|----------|------|------|
| 早期 | 0-10% | 0 | memory 积累中，不注入，纯单帧预测 |
| 过渡 | 10-30% | 线性 0→max | 逐步引入 temporal 信息 |
| 全量 | 30-100% | max | 全量注入 |

**为什么 obj gate(0.15) < rel gate(0.30)？**
- Object detection 单帧特征已足够 → 弱 temporal 依赖
- Relation prediction 需要多帧上下文 → 强 temporal 注入

---

## 7. MemoryEncoder 五路输入

```
rel_query [256] → rel_query_proj → [128]  ┐
obj_query [256] → obj_query_proj → [128]  │
                                           │
sub_box [4] ─┐                             │
obj_box [4]  ├─ cat(12) → box_proj → [64] │
union_box[4]─┘                             ├─ cat(416) → fusion → [128]
                                           │
rel_geom [8] ──────→ geom_proj ──→ [64]   │
                                           │
pred_dist[26] ────→ pred_proj ──→ [32]   ┘
```

| 输入 | 维度 | 编码内容 |
|------|------|---------|
| rel_query | 256→128 | decoder 对 predicate 的语义理解 |
| obj_query | 256→128 | decoder 对 object 的视觉表征 |
| sub/obj/union box | 12→64 | 两个物体的绝对位置+外接矩形 |
| relative_geometry | 8→64 | 物体间空间关系 |
| pred_dist | 26→32 | 谓词概率分布 |

---

## 8. relative_geometry 的八维含义

```python
dx = o.cx - s.cx              # object 在 subject 的右侧多远
dy = o.cy - s.cy              # object 在 subject 的上/下方多远
dw = log(o.w / s.w)           # 宽度比 (log)
dh = log(o.h / s.h)           # 高度比 (log)
area_ratio = log(o.area/s.area)   # 面积比 (log)
sub_union_ratio = s.area/u.area   # subject 占 union 比例
obj_union_ratio = o.area/u.area   # object 占 union 比例
center_dist = sqrt(dx²+dy²)  # 中心距离
```

**为什么用相对几何？** AG 相机移动时 `person holding cup` 中 cup 相对 person 的位置比绝对坐标更稳定。

---

## 9. Cross-Attention 注入机制

```python
queries [B,N,256]      memory [B,M,128]
    │                      │
    ▼                      ▼
Q = Linear→128         K,V = Linear→128
    │                      │
    └────── Q·K^T ────────┘
              │
              ▼
    attn = softmax(QK^T/√128)    [B, N, M]
    context = attn @ V            [B, N, 128]
    out = Linear(context, 256)    [B, N, 256]
    return queries + gate × out   ← 残差连接
```

- 残差连接: 只补充历史信息，不完全替代当前帧
- score-weighted memory: `memory × slot.score` 抑制低质量 slot

---

## 10. 完整数据流时序

以 `001YG.mp4` 连续 3 帧为例：

```
Frame 123 (第一帧)              Frame 124                       Frame 125
──────────────────────────────────────────────────────────────────────
1. forward(img_123)            1. forward(img_124)             1. forward(img_125)

2. get_batch_memory(vid,123)   2. get_batch_memory(vid,124)    2. get_batch_memory(vid,125)
   → bank empty                   → bank has 3 slots              → bank has 5 slots
   → (None,None)                  → [3,128], [3]                  → [5,128], [5]
                                    delta = 1                      delta: 1,2,1,2,1

3. injection: SKIP              3. injection:                   3. injection:
   (memory is None)                gate_obj=0.03  gate_rel=0.05    gate_obj=0.06  gate_rel=0.11
                                   hs_obj += 0.03×Xattn          hs_obj += 0.06×Xattn
                                   hs_rel += 0.05×Xattn          hs_rel += 0.11×Xattn

4. predict (no temporal)        4. predict (weak temporal)      4. predict (growing temporal)

5. update:                      5. update:                      5. update:
   → insert slot0:                 → match slot0: ema_update       → match slot0: ema_update
     (person,hold,cup)             → insert slot2:                  → match slot2: ema_update
   → insert slot1:                   (person,behind,door)           → slot1.miss++ (未匹配)
     (person,near,table)          → 3 slots total                  → slot1 可能被 expire
   → 2 slots total
```

**关键观察：**
- 第 1 帧: 无历史，gate=0，纯单帧预测
- 第 2 帧: 弱 temporal 信号出现
- 第 3 帧: memory 积累，gate 增大
- `(person, near, table)` 消失 → miss++ → 可能过期
- `(person, hold, cup)` 持续匹配 → EMA 平滑

---

## 11. 与 temporal_v2 并存

| | temporal_v2 | temporal_v3 |
|---|------------|------------|
| 注入时机 | decoder 之前 (query embed 级) | decoder 之后 (feature 级) |
| 注入位置 | subject/object query embed | hs_obj[-1], hs_rel[-1] |
| 记忆对象 | 单个 object query + box | 完整 triplet (sub+pred+obj) |
| 匹配方式 | IoU + class (可选) | signature + avg_IoU |
| 作用域 | 影响所有 decoder 层 | 只影响最后一层 |
| Gate | sigmoid 可学习参数 | warmup schedule |

两者互补：v2(encoder 先验) → decoder → v3(decoder 增强) → prediction
