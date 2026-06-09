# Temporal Triplet Memory (temporal_v3) 实现文档

> 分支: `temporal_v3`  
> 基 commit: `7f76e38` (temporal_memory_v2 final)  
> 最终 commit: `1714fb3 stage4: critical bugfixes`

```
1714fb3 stage4: critical bugfixes
baf3f5a stage3: obj_query removal + config passthrough
ec78308 stage2: relation-only fallback
1552882 stage1: core implementation
```

---

## 1. 设计目标

在 SAM3 + SpeaQ + Action Genome 框架中，增加 **Temporal Triplet Memory** 模块，
在视频连续帧之间保留 `(subject, predicate, object)` 三元组级别的历史信息。

与 temporal_v2 的关系：
- v2: ObjectMemoryBank → query_injector → query embedding 注入（decoder 之前）
- v3: TripletMemoryBank → TripletInjector → decoder 最后层 feature 注入（decoder 之后）
- 两者互补。当前推荐运行 **v3-only**（MODE=triplet_memory_v3，不启用 v2 的 ObjectMemoryBank/RelationMemoryBank）

---

## 2. 总体架构

### 2.1 数据流（含修复后的注入位置）

```
batched_inputs (含 video_id, frame_idx)
    │
    ├─→ meta_arch/detr.py: 提取 video_ids/frame_idxs → 附加到 images (NestedTensor)
    │
    └─→ transformer/detr.py: IterativeRelationDETR.forward()
            │
            ├─ Backbone (SAM3 frozen, avg_pool2d)
            ├─ Transformer Encoder + Decoder
            │       │
            │       ├─ output dict: hs_subject_last, hs_object_last, hs_relation_last
            │       ├─ relation_logits (L0..L5), relation_boxes
            │       ├─ relation_subject_logits, relation_object_logits
            │       └─ relation_subject_boxes, relation_object_boxes
            │
            ├─ ★ _apply_triplet_memory_to_output(output, samples, device)  ← L484, 在 ROI/out 组装之前
            │       │
            │       ├─ 读取 memory: Manager.get_batch_memory(video_ids, frame_idxs)
            │       ├─ 计算 gate: get_temporal_gate(iter, SOLVER.MAX_ITER, gate_max, ...)
            │       │
            │       ├─ 注入 hs_object_last:
            │       │     obj_q = injector_obj(hs_object_last[-1], memory, gate=gate_obj)
            │       │     → 重算 object_logits (通过 _predict_object_from_embeddings)
            │       │     → clone 原始 tensor → 替换最后一层 logits/boxes/coords
            │       │
            │       └─ 注入 hs_relation_last:
            │             rel_q = injector_rel(hs_relation_last[-1], memory, gate=gate_rel)
            │             → 重算 relation_logits (通过 transformer.relation_embed)
            │             → clone → 替换最后一层 logits/boxes/coords
            │
            ├─ person_score_scale（在注入之后执行）
            ├─ ROI refine（使用注入后的 hs_object_last）
            ├─ out 组装（logits/boxes 已包含注入后的最后一层）
            ├─ Loss computation
            │
            └─ ★ Triplet Memory Update (with torch.no_grad)
                  │
                  ├─ cxcywh → xyxy: box_cxcywh_to_xyxy(rel_sub/obj_boxes).clamp(0,1)
                  ├─ 构造 candidates: sub/obj/pred scores → quality → topk
                  ├─ TripletMemoryEncoder(rel_query, sub_box, obj_box, pred_prob)
                  │     (4 路输入: rel_query + box + geom + pred_dist → 288d → fusion → 128d)
                  └─ Manager.update_batch(video_ids, frame_idxs, candidates)
```

### 2.2 注入生效性的关键修复

**Stage 1-3 的问题**：`output["hs_object_last"]` 和 `output["hs_relation_last"]` 被注入修改，
但 `relation_logits` 在 decoder 内部已计算完毕（L582 提前提取）。当前帧预测**完全不受 injection 影响**。

**Stage 4 修复**：注入后重跑 prediction head，替换最后一层：
```python
# object 分支
obj_q = injector_obj(output["hs_object_last"], memory, gate=gate_obj)
obj_logits = _predict_object_from_embeddings(obj_q)
output["relation_object_logits"] = output["relation_object_logits"].clone()
output["relation_object_logits"][-1] = obj_logits

# relation 分支
rel_q = injector_rel(output["hs_relation_last"], memory, gate=gate_rel)
output["relation_logits"] = output["relation_logits"].clone()
output["relation_logits"][-1] = transformer.relation_embed(rel_q)
```
`clone()` 确保不破坏原 tensor 被 autograd 链到的中间结果。
仅替换最后（第 L 层），中间层 aux loss 保持原 decoder 输出。

---

## 3. 模块结构

### 3.1 TripletMemorySlot
```
字段: valid(bool), signature(tuple), feat[Tensor], sub_box/obj_box/union_box[Tensor]
      score/sub_score/obj_score/pred_score(float), frame_idx(int), miss(int), age(int)
```

### 3.2 TripletMemoryEncoder (4 路输入，Stage 3 删除 obj_query 分支)
```
rel_query [256] → rel_query_proj → [128] ─┐
sub_box [4] ─┐                             │
obj_box [4]  ├─ cat(12) → box_proj → [64] ├─ cat(288) → fusion → [128]
union_box[4]─┘                             │
rel_geom [8] ────→ geom_proj ─→ [64] ────┤
pred_dist[26] ───→ pred_proj ─→ [32] ────┘
```
fusion 输入: 128 + 64 + 64 + 32 = 288（Stage 1-2 为 416，多了错误的 obj_query 128）

### 3.3 TripletMemoryBank
- **匹配**: signature 相同 + (sub_iou+obj_iou+union_iou)/3 > match_thresh(0.3)
- **更新**: EMA(0.9)，score = max(old, new)
- **过期**: miss > 2 → valid=False
- **容量**: 32 slots，满时替换 score-0.05×miss 最低者

### 3.4 TemporalTripletInjector
```
queries[B,N,256] memory[B,M,128] gate:scalar
         │              │
    Q=Linear→128    K,V=Linear→128
         └── Q·K^T / √128 ──→ softmax → attn@V → Linear→256
                                           return queries + gate × out
```
gate 为外部 warmup schedule 传入的纯 scalar，非可学习参数。

### 3.5 TemporalDeltaEmbedding
bucket 化的帧差编码：{0}, {1}, {2}, {3}, {4-7}, {8-15}, {16+}，每 bucket 一个可学习 embedding。

### 3.6 TripletMemoryManager (nn.Module)
- per-video TripletMemoryBank 管理
- `maybe_clear_on_video_jump()`: 帧序号倒退 → 清空 bank
- `get_batch_memory()`: 返回 [B, M_max, D] + [B, M_max] mask
- `update_batch()`: 批量更新各 video 的 bank

---

## 4. Gate 机制（三类区分）

| | v2 Temporal Injector | ROI Refine Gate | **v3 Triplet Injector** |
|---|---|---|---|
| 参数形式 | `nn.Parameter` → sigmoid | `nn.Sequential` MLP | **纯 scalar, warmup schedule** |
| 可学习 | ✅ 梯度更新 | ✅ 梯度更新 | ❌ iter 比例查表 |
| GATE_LR_MULT 生效 | ✅ | ✅ | ❌ |
| 控制范围 | 单 scalar | per-query scalar | 全局 scalar |
| 初始化 | sigmoid(-4)≈0 | 随机 | 0（10% iter 内为 0） |

v3 选择调度函数的原因：训练早期 memory 为空或全是噪声，需要先积累再注入。调度函数提供明确的阶段策略。

---

## 5. 关键 Bug 修复记录

### 5.1 cxcywh→xyxy 坐标格式不匹配

**根因**：SpeaQ Decoder 输出 DETR 格式 `(cx, cy, w, h)`，但 `make_union_box()` / `relative_geometry()` / `find_match()` 按 `(x1, y1, x2, y2)` 实现。

**影响**：union_box min/max 算错 → relative_geometry 8 维全错 → Encoder 输入污染 → slot IoU 匹配失败。

**修复**（detr.py L729）：
```python
sub_bx = box_cxcywh_to_xyxy(rel_sub_boxes[b, r].unsqueeze(0))[0].clamp(0.0, 1.0)
obj_bx = box_cxcywh_to_xyxy(rel_obj_boxes[b, r].unsqueeze(0))[0].clamp(0.0, 1.0)
```

### 5.2 注入未影响当前帧 prediction

**根因**：logits 在 decoder 内已计算，L582 提前提取。injection 发生在之后，只改了 feature，没改 logits。

**修复**：注入后重跑 prediction head，clone → 替换最后一层。

### 5.3 Gate warmup 硬编码 max_iter=80000

**根因**：`get_temporal_gate(iter, 80000, ...)` 写死。MAX_ITER=16000 时 gate 只走 20%。

**修复**：从 `cfg.SOLVER.MAX_ITER` 读取 `self._triplet_max_iter`。

### 5.4 v3 构建依赖 v2 MODE

**根因**：v3 init 代码在 `if temporal_mode == "object_query_memory_v1"` 块内部。

**修复**：移到独立条件 `if temporal_enabled and TRIPLET_MEMORY_ENABLED`。

### 5.5 基类 DETR 中错误 memory update 块

**根因**：update 代码引用了 IterativeRelationDETR 才有的 `out["relation_logits"]` 和 `bs`，普通 DETR 加载即崩溃。

**修复**：从基类 DETR 删除，update 仅在 IterativeRelationDETR.forward 末尾执行。

### 5.6 Manager 未继承 nn.Module

**根因**：TemporalDeltaEmbedding 的 nn.Embedding 未被 model 管理。

**修复**：`class TripletMemoryManager(nn.Module)`

---

## 6. Memory Update Detach 策略（三层保护）

| 保护层 | 位置 | 实现 |
|--------|------|------|
| 1 | forward 调用处 | `with torch.no_grad():` |
| 2 | Encoder 内部 | `pred_feat = self.pred_proj(pred_prob.detach())` |
| 3 | Bank 写入 | `slot.feat = cand["feat"].detach().cpu()` |

所有 memory 数据存 CPU（.cpu()），不占用 GPU 显存。跨 iter 无计算图连接。

---

## 7. 配置项速查

### 核心开关
```python
TRIPLET_MEMORY_ENABLED = False  # 总开关
INJECT_OBJECT = True            # Stage 1/3, Stage 2=False
INJECT_RELATION = True          # 始终开启
```

### Memory 结构
```python
TRIPLET_MEMORY_DIM = 128        # encoder 输出维度
TRIPLET_MEMORY_SIZE = 32        # per-video 最大 slot 数
TRIPLET_MEMORY_TOPK_UPDATE = 16 # 每帧最多更新 k 个候选
TRIPLET_MEMORY_MAX_MISS = 2     # miss 上限
```

### Gate（warmup schedule，非学习参数）
```python
GATE_MAX_OBJECT = 0.15          # obj 分支 max gate
GATE_MAX_RELATION = 0.30        # rel 分支 max gate
GATE_ZERO_END_RATIO = 0.10      # gate=0 持续到 10% iter
GATE_WARMUP_END_RATIO = 0.30    # gate 线性增长到 30% iter
```

### Memory Update
```python
UPDATE_SCORE_THRESH = 0.10       # candidate quality 阈值 (Stage 3: 0.05→0.10)
UPDATE_EMA_MOMENTUM = 0.9       # EMA 更新动量
MATCH_IOU_THRESH = 0.3          # slot 匹配 IoU 阈值
```

### 调试 & 时序
```python
USE_DELTA_T_EMB = True           # 帧差 bucket embedding
MAX_DELTA_T_BUCKET = 7           # bucket 数量 (0/1/2/3/4-7/8-15/16+)
DEBUG_MEMORY = False             # 每 100 iter 打印 memory 统计
```

### 预留（config 已定义，forward 未消费）
```python
MEMORY_UPDATE_SCHEDULE = "gt_to_pred"     # gt_aligned/mixed/prediction 三阶段
GT_UPDATE_END_RATIO = 0.30               # GT phase 结束比例
MIXED_UPDATE_END_RATIO = 0.70            # mixed phase 结束比例
USE_GT_ALIGNED_MEMORY = True             # 是否启用 GT-aligned candidate
PRED_UPDATE_THRESH_START = 0.15          # prediction 初始阈值（高）
PRED_UPDATE_THRESH_END = 0.05            # prediction 最终阈值（低）
```

---

## 8. 训练脚本

### Overfit（快速验证）
```bash
# 单卡 1000 帧
bash tools/overfit_triplet_v3_only.sh
```

### 全量训练
```bash
# v3-only, 4 GPU
bash tools/fulltask_v3_only_dist_4gpu.sh

# v3+ROI, 4 GPU
bash tools/fulltask_v3_roi_dist_4gpu.sh
```

### 各 Stage 配置差异
| 配置 | Stage 1 (v3+v2) | Stage 2 (rel-only) | **Stage 4 (v3-only)** |
|------|----------------|-------------------|----------------------|
| INJECT_OBJECT | True | **False** | True |
| INJECT_RELATION | True | True | True |
| GATE_MAX_OBJECT | 0.15 | — | 0.15 |
| GATE_MAX_RELATION | 0.30 | 0.25 | 0.30 |
| UPDATE_SCORE_THRESH | 0.05 | 0.05 | **0.10** |
| v2 ObjectMemoryBank | 构建 | 构建 | **不构建** |
| Encoder obj_query | 有(错误索引) | 有(错误索引) | **已删除** |
| Gate max_iter | 硬编码 80000 | 硬编码 80000 | **SOLVER.MAX_ITER** |
| cxcywh→xyxy | ❌ 缺失 | ❌ 缺失 | **✅ 已修复** |
| 注入重算 prediction | ❌ 不生效 | ❌ 不生效 | **✅ 已修复** |

---

## 9. 实验结果

| 实验 | ng-R@20 | AP50 | Rec@50 | 说明 |
|------|---------|------|--------|------|
| v3+v2 Stage 1 overfit 16k | 0.7384 | 56.93 | 98.76% | Stage 1 代码（含多个 bug） |
| **v3-only Stage 4 overfit 16k** | **0.7776** | — | — | 全部 bug 修复后 |

v3-only 比 v3+v2 的 ng-R@20 高约 4 个百分点，且修复了 6 个 bug。

---

## 10. 文件改动清单

| 文件 | 改动 |
|------|------|
| `configs/defaults.py` | 新增 22 个 `TRIPLET_MEMORY_*` 配置项 |
| `modeling/temporal/triplet_memory.py` | **新建** — 全部核心模块 (~500 行) |
| `data/datasets/action_genome.py` | 新增 `frame_idx`、`is_keyframe` 字段 |
| `modeling/meta_arch/detr.py` | 传递 video_ids/frame_idxs 到 NestedTensor |
| `modeling/transformer/detr.py` | init 构建、`_apply_triplet_memory_to_output()`、update 块 |
| `tools/overfit_triplet_v3_only.sh` | v3-only 单卡 overfit |
| `tools/overfit_triplet_v3_16000.sh` | v3-only 16000 iter 长版 |
| `tools/overfit_triplet_memory_v3.sh` | Stage 1 短版（保留） |
| `tools/overfit_triplet_memory_v3_stage2.sh` | Stage 2 短版（保留） |
| `tools/fulltask_v3_only_dist_4gpu.sh` | v3-only 4GPU 全量 |
| `tools/fulltask_v3_roi_dist_4gpu.sh` | v3+ROI 4GPU 全量 |
