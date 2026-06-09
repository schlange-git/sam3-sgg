# Temporal v3 一轮修改记录

> 分支: `temporal_v3`
> 基 commit: `7f76e38` (temporal_memory_v2 final)

---

## Commit 记录

```
593f416 stage4: critical bugfixes
baf3f5a stage3: obj_query removal + config passthrough
ec78308 stage2: relation-only fallback
1552882 stage1: core implementation
```

---

## Stage 1: 核心模块实现 (`1552882`)

**目标**：落地 triplet-level memory 的最小闭环。

### 新增文件
- `modeling/temporal/triplet_memory.py` — 全部核心模块
- `tools/overfit_triplet_memory_v3.sh` — 单卡 overfit 启动脚本

### 修改文件
- `configs/defaults.py` — 新增 22 个 `TRIPLET_MEMORY_*` 配置项
- `data/datasets/action_genome.py` — 新增 `frame_idx`、`is_keyframe` 字段
- `modeling/meta_arch/detr.py` — 传递 video_id/frame_idx 到 NestedTensor
- `modeling/transformer/detr.py` — init 构建、forward 注入、forward 更新

### 实现内容
| 模块 | 功能 |
|------|------|
| `TripletMemorySlot` | 单个 triplet 记忆槽：signature + feat + boxes + scores + frame_idx/miss/age |
| `TripletMemoryEncoder` | 压缩 triplet 信息为 128-dim memory feature（原 5 路输入） |
| `TemporalDeltaEmbedding` | bucket 化的帧差编码（0/1/2/3/4-7/8-15/16+） |
| `TripletMemoryBank` | per-video 记忆库：signature+IoU 匹配、EMA 更新、miss 过期 |
| `TemporalTripletInjector` | gated cross-attention 注入（Q=queries, K/V=memory） |
| `TripletMemoryManager` | 多 video bank 管理 + batch padding |
| `get_temporal_gate()` | warmup schedule：0%(gate=0)→10%→30%(gate=gate_max) |

### 验证结果
- 单卡 overfit 1000 帧，iter 199 稳定运行，loss 284→244

---

## Stage 2: Relation-only Fallback (`ec78308`)

**目标**：只注入 relation branch，不注入 object branch。

### 修改
- 新建 `tools/overfit_triplet_memory_v3_stage2.sh`
- `INJECT_OBJECT=False`，`INJECT_RELATION=True`，`GATE_MAX_RELATION=0.25`

### 验证结果
- 单卡 overfit，2000 iter 完整跑完，0 Error

---

## Stage 3: obj_query 错配修复 + config 透传 (`baf3f5a`)

### 修复 1：删除错误的 obj_query 输入（**结构 bug**）

**问题**：`obj_query = hs_obj[b, r]` 中 `r` 是 relation query index，不是 object query index。
语义不匹配导致 Encoder 输入被污染。

**修复**：从 `TripletMemoryEncoder` 移除 `obj_query_proj` 分支。
Fusion 输入维度从 416 降至 288（`mem_dim + 64 + 64 + 32`）。

### 修复 2：config 硬编码透传

**问题**：gate_max_obj、gate_max_rel、topk_update、update_threshold、debug_memory 等参数在
forward 中写死为常量，训练脚本中的配置项不生效。

**修复**：在 `__init__` 中从 `cfg.MODEL.TEMPORAL` 读取并存储为 `self._triplet_*` 属性：
```python
self._triplet_gate_max_obj = float(getattr(cfg.MODEL.TEMPORAL, "GATE_MAX_OBJECT", 0.15))
self._triplet_topk_update = int(getattr(cfg.MODEL.TEMPORAL, "TRIPLET_MEMORY_TOPK_UPDATE", 16))
self._triplet_update_thresh = float(getattr(cfg.MODEL.TEMPORAL, "UPDATE_SCORE_THRESH", 0.10))
```

### 修复 3：DEBUG_MEMORY 变量作用域

**问题**：forward 中使用 `tcfg.DEBUG_MEMORY`，但 `tcfg` 仅存在于 `__init__` 作用域。

**修复**：改用 `self._triplet_debug_memory`。

### 修复 4：UPDATE_SCORE_THRESH 默认值上调

**原因**：`quality = sub_score × obj_score × pred_score` 乘积在训练早期偏低。
0.05 过激，可能让低质量 triplet 进入 memory。

**修复**：默认值 0.05 → 0.10。

### 新增文件
- `tools/overfit_triplet_v3_only.sh` — v3-only（不启用 v2 ObjectMemoryBank/RelationMemoryBank）、默认单卡
- `tools/overfit_triplet_v3_16000.sh` — 16000 iter 长版脚本

### 验证结果
- v3-only overfit 16000 iter：ng-R@20 = **0.7776**（比 v3+v2 的 0.7384 高 4 个百分点）

---

## Stage 4: 关键 Bugfix (`593f416`)

### Bug 1: cxcywh→xyxy 坐标格式不匹配（**最严重的 bug**）

**问题**：SpeaQ Decoder 输出的 `relation_subject_boxes` / `relation_object_boxes`
是 DETR 格式 `(cx, cy, w, h)`，但 triplet memory 所有几何函数都按 `(x1, y1, x2, y2)` 实现。

**影响链**：
```
cxcywh 被当作 xyxy →
  make_union_box() 的 min/max 算错 →
    relative_geometry() 的 dx,dy,dw,dh,area_ratio 全错 → Encoder 输入污染
    find_match() 的 3-box IoU 全错 → slot 匹配失败
    memory slot 存了错误位置 → 下一帧 cross-attention 读到垃圾
```

**修复**（`detr.py` L729-730）：
```python
sub_bx = box_cxcywh_to_xyxy(rel_sub_boxes[b, r].unsqueeze(0))[0].clamp(0.0, 1.0)
obj_bx = box_cxcywh_to_xyxy(rel_obj_boxes[b, r].unsqueeze(0))[0].clamp(0.0, 1.0)
union_bx = make_union_box(sub_bx, obj_bx)
```

### Bug 2: injection 未影响当前帧 prediction（**架构级修复**）

**问题**：旧代码中 injection 修改了 `output["hs_object_last"]` 和 `output["hs_relation_last"]`，
但 `relation_logits`、`relation_boxes` 等在 decoder 内部已算完。
当前帧预测完全不受 injection 影响——注入的 feature 只进了 memory bank，从未参与本帧的预测。

**修复**：新增 `_apply_triplet_memory_to_output()` 方法（`detr.py` L350）。
注入后重跑 prediction head：
```python
# 1. 注入 query
obj_q = self.triplet_injector_obj(output["hs_object_last"], mem, gate=gate_obj)

# 2. 重算 object logits（最后一层）
obj_logits = self._predict_object_from_embeddings(obj_q)

# 3. clone 原始 tensor 后替换最终层（避免 in-place 破坏 autograd）
output["relation_object_logits"] = output["relation_object_logits"].clone()
output["relation_object_logits"][-1] = obj_logits
```
relation 分支同理。

### Bug 3: Gate warmup 使用硬编码 max_iter

**问题**：`get_temporal_gate(iter, 80000, ...)` 中 80000 写死。
当 MAX_ITER=16000 时，gate 只走 20%，永不到达 full gate。

**修复**：在 `__init__` 读取真实配置：
```python
self._triplet_max_iter = int(getattr(cfg.SOLVER, "MAX_ITER", 80000))
```
gate 计算始终与真实训练长度对齐。

### Bug 4: v3 模块构建依赖 v2 MODE

**问题**：v3 模块（encoder/injector）的构建代码放在 `if temporal_mode == "object_query_memory_v1":`
条件块内部。如果 MODE 不设为 v2 模式，v3 的组件不会被创建。

**修复**：v3 构建移到独立的 `if temporal_enabled and TRIPLET_MEMORY_ENABLED:` 条件，
与 v2 MODE 完全解耦。

### Bug 5: 基类 DETR 中错误 memory update 块（崩溃级 bug）

**问题**：Stage 1-2 时把 triplet memory update 代码放在了基类 `DETR.forward()` 中
（非 `IterativeRelationDETR`），引用 `out["relation_logits"]` 和 `bs` 等
仅 IterativeRelationDETR 才有的变量。普通 DETR forward 执行到该块会因
`UnboundLocalError` 崩溃。

**修复**：删除基类中的错误代码。Update 逻辑正确放在 `IterativeRelationDETR.forward()` 末尾。

### Bug 6: TripletMemoryManager 未继承 nn.Module

**问题**：`TemporalDeltaEmbedding` 包含 `nn.Embedding`，需要被 `model.parameters()` 和
`model.to(device)` 管理。Manager 是普通 Python class 时，DeltaEmbedding 的参数不参与
optimizer 也不响应 device 迁移。

**修复**：`class TripletMemoryManager(nn.Module)`

### 脚本修改

| 文件 | 修改 |
|------|------|
| `tools/overfit_triplet_v3_only.sh` | 显式 `MODEL.TEMPORAL.MODE triplet_memory_v3`；默认单卡；NUM_WORKERS=0 |
| `tools/overfit_triplet_memory_v3.sh` | 加 `MODE triplet_memory_v3` |
| `tools/overfit_triplet_memory_v3_stage2.sh` | 加 `MODE triplet_memory_v3` |
| `tools/overfit_triplet_v3_16000.sh` | 加 `MODE triplet_memory_v3` |
| `tools/fulltask_v3_only_dist_4gpu.sh` | **新建** — v3-only 4GPU 全量训练 |
| `tools/fulltask_v3_roi_dist_4gpu.sh` | **新建** — v3+ROI 4GPU 全量训练 |

### 验证结果

- `python3 -m py_compile` 通过
- 所有脚本 `bash -n` 通过
- `git diff --check` 无 whitespace error
- v3-only overfit 16000 iter 完整运行：ng-R@20 = 0.7776

---

## 仍为预留（未正式接入）

| 功能 | 状态 |
|------|------|
| `MEMORY_UPDATE_SCHEDULE` 三阶段 GT curriculum | config 已定义，forward 未消费；当前只用 prediction-only 模式 |
| `GT_UPDATE_END_RATIO` / `MIXED_UPDATE_END_RATIO` | 同上 |
| `PRED_UPDATE_THRESH_START` / `_END` 动态阈值 | 同上 |
| `USE_GT_ALIGNED_MEMORY` | 同上 |
| `TripletMemoryEncoder` 被主 loss 训练 | 当前仅在 `no_grad` 的 memory update 中调用，无梯度更新 |
| DDP 跨 rank memory 同步 | 每 rank 独立维护 per-video memory |

---

## 各 Stage 的 Gate 区分

| 组件 | 参数形式 | 可学习 | 控制方式 |
|------|---------|--------|---------|
| v2 Temporal Injector | `nn.Parameter` → sigmoid | ✅ 梯度 | `GATE_LR_MULTIPLIER` |
| ROI Refine Gate | `nn.Sequential` MLP | ✅ 梯度 | `GATE_LR_MULTIPLIER` |
| **v3 Triplet Injector** | 纯 scalar, warmup schedule | ❌ 不可学习 | iter 比例查表 |
