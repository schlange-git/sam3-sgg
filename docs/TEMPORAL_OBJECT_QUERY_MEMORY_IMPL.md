# Temporal Object Query Memory 落地说明（按当前仓库实现）

本文档整理了本次在 `SAM3 + SpeaQ + Action Genome` 代码库中的时序记忆落地改动，并回答一个关键问题：**是否必须改成 clip 数据集**。

---

## 1. 本次实际落地的实现范围

本次实现采用“最小侵入、可运行优先”的策略，核心目标是：

1. 在不破坏现有训练/评测流程的前提下，引入 `object query memory`；
2. 保留旧时序逻辑（feature-level EMA）作为兼容模式；
3. 不强制改动 criterion/evaluator/trainer 主流程。

### 1.1 新增文件

- `SpeaQ/modeling/temporal/object_memory.py`
- `SpeaQ/modeling/temporal/__init__.py`

其中 `object_memory.py` 提供：

- `MemoryState`：存储 memory slots 的状态（`queries/boxes/scores/labels/ages/miss/valid_mask`）
- `ObjectMemoryBank`：
  - `init_empty()`
  - `get_memory_queries()`
  - `update()`

### 1.2 修改文件

- `SpeaQ/configs/defaults.py`
- `SpeaQ/modeling/transformer/transformer.py`
- `SpeaQ/modeling/transformer/detr.py`

---

## 2. 配置项与模式说明

在 `SpeaQ/configs/defaults.py` 中，`MODEL.TEMPORAL` 现在支持两种模式：

- `feature_ema`：旧逻辑（已有 `TemporalAggregator`，特征级两状态记忆）
- `object_query_memory_v1`：新逻辑（本次落地）

新增主要配置：

- `MODEL.TEMPORAL.MODE`
- `MODEL.TEMPORAL.NUM_MEMORY_SLOTS`
- `MODEL.TEMPORAL.MEMORY_TOPK`
- `MODEL.TEMPORAL.MEMORY_SCORE_THRESH`
- `MODEL.TEMPORAL.MEMORY_IOU_THRESH`
- `MODEL.TEMPORAL.MEMORY_MAX_MISS`
- `MODEL.TEMPORAL.DETACH_MEMORY`
- `MODEL.TEMPORAL.INJECT_METHOD`

同时预留了：

- `DATASETS.AG_TEMPORAL.*`（暂未强依赖，不影响现有 pipeline）

---

## 3. 核心执行逻辑（object_query_memory_v1）

### 3.1 Query 注入（前向前）

在 `SpeaQ/modeling/transformer/detr.py`：

- 新增 `TemporalQueryInjector`：
  - 对 base object queries 的前 `K` 个 slot 做加性注入；
  - 注入项为 `memory_proj(memory_query) + memory_type_embed`；
  - **不改变 query 数量**，避免牵动 matcher/loss/output shape。

### 3.2 Memory 更新（前向后）

同文件 `IterativeRelationDETR.forward()` 中：

- 从 transformer 输出中取：
  - `hs_subject_last`
  - `hs_object_last`
  - `relation_subject_logits / relation_object_logits`
  - `relation_subject_boxes / relation_object_boxes`
- 合并 subject/object 分支后写入 `ObjectMemoryBank.update()`。

### 3.3 视频级状态隔离

`IterativeRelationDETR` 内维护 `self._memory_states: Dict[video_id, MemoryState]`：

- 以 `video_id` 作为 key；
- 每个视频独立维护 memory 状态；
- 提供 `reset_temporal_memory()` 清理接口。

### 3.4 Transformer 输出补充

在 `SpeaQ/modeling/transformer/transformer.py` 中，`IterativeRelationTransformer.forward()` 新增返回：

- `hs_subject/hs_object/hs_relation`
- `hs_subject_last/hs_object_last/hs_relation_last`

用于 memory 更新，不影响原有关系预测输出结构。

---

## 4. 关于“是否必须用 clip 数据集”的结论

你的疑问：

> 现在的逻辑是不是还保留：一个 batch 来自同一个视频，因此可以不用 clip 形式？

结论：**是的，当前仓库可以在不引入 clip dataset 的情况下运行 object query memory。**

依据如下：

1. `SpeaQ/engine/trainer.py` 里 `JointTransformerTrainer.build_train_loader()`：
   - 当 `DATASETS.TYPE == "ACTION GENOME"` 且 `DATASETS.ACTION_GENOME.FORMAT_VID_WISE=True` 时，
   - 使用 `SameVideoBatchSampler`；
2. `SameVideoBatchSampler` 会按 `video_id` 分组并按视频内顺序产出帧；
3. `Detr.preprocess_image()` 会把 `video_id` 写入 `images.video_ids`；
4. `IterativeRelationDETR` 读取 `video_ids` 做视频级 memory 读写。

因此，**第一阶段**可采用“单帧数据结构 + 同视频顺序采样”实现时序记忆闭环，而不必立即改成 clip dict pipeline。

---

## 5. 第二部分：当前方案的优劣与边界

### 5.1 优点（为何先这样做）

1. 改动面小，能快速验证时序记忆是否有效；
2. 不改 criterion/evaluator 主逻辑，训练风险更低；
3. 与现有 `FORMAT_VID_WISE` 机制天然兼容；
4. 保留 `feature_ema` 与 `object_query_memory_v1` 两种模式，便于 A/B 对照。

### 5.2 当前边界（与完整 clip 方案的差距）

1. 还未引入显式 `key-frame / non-key-frame` 监督拆分；
2. 还未实现 clip 级 `forward_temporal_clip()`；
3. 还未做“仅关键帧 relation loss、非关键帧只更新状态”的硬约束。

这部分如果要继续推进，建议作为第二阶段改造（数据结构、trainer 调用、loss 聚合一起做）。

---

## 6. 启用方式（建议）

在你的 yaml 中先开：

```yaml
MODEL:
  TEMPORAL:
    ENABLED: True
    EVAL_ENABLED: True
    MODE: "object_query_memory_v1"
    NUM_MEMORY_SLOTS: 32
    MEMORY_TOPK: 16
    MEMORY_SCORE_THRESH: 0.5
    MEMORY_IOU_THRESH: 0.5
    MEMORY_MAX_MISS: 2
    DETACH_MEMORY: True

DATASETS:
  TYPE: "ACTION GENOME"
  ACTION_GENOME:
    FORMAT_VID_WISE: True
```

---

## 7. 关键代码入口索引

- 配置入口：`SpeaQ/configs/defaults.py`
- 训练采样器：`SpeaQ/engine/trainer.py` -> `SameVideoBatchSampler`
- video_id 传递：`SpeaQ/modeling/meta_arch/detr.py` -> `preprocess_image()`
- memory 模块：`SpeaQ/modeling/temporal/object_memory.py`
- query 注入与更新：`SpeaQ/modeling/transformer/detr.py` -> `IterativeRelationDETR.forward()`
- hidden states 输出：`SpeaQ/modeling/transformer/transformer.py` -> `IterativeRelationTransformer.forward()`

---

## 8. 后续建议（第二阶段）

如需完全对齐 `docs/temporal.md` 的完整方案，下一阶段建议：

1. 新增 temporal dataset（clip dict）；
2. 新增 temporal collate；
3. 在 meta-arch 引入 clip 级 forward；
4. 仅关键帧参与主 loss，非关键帧只更新 memory。

这四步应整体推进，避免“数据格式先变、loss 还没变”造成中间状态不稳定。


