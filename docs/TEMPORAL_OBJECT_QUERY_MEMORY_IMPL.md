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




### 现在这版代码里，loss 是怎么处理 keyframe / 非 keyframe 的？

**结论先说：按照你现在这套实现（`TEMPORAL_OBJECT_QUERY_MEMORY_IMPL.md` 描述的“最小侵入版”），loss 完全还是「单帧 baseline 的逻辑」，**当前训练里：

- **不存在真正意义上的 “非关键帧只更新 memory 不算 loss”** 的逻辑；
- **所有被 dataloader 采样到的帧，只要有标注，就正常算 object + relation 全部 loss；**
- 你额外抽出来的 **非 keyframe 图像目前根本没进训练 pipeline**，只是存在磁盘上，为下一阶段 `AG_TEMPORAL` clip dataset 做准备。

具体拆开说：

---

### 1. 现在用的是什么数据形式？

根据 `TEMPORAL_OBJECT_QUERY_MEMORY_IMPL.md` 第 1 节 和第 3 节的描述：

- 这次真正落地的是：
  - `ObjectMemoryBank`（`modeling/temporal/object_memory.py`）
  - `IterativeRelationDETR` 里：
    - 前向前：从 `self._memory_states` 取 memory，注入到 object queries；
    - 前向后：用 `hs_subject_last/hs_object_last + relation_*` 更新 `ObjectMemoryBank`。
- **数据侧没有换成 clip dict**：
  - 仍然是 `SameVideoBatchSampler` + 单帧样本：
    - 每个 batch 里多帧来自同一个视频、按时间顺序；
    - 但 loader 返回的还是「一帧一个 dict」，不是 `clip = [frame_t0_key, frame_t0_mid1, ...]`。

**所以：训练时看见的就是 AG 原始的“关键帧列表”，每帧都当成一个训练样本。**

---

### 2. “没有标注的帧” / “非 keyframe” 现在怎么处理？

分两种情况：

- **(1) AG 原始标注里就没有 ann 的帧**  
  - 如果 dataset 实现里保留了这类样本，一般会构造一个空的 `Instances`（无 GT box / relation）；
  - Detectron2 的 criterion 对“空 target”通常会返回 0 loss（或者只算存在的那部分 loss），**不会因为没标注报错**；
  - 换句话说：这类“无 ann 帧”在当前实现下，要么被过滤掉，要么就贡献 0 或很小的 loss，对训练影响极小。

- **(2) 你通过抽帧脚本多出来的“非 keyframe”图像**  
  - 目前还没有启用 `DATASETS.AG_TEMPORAL.ENABLED=True`、也没有真正创建 `ActionGenomeTemporalDataset` / clip collate；
  - **这些非 keyframe 图片现在不在 AG train.json 里，也不会被 dataloader 读到，更不会参与 loss。**
  - 它们现在唯一的作用，是未来如果按 `docs/temporal.md` 里说的，做 clip dataset 时，可以插进去当 `is_key_frame=False, has_ann=False` 的中间帧。

所以就你现在这批训练日志而言：

- **所有参与训练的帧，几乎可以认为都是 “关键帧”**（AG 标注原本就只在关键帧上）；
- **loss 对关键帧 / 非关键帧没有做任何区分**；
- temporal memory 只是 “读取上一帧/历史帧的状态，注入到当前 queries”，**不改变 loss 计算哪些帧**。

---

### 3. 和文档中“第二阶段”设计的区别

`docs/temporal.md` 中这几条：

- “关键帧算 loss，非关键帧只更新 memory，不算主监督”  
- “clip 返回 `frames` 列表，每一帧有 `is_key_frame` / `has_ann` 字段”  
- “`MODEL.TEMPORAL.RUN_RELATION_ON_NON_KEY = False`”

这些是 **目标设计**，对应的是「真正的 clip dataset + temporal collate + clip 级 forward」，文档也在 `TEMPORAL_OBJECT_QUERY_MEMORY_IMPL.md` 里强调：

- 目前只做了 **最小侵入版**：  
  - 不改 criterion / trainer 主流程；  
  - 不切换到 clip pipeline。
- 把「仅关键帧算 loss，非关键帧只更新 memory」作为 **第二阶段整体改造**。

所以：

- **当前仓库 = 单帧 baseline + 时序 memory 注入 / 更新，loss 统一按帧算；**
- **未来 clip 版本 = per clip 循环帧，`is_key_frame=True` 的帧才算 object / relation loss，`is_key_frame=False` 的帧只更新 memory，不对 loss 负责。**

---

### 总结成一句话

**现在这版代码里，训的是“所有训练帧都算正常 loss，temporal 只是帮你跨帧记忆”；你额外抽的非 keyframe 图片目前没进训练，更没有“非 keyframe 不算 loss、只更新 memory”这条逻辑，这部分要等第二阶段 clip dataset 一起落地。**



### 先把「现在」讲清楚：当前训练里发生了什么？

按你目前这套代码/配置（没开 `AG_TEMPORAL.ENABLED`、没 clip dataset）：

- **参与训练的数据**  
  - DataLoader 读的是 **AG 原始关键帧列表**（train.json 里的那批），一帧一个 sample；
  - 你额外抽出来的非关键帧图片，现在 **完全不在这个列表里**，所以：
    - 不进 DataLoader
    - 不进 backbone / transformer
    - 不更新 memory
    - 不算 loss

- **Temporal memory 现在怎么用？**  
  以「一条训练迭代」为例（batch 内多帧，同一视频）：

  1. 当前 batch 的关键帧进模型：`self.model(data)`；
  2. 在 `IterativeRelationDETR.forward()` 里：
     - 先读 `self._memory_states[video_id]`，拿到 **历史帧的 memory queries**；
     - 用 `TemporalQueryInjector` 把这些 memory 注入当前的 object queries；
  3. 模型前向解码出 `pred_logits/pred_boxes/hs_*`；
  4. 用当前帧的预测 (`hs_obj + pred_logits + pred_boxes`) **更新 `ObjectMemoryBank`**；  
     也就是说，**关键帧既“读 memory”又“写 memory”**；
  5. 对当前帧，按原来的 DETR/SpeaQ 方式 **正常算 object + relation loss**。

所以：

- 现在的「时序」其实是：**一串关键帧之间的跨帧记忆**（memory 在关键帧之间流动），  
- 但**没有真正的“非关键帧 only-forward-only-memory”参与**。

---

### 那么「额外抽的非关键帧」现在在干嘛？

目前它们只是：

- 以图片的形式躺在 `dataset/frames` 目录里；
- 为 **下一步引入 `AG_TEMPORAL` clip dataset** 做数据准备；
- 但 **还没“进入训练”**，因为：
  - 它们不在 annotation 的 index 里；
  - 也没有任何 loader 会去枚举这些帧。

可以理解为：**你先把原料备好了，但还没把它挂到 dataloader 上。**

---

### 你理解中的「非关键帧推理 -> topk 写入 memory」是对的，但没完全落地

你说的这条链路，其实就是完整设计里的 **“非关键帧只更新 memory”** 部分：

> 非关键帧会进行推理，然后推理结果中的 topk query 保存进入 memory。关键帧训练时，从 memory 取历史 query，完成时序训练。

在 **`docs/temporal.md` + TEMPORAL_OBJECT_QUERY_MEMORY_IMPL.md 第 7,8 节** 里，完整版本预期是：

1. dataset 返回一个 `clip`：

   ```python
   {
     "video_id": ...,
     "frames": [
       { image, instances, is_key_frame=True },
       { image, instances=None, is_key_frame=False },
       { image, instances=None, is_key_frame=False },
       { image, instances, is_key_frame=True },
       ...
     ]
   }
   ```

2. forward 里按时间循环 `for frame in clip["frames"]`：
   - 每一帧都：
     - 读 memory -> 注入 object query；
     - 前向 -> 得到 `hs_obj + pred_logits + pred_boxes`；
     - 更新 memory（`ObjectMemoryBank.update()`）。
   - **只有 `is_key_frame=True` 的帧才：**
     - 跑 relation 分支；
     - 构造 GT；
     - 参与主 loss（object + relation）。

3. 最终，这个 clip 的 loss = 所有 keyframe loss 的平均。

**这时候，你额外抽取的非关键帧才算真正“进入训练”（以：前向 + 更新 memory 的方式进入），只是不算 loss。**

但目前这 3 步你还没真正接上 —— 只落地了步骤 2 里的「memory bank + 注入 + 更新」，而 **步骤 1 的 clip dataset 和“key/non-key 区分 + loss 策略”还没做**。

---

### “什么叫真正进入训练？”——用你的语境翻译一下

可以这么划分三个层级：

1. **数据只在磁盘上**  
   - 额外抽帧完成，但 Dataset 完全不知道有它们；
   - 现在就是这个状态：**不参与任何 forward / loss / memory**。

2. **数据参与 forward + memory 更新，但不算 loss**  
   - 通过 `AG_TEMPORAL` clip dataset 把非关键帧纳入 clip；
   - forward 时，非关键帧：
     - 跑 backbone + transformer；
     - 生成预测；
     - 更新 memory（`ObjectMemoryBank`）；
     - **不构造 GT，不算 object / relation loss**。
   - **这是“时序训练”的关键一环**：non-key-frame 通过 memory 影响后续 keyframe 的预测。

3. **数据参与 full loss（关键帧）**  
   - 关键帧本身既 forward+update memory，又正常算 object / relation loss；
   - 和你现在的关键帧行为一致，只是被嵌在 clip 里。

你现在处在 **第 1 -> 2 之间**：

- memory 机制已经写好了；
- 关键帧已经在用 memory；
- 但非关键帧还没被挂到 dataset 上，也就谈不上“只更新 memory 不算 loss”。

---

### 接下来如果你想推进到“真正时序训练”，大概要做这几步

简单列一下你后续可以做的事（也方便我们下一步一起落地）：

1. **开启一个 `AG_TEMPORAL` dataset & collate**：
   - 新建 `ActionGenomeTemporalDataset`（文档里已经给了结构草稿）；
   - `DATASETS.AG_TEMPORAL.ENABLED=True` 时，train loader 用这个 dataset；
   - `__getitem__` 返回 `clip_dict`，包含 key + 非 key 帧。

2. **在 meta-arch 里写一个 clip 级 forward**（或者在现有 forward 里支持 List[frames]）：
   - 对一个 clip 中的 `frames` 按时间循环；
   - 每帧：
     - 读 memory -> 注入 -> forward -> 更新 memory；
     - 若 `is_key_frame`：
       - 构造 per-frame loss（object + relation）；
   - 汇总所有 keyframe 的 loss，返回一个标量。

3. **criterion 保持不变，只是被“多帧包了一圈”**  
   - 每个 keyframe 仍然用原来的单帧 loss 计算逻辑；
   - 只是现在 loss 是 clip 内多帧 keyframe loss 的平均。

到那一步，你就可以说：

> “非关键帧真的参与到了训练过程（影响了关键帧的监督结果），只是它自己不直接被 supervised；而关键帧既 supervised 又利用历史 non-key frame memory。”

---

### 最后一行直白总结

- **现在：** 只有关键帧在训练，temporal memory 让关键帧之间有“跨帧记忆”，额外抽取的非关键帧只是提前准备好的原始图像。  
- **以后（完整版）：** 非关键帧会进模型 forward，写入 memory，影响后续关键帧的 loss——这时我们才说“这些非关键帧真正进入了训练（以无监督时序约束的方式）”。