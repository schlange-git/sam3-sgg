
**只做 object / det feature 的时间记忆（object query memory）**，形成一个能训练、能验证、能推理、能评测的闭环；
**不改 backbone 输入维度，不改 relation query 的监督逻辑，不做 triplet memory，不做 feature-level 历史帧加权叠加。**

这样设计的原因是：你当前仓库已经是 **SAM3 backbone + DETR/SpeaQ 风格 object/relation 解码**，对象侧输出 `pred_logits/pred_boxes`，关系侧输出 `relation_logits/rel_pair_idx`，AG 数据也已经围绕 `(sub_idx, obj_idx, rel_id)` 组织，所以第一阶段最小侵入点就是 **object query 输入/输出**，而不是 backbone 输入端。
同时，TrackFormer 的核心做法就是把**上一帧输出 embedding 作为 track query** 带到下一帧，并与静态 object queries 一起解码；它依赖 decoder attention 持续更新对象身份与位置，而不是保存整张历史特征图。([开放获取CVF][1])
Action Genome 本身也就是把动作建模成**时空 scene graph**，对象与关系是随时间演化的，因此“非关键帧只做状态更新、关键帧做监督”这条路线与数据集本意是对齐的。([开放获取CVF][2])

---

# 总体原则

## 1. 第一阶段要做什么

你这一阶段只做四件事：

1. **数据侧把单帧样本改成时序 clip 样本**
2. **模型侧新增 object memory bank**
3. **每帧前向时，把 memory 注入 object queries**
4. **关键帧算 loss，非关键帧只更新 memory，不算主监督**

---

## 2. 第一阶段坚决不做什么

这一阶段先不要碰下面这些内容：

1. 不改 SAM3 输入为双帧/多帧
2. 不做 feature map 级别历史融合
3. 不让 relation query 跨帧传播
4. 不让 non-key frame 参与 relation loss
5. 不做跨整个 epoch 的全局 memory
6. 不做 BPTT 跨帧反传

最后一条尤其重要：
**第一阶段 memory 默认 `detach()`，把它当成“状态缓存”，不是“跨时间可反传图”**。
这样显存、代码复杂度、debug 难度都会低很多。

---

# 推荐的第一阶段最终形态

## 3. 你最终要得到的训练/推理范式

### 训练时

一个 sample 不再是单张图，而是一个**有序 clip**：

```python
clip = [
    frame_t0_key,      # 有标注
    frame_t0_mid1,     # 无标注
    frame_t0_mid2,     # 无标注
    frame_t1_key,      # 有标注
]
```

对 clip 按时间顺序循环：

1. 当前帧进 backbone + encoder
2. 取出当前 `object queries`
3. 把上一帧/历史的 `memory queries` 注入 object queries
4. 跑 object decoder，得到：

   * `pred_logits`
   * `pred_boxes`
   * `hs_obj`（最后一层 object hidden states）
5. 如果当前帧是关键帧：

   * 正常跑 relation branch
   * 正常算 object + relation loss
6. 不管是不是关键帧：

   * 用 `hs_obj + pred_logits + pred_boxes` 更新 memory
7. clip 结束后，把关键帧的 loss 求平均返回

### 推理时

同样顺序处理 clip：

1. non-key frame 也前向
2. non-key frame 不输出评测结果，只更新 memory
3. key frame 输出完整 scene graph 结果，沿用现有 evaluator

这就形成闭环了。

---

# 具体实施 plan

---

## 1. 新增 temporal 配置，不要先改老逻辑

### 要改的地方

建议加在你的默认配置文件里，比如：

* `config/defaults.py`
* 或项目自己的 `defaults.py / config.py / yaml`

### 新增配置

先加下面这组：

```python
MODEL.TEMPORAL = CN()
MODEL.TEMPORAL.ENABLED = False
MODEL.TEMPORAL.MODE = "object_query_memory_v1"

MODEL.TEMPORAL.NUM_MEMORY_SLOTS = 32
MODEL.TEMPORAL.MEMORY_DIM = 256
MODEL.TEMPORAL.DETACH_MEMORY = True

MODEL.TEMPORAL.INJECT_METHOD = "add_first_k"   # v1 固定用这个
MODEL.TEMPORAL.MEMORY_SCORE_THRESH = 0.5
MODEL.TEMPORAL.MEMORY_TOPK = 16
MODEL.TEMPORAL.MEMORY_MAX_MISS = 2
MODEL.TEMPORAL.MEMORY_NMS_THRESH = 0.7

MODEL.TEMPORAL.RUN_RELATION_ON_NON_KEY = False
MODEL.TEMPORAL.RETURN_FRAME_LOSSES = False

DATASETS.AG_TEMPORAL = CN()
DATASETS.AG_TEMPORAL.ENABLED = False
DATASETS.AG_TEMPORAL.NUM_INTERMEDIATE_FRAMES = 1   # 先从 1 开始
DATASETS.AG_TEMPORAL.CLIP_MODE = "between_keyframes"
DATASETS.AG_TEMPORAL.RETURN_CLIP = True
```

### 为什么这样做

因为第一阶段最好做到：

* `TEMPORAL.ENABLED=False` 时，完全退回当前单帧 baseline
* `TEMPORAL.ENABLED=True` 时，才走新逻辑

这样你可以快速做 A/B 对照。

---

## 2. 新建一个 temporal dataset，不要直接把老单帧 dataset 改烂

你当前 AG 数据加载本质还是按帧组织，关系字段也是标准 `(sub_idx, obj_idx, rel_id)`，这非常适合继续复用 target 构造逻辑。

### 新增文件

建议新增，不要一开始就魔改原文件：

```python
SpeaQ/data/datasets/action_genome_temporal.py
```

### 这个文件做什么

它不负责重写 GT 构造，而是：

1. 复用原 `action_genome.py` 的单帧 annotation 解析逻辑
2. 在 `__init__` 阶段把 frame 按 `video_id` 和 `frame_id` 排序
3. 按“相邻关键帧之间 + 中间补帧”的规则构建 clip 索引
4. `__getitem__` 返回一个 clip，而不是一张图

### 返回的数据结构

建议统一成这种结构：

```python
{
    "video_id": str,
    "clip_id": str,
    "frames": [
        {
            "image": image_tensor,
            "instances": target_or_none,
            "frame_id": int,
            "is_key_frame": bool,
            "has_ann": bool,
            "file_name": str,
        },
        ...
    ]
}
```

### 关键点

* `is_key_frame=True` 的帧，`instances` 正常构造
* `is_key_frame=False` 的帧，`instances=None`
* `has_ann = is_key_frame`

### 你应该怎么改

不是删原 `action_genome.py`，而是：

* 保留原单帧 dataset
* 新增 temporal dataset
* 在 `build_dataset` 或 `register_dataset` 处加一个 temporal 开关

### 推荐做法

如果你当前数据 builder 在：

```python
SpeaQ/data/build.py
```

那就加：

```python
if cfg.DATASETS.AG_TEMPORAL.ENABLED:
    dataset = ActionGenomeTemporalDataset(cfg, split)
else:
    dataset = ActionGenomeDataset(cfg, split)
```

---

## 3. clip 级 collate function 也要单独加

### 新增文件

如果你有自定义 dataloader utils，建议加：

```python
SpeaQ/data/temporal_collate.py
```

### 目标

让 dataloader 不要把 clip 拍扁成 frame。

### 推荐输出

保持 batch 是：

```python
List[clip_dict]
```

不要强行 stack 成 5D tensor，第一阶段没必要。

### 原因

因为每个 clip 内你本来就要按时间循环，保留 Python list 更直接。

---

## 4. 增加一个独立的 object memory bank 模块

### 新增文件

建议新建：

```python
modeling/temporal/object_memory.py
```

### 里面至少有两个核心对象

#### 4.1 MemoryState

```python
from dataclasses import dataclass

@dataclass
class MemoryState:
    queries: torch.Tensor   # [M, D]
    boxes: torch.Tensor     # [M, 4], normalized cxcywh
    scores: torch.Tensor    # [M]
    labels: torch.Tensor    # [M]
    ages: torch.Tensor      # [M]
    miss: torch.Tensor      # [M]
    valid_mask: torch.Tensor # [M]
```

#### 4.2 ObjectMemoryBank

```python
class ObjectMemoryBank(nn.Module):
    def __init__(self, cfg):
        ...

    def init_empty(self, device):
        ...

    def get_memory_queries(self, state):
        ...

    def update(self, state, hs_obj, pred_logits, pred_boxes):
        ...

    def _select_candidates(self, hs_obj, pred_logits, pred_boxes):
        ...

    def _associate(self, state, cand_queries, cand_boxes, cand_scores, cand_labels):
        ...
```

---

## 5. memory 里到底存什么

第一阶段只存：

1. `最后一层 object decoder hidden state`
2. `对应 box`
3. `对应 score`
4. `对应 label`
5. `age / miss`

### 不存什么

第一阶段先不要存：

* relation query
* triplet feature
* image feature map
* SAM3 backbone feature
* relation logits

### 为什么存最后一层 object hidden state

因为你当前对象头本来就是用 hidden state 经过线性头输出 `class_embed / bbox_embed`，所以**最自然的 memory 载体就是 head 之前的 object hidden feature**。
TrackFormer 也是把上一帧的**输出 embedding**拿来作为下一帧的 track query，而不是存整张特征图。([开放获取CVF][1])

---

## 6. 改 transformer / DETR，让它能把 object hidden states 吐出来

这个地方非常关键。

### 你大概率要改的文件

优先看这几个：

* `modeling/transformer/detr.py`
* `modeling/transformer/transformer.py`
* `modeling/meta_arch/detr.py`

### 你要拿到什么

你需要模型在单帧 forward 时，除了原本的：

* `pred_logits`
* `pred_boxes`

还返回：

* `hs_obj_last`，形状 `[B, Nq, D]`

如果你现在已经有 decoder 输出 `hs`，那就直接拿最后一层；
如果现在只返回 head 结果，那就把 `hs[-1]` 暴露出来。

### 推荐输出格式

让单帧 object branch 输出：

```python
{
    "pred_logits": ...,
    "pred_boxes": ...,
    "hs_obj_last": ...,
}
```

如果 relation branch 还在同一个 forward 里，再附加：

```python
{
    "relation_logits": ...,
    "rel_pair_idx": ...,
}
```

### 为什么必须这样改

因为 memory update 不能用分类头之后的 one-hot 结果，它需要的是**可继续作为 query 使用的 dense embedding**。

---

## 7. query 注入方式：第一阶段不要 concat，直接“前 K 个 slot 加 memory”

这是这版方案最重要的工程取舍。

### 第一阶段推荐方案

不要把 query 数量改成 `Nq + Nm`，
而是保留原来的 `NUM_OBJECT_QUERIES` 不变，只改 query 初始化：

```python
q_base = learned_object_queries            # [B, Nq, D]
q_mem  = memory_bank.get_memory_queries()  # [B, M, D]

k = min(M, NUM_MEMORY_SLOTS, Nq)
q_base[:, :k, :] = q_base[:, :k, :] + memory_proj(q_mem[:, :k, :]) + memory_type_embed
```

### 新增小模块

建议在 `modeling/temporal/object_memory.py` 或 `modeling/transformer/detr.py` 里加一个小 adapter：

```python
class TemporalQueryInjector(nn.Module):
    def __init__(self, d_model):
        self.memory_proj = nn.Linear(d_model, d_model)
        self.memory_type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.memory_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, base_queries, memory_queries):
        q = base_queries.clone()
        k = min(base_queries.size(1), memory_queries.size(1))
        if k > 0:
            q[:, :k] = q[:, :k] + self.memory_scale * self.memory_proj(memory_queries[:, :k]) + self.memory_type_embed
        return q
```

### 为什么不用 concat

因为 concat 会连带修改：

* decoder query 数
* matcher 输入长度
* relation pair 生成逻辑
* loss shape
* 后处理 shape

而 additive slot injection：

* 不改 query 数
* 不改 matcher
* 不改 evaluator
* 不改 relation 分支 shape

这是最适合第一阶段闭环的方案。

---

## 8. 修改 object 分支 forward，让它支持 memory 注入

### 你大概率要改的文件

```python
modeling/meta_arch/detr.py
```

### 改造方式

把你原来的单帧 object forward，拆成两个层级：

#### 8.1 单帧基础 forward

```python
def forward_single_frame(self, frame_dict, memory_state=None, run_relation=True):
    # 1. backbone / encoder
    # 2. build object queries
    # 3. inject memory
    # 4. object decoder
    # 5. object heads
    # 6. optional relation branch
    # 7. return outputs
```

#### 8.2 clip 级 temporal forward

```python
def forward_temporal_clip(self, clip_dict):
    state = self.memory_bank.init_empty(device)
    outputs_for_loss = []
    total_losses = {}

    for frame in clip_dict["frames"]:
        run_relation = frame["has_ann"] or self.cfg.MODEL.TEMPORAL.RUN_RELATION_ON_NON_KEY
        outputs = self.forward_single_frame(frame, state, run_relation=run_relation)

        state = self.memory_bank.update(
            state,
            hs_obj=outputs["hs_obj_last"].detach() if self.cfg.MODEL.TEMPORAL.DETACH_MEMORY else outputs["hs_obj_last"],
            pred_logits=outputs["pred_logits"].detach(),
            pred_boxes=outputs["pred_boxes"].detach(),
        )

        if frame["has_ann"]:
            outputs_for_loss.append((outputs, frame["instances"], frame))

    # 聚合关键帧 loss
    return self.compute_temporal_losses(outputs_for_loss)
```

---

## 9. 关键帧算 loss，非关键帧完全不进 criterion

这个点我建议你做得非常硬。

### 为什么

你当前对象和关系损失都还是 DETR/Relation criterion 路线，核心是 Hungarian matching + relation supervision。
non-key frame 没有标注，硬塞进去只会制造假 negative。

### 实现方式

最简单的方法不是改 criterion，而是：

* non-key frame 不构造 loss 输入
* 只把 key frame 的 `outputs + targets` 收集起来
* 最后按关键帧数平均 loss

### 好处

这样你几乎不用改：

* `SetCriterion`
* `IterativeRelationCriterionBase`
* matcher

### 代码上怎么做

如果当前 `model.forward()` 返回 `loss_dict`，那 temporal 模式下就变成：

```python
loss_dict_sum = defaultdict(float)
num_supervised = 0

for outputs, targets, frame_meta in outputs_for_loss:
    loss_dict = self.criterion(outputs, targets)
    for k, v in loss_dict.items():
        loss_dict_sum[k] += v
    num_supervised += 1

for k in loss_dict_sum:
    loss_dict_sum[k] /= max(num_supervised, 1)
```

---

## 10. relation branch：第一阶段只在关键帧运行

你当前关系评估和 triplet 生成依赖 `relation_logits + rel_pair_idx`，而且训练/评估已经围绕去重后的 triplet 在跑。
因此第一阶段最稳的做法是：

### 保持不变

* relation decoder 结构不改
* relation loss 不改
* triplet evaluator 不改

### 只加一个行为开关

* `frame.has_ann == True` 时：正常跑 relation 分支
* `frame.has_ann == False` 时：跳过 relation 分支

### 结果

时序 memory 只通过**改良 object queries**间接帮助关系预测。
这是第一阶段最干净的实验变量。

---

## 11. memory update 规则要写死，别一开始做太花

### 第一阶段推荐 update 流程

#### 11.1 从当前 object outputs 选候选

对 `pred_logits` 做：

```python
prob = pred_logits.softmax(-1)
scores, labels = prob[..., :-1].max(-1)   # 去掉 no-object
```

过滤规则：

1. `scores > MEMORY_SCORE_THRESH`
2. 去掉背景类
3. 做一次 class-aware NMS
4. 取 top-k

得到：

* `cand_queries`
* `cand_boxes`
* `cand_scores`
* `cand_labels`

#### 11.2 和旧 memory 关联

第一阶段用最简单方案：

```python
cost = lambda_iou * (1 - IoU(mem_boxes, cand_boxes)) + lambda_cls * (mem_labels != cand_labels).float()
```

然后做 greedy match 或 Hungarian 都行。
我建议第一阶段先 greedy，代码更短。

#### 11.3 更新规则

* matched：

  * `memory_query = current_query`
  * `memory_box = current_box`
  * `memory_score = current_score`
  * `memory_label = current_label`
  * `miss = 0`
  * `age += 1`

* unmatched current：

  * 如果还有空 slot，直接写入
  * 如果没有空 slot，用“最低 score 或最高 miss”的 slot 替换

* unmatched memory：

  * `miss += 1`
  * `miss > MEMORY_MAX_MISS` 就失效

### 注意

第一阶段我建议：

* **query 直接替换，不做 EMA**
* box 也直接替换
* 只保留最简单稳定的生命周期管理

这样便于 debug。

---

## 12. 推理闭环：evaluator 尽量不改，只改输入组织

### 最理想做法

让 temporal model 在 eval 时也收 clip：

```python
clip -> sequential forward -> only key frame predictions returned
```

### 返回结果格式

关键帧输出的 prediction 结构必须和你当前 evaluator 吃的一样：

```python
{
    "pred_boxes": ...,
    "pred_classes": ...,
    "pred_scores": ...,
    "relation_logits": ...,
    "rel_pair_idx": ...,
}
```

这样 `evaluation/sg_evaluation.py::_triplet` 和后续 `_compute_pred_matches` 不用动。你当前 evaluator 本来就是围绕 triplet 在做匹配和评测。

### 非关键帧怎么处理

non-key frame 前向以后：

* 不进入 results list
* 不进入 evaluator
* 只更新 memory

---

## 13. 训练脚本层面，最好尽量不改，只保证 model 能直接返回 loss_dict

### 最小改法

让 `train_net.py` / `tools/train.py` 这种入口保持不变：

* dataloader 现在吐 `clip_dict`
* model.forward(clip_dict) 返回 `loss_dict`
* trainer 继续 `sum(loss_dict.values())`

### 不建议一开始改 trainer 的原因

一旦 trainer 改太多，你后面 debug 会分不清是：

* dataset 错了
* model 错了
* trainer 错了
* DDP / collate 错了

---

## 14. 推荐你新增的文件与修改清单

下面这个可以直接给 Cursor。

### 新增文件

```python
SpeaQ/data/datasets/action_genome_temporal.py
SpeaQ/data/temporal_collate.py
modeling/temporal/object_memory.py
```

### 主要修改文件

```python
config/defaults.py
SpeaQ/data/build.py
modeling/meta_arch/detr.py
modeling/transformer/detr.py
modeling/transformer/transformer.py   # 如果 hs_obj 需要从这里暴露
```

### 暂时不要改的文件

```python
evaluation/sg_evaluation.py
modeling/transformer/criterion.py
```

### 只需要绕开的旧逻辑

你现在“历史帧 weighted feature fusion”那条路先不要和 query memory 混用。
不要真删代码，但要在 config 里把它关掉。否则实验变量不干净。

---

## 15. 你应该要求 Cursor 按什么顺序生成代码

这个顺序最稳：

### Step A

先做 temporal dataset + temporal collate
验证 dataloader 输出：

```python
batch[0]["frames"][i]["is_key_frame"]
batch[0]["frames"][i]["has_ann"]
batch[0]["frames"][i]["instances"]
```

### Step B

再做 `ObjectMemoryBank` 单元模块
先本地喂 fake tensor，验证：

* init_empty 正常
* update 正常
* get_memory_queries 正常

### Step C

改单帧 object forward，暴露 `hs_obj_last`

### Step D

加 query injector，让单帧能吃 `memory_state`

### Step E

加 clip-level temporal forward

### Step F

只对关键帧算 loss

### Step G

跑一个 overfit 小实验
只取 1 个 video / 10 个 clip，确认 loss 能下降

### Step H

再接 evaluator

---

## 16. 第一阶段建议的默认超参数

第一版别搜太大空间，直接用这一组：

```python
NUM_MEMORY_SLOTS = 32
MEMORY_TOPK = 16
MEMORY_SCORE_THRESH = 0.5
MEMORY_MAX_MISS = 2
MEMORY_NMS_THRESH = 0.7
DETACH_MEMORY = True
RUN_RELATION_ON_NON_KEY = False
NUM_INTERMEDIATE_FRAMES = 1
```

如果显存紧：

```python
NUM_MEMORY_SLOTS = 16
MEMORY_TOPK = 8
```

---

## 17. 第一阶段必须打的 debug log

这个非常重要，不然你会调不出来。

每个 clip 至少打印一次：

```python
video_id
clip_id
num_frames
num_key_frames
num_non_key_frames
```

每一帧至少可选打印：

```python
frame_id
is_key_frame
memory_valid_count_before
selected_candidate_count
memory_valid_count_after
max_score
mean_score
```

关键帧额外打印：

```python
num_pred_boxes
num_rel_pairs
loss_ce
loss_bbox
loss_giou
loss_relation
```

---

## 18. 第一阶段的验收标准

你不要上来就看最终 R@50。

先看这 6 个标准：

### 标准 1

`TEMPORAL.ENABLED=False` 时，结果和你当前 baseline 一致

### 标准 2

`TEMPORAL.ENABLED=True` 时，训练不报错，loss 能下降

### 标准 3

non-key frame 确实参与前向，但不参与 loss

### 标准 4

关键帧 relation 分支 shape 和旧代码一致

### 标准 5

evaluator 不需要改或只改很少

### 标准 6

memory slot 数量在视频内动态变化合理，不会永远是 0，也不会爆满不清空

---

## 19. 这一阶段最容易出错的地方

### 错误 1：把 non-key frame 也送进 criterion

结果就是假负样本爆炸。

### 错误 2：memory 不 detach

结果是跨帧图越来越大，显存和训练稳定性都会炸。

### 错误 3：query 数量改了

一旦改 `Nq`，matcher、relation pair、head shape 都要连锁修改，第一阶段不值得。

### 错误 4：memory 用分类后结果，不用 hidden state

这样下一帧根本没法当 query 用。

### 错误 5：同时开 feature fusion + query memory

会让实验解释失效。

---

# 可直接交给 Cursor 的任务描述

下面这段你可以直接贴给 Cursor：

```python
目标：在当前 SAM3 + SpeaQ + Action Genome 代码库中，实现第一阶段 temporal object query memory。
约束：
1. 不修改 backbone 输入维度，不做双帧图像输入。
2. 不修改 relation query 结构，不做 triplet memory。
3. 非关键帧参与前向和 memory update，但不参与主 loss。
4. 关键帧正常计算 object + relation loss。
5. memory 默认 detach，不做跨帧反传。
6. 尽量不修改 evaluator 和 criterion。

具体任务：
1. 新建 temporal dataset：
   - 文件：SpeaQ/data/datasets/action_genome_temporal.py
   - 返回 clip dict，而不是单帧 dict。
   - 每个 clip 内 frames 按时间排序。
   - 每个 frame 包含 image, instances, frame_id, is_key_frame, has_ann。
   - 非关键帧 instances=None。
   - 复用原 action_genome.py 的 annotation parsing 逻辑。

2. 新建 temporal collate：
   - 文件：SpeaQ/data/temporal_collate.py
   - 返回 List[clip_dict]。

3. 新建 object memory 模块：
   - 文件：modeling/temporal/object_memory.py
   - 定义 MemoryState 和 ObjectMemoryBank。
   - MemoryState 包含 queries, boxes, scores, labels, ages, miss, valid_mask。
   - ObjectMemoryBank 需要提供 init_empty, get_memory_queries, update。
   - update 内部包含 candidate 选择、NMS、memory-current 关联、slot 更新。

4. 修改配置：
   - 在 config/defaults.py 中新增 MODEL.TEMPORAL 和 DATASETS.AG_TEMPORAL 配置。
   - 默认关闭 temporal。

5. 修改 object detection forward：
   - 在 modeling/transformer/detr.py 或 modeling/transformer/transformer.py 中暴露最后一层 object decoder hidden state hs_obj_last。
   - 在 modeling/meta_arch/detr.py 中新增 forward_single_frame(frame_dict, memory_state=None, run_relation=True)。
   - forward_single_frame 返回 pred_logits, pred_boxes, hs_obj_last，以及可选 relation_logits, rel_pair_idx。

6. 新增 query memory 注入：
   - 在 object query 初始化阶段，将 memory queries 投影后加到前 k 个 object query slots 上。
   - 保持 object query 总数不变。
   - 新增一个 TemporalQueryInjector 小模块即可。

7. 新增 temporal clip forward：
   - 在 modeling/meta_arch/detr.py 中新增 forward_temporal_clip(clip_dict)。
   - 顺序处理 clip 内每一帧。
   - 每帧跑 object branch。
   - 非关键帧默认不跑 relation branch。
   - 每帧结束后用 hs_obj_last + pred_logits + pred_boxes 更新 memory。
   - 只收集关键帧 outputs 进入 loss。

8. loss 聚合：
   - 不修改 criterion 本身。
   - 在 temporal forward 中，对关键帧逐帧调用现有 criterion。
   - 最终对关键帧 loss 求平均返回。

9. 数据构建入口：
   - 在 SpeaQ/data/build.py 中根据 cfg.DATASETS.AG_TEMPORAL.ENABLED 选择 temporal dataset。
   - dataloader 使用 temporal collate。

10. 日志和断言：
   - 打印 clip/frame 级 debug 信息。
   - 确保 temporal 关闭时退化为当前单帧逻辑。
```

---

如果你要，我下一条可以继续直接给你写：

**“第一阶段各文件的伪代码骨架”**，
我会按这几个文件分别写出函数签名和函数体逻辑：

* `action_genome_temporal.py`
* `object_memory.py`
* `detr.py / meta_arch/detr.py`
  这样就更接近 Cursor 可直接补全的状态了。

[1]: https://openaccess.thecvf.com/content/CVPR2022/papers/Meinhardt_TrackFormer_Multi-Object_Tracking_With_Transformers_CVPR_2022_paper.pdf "TrackFormer: Multi-Object Tracking With Transformers"
[2]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Ji_Action_Genome_Actions_As_Compositions_of_Spatio-Temporal_Scene_Graphs_CVPR_2020_paper.pdf "Action Genome: Actions As Compositions of Spatio-Temporal Scene Graphs"
