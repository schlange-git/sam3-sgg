# 对 temporal_v3 当前实现的评估与后续任务建议

我看完你现在的实现文档后，整体判断是：**当前 temporal_v3 已经达到了可以进入小规模训练验证的程度**。它不是停留在方案层面，而是已经完成了核心链路：`TripletMemoryBank → TripletInjector → hs_obj[-1] / hs_rel[-1] 注入 → prediction head → memory update`。你现在不一定需要马上做 GT-aligned / mixed update，反而应该先把当前 prediction-only 版本修到工程闭环稳定，然后跑短实验验证趋势。

---

## 1. 当前实现的主要优点

### 1.1 插入位置比较稳

你现在的 temporal_v3 是在 decoder 输出之后、prediction head 之前注入：

```text
TripletMemoryBank → TripletInjector → decoder 输出层 hs_obj[-1] / hs_rel[-1]
```

这比直接改 encoder 或改 decoder 内部层间交互更稳。文档中也明确 temporal_v2 是 query embedding 级别、decoder 前注入，而 temporal_v3 是 decoder feature 级别、decoder 后注入，两者是互补而非替代。

这个设计的优点是：

```text
1. 不破坏 SpeaQ 原始 decoder 主体；
2. 不改 matcher；
3. 不改 loss；
4. 不影响中间层 aux loss 的基本结构；
5. 如果失败，可以通过 gate 或开关快速回退。
```

这对你这种训练成本很高的项目非常重要。

---

### 1.2 记忆槽设计已经比较完整

当前 `TripletMemorySlot` 已经包含：

```text
signature = (sub_label, pred_label, obj_label)
feat
sub_box / obj_box / union_box
score / sub_score / obj_score / pred_score
frame_idx / miss / age
```

这说明你不是单纯存 query，而是已经把 triplet 级身份、几何、质量、时间信息都保存下来了。文档中 memory manager 也已经支持 per-video bank、TemporalDeltaEmbedding、batch memory padding、EMA update 和 miss 过期机制。

这部分是合理的，可以作为论文里的正式方法模块。

---

### 1.3 相对几何和 temporal delta embedding 已经实现

`TripletMemoryEncoder` 现在包含：

```text
rel_query
obj_query
sub_box + obj_box + union_box
relative_geometry
predicate distribution
```

并通过 MLP 压缩到 `mem_dim=128`。文档中也实现了 `TemporalDeltaEmbedding`，bucket 为：

```text
0, 1, 2, 3, 4-7, 8-15, 16+
```

这两个设计是必要的。尤其是 AG 相机移动、抽帧稀疏时，不能只靠绝对 box 坐标。你现在的 memory feature 至少已经具备“相对空间结构 + 时间间隔”的表达能力。

---

### 1.4 Gate warmup 是正确的

当前设计中：

```text
0% - 10%: gate = 0
10% - 30%: gate 线性增长
30% 之后: gate = gate_max
```

这是很重要的稳定性保护。即使你不用 GT-aligned memory，gate warmup 也可以避免训练初期 prediction memory 直接强干预 query。

所以你说“detection 有预训练，开始阶段应该没有那么差”，这个判断成立。**在已有 detection 预训练的前提下，prediction-only memory 可以先试，不必马上实现 mixed update。**

---

## 2. 关于“不一定需要 mixed update”的判断

我同意你现在可以暂时不做 mixed update。

原因是你当前条件和我之前假设不同：

```text
之前假设：模型从较弱初始化开始训练，prediction memory 早期可能全错；
现在条件：detection 部分已有预训练，object/sub/object box 初期不是完全随机。
```

因此现在更推荐：

```text
先跑 prediction-only + gate warmup
不要马上加 GT-aligned / mixed update
```

但是需要保留两个保护：

```text
1. update threshold 不要太低；
2. gate warmup 必须保留。
```

我建议当前 prediction-only 的第一轮训练配置为：

```text
UPDATE_SCORE_THRESH = 0.10 或 0.15 起步
GATE_ZERO_END_RATIO = 0.10
GATE_WARMUP_END_RATIO = 0.30
GATE_MAX_OBJECT = 0.10 或 0.15
GATE_MAX_RELATION = 0.25 或 0.30
```

你文档里现在 threshold 硬编码为 `0.05`，这个对于训练初期可能偏低。虽然 detection 预训练了，但 predicate 仍然可能不稳定。SpeaQ 的质量感知多分配本身也是基于 subject、object、predicate 的整体质量来决定正样本分配，而不是盲信所有 prediction。 所以 memory update 也应该更偏保守。

我的建议是：

```text
第一轮不要做 mixed update；
但把 prediction update threshold 从 0.05 提高到 0.10 / 0.15。
```

---

## 3. 当前实现中最需要优先修的点

你文档最后列出的 TODO 很准确，但我会重新排序优先级。

---

## Priority 1：修正 obj_query 映射问题

你文档里写到：

```text
obj_query 索引简化：update 中直接用 hs_obj[b, r] 匹配 relation index r，未经过 rel_pair_idx 映射
```

这个问题我认为不是“中优先级”，而是**高优先级**。因为如果 `r` 是 relation query index，而 `hs_obj[b, r]` 是 object query index，那么两者语义并不一定对应。这样会导致：

```text
memory signature 来自 relation r；
sub_box / obj_box 来自 relation r；
rel_query 来自 relation r；
但 obj_query 可能来自完全无关的 object query r。
```

这会污染 `TripletMemoryEncoder` 的输入。

### 推荐修法 A：如果当前使用 relation_subject/object 输出

如果 `relation_subject_logits`、`relation_object_logits`、`relation_subject_boxes`、`relation_object_boxes` 本身就是 relation query 对应的 subject/object 预测，那么第一版可以直接**不使用 object branch 的 hs_obj**，而是改成：

```python
obj_query_for_memory = hs_rel[b, r]
```

也就是 memory encoder 暂时只用 relation query 表征：

```text
rel_query = hs_rel[b, r]
obj_query = hs_rel[b, r]  # fallback
```

这样虽然 object feature 不够独立，但至少不会错配。

### 推荐修法 B：增加 relation-object projection

更干净的方式是改 `TripletMemoryEncoder`：

```python
self.obj_query_proj = None
```

第一版 memory encoder 改成：

```text
rel_query
sub_box
obj_box
union_box
relative_geometry
pred_prob
```

也就是先去掉 `obj_query` 依赖。

原因是你的 triplet candidates 当前主要从 relation-specific outputs 构造，而不是从 object branch 的真实 pair mapping 构造。既然没有可靠 `rel_pair_idx → obj_query_idx`，就不要强行塞 `hs_obj[b, r]`。

### 推荐修法 C：如果已有 rel_pair_idx

如果你的 out 里确实有可靠的：

```text
rel_pair_idx[r] = (sub_query_idx, obj_query_idx)
```

那就应该改成：

```python
sub_q_idx, obj_q_idx = rel_pair_idx[b][r]
obj_query = hs_obj[b, obj_q_idx]
```

这是最理想的版本。

### 我的建议

先采用 **修法 B 或 C**。

如果你想最快稳定验证：

```text
第一轮：去掉 obj_query 输入，只用 rel_query + geometry + pred_prob。
```

等跑通后再加入正确映射的 obj query。

---

## Priority 2：所有硬编码配置必须透传

你文档中列出：

```text
gate_max_iter 硬编码 80000
topk_update 硬编码 16
threshold 硬编码 0.05
```

这些必须在正式训练前修掉。否则你后续跑 16k overfit、80k full training 或不同 stage 时，gate schedule 都会错。

建议立即改成：

```python
max_iter = cfg.SOLVER.MAX_ITER
topk = cfg.MODEL.TEMPORAL.TRIPLET_MEMORY_TOPK_UPDATE
threshold = cfg.MODEL.TEMPORAL.UPDATE_SCORE_THRESH
gate_obj_max = cfg.MODEL.TEMPORAL.GATE_MAX_OBJECT
gate_rel_max = cfg.MODEL.TEMPORAL.GATE_MAX_RELATION
```

尤其是 `max_iter=80000` 不能硬编码。因为你文档里新建了 `tools/overfit_triplet_v3_16000.sh`，如果 max_iter 仍然写死 80000，那么 16000 iter 训练中 gate 只会走到：

```text
16000 / 80000 = 20%
```

这意味着 gate 可能还没完全打开，实验结果会被误判。

---

## Priority 3：memory update 阈值建议先调高

当前硬编码：

```text
threshold = 0.05
```

不建议第一轮就这么低。

虽然 detection 预训练较好，但 predicate 早期还是可能不稳定。triplet quality 是：

```text
quality = sub_score × obj_score × pred_score
```

这个乘积会比较小。若 threshold 太高可能没有 memory，太低又会有噪声。建议用两个阶段：

```text
单视频 overfit / debug:
    UPDATE_SCORE_THRESH = 0.05

正式训练前 30%:
    UPDATE_SCORE_THRESH = 0.10 或 0.15

如果 memory slot 太少:
    降到 0.05
```

更稳的简化版本：

```python
threshold = cfg.MODEL.TEMPORAL.UPDATE_SCORE_THRESH
```

先不用动态 threshold，但要能配置。

---

## Priority 4：确认 prediction head 是否真的使用了注入后的 query

文档中数据流写的是：

```text
Triplet Memory Injection → Prediction heads 使用注入后的 query
```

这是正确目标。

但代码上需要仔细确认：

```text
output["hs_object_last"] / output["hs_relation_last"] 被修改后，
后续 relation_logits / relation_subject_logits / relation_object_logits
是否真的是基于修改后的 query 重新计算？
```

如果当前 prediction head 在注入前已经算完 logits，然后你只是改了 `output["hs_relation_last"]`，那 memory 对最终预测没有实际影响。

你需要检查 `DETR.forward()` 的顺序：

```text
正确顺序：
decoder outputs
→ triplet injection 修改 hs_*_last
→ prediction heads
→ logits / boxes
→ loss / update
```

错误顺序：

```text
decoder outputs
→ prediction heads 已经生成 logits / boxes
→ triplet injection 只改 output 里的 hs_*_last
→ logits 不变
```

这是最容易出现的“看起来接入了，但训练完全没变化”的问题。

---

## Priority 5：debug memory 日志要修

你文档中写 `DEBUG_MEMORY` 日志不生效，因为 forward 中 `tcfg` 变量不可用。这个优先级不如上面三个高，但在正式训练前必须修。

建议 debug 每 100 或 500 iter 打印：

```text
iter
video_id
frame_idx
gate_obj / gate_rel
memory_slots_mean
update_candidates_mean
quality_mean / max / min
num_empty_memory_in_batch
```

如果你发现：

```text
memory_slots_mean 长期为 0
update_candidates_mean 长期为 0
quality_mean 极低
gate 一直没打开
```

那训练结果就不用等三天，可以提前停。

---

## 4. 当前不建议马上做的内容

### 4.1 暂时不做 GT-aligned / mixed update

你现在 detection 预训练过，而且已经有 gate warmup，所以可以先跳过。文档中已经预留了 `gt_aligned / mixed / prediction` helper，但当前仅 prediction 生效。

我建议：

```text
GT-aligned memory 暂时作为 Plan B；
只有 prediction-only 明显不稳定时再做。
```

触发条件：

```text
1. memory 开启后 loss 明显爆炸；
2. object AP 或 recall 大幅下降；
3. memory candidates 大量低质量；
4. relation loss 比 baseline 更不稳定；
5. 单视频 overfit 都无法收敛。
```

如果这些都没有出现，就没有必要先做 mixed update。

---

### 4.2 暂时不做 DDP memory 同步

文档里写 DDP 下 memory 行为未测试。这个确实是问题，但不是第一优先级。因为你可以先在：

```text
单卡 overfit
单卡短训
双卡但每 rank 独立 memory
```

上验证趋势。

严格的跨 rank memory 同步实现复杂，而且 AG dataloader 如果每个 rank 分到不同视频，强行同步反而未必合理。

第一版建议：

```text
DDP 下每个 rank 维护自己的 per-video memory，不跨 rank broadcast。
```

只要同一视频帧序列不要被切到多个 rank 之间即可。如果 dataloader 保证每个 iter 来自一个视频连续帧，那就可以先接受。

---

### 4.3 暂时不做 temporal_v2 + temporal_v3 双开的大实验

文档里写 temporal_v2 和 temporal_v3 共存。这个作为最终实验可以，但第一轮验证最好分清楚：

```text
Baseline A: 原始 SpeaQ + SAM3
Baseline B: temporal_v2 only
Experiment C: temporal_v3 only
Experiment D: temporal_v2 + temporal_v3
```

但你算力有限，不要全跑。建议先跑：

```text
当前已有最好 baseline
vs
当前 temporal_v3
```

如果 temporal_v3 不涨，再考虑是否 temporal_v2 干扰了 temporal_v3。

---

## 5. 下一步最小任务列表

下面是我建议你接下来交给 Cursor 的任务，不超过必要范围。

---

## Task 1：修 config 透传与硬编码

目标：让所有训练脚本可控。

必须修：

```text
1. max_iter 从 cfg.SOLVER.MAX_ITER 读取；
2. topk_update 从 TRIPLET_MEMORY_TOPK_UPDATE 读取；
3. update threshold 从 UPDATE_SCORE_THRESH 读取；
4. gate max 从 GATE_MAX_OBJECT / GATE_MAX_RELATION 读取；
5. DEBUG_MEMORY 从 self.cfg 或 self.temporal_cfg 读取。
```

---

## Task 2：修 obj_query 错配

目标：避免 relation index 误用 object query index。

优先方案：

```text
如果没有可靠 rel_pair_idx：
    TripletMemoryEncoder 暂时去掉 obj_query 输入；
    只使用 rel_query + box/geometry + pred_prob。

如果有可靠 rel_pair_idx：
    使用 obj_query = hs_obj[b, obj_q_idx]。
```

这是目前最需要修的结构性问题。

---

## Task 3：确认 injection 是否发生在 prediction head 之前

目标：确保 temporal module 真的影响 logits。

检查 forward 顺序：

```text
decoder output
→ temporal injection
→ class_embed / bbox_embed / relation heads
→ loss
```

如果不是这个顺序，必须调整。

---

## Task 4：加入 memory debug 统计

目标：快速判断是否值得继续训练。

每 100 或 500 iter 打印一次：

```text
gate_obj
gate_rel
avg_memory_slots
avg_update_candidates
avg_candidate_quality
num_empty_memory
```

---

## Task 5：先跑单视频 / 小 batch overfit

不要直接双卡三天。

建议先跑：

```text
1. TRIPLET_MEMORY_ENABLED=False baseline overfit
2. TRIPLET_MEMORY_ENABLED=True temporal_v3 overfit
```

看：

```text
loss 是否下降；
memory slot 是否正常增长；
relation loss 是否比 baseline 更稳；
是否出现 object loss 变坏；
R@20/R@50 是否有趋势。
```

---

## 6. 建议当前训练配置

如果你现在马上要跑第一版，我建议：

```bash
MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED True
MODEL.TEMPORAL.INJECT_OBJECT True
MODEL.TEMPORAL.INJECT_RELATION True

MODEL.TEMPORAL.GATE_MAX_OBJECT 0.10
MODEL.TEMPORAL.GATE_MAX_RELATION 0.25

MODEL.TEMPORAL.GATE_ZERO_END_RATIO 0.10
MODEL.TEMPORAL.GATE_WARMUP_END_RATIO 0.30

MODEL.TEMPORAL.UPDATE_SCORE_THRESH 0.10
MODEL.TEMPORAL.TRIPLET_MEMORY_TOPK_UPDATE 16
MODEL.TEMPORAL.TRIPLET_MEMORY_SIZE 32
MODEL.TEMPORAL.TRIPLET_MEMORY_MAX_MISS 2
MODEL.TEMPORAL.USE_DELTA_T_EMB True
MODEL.TEMPORAL.DEBUG_MEMORY True
```

如果单视频 overfit 中 memory slot 太少，再改：

```bash
MODEL.TEMPORAL.UPDATE_SCORE_THRESH 0.05
```

如果 object loss 明显变坏，Stage 2：

```bash
MODEL.TEMPORAL.INJECT_OBJECT False
MODEL.TEMPORAL.INJECT_RELATION True
MODEL.TEMPORAL.GATE_MAX_RELATION 0.25
```

---

## 7. 最终判断

你现在实现的 temporal_v3 方向是对的，已经具备实验价值。当前不需要优先做 mixed update；在 detection 有预训练、gate 有 warmup 的情况下，prediction-only memory 是合理的第一版。

但在正式训练前，必须先修这三件事：

```text
1. 修掉硬编码配置；
2. 修掉 obj_query 与 relation index 的错配；
3. 确认 injection 发生在 prediction head 之前。
```

这三件不修，实验结果可能不可解释。修完之后，再跑小规模 overfit 验证 memory 是否真的进入模型，再决定是否需要 GT-aligned / mixed update。
