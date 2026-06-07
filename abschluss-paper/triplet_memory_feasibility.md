deepseekv4方案


# SAM 3 Memory 机制改造为 Triplet Memory：SGG 时序建模落地方案

## 1. 问题来源

学长的构想：
> 把 SAM 3 的 memory bank + memory attention 机制移植到 SGG 任务，将 SAM 3 的 mask query 替换为 triplet query。

本文档分析该构想的可行性与具体实施方案。

---

## 2. 已有基础 (temporal_memory_v2 分支现状)

当前分支已经实现了三层时序机制，为改造提供了良好的代码基础。

### 2.1 架构总览

```
                      Perception Encoder (SAM 3 backbone)
                                │
                                ▼
                      IterativeRelationDecoder
                                │
                  ┌─────────────┼─────────────┐
                  ▼             ▼             ▼
            Subject Head   Object Head   Relation Head
                  │             │             │
                  └─────────────┼─────────────┘
                                │
                    时序模块 (当前已有)
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  TemporalAggregator    ObjectMemoryBank    TemporalQueryInjector
  (feature_ema)         (query_memory)      (cross-attention)
```

### 2.2 已实现组件详解

**TemporalAggregator** (`feature_ema` 模式)：

```python
# 特征图级别的 EMA 时序平滑
H_t = sigmoid(alpha) * H_{t-1} + (1 - sigmoid(alpha)) * conv(F_t)
# alpha: 可学习参数, 初始值 0.7
# 输出: gate * H_expanded + (1-gate) * F_projected
```

**ObjectMemoryBank** (`object_query_memory_v1` 模式)：

```python
class ObjectMemoryBank:
    """
    32 槽固定大小记忆库
    每槽存储: query_embedding(256-dim) + box(4-dim) + score + label + age + miss
    """
    def update(state, hs_obj, pred_logits, pred_boxes):
        # 1. 从当前帧预测中按 score 筛选候选 (topk=16)
        candidates = select_topk_by_score(hs_obj, pred_logits, pred_boxes)

        # 2. 对每个候选, 在已有槽中找 IoU 匹配 (同类+高IoU)
        for cand in candidates:
            match = find_best_iou_match(cand_box, memory_boxes[valid_slots])
            if match.found:
                # EMA 更新已有槽
                slot.query = 0.9 * slot.query + 0.1 * cand.query
            else:
                # 未匹配: 找空槽, 或替换最弱槽
                slot = find_empty_or_weakest()

        # 3. 未匹配的旧槽 miss 计数器 +1
        #    miss > max_miss(2) → 清空该槽
        expire_unmatched_slots()

    def get_memory_queries(state):
        # Score 加权: 低分记忆自然衰减
        return score_weighted_topk(state.queries, k=32)
```

**TemporalQueryInjector** (单层 Cross-Attention)：

```python
class TemporalQueryInjector:
    """
    Q = query_proj(current_queries)   # 当前帧的 triplet query
    K = memory_proj_k(memory_queries) # 记忆库中的历史 query
    V = memory_proj_v(memory_queries)

    attn = softmax(QK^T / sqrt(d))     # einsum: 'qbd,mbd->qbm'
    output = gate * (attn @ V) + (1-gate) * current_queries

    gate: 可学习 sigmoid 参数, 随 iter 从 gate_min 到 gate_max warmup
    """
```

**Relation Memory** (与 Object Memory 并列)：

```python
# 配置项
RELATION_MEMORY_ENABLED: True/False
RELATION_MEMORY_SOURCE: "object" | "relation"  # 使用哪个分支的特征
RELATION_MEMORY_UPDATE_MODE: "prediction" | "matched_gt"

# 使用方式: 与 ObjectMemoryBank 共享同一个类
self.relation_memory_bank = ObjectMemoryBank(transformer.d_model, cfg)
```

---

## 3. 与 SAM 2/3 的差异分析

### 3.1 详细对比表

| 维度 | 当前 temporal_memory_v2 | SAM 2 | SAM 3 |
|---|---|---|---|
| **记忆内容** | query embedding 向量 (256-dim) | 空间特征图 (64-dim × H/16 × W/16) + mask | 同 SAM 2 |
| **记忆编码器** | 无（直接存储 decoder query） | **SimpleMaskEncoder**: mask+vision_feat→64-dim | 同 SAM 2 |
| **Cross-Attn 方向** | 当前 query → 历史 query | 当前帧特征 → 记忆空间特征 | 同 SAM 2 |
| **Cross-Attn 层数** | 1 层 | **4 层** encoder-only Transformer | 同 SAM 2 |
| **时序位置编码** | 无 | **可学习 maskmem_tpos_enc** (每帧位置独立参数) | 同 SAM 2 |
| **记忆独立性** | 32 槽共享池 + IoU 匹配分配 | **每个对象独立记忆库** (无需匹配) | 同 SAM 2 |
| **置信度门控** | gate_min/max 调度 | `cal_mem_score > mf_threshold` (默认 0.01) | 同 SAM 2 |
| **Object Pointer** | 无 | MaskDecoder output token → MLP → (256,) | 同 SAM 2 |
| **遗忘策略** | miss 计数器 > max_miss | FIFO 滑动窗口 + 质量过滤 | 同 SAM 2 |

### 3.2 SAM 2 到 SAM 3 的核心演进 (记忆部分)

```
SAM 2:  输入 = 空间提示 (点/框/掩码)
        跟踪 = 纯几何跟踪 (无语义理解)
        记忆 = 视觉特征 + 掩码

SAM 3:  输入 = 概念提示 (文本 + 图像示例)
        跟踪 = 概念引导跟踪 (语义理解 + 几何跟踪)
        记忆 = 继承 SAM 2 (+ 概念特征条件化)

共同点: 记忆库 + Cross-Attention + Memory Encoder 核心架构不变
```

---

## 4. 可行性评估

### 4.1 结论：可行，但需要走自己的路

**不能机械照搬 SAM 3 的空间记忆**，因为 SGG triplet 没有空间掩码。但 **query-level 的记忆库 + Cross-Attention 范式可以直接迁移**。

### 4.2 天然契合的对应关系

```
SAM 3 的对象跟踪                    SGG 的关系跟踪
─────────────────                   ─────────────────
对象 = 一个物理实例                   关系 = 一个三元组 (s, p, o)

Object Pointer (256-dim)             Relation Query (256-dim)  ← 已存在！
  从 MaskDecoder output token         从 Relation Decoder output
  通过 MLP 投影得出                    天然就是 256-dim query

Memory Bank (每对象 7 帧)             Triplet Memory (每三元组 N 帧)
  存储: 空间特征 + 掩码                 存储: query + sub/obj box + pred dist
  编码器: SimpleMaskEncoder            编码器: TripletMemoryEncoder (需新建)

Cross-Attention                     Cross-Attention (已实现!)
  Q = current_frame_features          Q = current_triplet_queries
  K/V = memory_spatial_features       K/V = memory_triplet_queries
  4 层 encoder-only Transformer       1 层 (需扩展)
```

### 4.3 可行性依据

1. **核心抽象已存在**：你的 relation query 天然就是 SAM 3 "object pointer" 的角色。不需要新造基础架构。

2. **Cross-Attention 已跑通**：`TemporalQueryInjector` 已经实现了 Q=query, K/V=memory 的 einsum attention + learnable gate。

3. **Memory Bank 已跑通**：`ObjectMemoryBank` 的 slot/EMA/IoU/expire 逻辑可以直接复用到 triplet 级别。

4. **配置化设计**：现有代码通过 `RELATION_MEMORY_ENABLED` / `RELATION_MEMORY_SOURCE` 等 flag 控制，新功能可以同样默认关闭。

---

## 5. 三个关键缺口

从当前实现到 SAM 3 风格完整方案，有三个核心缺失：

### 缺口 1：无记忆编码器 (Memory Encoder)

```
当前做法:
  memory = relation_query  # 直接把 256-dim query 塞进记忆库
  # 问题: 256 维空间中包含了大量"当前帧特定"的噪声信息
  # 没有显式编码"哪些信息值得跨帧记住"

SAM 3 风格做法:
  memory = TripletMemoryEncoder(
      query=relation_query,          # (256,) 关系 query
      sub_box=pred_sub_box,          # (4,)   subject 框
      obj_box=pred_obj_box,          # (4,)   object 框
      pred_logits=pred_logits,       # (num_pred) 谓词分布
      union_feat=union_roi_feat,     # (256,) 联合区域视觉特征
  )
  # → (64,) 压缩后的记忆向量
  # 编码器学会: 保留"这对实体的空间关系模式", 丢弃"当前帧的噪声"
```

**实现**：
```python
class TripletMemoryEncoder(nn.Module):
    def __init__(self, d_model=256, mem_dim=64):
        self.query_proj = nn.Linear(d_model, mem_dim)      # query → 64
        self.box_proj_s = nn.Linear(4, 32)                  # sub box → 32
        self.box_proj_o = nn.Linear(4, 32)                  # obj box → 32
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 32, mem_dim),
            nn.LayerNorm(mem_dim),
            nn.GELU(),
            nn.Linear(mem_dim, mem_dim),
        )
    def forward(self, query, sub_box, obj_box):
        return self.fusion(torch.cat([
            self.query_proj(query),
            self.box_proj_s(sub_box),
            self.box_proj_o(obj_box),
        ], dim=-1))
```

### 缺口 2：无时序位置编码 (Temporal Position Encoding)

```
当前做法:
  all_memories = [mem_t3, mem_t2, mem_t1]  # 直接拼接
  # 问题: Cross-Attention 不知道哪条是 3 帧前的, 哪条是当前帧的

SAM 3 风格做法:
  tpos_enc = nn.Parameter(randn(num_maskmem, 1, mem_dim))  # (7, 1, 64)

  for i, mem in enumerate(all_memories):
      mem += tpos_enc[i]  # 每条记忆被打上"距离当前帧多远"的标记
```

**效果**：Cross-Attention 学会对近期记忆分配更高权重，同时仍能从远期记忆中提取稳定模式。

### 缺口 3：共享槽 vs 独立记忆库

```
当前:  32 个槽的共享池
      槽[0]: person-eat-pizza (有些帧)
      槽[1]: person-hold-cup (有些帧)
      槽[2]: person-near-table (有些帧)
      ...
      通过 IoU 匹配决定新候选放入哪个槽

SAM 3: 每个对象独立记忆库
      Object_A: MemoryBank([mem_t0, mem_t1, ..., mem_t6])
      Object_B: MemoryBank([mem_t0, mem_t1, ..., mem_t6])
      不需要 IoU 匹配 —— 身份在创建时确定

对 SGG 的启示:
  共享池更适合 SGG —— 关系是类别级概念, 不是实例级概念
  同一语义关系 "person-eating-pizza" 在视频中出现多次
  应该共享同一个记忆槽, 累积该关系类别的跨帧知识
```

---

## 6. 推荐落地方案 (三期)

### 6.1 第一期：Memory Encoder + Temporal PE (低风险, ~200 行)

```
目标: 让记忆更紧凑、有时序感知

新增文件:
  modeling/temporal/triplet_memory_encoder.py

改动文件:
  modeling/temporal/object_memory.py
    └─ store() 调用 TripletMemoryEncoder 而非直接存 query
    └─ get_memory_queries() 添加 tpos_enc

  modeling/transformer/detr.py
    └─ 传入 tpos_enc 参数

配置文件:
  configs/defaults.py
    └─ MODEL.TEMPORAL.MEMORY_ENCODER_ENABLED (默认 False)
    └─ MODEL.TEMPORAL.TPOS_ENC_ENABLED (默认 False)

验证方案:
  单视频过拟合 → 全量 16000 iter 对比 baseline
```

### 6.2 第二期：Multi-Layer Memory Transformer (中等风险, ~300 行)

```
目标: 让三元组间信息与历史信息多层交替融合

新增:
  class MemoryTransformerLayer(nn.Module):
      self_attn = nn.MultiheadAttention(...)  # triplet queries 互相看
      cross_attn = nn.MultiheadAttention(...) # 看 memory bank
      ffn = nn.Sequential(...)

  class MemoryTransformer(nn.Module):
      layers = [MemoryTransformerLayer * 2..4]

替换:
  单层 TemporalQueryInjector → MemoryTransformer

门控保留:
  初始阶段 gate=0 (纯原始 query)
  随 iter warmup 到 target gate
```

### 6.3 第三期：Per-Triplet Independent Memory (高风险, 大改动)

```
目标: 最接近 SAM 3 完整设计

新增:
  class TripletMemoryBank:  # 替代共享 ObjectMemoryBank
      triplet_id: str       # triplet 唯一标识
      memories: deque(7)    # 独立的 7 帧窗口
      obj_ptr: Tensor       # triplet 的紧凑描述符
      state: ACTIVE | OCCLUDED | DEAD

  匹配逻辑:
      当前帧 triplet_queries
          ↓ (无需 IoU 匹配)
      每个 triplet 直接用自己的 memory_bank

风险:
  - SGG triplet identity 概念弱（见附录 identity 分析）
  - 仅在 GT 标注足够稠密时考虑
```

---

## 7. 风险分析

| 风险 | 严重度 | 缓解策略 |
|---|---|---|
| **AG 视频短** (~30-100 帧) | 中 | SAM 3 的 7-frame window 过大，用 3-5 帧窗口 |
| **数据量小** (~10K 视频) | 高 | 先从单视频过拟合验证，有效再全量 |
| **Triplet identity 模糊** | 高 | 共享记忆方案 (1/2 期) 比独立记忆 (3 期) 更安全 |
| **训练不稳定** | 中 | Gate warmup 从 0 开始，训练过半才开启 memory |
| **显存** | 低 | 32 槽 × 64-dim = 2K 参数，加 encoder 也远小于 backbone |
| **谓词变化导致身份混乱** | 高 | 见附录 identity 分析 |

---

## 附录 A：Triplet Identity 设计分析

### A.1 核心问题

> 如果 sub 和 obj 不变，但时序中存在 relation 改变（如 eating→holding），
> triplet 的 id 应该改变还是不变？

### A.2 两种方案

**方案 A：Triplet 身份 = (sub, obj, pred)，predicate 变了就换 id**

```
帧 t:   person(A)-eating-pizza(B)   → triplet_id=1
帧 t+1: person(A)-holding-pizza(B)  → triplet_id=2 (新身份)
帧 t+2: person(A)-eating-pizza(B)   → triplet_id=3 (又一个新身份)

问题:
  - 身份碎片化：同一对实体产生多个短暂身份
  - 记忆无法累积："这个人-披萨的历史交互模式" 被拆散
  - 时序连续性丢失：模型不知道 "他刚才在吃，现在在拿" 是同一对实体
```

**方案 B：Triplet 身份 = (sub, obj)，predicate 变化不换 id**

```
帧 t:   person(A)-eating-pizza(B)   → triplet_id=1
帧 t+1: person(A)-holding-pizza(B)  → triplet_id=1 (同身份)
帧 t+2: person(A)-eating-pizza(B)   → triplet_id=1 (同身份)

优势:
  - 身份稳定，记忆可累积
  - 模型看到 "这个实体对的历史交互模式"
  - SAM 3 的设计哲学正是如此

潜在风险:
  - 记忆中的旧 predicate 特征可能干扰当前 predicate 判断
  - 如果 EMA 衰减不够快，可能 "黏" 在旧 relation 上
```

### A.3 SAM 3 的答案

SAM 3 用了一个非常清晰的设计来解决类似问题：

```
SAM 3 中:
  - Object identity = 物理实例 (同一个人, 跨帧跟踪)
  - Concept = 语义属性 ("穿红色衣服的人", 由一个固定概念提示定义)
  - 概念提示在整个视频中不变 → 不需要处理 "概念改变" 的问题

SGG 中 (类比):
  - Entity-pair identity = (同一个人, 同一个披萨) 跨帧跟踪
  - Relation/Predicate = 语义属性 (eating, holding, ...)
  - 但 relation 在整个视频中会变！这与 SAM 3 不同

关键区别:
  SAM 3:  概念提示是外部给定的 (整个视频不变)
  SGG:    relation 是每一帧内部预测的 (字帧都可能变)
```

### A.4 推荐方案：Entity-Pair Identity + Per-Frame Predicate Prediction

```
核心设计:

  identity = (sub_instance, obj_instance)     ← 稳定的实体对身份
  memory   = 存储该实体对的跨帧交互特征       ← 提供时序上下文
  predicate = 基于 memory + 当前帧特征预测     ← 每帧独立预测

数据流:
  帧 t:
    entity_pair_id: pair_1 (person_A, pizza_B)
    当前帧特征 + pair_1 的记忆库
      ↓
    predicate head 预测: eating (0.7), holding (0.2), ...

  帧 t+1:
    entity_pair_id: pair_1 (person_A, pizza_B)  ← 身份不变
    当前帧特征 + pair_1 的记忆库 (包含 t 帧的 "eating" 记忆)
      ↓
    predicate head 预测: holding (0.6), eating (0.3), ...
    记忆库帮助: "这个实体对之前是 eating, 现在空间关系变了 → holding"
```

### A.5 这个设计如何消除负面影响

**问题**：旧 predicate 的记忆会不会 "污染" 新 predicate 的判断？

**回答**：不会，原因如下：

1. **记忆存的是特征，不是标签**

```
记忆编码器的输入: query_embedding (256-dim, 实体对的交互模式编码)
                   + sub_box + obj_box (空间关系)

它不是存 "这个三元组的 predicate = eating"
而是存 "这对实体在帧 t 的空间+语义交互模式"

当帧 t+1 关系变为 holding 时:
  帧 t 的记忆说: "这两个对象当时在这样交互 (eating 的空间模式)"
  帧 t+1 的 query 说: "这两个对象现在在这样交互 (holding 的空间模式)"
  Cross-Attention 能区分两者的差异 → 产生 "关系改变了" 的信号
```

2. **时序位置编码提供时间先后**

```
模型知道 "eating 的记忆是 1 帧前的" vs "当前帧的新特征"
不会把 3 帧前的旧特征和当前帧的新特征等权重对待
```

3. **EMA 衰减让旧记忆自然淡出**

```
q_t_new = 0.9 * q_t_old + 0.1 * q_candidate

如果 predicate 持续是 holding:
  帧 t+1 记忆: 10% eating + 90% 空
  帧 t+2 记忆: 1% eating + 19% holding + 80% 空
  帧 t+3 记忆: 0.1% eating + 2.7% holding + 97.2% 空

eating 的特征被指数级衰减 → 不会持久干扰
```

4. **Gate 机制控制记忆注入比例**

```
output = gate * memory_output + (1-gate) * current_query

如果当前帧的 predicate 与记忆中的 predator 冲突:
  gate 可以学得很小 → 模型更信任当前帧的 query
如果当前帧模糊 (如遮挡):
  gate 可以学得较大 → 参考历史记忆
```

### A.6 配置建议

```python
# 为 triplet memory 专门设置
MODEL.TEMPORAL.TRIPLET_IDENTITY_MODE: "entity_pair"  # entity_pair | full_triplet
MODEL.TEMPORAL.TRIPLET_MEM_NUM_FRAMES: 3  # AG 视频短，3 帧窗口足够
MODEL.TEMPORAL.TRIPLET_EMA_MOMENTUM: 0.85  # 略低于 object 的 0.9，predicate 变化更快
MODEL.TEMPORAL.TRIPLET_GATE_MIN: 0.0
MODEL.TEMPORAL.TRIPLET_GATE_MAX: 0.3  # predicate 变化快，memory 占比不宜过高
```
