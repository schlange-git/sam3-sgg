# ROI / 时序 Stage-4 / PatchMerge 三 Flag 设定下的实际生效参数

本文档总结：在 X-SAM 预训练基线之上**仅开启三个结构组件**时，配置中**真正生效**的参数。三个组件为：(1) ROI 特征精修（`ROI_REFINE`）；(2) Stage-4 时序建模（`TEMPORAL`，triplet-memory v3）；(3) 特征图块合并（`SAM3.USE_PATCH_MERGE`）。

参考配置文件：`configs/fulltask_roiresid_pm_clip_v3_xsam_bs24_20w.yaml`。「生效值」指 YAML 显式覆盖、`configs/defaults.py` 默认值与代码实际消费路径三者合并后，真正支配运行时行为的取值。对于「配置中写了但代码从未读取」的项，标注为**惰性（inert）**，不应作为有效机制汇报。

---

## 1. ROI 特征精修（`MODEL.ROI_REFINE`）

ROI 精修模块对每个关系实体 query（subject / object）用 RoIAlign 重新池化一块高分辨率特征区域，经一个小 MLP 投影后，通过一个逐 query 的可学习门控融合回 query 嵌入。其目的是恢复粗粒度全局特征步长所丢失的精细空间细节，对小目标尤为有效。

### 1.1 生效参数

| 参数 | 生效值 | 默认值 | 作用 / 控制内容 |
|---|---|---|---|
| `ENABLED` | `True` | `False` | ROI 精修分支总开关。 |
| `STRIDE` | `14` | `14` | RoIAlign 作用的特征图步长；ROI 采样原生 stride-14 缓存特征（位于 patch-merge 下采样之前）。 |
| `POOL_SIZE` | `7` | `7` | RoIAlign 输出网格；每个 ROI 池化为 `7×7`，展平为 `H·7·7` 后投影。 |
| `RESNET_FPN_LEVEL` | `0` | `0` | 选择源特征层级（0 = 最粗 / stride-32 / p5）。 |
| `APPLY_TO` | `all` | `small_only` | 哪些 query 接受 ROI 精修。`all` → 全部 query 都精修（keep-mask 全 True）；`small_only` → 仅面积 `< SMALL_AREA_THRESH` 的框。 |
| `SMALL_AREA_THRESH` | `0.02` | `0.02` | 「小框」的归一化面积阈值。在 `APPLY_TO=all` 下它不再决定*哪些* query 被精修；但仍用于：(a) 日志中区分小/大框的门控统计；(b) ROI 分类损失筛选小框 query。阈值由全数据集面积统计得出。 |
| `DETACH_BOXES` | `True` | `True` | 构建 ROI 所用的框被 detach，梯度不经 ROI 采样回流到框回归。 |
| `USE_GATE` | `True` | `True` | 启用可学习融合门控（见下）。 |
| `FUSION` | `residual` | `convex` | 融合规则。`residual`：`refined = e + g·roi`；`convex`：`refined = (1−g)·e + g·roi`。 |
| `LOSS_ENABLED` | `True` | `False` | 增加一项 ROI 辅助分类损失。 |
| `LOSS_WEIGHT` | `1.0` | `1.0` | ROI 分类损失在总目标中的权重。 |
| `ONLY_ROI_CLS` | `False` | `False` | 若为真则用 ROI 投影完全替换 query 嵌入；此处关闭，故采用上述门控融合。 |
| `EVAL_DUAL` | `True` | `False` | 评估时单次前向同时输出精修后（`override`）与原始（`origin`，`raw_` 前缀）两套指标以便直接对比。 |
| `REPLACE_BEFORE_MATCHER` | `False` | `False` | 精修后的 logits 是否在匈牙利匹配前替换原始 logits。 |

### 1.2 实际生效机制

对每个保留的 query，模块计算区域嵌入 `roi = roi_proj(RoIAlign(feat, box))`，其中 `roi_proj` 为 `Linear(H·7·7 → H) → ReLU → LayerNorm → Linear(H → H)`。门控是一个 MLP `Linear(2H → H) → ReLU → Linear(H → 1) → Sigmoid`，输入为原始 query 嵌入与 ROI 嵌入的拼接，逐 query 产生标量 `g ∈ (0,1)`。本配置采用残差融合，精修嵌入为 `e + g·roi`，门控学习的是*注入多少*纠正性 ROI 细节、而不破坏原始 query 内容。由于 `DETACH_BOXES=True`，空间采样位置被视为常量；ROI 辅助分类损失（仅在面积 `< 0.02` 的小框 query 上计算）监督精修以提升对小实体的识别。

---

## 2. Stage-4 时序建模（`MODEL.TEMPORAL`，triplet-memory v3）

Stage-4 为每个视频维护一个三元组级（subject, predicate, object）记忆库，通过门控交叉注意力将聚合的时序上下文注入到 object 与 relation query 中，并以一套课程在训练中同时渐进提升*注入强度*与*记忆写入保真度*。

### 2.1 生效参数

| 参数 | 生效值 | 默认值 | 作用 / 控制内容 |
|---|---|---|---|
| `ENABLED` / `EVAL_ENABLED` | `True` / `True` | `False` / `False` | 训练与评估均启用时序建模。 |
| `MODE` | `triplet_memory_v3` | `feature_ema` | 选择三元组记忆变体（区别于 v1 的 object-query 记忆）。 |
| `TRIPLET_MEMORY_ENABLED` | `True` | `False` | 激活三元组记忆库 + 编码器 + 注入器。 |
| `INJECT_OBJECT` / `INJECT_RELATION` | `True` / `True` | `True` / `True` | 向 object query 与 relation query 注入时序记忆。 |
| `INJECT_SUBJECT` | `False`（默认） | `False` | 关闭 subject query 注入。 |
| `DETACH_MEMORY` | `True` | `True` | 记忆条目以 detach 形式存于 CPU；记忆库永不参与反传。 |
| `TRIPLET_MEMORY_DIM` | `128` | `128` | 单条记忆特征的维度。 |
| `TRIPLET_MEMORY_SIZE` | `32` | `32` | 每视频最大槽位数；满时替换最弱槽。 |
| `TRIPLET_MEMORY_TOPK_UPDATE` | `16` | `16` | 每帧写入记忆库的 top 候选数。 |
| `TRIPLET_MEMORY_MAX_MISS` | `2` | `2` | 连续未匹配达该帧数后槽位失效。 |
| `GATE_MAX_OBJECT` | `0.15` | `0.15` | object query 的最大注入强度。 |
| `GATE_MAX_RELATION` | `0.30` | `0.30` | relation query 的最大注入强度（关系更依赖时序上下文）。 |
| `GATE_ZERO_END_RATIO` | `0.10` | `0.10` | 门控严格保持为 0 的训练占比（纯预热、不注入）。 |
| `GATE_WARMUP_END_RATIO` | `0.30` | `0.30` | 门控线性爬升至最大值结束时的训练占比。 |

### 2.2 注入强度调度（时序课程核心）

门控值是训练进度 `r = iter / MAX_ITER` 的函数，由 `get_temporal_gate`（`modeling/temporal/triplet_memory.py`）计算：

```
r < 0.10              → gate = 0                                        （不注入；query 原样返回）
0.10 ≤ r < 0.30       → gate = gate_max · (r − 0.10) / (0.30 − 0.10)    （线性爬升）
r ≥ 0.30              → gate = gate_max                                 （满强度）
```

注入器执行 `q' = q + gate · CrossAttn(q, memory)`。在 `MAX_ITER = 200000` 下，门控在前 20000 步严格为 0，于第 20000–60000 步线性爬升，之后恒定在最大值（object 0.15 / relation 0.30）。这种延迟预热让检测器先稳定，再混入任何时序信号。

另有两套配套课程控制*写入什么*到记忆：

- **记忆更新模式**（`get_memory_update_mode`）：`r < 0.30` 用 `gt_aligned`，`0.30 ≤ r < 0.70` 用 `mixed`，`r ≥ 0.70` 用 `prediction`。早期写入 GT 对齐三元组以求稳定，后期写入模型自身预测。
- **预测质量阈值**（`get_prediction_threshold`）：在 `r ∈ [0, 0.70]` 上由 `0.15` 线性衰减到 `0.05`，随着模型成熟逐步放低门槛、纳入更低置信度的预测。

槽位匹配使用伪身份签名 `(sub_label, pred_label, obj_label)` 加上 subject/object/union 框平均 IoU 阈值 `0.3`，匹配槽以 EMA 动量 `0.9` 更新。读取记忆时用一个分桶时序差嵌入（7 桶：0, 1, 2, 3, 4–7, 8–15, 16+）编码帧间隔。

> 注（惰性 flag）：`NON_KEY_SKIP_LOSS`、`NON_KEY_SKIP_EVAL`、`NON_KEY_RUN_OBJECT_ONLY` 在配置中被设为 `True`，但它们**仅在 `configs/defaults.py` 中声明、未被任何代码路径消费**；在当前实现下无运行时效果，不应作为有效机制汇报。

---

## 3. 特征图块合并（`MODEL.SAM3.USE_PATCH_MERGE`）

块合并用一个可学习下采样算子降低冻结的 SAM3 特征图的空间分辨率，以空间粒度换取与检测头匹配的更长有效步长，且该算子保持可训练。

### 3.1 生效参数

| 参数 | 生效值 | 默认值 | 作用 / 控制内容 |
|---|---|---|---|
| `USE_PATCH_MERGE` | `True` | `False` | 启用对 SAM3 特征图的可学习块合并下采样。 |
| `TARGET_STRIDE` | `32` | `32` | 合并后的输出步长；特征由原生步长合并到 stride 32。 |
| `IMAGE_SIZE` | `1008` | `1008` | SAM3 输入分辨率；决定合并前的特征图尺寸。 |
| `FEATURE_DIM` | `256` | `256` | 送入检测 transformer 的 SAM3 特征通道维度。 |
| `CHANNEL_REPEAT` | `1` | `1` | 对 SAM3 特征的通道重复因子（不重复）。 |
| `FREEZE` | `True` | — | 冻结 SAM3 主干。**例外：** 块合并卷积被显式保持 `requires_grad=True`，即便主干冻结仍参与训练。 |

### 3.2 实际生效机制

块合并沿用 X-SAM 设计：先用 pixel-unshuffle 将空间块折叠进通道维，再用一个可学习 `1×1` 卷积投影回 `FEATURE_DIM=256`，输出在更长的 `TARGET_STRIDE=32`。尽管 `SAM3.FREEZE=True` 冻结了主干其余部分，块合并卷积仍可训练，因此合并表征会适配下游关系任务。ROI 精修分支（第 1 节）刻意在此下采样*之前*抽取原生 stride-14 缓存特征，故两模块互补：块合并提供高效的全局 stride-32 表征，ROI 精修在需要处恢复精细局部细节。

---

## 4. 小结

在该三-flag 设定下，模型：(i) 用可训练块合并卷积将冻结的 SAM3 特征下采样为 stride-32 全局表征；(ii) 通过门控残差 ROI 精修（`e + g·roi`，逐 query 门控 ∈ (0,1)）配合小目标辅助分类损失，为关系实体恢复精细局部细节；(iii) 通过门控交叉注意力将每视频三元组级时序上下文注入 object 与 relation query，其门控遵循延迟线性预热（前 10% 训练为零，至 30% 爬升到 object 0.15 / relation 0.30），并耦合 GT→prediction 的记忆写入课程。
