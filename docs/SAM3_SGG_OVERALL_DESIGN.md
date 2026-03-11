## 整体框架设计概述（SAM3 + SpeaQ + Action Genome）

本文件面向「让大模型辅助分析/改进方案」的场景，总结当前代码库中 \*\*场景图生成（Scene Graph Generation, SGG）\*\* 的整体设计。重点包括：

- **视觉主干与特征编码**：SAM3 作为 backbone 接入 SpeaQ/DETR；
- **注意力与 Transformer 结构**：编码器/解码器以及 RoPE attention 的使用；
- **关系建模与 triplet 构造**：\((\text{subj}, \text{pred}, \text{obj})\) 三元组的生成与评估接口；
- **分类头设计**：对象分类头、关系谓词头以及几何关系 MLP 头；
- **损失函数设计**：DETR 风格匹配损失 + 关系重加权 + 几何关系 CE/Focal 等。

下文以模块为中心进行拆解，方便 ChatGPT 按模块给出改进建议。

---

## 1. 视觉主干：SAM3 Backbone 与特征接口

- **主干实现位置**：`modeling/backbone/sam3_backbone.py` 与 `sam3/sgg/precompute/sam3_adapter.py`。
- **核心思想**：将原生 SAM3 的多模态编码器简化为「仅视觉分支」，输出多尺度特征图，适配 SpeaQ/DETR 所需的 `NestedTensor` 接口。

**输入规范与预处理**

- 所有输入图像在进入 SAM3 之前会被：
  - 转为 `uint8`；
  - resize 到统一尺寸 \(\texttt{IMAGE_SIZE} \times \texttt{IMAGE_SIZE}\)（当前为 \(1008 \times 1008\)）；
  - 转为 `float32` 并归一化到 \([-1, 1]\)。
- 这样可以保证：无论上游数据增强如何设置，进入 SAM3 的空间分辨率和分布是稳定的。

**多尺度特征与 stride 估计**

- SAM3 输出一组候选特征（例如 `backbone_fpn` 不同层，或单尺度 `vision_features`）。
- 对于每一层特征 \(\mathbf{F}_l \in \mathbb{R}^{B \times C_l \times H_l \times W_l}\)，通过输入/输出高度估计 stride：
  \[
  s_l = \max\left(1, \mathrm{round}\left(\frac{H_{\text{in}}}{H_l}\right)\right).
  \]
- 通道统一：每一层特征先通过 \(1\times 1\) 卷积映射到统一通道数 \(\texttt{FEATURE\_DIM}=256\)。
- 目标 stride 对齐：若当前 stride 小于目标值（如 32），会通过 `AvgPool2d` 进一步下采样，使主干输出约为 \(31 \times 31\) 的特征图（给 Transformer 使用）。

**接口到 SpeaQ/DETR**

- 最终 backbone 输出字典 `Dict[str, NestedTensor]`，每个 `NestedTensor` 包含：
  - `tensors`: \(\mathbb{R}^{B \times C \times H \times W}\)；
  - `mask`: \(\mathbb{R}^{B \times H \times W}\)。
- `Joiner` 将其转为列表 `(features, pos)`，其中：
  - `features[-1]` 是供 DETR 使用的主特征图；
  - `pos[-1]` 是对应的位置编码（见后文 attention 部分）。

**冻结策略**

- 配置 `MODEL.SAM3.FREEZE` 控制是否冻结 SAM3 参数：
  - 冻结时：`eval()` + `requires_grad=False`，并用 `torch.set_grad_enabled(False)` 包裹前向；
  - 解冻时：恢复 `train()` + 梯度。
- 这种设计允许实验中灵活切换「SAM3 冻结特征」与「端到端微调」两种范式。

---

## 2. Transformer 与注意力结构

### 2.1 编码器/解码器总体结构

- Transformer 搭建入口：`modeling/transformer/transformer.py` 与 `sam3/sam3/model_builder.py`。
- 采用 DETR 风格的结构：
  - **编码器（Encoder）**：多层自注意力 + FFN，对 backbone 特征进行全局建模；
  - **解码器（Decoder）**：多层 query 自注意力 + 对图像特征的 cross-attention，用于预测对象框和关系。
- 关键参数：
  - 维度 `d_model = cfg.MODEL.DETR.HIDDEN_DIM`；
  - 头数 `n_heads = cfg.MODEL.DETR.NHEADS`；
  - 编码层数 `ENC_LAYERS`，解码层数 `DEC_LAYERS`；
  - 对象 query 数量 `NUM_OBJECT_QUERIES`，关系 query 数量 `NUM_RELATION_QUERIES`，并支持 `MULTIPLY_QUERY` 一对多扩展。

### 2.2 注意力实现与 RoPE

- 基础注意力模块主要分三类：
  - `sam3/sam3/sam/transformer.py::Attention` 与 `TwoWayAttentionBlock`：
    - 用于原生 SAM3 内部的 token-to-image / image-to-token 双向注意力；
    - 支持多头注意力、残差连接和前馈网络。
  - `sam3/sam3/sam/transformer.py::RoPEAttention`：
    - 在标准多头注意力上加入二维 RoPE（Rotary Position Embedding）；
    - 保留相对位置信息，对大分辨率下的图像更友好。
  - `sam3/sam3/model/model_misc.py::MultiheadAttentionWrapper`：
    - 对 `nn.MultiheadAttention` 的薄封装，统一 batch 维和 mask 接口。

- 在 `sam3/sam3/model/decoder.py` 与 `sam3/sam3/model/encoder.py` 中：
  - `self_attn`：对当前 token 序列做自注意力（包含 RoPE 版本）；
  - `cross_attn_image`：对图像特征做 cross-attention，使 query 能「看见」全图；
  - 可选 `use_text_cross_attention=True` 时，再引入 `ca_text` 对文本/语义先验进行 cross-attention。

**直观理解**：

- \*\*编码器\*\*：把 `SAM3` backbone 输出的空间特征 flatten 成序列，叠加二维位置编码，经若干层 self-attention 后获得全局上下文特征。
- \*\*解码器\*\*：
  - 先在 object queries 内部做 self-attention，建模 object-query 间的交互；
  - 再通过 cross-attention 把每个 query 与整幅图像特征对齐，用于预测 box+class；
  - 关系分支则使用 relation queries，对应「潜在关系对」或「关系 anchor」。

---

## 3. Triplet（关系三元组）的生成与使用

### 3.1 GT 与预测 triplet 结构

- 评估侧 triplet 构造函数：`evaluation/sg_evaluation.py::_triplet`。
- 输入：
  - `relations`: \((\text{sub\_id}, \text{obj\_id}, \text{pred\_label})\)；
  - `classes`: 每个检测框对应的类别标签；
  - `boxes`: 每个检测框的边界框。
- 输出：
  - `triplets`: 形状 \((\#\text{rel}, 3)\)，每行为 `(sub_label, pred_label, obj_label)`；
  - `triplet_boxes`: 形状 \((\#\text{rel}, 8)\)，拼接 subject/object 的 \([x1,y1,x2,y2]\)；
  - 可选 `triplet_scores`: \((\#\text{rel}, 3)\)，分别为 subject/object/predicate 的分数。

这一定义同时适用于：

- **GT 三元组**：由 AG 注释中的 subject/object index 与关系标签构造；
- **预测三元组**：由 DETR 输出的目标框、关系 logits 经后处理得到。

### 3.2 训练侧关系对与 triplet 的连接

- 在 `modeling/meta_arch/detr.py` 中，模型前向阶段会为每张图像产生：
  - 对象级输出：`pred_logits`, `pred_boxes`；
  - 关系级输出：`relation_logits` 以及 `rel_pair_idx`，后者指明了每个关系预测对应的 (sub_idx, obj_idx)。
- 中间逻辑会构造：
  - `triplets = cat((rel_pair_idx, rel_labels.unsqueeze(-1)), -1)`；
  - 对 triplets 做 `torch.unique` 去重，得到 `keep_triplet` mask，使同样的 (sub, pred, obj) 只保留一条。
- 这样保证：
  - 关系分支的训练与评估都围绕「去重后的关系 triplet」展开；
  - 后续 `_compute_pred_matches` 可直接在 triplet 级别与 GT 对齐，计算 mR / R@K 等指标。

### 3.3 零样本 triplet 与 reweight

- `evaluation/sg_evaluation.py` 中还定义了 zero-shot triplet 集合，用于评估「训练集中未出现过的 (sub, pred, obj)」的性能。
- triplet 层面的 re-weight 主要体现在：
  - relation loss 中的 `empty_rel_weight`（见后文 Criterion 中的 reweight 逻辑）；
  - 以及评估时对 zero-shot subset 的单独统计。

---

## 4. 分类头设计

### 4.1 DETR 对象分类头

- 对象分类头主要位于：
  - `modeling/transformer/detr.py::DETR`：
    - `self.class_embed = nn.Linear(hidden_dim, num_classes + 1)`；
  - Transformer 内部还存在 `object_embed`：
    - `self.object_embed = nn.Linear(d_model, num_classes + 1)`，用于中间层监督或特定 one-to-many 方案。
- 输出形状通常为：
  \[
  \text{pred\_logits} \in \mathbb{R}^{B \times N_q \times (C+1)},
  \]
  其中 \(C\) 为对象类别数（不含 no-object），额外一维为背景/空框类。

### 4.2 关系谓词分类头

- 关系谓词头存在两种形态：
  - DETR-based 关系头（在 `transformer` 内部或 `Detr` MetaArch 中）：
    - 基于 relation queries，对应的 `relation_class_embed` 或 `relation_embed`；
    - 输出 `relation_logits`，形状 \([B, N_{\text{rel\_q}}, C_{\text{rel}}+1]\)。
  - 纯几何 MLP 关系头：`sam3/sgg/models/relation_head_geom.py::RelationHeadMLP`：
    - 仅依赖 box/mask 的几何特征，适合作为「轻量关系 baseline」或 ablation。

**RelationHeadMLP 结构**

- 输入：`geom_feat`，形状 \([P, G_d]\)，其中 \(G_d=11\)（6 维 box + 5 维 mask 几何）；
- 结构：
  - `Linear(in_dim, hidden) -> ReLU -> Dropout`；
  - `Linear(hidden, hidden) -> ReLU -> Dropout`；
  - `Linear(hidden, num_classes)`；
- 输出：`logits` 形状 \([P, C_{\text{rel}}]\)，其中包含背景类。

### 4.3 头部参数加载与重建策略

- `modeling/meta_arch/detr.py::_load_detr_head_only` 与 `_load_detr_full`：
  - 支持从外部 DETR 权重加载 backbone+transformer；
  - 可选是否加载原始分类头：
    - 若不加载，则会重建 `class_embed`、`relation_class_embed` 等线性层，并重新初始化；
    - 这样保证 AG 或其他数据集的类别数可以与预训练权重解耦。

---

## 5. 损失函数与训练目标设计

### 5.1 DETR 主损失（对象级）

- 主损失入口：`modeling/transformer/criterion.py::SetCriterion`。
- 训练过程分两步：
  1. 通过 Hungarian matcher（`build_matcher`）在预测与 GT 之间做一一匹配；
  2. 在匹配对上计算分类与回归损失。

**对象分类损失**

- `loss_labels` 中：
  - 先构造 `target_classes`，默认填充为 `num_classes` 作为 no-object 类；
  - 使用带类别权重的 `F.cross_entropy`：
    - 背景权重（包括 no-object）由 `eos_coef` 控制；
    - 特别地，对 AG 中的 `person` 类（索引 0）单独设置 `person_class_weight`，以平衡前景/背景。

**边框回归与 GIoU 损失**

- `loss_boxes`：
  - L1 损失：`loss_bbox`；
  - Generalized IoU 损失：`loss_giou`；
  - 二者通过 `weight_dict` 中的 `l1_weight`、`giou_weight` 与主损汇总。

**Mask 损失（可选）**

- 若启用实例分割，则在 `loss_masks` 中计算：
  - `sigmoid_focal_loss` + `dice_loss`；
  - 用于对预测 mask 与 GT mask 的二值监督。

### 5.2 关系/对象联合 Criterion

- 扩展版本 Criterion：`IterativeRelationCriterionBase`：
  - 同时管理对象级与关系级的损失；
  - `empty_weight_obj`、`empty_rel_weight` 分别为对象和关系类别的权重向量：
    - 对象侧同样用 `person_class_weight` 提升人类主体权重；
    - 关系侧可使用统计量 `statistics['fg_rel_count']` 对稀有关系上采样；
      \[
      \text{weight}_r \propto \frac{\sum_c \text{fg\_rel\_count}[c]}{\text{fg\_rel\_count}[r] + \epsilon},
      \]
      并可选取 log 形式或线性形式。
  - `rel_eos_coef` 与 `reweight_rel_eos_coef` 控制关系背景类的相对权重。

### 5.3 几何关系头损失

- 对于 `RelationHeadMLP`，主要使用 `sam3/sgg/train/loss.py::MaskedCrossEntropy`：
  - 接口：
    - `logits`: \([N, C]\)；
    - `labels`: \([N]\)，其中 -1 表示 padding 或无效；
    - `mask`: \([N]\) bool，有效关系对为 True。
  - 实现：
    - 首先计算所有位置的 CE；
    - 通过 `valid_mask = mask & (labels >= 0)` 筛选有效样本；
    - 在有效集合上求平均，若无有效样本返回 0 loss。
  - 通过 `bg_weight` 对背景关系（类 0）降权，缓解前景/背景极度不平衡问题。

### 5.4 其他损失项与权重

- 在 `modeling/meta_arch/detr.py` 中，`weight_dict` 还包含：
  - `loss_relation`, `loss_bbox_relation`, `loss_giou_relation`：关系框/关系分类的一致性；
  - `loss_selection_subject`, `loss_selection_object`：与 one-to-many 方案相关的选择损失；
  - `loss_nms`：后处理/抑制相关的正则项。
- 所有损失最终通过 `build_criterion(...)` 汇总为一个总损失，支持 deep supervision（对中间 decoder layer 也计算损失）。

---

## 6. Action Genome 数据与抽帧/GT 流程（简要）

> 更详细的数据与抽帧结构已在 `SAM3_BACKBONE_SPEAQ_AG_THESIS_NOTES.md` 描述，本节只保留与 SGG 设计强相关的要点。

- 抽帧脚本：`prepare_actiongenome_frames.sh` 调用 `data/ActionGenome/tools/dump_frames.py`：
  - 先 `ffmpeg` 全量解码到 `frames/video_id/%06d.png`；
  - 若未启用 `--all_frames`，则基于 `frame_list.txt` 保留标注帧；
  - \*\*当前扩展逻辑\*\*：在每两个关键帧之间，额外均匀采样若干中间帧（默认 2 帧），以缓解「仅关键帧太稀疏、全帧太密集」的矛盾。
- 数据加载：`SpeaQ/data/datasets/action_genome.py`：
  - 从 `frame_list.txt` 和 bbox/关系 pkl 中构造 Detectron2 风格 GT 字典；
  - `relations` 字段为 \([N_{\text{rel}}, 3]\) int64，元素为 (sub_idx, obj_idx, rel_id)；
  - 下游训练与评估都通过该结构连接到 triplet 与 Criterion。

---

## 7. 可供 ChatGPT 深挖/改进的关键切入点

为了让 ChatGPT 更高效地给出设计建议，可以重点围绕以下问题展开分析：

- **Attention / Transformer 设计**
  - 是否需要在编码器中引入额外的「关系感知」或「人物中心」注意力（例如对 person 区域加权）？
  - RoPE 的使用是否需要在大分辨率/长序列场景下做截断或分块，以减小显存占用？
- **Triplet 构造与采样**
  - 当前 triplet 是「遍历所有有效关系对再去重」，是否需要在训练时对负关系对做更精细的采样策略（如基于 IoU/语义相似度）？
  - 是否可以在 Criterion 中直接对 triplet 级别建模，而非仅在评估时构造三元组？
- **分类头与表征**
  - 对象头目前是单层线性，是否需要在 object query 输出前加入更强的 MLP 或门控机制？
  - 几何关系头与视觉关系头如何融合：例如几何 logits 与视觉 logits 的 late fusion / gating 方案。
- **Loss 设计**
  - 关系重加权目前基于频次统计，是否可以改为在线难例挖掘（hard negative mining）或基于 uncertainty 的自适应重加权？
  - 对于时序场景（多帧 AG 抽帧），是否需要在 loss 中显式加入「时间一致性」约束？

如果需要，我可以基于以上结构再额外输出一份 LaTeX 版本（`tex` 环境），或为某一模块单独画出流程图（方便直接放入论文）。

