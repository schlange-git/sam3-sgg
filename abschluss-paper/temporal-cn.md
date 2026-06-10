# 面向场景图生成的时序三元组记忆模块

## 摘要

本文提出一种用于视频场景图生成的时序三元组记忆模块。该模块在 SpeaQ 场景图生成框架与 SAM3 视觉骨干网络之上，引入以视频为单位的记忆库来存储 `(subject, predicate, object)` 三元组的压缩表示，并通过门控交叉注意力机制将历史记忆注入解码器特征。模块以轻量级插件形式接入现有 DETR 式解码器，无需修改骨干网络、编码器、匹配器或损失函数。在 Action Genome 数据集上，时序三元组记忆显著提升了无图约束召回率。

## 1. 引言

场景图生成旨在将视觉场景解析为结构化图表示。在视频领域中，同一组 `(主体, 谓词, 客体)` 三元组常跨帧持续出现，但逐帧独立的模型无法利用这种时序先验。

现有视频场景图生成的时序方法主要在特征级或查询嵌入级操作，例如对骨干特征做指数移动平均，或将历史目标查询注入解码器输入嵌入。这些方法虽然有效，但记忆粒度停留在单个目标或特征层面，未显式建模场景图所定义的三元组级结构。

本文提出**时序三元组记忆**模块，以三元组粒度存储与检索信息。每个记忆槽编码一条完整的 `(主体, 谓词, 客体)` 交互——包含关系查询特征、主体与客体边界框、相对空间几何关系以及谓词概率分布。门控交叉注意力机制将此历史上下文注入解码器末层目标分支与关系分支，使模型在当前帧预测中利用时序先验。

## 2. 架构总览

本模块位于 DETR 式解码器输出与最终预测头之间，不修改骨干网络、Transformer 编码器、匹配器或损失准则。图 1 给出了数据流示意。

```
输入帧 I_t
      │
      ▼
┌─────────────────┐
│  SAM3 骨干网络   │  (冻结 ViT，avg_pool2d 降采样)
└────────┬────────┘
         ▼
┌─────────────────┐
│ 编码器+解码器     │  (IterativeRelationTransformer)
│                  │
│  hs_sub[-1]     │  主体查询特征 [B, N_s, D]
│  hs_obj[-1]     │  客体查询特征 [B, N_o, D]
│  hs_rel[-1]     │  关系查询特征 [B, N_r, D]
│  logits, boxes  │  预测输出
└────────┬────────┘
         │
         ▼
╔═════════════════════════════════════════════╗
║         时序三元组记忆模块                   ║
║                                            ║
║  ┌──────────────────────────┐              ║
║  │  三元组记忆管理器          │              ║
║  │   按视频查找记忆库         │              ║
║  └───────────┬──────────────┘              ║
║              │ memory [B, M, D_m]          ║
║              ▼                              ║
║  ┌──────────────────────────┐              ║
║  │  门控调度                  │              ║
║  │   g_obj ∈ [0, α_obj]     │              ║
║  │   g_rel ∈ [0, α_rel]     │              ║
║  └───────────┬──────────────┘              ║
║              │                              ║
║     ┌────────┴────────┐                    ║
║     ▼                 ▼                    ║
║  ┌──────────┐   ┌──────────┐              ║
║  │客体注入器 │   │关系注入器 │              ║
║  │MHA(Q,K,V)│   │MHA(Q,K,V)│              ║
║  └────┬─────┘   └────┬─────┘              ║
║       │              │                    ║
║       ▼              ▼                    ║
║  重算末层 logits/boxes                    ║
║  (clone 原始张量 → 替换末层)              ║
╚═══════════════════════════════════════════╝
         │
         ▼
┌─────────────────┐
│  预测头           │
└────────┬────────┘
         ▼
┌─────────────────┐
│  损失 / 输出      │
└────────┬────────┘
         │
         ▼
╔═════════════════════════════════════════════╗
║       记忆更新 (no_grad)                     ║
║  构造三元组候选                              ║
║  cxcywh → xyxy 转换                         ║
║  三元组记忆编码器                            ║
║  记忆库更新 (匹配→EMA / 插入→过期)           ║
╚═════════════════════════════════════════════╝
```

**图 1：时序三元组记忆架构。** 模块截取解码器输出，按视频读取历史记忆，通过门控交叉注意力注入时序增强上下文，重算末层预测，然后以当前帧三元组候选更新记忆库。

## 3. 三元组记忆库

### 3.1 记忆槽结构

每个视频维护独立记忆库，容量上限 $K=32$。每个记忆槽 $\mathbf{s}_k$ 存储一条历史三元组的压缩表示：

$$\mathbf{s}_k = \left( \sigma_k, \mathbf{f}_k, \mathbf{b}_k^{\text{sub}}, \mathbf{b}_k^{\text{obj}}, \mathbf{b}_k^{\text{union}}, q_k, t_k, m_k, a_k \right)$$

其中 $\sigma_k = (c_{\text{sub}}, c_{\text{pred}}, c_{\text{obj}})$ 为三元组签名，$\mathbf{f}_k \in \mathbb{R}^{D_m}$ 为压缩记忆特征（$D_m=128$），$\mathbf{b}_k^{*} \in [0,1]^4$ 为 xyxy 格式边界框，$q_k \in [0,1]$ 为质量分数，$t_k$ 为来源帧序号，$m_k$ 为连续未命中计数器，$a_k$ 为槽位年龄。

三元组签名 $\sigma_k$ 作为伪身份标识，利用了 Action Genome 中完全相同的三元组极少同帧出现这一特性，消除了对外部跟踪 ID 的依赖。

### 3.2 槽位匹配

当新三元组候选到达时，记忆库通过两阶段准则寻找最佳匹配槽位：

$$\text{match}(k, c) = \begin{cases} 1 & \text{若 } \sigma_k = \sigma_c \land \bar{\text{IoU}}(\mathbf{b}_k, \mathbf{b}_c) > \tau_{\text{match}} \\ 0 & \text{否则} \end{cases}$$

其中 $\bar{\text{IoU}} = \frac{1}{3}(\text{IoU}_{\text{sub}} + \text{IoU}_{\text{obj}} + \text{IoU}_{\text{union}})$ 为配对标框的平均重叠度，$\tau_{\text{match}} = 0.3$ 为匹配阈值。

### 3.3 槽位更新

匹配成功的槽位通过指数移动平均更新（动量 $\beta = 0.9$）。质量分数取历史最大值，$m_k$ 重置为零。未匹配槽位的 $m_k$ 递增；$m_k > 2$ 的槽位标记为无效并从活跃集中移除。无匹配槽位的新三元组插入；库满时淘汰 $q_k - 0.05 \cdot m_k$ 最低的槽位。

### 3.4 时序距离嵌入

将帧差 $\Delta_t$ 离散化为 7 个桶：

$$\text{bucket}(\Delta_t) = \begin{cases} 0 & \Delta_t = 0 \\ 1 & \Delta_t = 1 \\ 2 & \Delta_t = 2 \\ 3 & \Delta_t = 3 \\ 4 & 4 \leq \Delta_t \leq 7 \\ 5 & 8 \leq \Delta_t \leq 15 \\ 6 & \Delta_t \geq 16 \end{cases}$$

嵌入在交叉注意力之前加到记忆特征上：$\mathbf{f}_k \leftarrow \mathbf{f}_k + \mathbf{e}_{\text{emb}}(\text{bucket}(\Delta_t))$。

## 4. 记忆编码

### 4.1 编码器结构

三元组记忆编码器将三元组候选压缩为 $D_m$ 维特征，包含四路输入模态：

1. **关系查询特征** $\mathbf{h}_{\text{rel}} \in \mathbb{R}^{256}$：解码器末层对谓词的表示。
2. **边界框几何** $[\mathbf{b}_{\text{sub}}, \mathbf{b}_{\text{obj}}, \mathbf{b}_{\text{union}}] \in \mathbb{R}^{12}$：主体、客体及其并集的绝对位置。
3. **相对空间几何** $\mathbf{g} \in \mathbb{R}^{8}$：主体与客体的空间关系（见下节）。
4. **谓词分布** $\mathbf{p} \in \mathbb{R}^{C_{\text{rel}}}$：关系类上的 softmax 概率，从计算图中分离。

各路经独立 MLP 投影后拼接融合：

$$\mathbf{f} = \text{MLP}_{\text{fuse}}\left( \left[ \text{proj}_{\text{rel}}(\mathbf{h}_{\text{rel}}); \text{proj}_{\text{box}}(\mathbf{b}); \text{proj}_{\text{geom}}(\mathbf{g}); \text{proj}_{\text{pred}}(\mathbf{p}) \right] \right)$$

融合层输入维度为 $128 + 64 + 64 + 32 = 288$。

### 4.2 相对空间几何

给定主体框 $\mathbf{b}_{\text{sub}}$ 与客体框 $\mathbf{b}_{\text{obj}}$（xyxy 归一化坐标），先转换为中心-尺寸表示 $(c_x, c_y, w, h)$，再计算：

$$\mathbf{g} = \begin{bmatrix} c_x^o - c_x^s \\ c_y^o - c_y^s \\ \log(w^o / w^s) \\ \log(h^o / h^s) \\ \log(A^o / A^s) \\ A^s / A^{\text{union}} \\ A^o / A^{\text{union}} \\ \sqrt{(c_x^o - c_x^s)^2 + (c_y^o - c_y^s)^2} \end{bmatrix}$$

在相机移动频繁的 Action Genome 视频中，相对几何较绝对坐标更为稳定。

## 5. 时序注入

### 5.1 门控交叉注意力

时序三元组注入器使用多头交叉注意力将历史记忆融入当前帧解码器特征。对查询分支（客体或关系）有特征 $\mathbf{Q} \in \mathbb{R}^{B \times N \times D}$ 和记忆 $\mathbf{M} \in \mathbb{R}^{B \times M \times D_m}$：

$$\mathbf{Q}' = \mathbf{Q} + \gamma \cdot \text{Proj}_{\text{out}}\left( \text{MHA}\left( \text{Proj}_Q(\mathbf{Q}), \text{Proj}_{KV}(\mathbf{M}), \text{Proj}_{KV}(\mathbf{M}) \right) \right)$$

其中 MHA 为 $H=8$ 头注意力，$\gamma$ 为门控标量，残差连接确保当前帧信息在加入时序上下文时得以保留。

记忆特征在交叉注意力前经过质量加权：$\mathbf{M} \leftarrow \mathbf{M} \odot \mathbf{s}$，抑制低置信度记忆。

### 5.2 门控调度

采用确定性预热调度策略：

$$\gamma(\tau) = \begin{cases} 0 & \tau < 0.10 \\ \gamma_{\max} \cdot \frac{\tau - 0.10}{0.20} & 0.10 \leq \tau < 0.30 \\ \gamma_{\max} & \tau \geq 0.30 \end{cases}$$

前 10% 迭代 $\gamma = 0$，模型先建立可靠单帧预测；随后 20% 迭代线性增长至最大值。设 $\gamma_{\max}^{\text{obj}} = 0.15$，$\gamma_{\max}^{\text{rel}} = 0.30$，关系分支取更大值反映谓词识别对时序上下文的更强依赖。

### 5.3 注入与预测头重算

注入后重跑对应预测头，克隆原始张量防止破坏 autograd 链中的中间结果。仅替换末层，中间层辅助损失保持原始解码器输出。

## 6. 记忆更新

### 6.1 候选构造

每帧前向结束后，从当前帧预测构造三元组候选。先完成坐标转换（cxcywh → xyxy），再计算三元组质量 $q_r = s_{\text{sub}}(r) \cdot s_{\text{obj}}(r) \cdot s_{\text{pred}}(r)$。选取 $q_r > 0.10$ 的候选（至多 $K_{\text{topk}}=16$ 个/帧）。

### 6.2 梯度隔离

整个记忆更新管线在 `torch.no_grad()` 下运行，编码器内显式 `.detach()` 谓词分布，所有写入槽位的张量经 `.detach().cpu()` 处理。三层梯度保护确保记忆更新不参与反向传播，避免跨迭代计算图导致显存爆炸。记忆数据存储在 CPU 上以节省 GPU 显存。

## 7. 实现说明

模块作为 SpeaQ 框架的轻量插件实现，关键接入点：(1) 模型构建时初始化 TripletMemoryManager；(2) Transformer 解码器输出后、ROI 精修与输出组装前调用注入方法；(3) 前向末尾执行记忆更新。

视频标识与帧索引从数据加载器经元架构传递至 Transformer 模块。按视频记忆库在首次访问时自动创建，帧序号倒退（表明视频切换）时自动清空。

模块通过 `MODEL.TEMPORAL.TRIPLET_MEMORY_*` 命名空间配置。所有功能默认关闭，确保与现有训练流程的向后兼容。

## 8. 相关工作

**场景图生成。** 基于 DETR 的 SGG 方法将关系检测重构为集合预测问题，通过二分匹配将预测分配至真值三元组。SpeaQ 在此基础上引入匈牙利匹配与质量感知多分配，提升三元组召回率。

**视频理解中的时序建模。** SAM 3 展示了记忆条件 Transformer 在视频目标分割中的有效性，采用以目标为单位的记忆库与交叉注意力融合及目标指针表示。本文的三元组记忆设计借鉴了该架构，并将其适配至场景图生成领域。

**记忆增强 Transformer。** 记忆网络及其 Transformer 变体广泛应用于需要长程上下文的任务。本文将这一范式应用于结构化视觉关系预测，以语义三元组而非原始特征或 token 为记忆粒度。

## 参考文献

[1] Carion N, Massa F, Synnaeve G, et al. End-to-End Object Detection with Transformers. *ECCV*, 2020.

[2] Carion N, et al. SAM 3: Segment Anything with Concepts. *arXiv:2511.16719*, 2025.

[3] Ravi N, et al. SAM 2: Segment Anything in Images and Videos. *ICLR*, 2025.

[4] Ji J, Krishna R, Fei-Fei L, et al. Action Genome: Actions As Compositions of Spatio-Temporal Scene Graphs. *CVPR*, 2020.

[5] Vaswani A, et al. Attention Is All You Need. *NeurIPS*, 2017.

[6] Cong Y, et al. SpeaQ: Quality-Aware Multi-Assignment for Scene Graph Generation. *CVPR*, 2023.

[7] Sukhbaatar S, Szlam A, Weston J, et al. End-To-End Memory Networks. *NeurIPS*, 2015.
