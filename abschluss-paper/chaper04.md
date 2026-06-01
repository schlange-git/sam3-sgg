# Chapter 4: Methodological Components for Efficient Video Scene Graph Generation

---

## 4.1 Region-of-Interest Refinement at Native Feature Stride

### 4.1.1 Problem Statement and Motivation

In the DETR-based Scene Graph Generation (SGG) pipeline, object detection and relation prediction share a common transformer encoder-decoder architecture. The SAM3 visual backbone produces feature maps at a native stride of approximately 14 pixels (72 × 72 spatial resolution for a 1008 × 1008 input). While this stride provides a reasonable balance between spatial resolution and computational cost for the DETR encoder, the subsequent decoder processes object queries at a fixed, relatively coarse representation. Small objects—whose bounding boxes occupy a limited area in the original image—are particularly disadvantaged: their feature representations at stride 14 may contain only a handful of activated positions, making fine-grained localization and classification difficult.

Prior work has demonstrated that region-level feature refinement can substantially improve detection quality, especially for small instances. Mask R-CNN introduced RoIAlign, which extracts fixed-size feature descriptors from region proposals using bilinear interpolation, avoiding the quantization errors of earlier RoIPool layers and achieving significant gains at high IoU thresholds \cite{he2017mask}. Feature Pyramid Networks (FPN) route region proposals of different scales to feature maps at different strides, enabling small objects to be processed at higher spatial resolutions \cite{lin2017feature}. Both principles—spatially precise feature extraction and scale-adaptive feature routing—motivate the ROI refinement module introduced in this chapter.

### 4.1.2 Design Rationale and Architecture

The ROI refinement module, denoted ROI14 throughout this thesis, operates as a lightweight post-detection refinement head. It is designed to complement—rather than replace—the standard DETR detection head. The key design decisions are as follows.

**1. Feature Source: Native Stride-14 Auxiliary Features.** The SAM3 backbone's forward pass caches intermediate features before any downsampling operation. These features are at the native SAM3 output stride of approximately 14 (72 × 72 spatial grid). Unlike the downsampled features fed to the DETR encoder (which are at stride 28 after pixel-unshuffle or average pooling), the native stride-14 features preserve finer spatial detail that is critical for small-object localization. The cached features are exposed via a dedicated `get_last_aux_features()` interface on the SAM3 backbone wrapper, returning a dictionary keyed by stride.

**2. RoI Pooling.** For each predicted bounding box from the DETR decoder—specifically boxes from the subject and object heads—the module extracts a fixed-size feature descriptor (default: 7 × 7) from the stride-14 feature map using RoIAlign-style bilinear interpolation. The pooled features are processed by a small convolutional head (two Conv2d layers with LayerNorm and ReLU activation) to produce a compact region representation.

**3. Classification-Only Refinement.** The ROI refinement head outputs refined classification logits for subject and object entities. It does not regress new bounding boxes; instead, it reuses the box coordinates from the DETR decoder. This design choice keeps the module lightweight and focuses its capacity on improving class discrimination for challenging cases—particularly small objects and tail categories. The refined logits replace the original logits only for entities whose predicted box area falls below a configurable threshold `SMALL_AREA_THRESH`.

**4. Learned Gating (Convex Fusion).** The final classification logit is a convex combination of the original decoder logit and the ROI-refined logit:

\begin{equation}
\text{logit}_{\text{final}} = \gamma \cdot \text{logit}_{\text{roi}} + (1-\gamma) \cdot \text{logit}_{\text{orig}},
\end{equation}

where $\gamma \in [0,1]$ is a learnable gating parameter. At initialization, $\gamma \approx 0$, meaning the model starts by trusting the original decoder prediction. During training, $\gamma$ can increase if ROI refinement proves beneficial, allowing the model to autonomously determine the optimal fusion ratio. This gating mechanism is inspired by the learned skip-connection gating in Gated Feedback RNNs \cite{chung2015gated}.

**Architecture Diagram:**
```
DETR Decoder Output
  │
  ├── subject_boxes, object_boxes (cxcywh)
  ├── subject_logits, object_logits
  │
  ▼
ROI Refine Head:
  ┌─────────────────────────────────┐
  │ 1. Box Projection (cxcywh→xyxy) │
  │ 2. RoIAlign (stride-14 feat)   │  ← get_last_aux_features()
  │    pool_size=7×7                │
  │ 3. Conv Head (2× Conv2d+LN+ReLU)│
  │ 4. Linear Classifier            │
  │ 5. Learned Gate γ (sigmoid)     │
  │    logit = γ·logit_roi + (1-γ)·logit_orig │
  └─────────────────────────────────┘
  │
  ▼
Refined subject_logits, object_logits → NMS → Final Detections
```

### 4.1.3 Training and Inference

The ROI refinement head is trained jointly with the main SGG objective. Its loss consists of cross-entropy classification losses for subject and object branches:
\begin{equation}
\mathcal{L}_{\text{roi}} = \mathcal{L}_{\text{roi\_subject\_cls}} + \mathcal{L}_{\text{roi\_object\_cls}}.
\end{equation}
The loss weight is controlled by `ROI_REFINE.LOSS_WEIGHT` (default: 1.0). At inference time, the refined logits replace the original logits in the NMS post-processing pipeline for qualifying entities.

---

## 4.2 Cross-Attention Temporal Query Injection with Learned Gating

### 4.2.1 Problem Statement and Motivation

Video Scene Graph Generation (Video SGG) extends the single-image SGG task to video sequences. In video data, adjacent frames exhibit strong temporal correlations: (1) *object persistence*—the same physical entity appears across multiple frames with stable appearance, location, and category; (2) *relation persistence*—spatial relationships and interaction patterns such as $\langle\text{person}, \text{holding}, \text{cup}\rangle$ typically remain unchanged or change smoothly between adjacent frames; (3) *complementary information*—individual frames may suffer from occlusion, motion blur, or viewpoint changes, while neighboring frames can provide supplementary cues. Effectively exploiting this temporal information can benefit both object detection and relation prediction sub-tasks.

A straightforward approach to temporal modeling is additive query injection (V1 design): historical object queries are projected and directly added to current-frame queries with a fixed hyperparameter controlling the blending ratio. This method has three limitations: (1) the fixed blending ratio cannot adapt to different queries' varying needs for temporal context; (2) a hard-replacement memory update (overwriting memory slots with new detections) causes abrupt memory state changes when detection quality fluctuates; (3) low-confidence detections are indiscriminately stored in memory, potentially propagating erroneous information to subsequent frames.

The V2 design proposed in this section addresses all three limitations through cross-attention fusion, exponential moving average (EMA) memory updates, and confidence-weighted memory retrieval. The total modification amounts to approximately 40 lines of code, does not alter the backbone or transformer structure, and is fully backward-compatible with the V1 configuration.

### 4.2.2 Design Overview

The temporal memory system consists of two cooperating components: (1) an *Object Memory Bank* that maintains a fixed-size set of learnable memory slots, each storing a historical object query feature, a confidence score, and a validity mask; and (2) a *Temporal Query Injector* that fuses memory features with current-frame queries through cross-attention with learned gating.

**Architecture Diagram:**
```
Frame t-1                          Frame t
┌──────────┐                      ┌──────────┐
│ DETR     │                      │ DETR     │
│ Decoder  │                      │ Decoder  │
│ Output:  │                      │ Query:   │
│ hs, ho   │──► Memory Update ──► │ Q_subject│
│ scores   │    (EMA + matched_gt)│ Q_object │
└──────────┘                      │ Q_rel    │
                                  └────┬─────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │  Temporal Query Injector            │
                    │                                     │
                    │  Q_proj ──► Cross-Attn(Q, K, V) ──► │
                    │              ▲                       │
                    │  K_proj ─────┘                       │
                    │  V_proj ─────┘                       │
                    │              │                       │
                    │  Memory Bank (M slots × D dim)      │
                    │  ┌─────────────────────────┐        │
                    │  │ m₀: [feat, score, valid]│        │
                    │  │ m₁: [feat, score, valid]│        │
                    │  │ ...                      │        │
                    │  │ m₃₁: [feat, score, valid]│       │
                    │  └─────────────────────────┘        │
                    │                                     │
                    │  Gate: γ·mem_out + (1-γ)·Q_base     │
                    │  γ = sigmoid(gate_param)            │
                    └─────────────────────────────────────┘
                                       │
                                       ▼
                              Injected Q_subject, Q_object
                              (optionally Q_relation)
```

### 4.2.3 Component 1: Cross-Attention Temporal Query Injector

Let $\mathbf{Q} \in \mathbb{R}^{Q \times B \times D}$ denote the current frame's object queries and $\mathbf{M} \in \mathbb{R}^{B \times M \times D}$ denote the memory bank with $M$ slots retrieved from previous frames. The injector performs:

**Step 1 — Cross-Attention:**
\begin{equation}
\begin{aligned}
\mathbf{Q}_{\text{proj}} &= \text{Linear}_Q(\mathbf{Q}), \\
\mathbf{K}_{\text{proj}} &= \text{Linear}_K(\mathbf{M}), \\
\mathbf{V}_{\text{proj}} &= \text{Linear}_V(\mathbf{M}).
\end{aligned}
\end{equation}
\begin{equation}
\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}_{\text{proj}} \mathbf{K}_{\text{proj}}^\top}{\sqrt{D}}\right) \mathbf{V}_{\text{proj}}.
\end{equation}

Each object query attends to all $M$ memory slots, allowing flexible retrieval of relevant historical information. The attention output is projected through a final linear layer:
\begin{equation}
\mathbf{Q}_{\text{mem}} = \text{Linear}_O(\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})).
\end{equation}

**Step 2 — Learned Gating:**
\begin{equation}
\gamma = \sigma(w_g), \quad w_g \in \mathbb{R},
\end{equation}
\begin{equation}
\mathbf{Q}_{\text{out}} = \gamma \cdot \mathbf{Q}_{\text{mem}} + (1-\gamma) \cdot \mathbf{Q}_{\text{base}}.
\end{equation}

The gating parameter $w_g$ is initialized to $0$ (yielding $\gamma \approx 0.5$), giving equal weight to memory and current queries at the start of training. Through gradient-based optimization, $w_g$ adapts to learn the optimal fusion ratio. The residual connection $\mathbf{Q}_{\text{out}} = \gamma \mathbf{Q}_{\text{mem}} + (1-\gamma) \mathbf{Q}_{\text{base}}$ ensures that if the memory output is unhelpful, the model can degenerate to pure object queries ($\gamma \to 0$).

**Relation to Prior Work:** The cross-attention mechanism follows the scaled dot-product attention formulation from the Transformer architecture \cite{vaswani2017attention}. The gating mechanism is inspired by the learned gates in Gated Recurrent Units (GRU) and Gated Feedback RNNs \cite{chung2015gated, chung2014empirical}. The use of external memory with content-based addressing relates to Memory Networks \cite{weston2015memory} and Neural Turing Machines \cite{graves2014neural}, though our memory bank is significantly simpler (fixed-size slots with EMA-based writing instead of differentiable read-write heads).

### 4.2.4 Component 2: EMA Memory Update with Matched-GT Filtering

The memory bank is updated after each frame using an exponential moving average (EMA):

\begin{equation}
\mathbf{m}_s^{(t)} = \alpha \cdot \mathbf{m}_s^{(t-1)} + (1-\alpha) \cdot \mathbf{q}^{(t)},
\end{equation}
where $\alpha = 0.9$ is the EMA momentum and $\mathbf{q}^{(t)}$ is the new query feature. The EMA update acts as a temporal low-pass filter:
\begin{equation}
\mathbf{m}_s^{(t)} = \alpha^t \mathbf{m}_s^{(0)} + (1-\alpha) \sum_{i=1}^{t} \alpha^{t-i} \mathbf{q}^{(i)},
\end{equation}
giving recent frames higher weight with exponential decay.

A *matched-GT* filtering mechanism is applied before writing to memory: only predictions that can be associated with ground-truth objects (via IoU ≥ 0.5 and class agreement) are stored. This prevents low-quality or false-positive detections from polluting the memory bank. For each ground-truth target, only the strongest matching prediction is kept to avoid filling memory slots with duplicates.

### 4.2.5 Gating Constraints and Warmup Schedule

To prevent the gate from oscillating during early training, bounded gating is employed:

\begin{equation}
\gamma_{\text{effective}} = \gamma_{\text{min}} + (\gamma_{\text{max}} - \gamma_{\text{min}}) \cdot \sigma(w_g) \cdot \lambda_{\text{warmup}}(t),
\end{equation}
where $\gamma_{\text{min}}, \gamma_{\text{max}}$ define the allowed range and $\lambda_{\text{warmup}}(t) = \min(t / T_{\text{warmup}}, 1)$ smoothly activates the gate over the first $T_{\text{warmup}}$ iterations. Separate gate parameters and bounds are maintained for object queries ($\gamma_{\text{max}} = 0.20$) and relation queries ($\gamma_{\text{max}} = 0.10$), reflecting the observation that relation predictions benefit less from entity-level temporal memory. The gate parameter receives a 5× learning rate multiplier to ensure it escapes the sigmoid saddle point at initialization.

### 4.2.6 Video-Level State Isolation

Each video maintains an independent memory state, keyed by `video_id` in a dictionary `MemoryStates`. When the video switches (detected via `CACHE_RESET_ON_VIDEO_SWITCH`), the memory state is automatically reset, preventing cross-video memory contamination. This design is essential for datasets like Action Genome \cite{ji2020action}, where videos are semantically independent.

---

## 4.3 Pixel-Unshuffle Patch Merge for Information-Preserving Downsampling

### 4.3.1 Problem Statement and Motivation

The SAM3 backbone, when operating at its standard input resolution of $1008 \times 1008$, produces feature maps of spatial size $72 \times 72$, corresponding to a stride of approximately 14 pixels. This yields $5{,}184$ visual tokens per image. DETR-series transformer encoders have $O(N^2)$ self-attention complexity in the number of input tokens $N$. With $N \approx 5{,}000$, the self-attention computation and memory consumption exceed the capacity of a single consumer GPU (24--48 GB).

The conventional solution is to apply $2 \times 2$ average pooling (stride 2) to reduce the spatial resolution from $72 \times 72$ to $36 \times 36$, cutting the token count by a factor of 4. However, average pooling irreversibly discards fine-grained spatial structure: each $2 \times 2$ window is compressed into a single scalar mean, permanently erasing the sub-pixel activation pattern. This information loss disproportionately harms small objects—whose feature activations may span only 1--3 spatial positions—and tail categories—whose sparse training samples provide weak gradient signals that are further diluted by pooling.

### 4.3.2 Proposed Method: Pixel-Unshuffle Patch Merge

Inspired by the Pixel-Unshuffle Connector introduced in X-SAM \cite{xsam2026}, we propose replacing average pooling with a two-stage *space-to-depth* pipeline:

**Stage 1 — Lossless Rearrangement (Pixel-Unshuffle):** The $2 \times 2$ spatial neighborhood is folded into the channel dimension via the pixel-unshuffle operation (equivalent to space-to-depth):
\begin{equation}
\mathbb{R}^{B \times C \times H \times W} \xrightarrow{\text{pixel\_unshuffle}(f=2)} \mathbb{R}^{B \times (C \cdot f^2) \times (H/f) \times (W/f)}.
\end{equation}
This is a bijection—every scalar value is preserved, with zero information loss. The operation is the inverse of the pixel shuffle operation introduced for image super-resolution \cite{shi2016real}, and is analogous to the Focus layer in YOLOv5 \cite{ultralytics2020yolov5}.

**Stage 2 — Learnable Projection (1×1 Convolution):** A $1 \times 1$ convolution projects the expanded channel dimension back to the target dimension:
\begin{equation}
\mathbf{F}^{\text{patch}}_{b,c,i,j} = \sum_{k=0}^{C \cdot f^2 - 1} \mathbf{W}_{c,k} \cdot \mathbf{F}^{\prime}_{b,k,i,j},
\end{equation}
where $\mathbf{W} \in \mathbb{R}^{C \times (C \cdot f^2)}$. 

**Weight Initialization (Critical Design):** The convolution weights are initialized to exactly mimic average pooling behavior at the start of training:
\begin{equation}
\mathbf{W}_{c, k} = \begin{cases} \frac{1}{f^2}, & \text{if } k = c + (\text{dy} \cdot f + \text{dx}) \cdot C \\ 0, & \text{otherwise} \end{cases}
\end{equation}
for $\text{dy}, \text{dx} \in [0, f-1]$. Under this initialization, the pixel-unshuffle + 1×1 conv pipeline produces numerically identical output to average pooling, ensuring that training starts from a known, stable baseline. During training, the 1×1 convolution weights are always kept trainable (`requires_grad=True`) regardless of the SAM3 backbone freeze state, allowing the model to learn non-uniform sub-pixel fusion weights that can amplify discriminative features for small objects.

**Architecture Diagram:**
```
Baseline (avg_pool2d):
  [B, 256, 72, 72] ── avg_pool2d(2×2, stride=2) ──► [B, 256, 36, 36]
  Information: 4→1 compression per window. Irreversible.

Proposed (Patch Merge):
  [B, 256, 72, 72]
      │
      ▼ pixel_unshuffle(factor=2)
  [B, 1024, 36, 36]     ← lossless rearrangement
      │
      ▼ 1×1 Conv(1024→256), init=avg_pool equivalent
  [B, 256, 36, 36]
      │
      ▼ gradient descent updates 1×1 conv weights
  Final: learned sub-pixel fusion, information-preserving
```

### 4.3.3 Integration and Configurable Behavior

The patch merge module is integrated into the SAM3 backbone wrapper and controlled by a single boolean flag `USE_PATCH_MERGE`. When enabled, the backbone forward pass:

1. Extracts native stride-14 features from SAM3,
2. Applies an optional $1 \times 1$ projection layer to adjust channels,
3. Computes the native feature stride as $\text{round}(\text{input\_h} / \text{feature\_h})$,
4. If the native stride is smaller than `TARGET_STRIDE`, computes the downsampling factor $f = \text{TARGET\_STRIDE} / \text{native\_stride}$,
5. Applies pixel-unshuffle (if $f$ is a power of two) and the 1×1 conv; otherwise falls back to average pooling.

For the standard configuration with SAM3 native stride 14 and `TARGET_STRIDE = 32` (or equivalently 28), the factor is $f = 2$, yielding a $2 \times 2$ patch merge with $f^2 = 4\times$ channel expansion before projection.

### 4.3.4 Relationship to Related Work

The pixel-unshuffle operation was originally introduced as the inverse of pixel shuffle (sub-pixel convolution) for efficient image super-resolution, where it rearranges channel dimensions into spatial dimensions for upsampling \cite{shi2016real}. In this work, the operation is used in the opposite direction—rearranging spatial dimensions into channels for lossless downsampling—which is the same direction used by the Focus layer in YOLOv5 \cite{ultralytics2020yolov5} and the patch embedding in Vision Transformers \cite{dosovitskiy2021image}. The key distinction from prior work is the combination with a learnable 1×1 convolution initialized to mimic average pooling, which provides a smooth optimization landscape: the model can start from a known-good baseline and gradually discover non-uniform sub-pixel fusion patterns through gradient descent.

The most direct inspiration is the Pixel-Unshuffle Connector proposed in X-SAM (AAAI 2026) for segmentation tasks \cite{xsam2026}. While X-SAM uses this connector to bridge multi-scale features for mask prediction, this work adapts the same principle to the scene graph generation domain, where the downstream task is object detection and relation prediction rather than pixel-level segmentation. The training dynamics differ accordingly: in segmentation, the 1×1 conv receives dense pixel-level supervision; in SGG, it receives sparse object-level gradients propagated through the DETR decoder, making the learning signal weaker and the initialization strategy more critical.

---

## References

\begin{thebibliography}{99}

\bibitem{carion2020detr}
N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko,
``End-to-End Object Detection with Transformers,''
in \emph{Proc. European Conference on Computer Vision (ECCV)}, 2020.

\bibitem{chung2014empirical}
J. Chung, C. Gulcehre, K. Cho, and Y. Bengio,
``Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling,''
in \emph{NeurIPS Workshop on Deep Learning}, 2014.

\bibitem{chung2015gated}
J. Chung, C. Gulcehre, K. Cho, and Y. Bengio,
``Gated Feedback Recurrent Neural Networks,''
in \emph{Proc. International Conference on Machine Learning (ICML)}, 2015.

\bibitem{dosovitskiy2021image}
A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby,
``An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,''
in \emph{Proc. International Conference on Learning Representations (ICLR)}, 2021.

\bibitem{graves2014neural}
A. Graves, G. Wayne, and I. Danihelka,
``Neural Turing Machines,''
arXiv preprint arXiv:1410.5401, 2014.

\bibitem{he2017mask}
K. He, G. Gkioxari, P. Dollár, and R. Girshick,
``Mask R-CNN,''
in \emph{Proc. IEEE International Conference on Computer Vision (ICCV)}, 2017.

\bibitem{ji2020action}
J. Ji, R. Krishna, L. Fei-Fei, and J. C. Niebles,
``Action Genome: Actions as Compositions of Spatio-Temporal Scene Graphs,''
in \emph{Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 2020.

\bibitem{kim2024speaq}
J. Kim, J. Park, J. Lee, and S. Kim,
``Groupwise Query Specialization and Quality-Aware Multi-Assignment for Transformer-based Visual Relationship Detection,''
in \emph{Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 2024.

\bibitem{lin2017feature}
T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie,
``Feature Pyramid Networks for Object Detection,''
in \emph{Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 2017.

\bibitem{shi2016real}
W. Shi, J. Caballero, F. Huszár, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert, and Z. Wang,
``Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network,''
in \emph{Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 2016.

\bibitem{ultralytics2020yolov5}
Ultralytics,
``YOLOv5,''
\emph{GitHub Repository}, 2020. [Online]. Available: \texttt{https://github.com/ultralytics/yolov5}

\bibitem{vaswani2017attention}
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin,
``Attention Is All You Need,''
in \emph{Proc. Advances in Neural Information Processing Systems (NeurIPS)}, 2017.

\bibitem{weston2015memory}
J. Weston, S. Chopra, and A. Bordes,
``Memory Networks,''
in \emph{Proc. International Conference on Learning Representations (ICLR)}, 2015.

\bibitem{xsam2026}
X. Chen, Y. Wang, Z. Li, H. Lu, and S. Liu,
``X-SAM: From Segment Anything to Any Segmentation,''
in \emph{Proc. AAAI Conference on Artificial Intelligence (AAAI)}, 2026.

\end{thebibliography}

---

# 第四章：面向高效视频场景图生成的方法组件

---

## 4.1 基于原生特征步长的感兴趣区域细化

### 4.1.1 问题定义与研究动机

在基于DETR的场景图生成流水线中，目标检测与关系预测共享一个公共的Transformer编码器-解码器架构。SAM3视觉骨干网络在标准输入分辨率$1008 \times 1008$下输出空间尺寸为$72 \times 72$的特征图，对应的原生步长约为14像素。虽然该步长为DETR编码器提供了空间分辨率与计算开销之间的合理平衡，但后续的解码器在固定且相对粗糙的表示上处理目标查询。小目标——其边界框在原始图像中占据有限面积——尤其不利：它们在步长14下的特征表示可能仅包含少量激活位置，使得细粒度定位和分类变得困难。

先前工作已证明，区域级特征细化可以显著提升检测质量，尤其对于小实例。Mask R-CNN引入了RoIAlign，通过双线性插值从区域提议中提取固定尺寸的特征描述符，避免了早期RoIPool层的量化误差，并在高IoU阈值下取得了显著增益\cite{he2017mask}。特征金字塔网络将不同尺度的区域提议路由到不同步长的特征图，使小目标能够在更高空间分辨率下处理\cite{lin2017feature}。这两个原则——空间精确的特征提取和尺度自适应的特征路由——共同motivate了本章引入的ROI细化模块。

### 4.1.2 设计思路与架构

ROI细化模块（本文中记为ROI14）作为一个轻量级的后检测细化头运行。它旨在补充——而非替代——标准的DETR检测头。关键设计决策如下。

**1. 特征来源：原生步长14的辅助特征。** SAM3骨干网络的前向传播在任何下采样操作之前缓存中间特征。这些特征处于SAM3的原生输出步长约14（72×72空间网格）。与被馈送到DETR编码器的下采样特征（经过像素反洗牌或平均池化后步长为28）不同，原生步长14的特征保留了更精细的空间细节，这对小目标定位至关重要。缓存的特征通过SAM3骨干网络封装器上的专用`get_last_aux_features()`接口暴露，返回一个以步长为键的字典。

**2. RoI池化。** 对于DETR解码器输出的每个预测边界框——特别是来自主体和客体头的边界框——该模块使用RoIAlign风格的双线性插值从步长14的特征图中提取固定尺寸的特征描述符（默认：7×7）。池化后的特征经过一个小型卷积头（两层Conv2d，配合LayerNorm和ReLU激活）处理，产生紧凑的区域表示。

**3. 仅分类细化。** ROI细化头输出主体和客体实体的细化分类logit。它不回归新的边界框；而是复用DETR解码器的框坐标。这一设计选择保持模块轻量化，并将其容量集中于改善具有挑战性情况的类别判别——特别是小目标和尾类别。细化后的logit仅替换那些预测框面积低于可配置阈值`SMALL_AREA_THRESH`的实体的原始logit。

**4. 可学习门控（凸融合）。** 最终分类logit是原始解码器logit与ROI细化logit的凸组合（公式4.1），其中$\gamma \in [0,1]$是可学习的门控参数。初始化时$\gamma \approx 0$，意味着模型从信任原始解码器预测开始。训练过程中，如果ROI细化被证明有益，$\gamma$可以增大，使模型能够自主确定最优融合比例。该门控机制受门控反馈RNN中可学习跳跃连接门控的启发\cite{chung2015gated}。

### 4.1.3 训练与推理

ROI细化头与主SGG目标联合训练。其损失由主体和客体分支的交叉熵分类损失组成（公式4.2）。损失权重由`ROI_REFINE.LOSS_WEIGHT`控制（默认：1.0）。推理时，细化后的logit在NMS后处理流水线中替换合格实体的原始logit。

---

## 4.2 基于交叉注意力的时序查询注入与可学习门控

### 4.2.1 问题定义与研究动机

视频场景图生成将单图像SGG任务扩展到视频序列。在视频数据中，相邻帧表现出强时序相关性：（1）目标持续性——同一物理实体在多个帧中出现，外观、位置和类别保持稳定；（2）关系持续性——空间关系和交互模式（如⟨人，持，杯子⟩）通常在相邻帧之间不变或平滑变化；（3）互补信息——单帧可能因遮挡、运动模糊或视角变化丢失部分信息，而相邻帧可提供补充线索。有效利用这些时序信息可以使目标检测和关系预测两个子任务都受益。

一种直接的时序建模方法是加法查询注入（V1方案）：历史目标查询被投影后直接添加到当前帧查询中，由一个固定超参数控制混合比例。该方法存在三个局限：（1）固定混合比例无法适应不同查询对时序上下文的不同需求；（2）硬替换记忆更新（用新检测覆盖记忆槽）在检测质量波动时导致记忆状态突变；（3）低置信度检测被无差别存入记忆，可能在后续帧中传播错误信息。

本节提出的V2方案通过交叉注意力融合、指数移动平均记忆更新和置信度加权记忆检索来解决全部三个局限。总修改量约为40行代码，不改变骨干网络和Transformer结构，完全向后兼容V1配置。

### 4.2.2 设计概述

时序记忆系统由两个协作组件组成：（1）一个对象记忆库，维护固定大小的可学习记忆槽集合，每个槽存储历史目标查询特征、置信度分数和有效性掩码；（2）一个时序查询注入器，通过带学习门控的交叉注意力将记忆特征与当前帧查询融合。

### 4.2.3 组件一：交叉注意力时序查询注入器

设$\mathbf{Q} \in \mathbb{R}^{Q \times B \times D}$表示当前帧的目标查询，$\mathbf{M} \in \mathbb{R}^{B \times M \times D}$表示从前序帧检索的具有$M$个槽的记忆库。

第一步——交叉注意力（公式4.3—4.5）。每个目标查询关注所有$M$个记忆槽，允许灵活检索相关历史信息。

第二步——学习门控（公式4.6—4.7）。门控参数$w_g$初始化为$0$（得到$\gamma \approx 0.5$），在训练开始时给予记忆和当前查询相等权重。通过基于梯度的优化，$w_g$自适应学习最优融合比例。残差连接确保如果记忆输出无帮助，模型可以退化为纯目标查询（$\gamma \to 0$）。

**与先前工作的关系：** 交叉注意力机制遵循Transformer架构中的缩放点积注意力公式\cite{vaswani2017attention}。门控机制受门控循环单元和门控反馈RNN中可学习门控的启发\cite{chung2015gated, chung2014empirical}。使用具有基于内容寻址的外部记忆与记忆网络\cite{weston2015memory}和神经图灵机\cite{graves2014neural}相关，但我们的记忆库显著更简单（固定大小槽，基于EMA的写入而非可微读写头）。

### 4.2.4 组件二：EMA记忆更新与匹配GT过滤

记忆库在每帧后使用指数移动平均更新（公式4.8—4.9），其中$\alpha = 0.9$是EMA动量。EMA更新的作用相当于时序低通滤波，给予近期帧更高权重并呈指数衰减。

在写入记忆之前应用匹配GT过滤机制：只有能够与真实标注目标关联（通过IoU≥0.5且类别一致）的预测才被存储。这防止低质量或假阳性检测污染记忆库。对于每个真实标注目标，仅保留最强匹配预测以避免重复填充记忆槽。

### 4.2.5 门控约束与预热调度

为防止门控在早期训练中振荡，采用有界门控（公式4.10），其中$\gamma_{\text{min}}, \gamma_{\text{max}}$定义允许范围，$\lambda_{\text{warmup}}(t) = \min(t / T_{\text{warmup}}, 1)$在前$T_{\text{warmup}}$次迭代中平滑激活门控。对目标查询（$\gamma_{\text{max}} = 0.20$）和关系查询（$\gamma_{\text{max}} = 0.10$）分别维护门控参数和界限，反映了关系预测从实体级时序记忆中获益较少的观察。门控参数获得5倍学习率乘数，以确保其逃离初始化时的sigmoid鞍点。

### 4.2.6 视频级状态隔离

每个视频维护独立的记忆状态，以`video_id`为键存储在`MemoryStates`字典中。当视频切换时，记忆状态自动重置，防止跨视频记忆污染。这一设计对于Action Genome等视频在语义上独立的数据集至关重要\cite{ji2020action}。

---

## 4.3 基于像素反洗牌的块合并用于信息保持下采样

### 4.3.1 问题定义与研究动机

SAM3骨干网络在标准输入分辨率$1008 \times 1008$下产生空间尺寸为$72 \times 72$的特征图，对应步长约14像素。这导致每张图像产生5184个视觉token。DETR系列Transformer编码器的自注意力复杂度为$O(N^2)$。当$N \approx 5000$时，自注意力计算和内存消耗超出单张消费级GPU（24--48 GB）的容量。

传统解决方案是应用$2 \times 2$平均池化（步长2）将空间分辨率从$72 \times 72$降至$36 \times 36$，将token数量减少4倍。然而，平均池化不可逆地丢弃了细粒度空间结构：每个$2 \times 2$窗口被压缩为单一标量均值，永久抹除了子像素激活模式。这种信息损失对小目标（其特征激活可能仅跨越1--3个空间位置）和尾类别（稀疏训练样本提供的梯度信号被池化进一步稀释）的伤害尤为严重。

### 4.3.2 提出方法：像素反洗牌块合并

受X-SAM中引入的像素反洗牌连接器启发\cite{xsam2026}，我们提出用两阶段空间到深度流水线替代平均池化。

**第一阶段——无损重排（像素反洗牌）：** $2 \times 2$空间邻域通过像素反洗牌操作（等价于space-to-depth）折叠到通道维度（公式4.11）。这是一个双射——每个标量值都被保留，零信息损失。该操作是图像超分辨率中引入的像素洗牌操作的逆操作\cite{shi2016real}，且类似于YOLOv5中的Focus层\cite{ultralytics2020yolov5}。

**第二阶段——可学习投影（1×1卷积）：** 一个$1 \times 1$卷积将扩展的通道维度投影回目标维度（公式4.12）。

**权重初始化（关键设计）：** 卷积权重被初始化以精确模拟训练开始时的平均池化行为（公式4.13）。在此初始化下，像素反洗牌+1×1卷积流水线产生与平均池化数值相同的输出，确保训练从一个已知的稳定基线开始。训练过程中，1×1卷积权重始终保持可训练（`requires_grad=True`），无论SAM3骨干网络的冻结状态如何，使模型能够学习可放大对小目标具有判别性特征的非均匀子像素融合权重。

### 4.3.3 集成与可配置行为

块合并模块集成在SAM3骨干网络封装器中，由单个布尔标志`USE_PATCH_MERGE`控制。启用时，骨干网络前向传播依次执行：从SAM3提取原生步长14特征、应用可选的1×1投影层调整通道、计算原生特征步长、如原生步长小于目标步长则计算下采样因子、应用像素反洗牌和1×1卷积（如果因子是2的幂），否则回退到平均池化。对于标准配置（SAM3原生步长14且`TARGET_STRIDE = 32`或等价地28），因子$f = 2$，产生$2 \times 2$块合并，投影前通道扩展4倍。

### 4.3.4 与相关工作的关系

像素反洗牌操作最初作为像素洗牌（子像素卷积）的逆操作被引入，用于高效的图像超分辨率\cite{shi2016real}。本工作中，该操作被用于相反方向——将空间维度重排为通道以实现无损下采样——这与YOLOv5中的Focus层\cite{ultralytics2020yolov5}和Vision Transformer中的块嵌入\cite{dosovitskiy2021image}使用相同的方向。与先前工作的关键区别在于与初始化为模拟平均池化的可学习1×1卷积的组合，这提供了平滑的优化景观：模型可以从已知的良好基线开始，通过梯度下降逐步发现非均匀子像素融合模式。

最直接的灵感是X-SAM（AAAI 2026）中为分割任务提出的像素反洗牌连接器\cite{xsam2026}。X-SAM使用该连接器桥接多尺度特征以进行掩码预测，而本工作将相同原理适配到场景图生成领域，其中下游任务是目标检测和关系预测而非像素级分割。训练动力学相应不同：在分割中，1×1卷积接收密集的像素级监督；在SGG中，它接收通过DETR解码器传播的稀疏目标级梯度，使学习信号更弱，初始化策略更为关键。

---

## 参考文献

（见英文部分参考文献列表）

