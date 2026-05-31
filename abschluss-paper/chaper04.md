# Chapter 4: Methodological Components for Efficient Video Scene Graph Generation

# 第四章：面向高效视频场景图生成的方法组件

---

## 4.1 Region-of-Interest Refinement at Native Feature Stride

## 4.1 基于原生特征步长的感兴趣区域细化

---

### 4.1.1 Problem Statement and Motivation

**EN:** In the DETR-based Scene Graph Generation (SGG) pipeline, object detection and relation prediction share a common transformer encoder-decoder architecture. The SAM3 visual backbone produces feature maps at a native stride of approximately 14 pixels (72 × 72 spatial resolution for a 1008 × 1008 input). While this stride provides a reasonable balance between spatial resolution and computational cost for the DETR encoder, the subsequent decoder processes object queries at a fixed, relatively coarse representation. Small objects—whose bounding boxes occupy a limited area in the original image—are particularly disadvantaged: their feature representations at stride 14 may contain only a handful of activated positions, making fine-grained localization and classification difficult.

Prior work has demonstrated that region-level feature refinement can substantially improve detection quality, especially for small instances. Mask R-CNN introduced RoIAlign, which extracts fixed-size feature descriptors from region proposals using bilinear interpolation, avoiding the quantization errors of earlier RoIPool layers and achieving significant gains at high IoU thresholds \cite{he2017mask}. Feature Pyramid Networks (FPN) route region proposals of different scales to feature maps at different strides, enabling small objects to be processed at higher spatial resolutions \cite{lin2017feature}. Both principles—spatially precise feature extraction and scale-adaptive feature routing—motivate the ROI refinement module introduced in this chapter.

**DE:** In der DETR-basierten Scene Graph Generation (SGG) Pipeline teilen sich Objekterkennung und Relationsvorhersage eine gemeinsame Transformer-Encoder-Decoder-Architektur. Das visuelle SAM3-Backbone erzeugt Feature-Maps mit einer nativen Schrittweite (Stride) von etwa 14 Pixeln (72 × 72 räumliche Auflösung bei 1008 × 1008 Eingabe). Während diese Schrittweite ein vernünftiges Gleichgewicht zwischen räumlicher Auflösung und Rechenaufwand für den DETR-Encoder bietet, verarbeitet der nachfolgende Decoder Objekt-Queries auf einer festen, relativ groben Repräsentation. Kleine Objekte – deren Bounding-Boxen nur eine begrenzte Fläche im Originalbild einnehmen – sind besonders benachteiligt: Ihre Feature-Repräsentationen bei Stride 14 können nur eine Handvoll aktivierter Positionen enthalten, was eine feinkörnige Lokalisierung und Klassifizierung erschwert.

Frühere Arbeiten haben gezeigt, dass die Verfeinerung von Merkmalen auf Regionsebene die Erkennungsqualität erheblich verbessern kann, insbesondere für kleine Instanzen. Mask R-CNN führte RoIAlign ein, das Feature-Deskriptoren fester Größe aus Regionsvorschlägen mittels bilinearer Interpolation extrahiert und so die Quantisierungsfehler früherer RoIPool-Schichten vermeidet \cite{he2017mask}. Feature Pyramid Networks (FPN) leiten Regionsvorschläge unterschiedlicher Skalen zu Feature-Maps mit unterschiedlichen Schrittweiten und ermöglichen so die Verarbeitung kleiner Objekte mit höherer räumlicher Auflösung \cite{lin2017feature}. Beide Prinzipien – räumlich präzise Feature-Extraktion und skalierungsadaptives Feature-Routing – motivieren das in diesem Kapitel vorgestellte ROI-Refinement-Modul.

---

### 4.1.2 Design Rationale and Architecture

**EN:** The ROI refinement module, denoted ROI14 throughout this thesis, operates as a lightweight post-detection refinement head. It is designed to complement—rather than replace—the standard DETR detection head. The key design decisions are as follows.

**1. Feature Source: Native Stride-14 Auxiliary Features.** The SAM3 backbone's forward pass caches intermediate features before any downsampling operation. These features are at the native SAM3 output stride of approximately 14 (72 × 72 spatial grid). Unlike the downsampled features fed to the DETR encoder (which are at stride 28 after pixel-unshuffle or average pooling), the native stride-14 features preserve finer spatial detail that is critical for small-object localization. The cached features are exposed via a dedicated `get_last_aux_features()` interface on the SAM3 backbone wrapper, returning a dictionary keyed by stride.

**2. RoI Pooling.** For each predicted bounding box from the DETR decoder—specifically boxes from the subject and object heads—the module extracts a fixed-size feature descriptor (default: 7 × 7) from the stride-14 feature map using RoIAlign-style bilinear interpolation. The pooled features are processed by a small convolutional head (two Conv2d layers with LayerNorm and ReLU activation) to produce a compact region representation.

**3. Classification-Only Refinement.** The ROI refinement head outputs refined classification logits for subject and object entities. It does not regress new bounding boxes; instead, it reuses the box coordinates from the DETR decoder. This design choice keeps the module lightweight and focuses its capacity on improving class discrimination for challenging cases—particularly small objects and tail categories. The refined logits replace the original logits only for entities whose predicted box area falls below a configurable threshold `SMALL_AREA_THRESH`.

**4. Learned Gating (Convex Fusion).** The final classification logit is a convex combination of the original decoder logit and the ROI-refined logit:

\begin{equation}
\text{logit}_{\text{final}} = \gamma \cdot \text{logit}_{\text{roi}} + (1-\gamma) \cdot \text{logit}_{\text{orig}},
\end{equation}

where $\gamma \in [0,1]$ is a learnable gating parameter. At initialization, $\gamma \approx 0$, meaning the model starts by trusting the original decoder prediction. During training, $\gamma$ can increase if ROI refinement proves beneficial, allowing the model to autonomously determine the optimal fusion ratio. This gating mechanism is inspired by the learned skip-connection gating in Gated Feedback RNNs \cite{chung2015gated}.

**Architecture Diagram (ASCII):**
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

**DE:** Das ROI-Refinement-Modul, in dieser Arbeit als ROI14 bezeichnet, arbeitet als leichtgewichtiger Post-Detection-Verfeinerungskopf. Es ist so konzipiert, dass es den standardmäßigen DETR-Erkennungskopf ergänzt – nicht ersetzt. Die wichtigsten Designentscheidungen sind wie folgt.

**1. Feature-Quelle: Native Stride-14 Hilfsfeatures.** Der Forward-Pass des SAM3-Backbones speichert Zwischen-Features vor jeder Downsampling-Operation zwischen. Diese Features haben die native SAM3-Ausgabe-Schrittweite von etwa 14 (72 × 72 räumliches Gitter). Im Gegensatz zu den heruntergerechneten Features, die dem DETR-Encoder zugeführt werden (Stride 28 nach Pixel-Unshuffle oder Average Pooling), bewahren die nativen Stride-14-Features feinere räumliche Details, die für die Lokalisierung kleiner Objekte entscheidend sind.

**2. RoI-Pooling.** Für jede vom DETR-Decoder vorhergesagte Bounding-Box extrahiert das Modul einen Feature-Deskriptor fester Größe (Standard: 7 × 7) aus der Stride-14-Feature-Map mittels bilinearer Interpolation im RoIAlign-Stil.

**3. Klassifikations-Only-Verfeinerung.** Der ROI-Refinement-Kopf gibt verfeinerte Klassifikations-Logits für Subjekt- und Objekt-Entitäten aus, ohne neue Bounding-Boxen zu regressieren. Die verfeinerten Logits ersetzen die ursprünglichen Logits nur für Entitäten, deren vorhergesagte Box-Fläche unter einem konfigurierbaren Schwellenwert `SMALL_AREA_THRESH` liegt.

**4. Lernbares Gating (Konvexe Fusion).** Das endgültige Klassifikations-Logit ist eine konvexe Kombination aus dem ursprünglichen Decoder-Logit und dem ROI-verfeinerten Logit (Gleichung 4.1).

---

### 4.1.3 Training and Inference

**EN:** The ROI refinement head is trained jointly with the main SGG objective. Its loss consists of cross-entropy classification losses for subject and object branches:
\begin{equation}
\mathcal{L}_{\text{roi}} = \mathcal{L}_{\text{roi\_subject\_cls}} + \mathcal{L}_{\text{roi\_object\_cls}}.
\end{equation}
The loss weight is controlled by `ROI_REFINE.LOSS_WEIGHT` (default: 1.0). At inference time, the refined logits replace the original logits in the NMS post-processing pipeline for qualifying entities.

**DE:** Der ROI-Refinement-Kopf wird gemeinsam mit dem Haupt-SGG-Ziel trainiert. Sein Verlust besteht aus Cross-Entropy-Klassifikationsverlusten für Subjekt- und Objektzweige (Gleichung 4.2). Zur Inferenzzeit ersetzen die verfeinerten Logits die ursprünglichen Logits in der NMS-Post-Processing-Pipeline für qualifizierende Entitäten.

---

## 4.2 Cross-Attention Temporal Query Injection with Learned Gating

## 4.2 Cross-Attention Temporale Query-Injektion mit lernbarem Gating

---

### 4.2.1 Problem Statement and Motivation

**EN:** Video Scene Graph Generation (Video SGG) extends the single-image SGG task to video sequences. In video data, adjacent frames exhibit strong temporal correlations: (1) *object persistence*—the same physical entity appears across multiple frames with stable appearance, location, and category; (2) *relation persistence*—spatial relationships and interaction patterns such as $\langle\text{person}, \text{holding}, \text{cup}\rangle$ typically remain unchanged or change smoothly between adjacent frames; (3) *complementary information*—individual frames may suffer from occlusion, motion blur, or viewpoint changes, while neighboring frames can provide supplementary cues. Effectively exploiting this temporal information can benefit both object detection and relation prediction sub-tasks.

A straightforward approach to temporal modeling is additive query injection (V1 design): historical object queries are projected and directly added to current-frame queries with a fixed hyperparameter controlling the blending ratio. This method has three limitations: (1) the fixed blending ratio cannot adapt to different queries' varying needs for temporal context; (2) a hard-replacement memory update (overwriting memory slots with new detections) causes abrupt memory state changes when detection quality fluctuates; (3) low-confidence detections are indiscriminately stored in memory, potentially propagating erroneous information to subsequent frames.

The V2 design proposed in this section addresses all three limitations through cross-attention fusion, exponential moving average (EMA) memory updates, and confidence-weighted memory retrieval. The total modification amounts to approximately 40 lines of code, does not alter the backbone or transformer structure, and is fully backward-compatible with the V1 configuration.

**DE:** Video Scene Graph Generation (Video SGG) erweitert die Einzelbild-SGG-Aufgabe auf Videosequenzen. In Videodaten weisen benachbarte Frames starke zeitliche Korrelationen auf. Ein einfacher Ansatz zur zeitlichen Modellierung ist die additive Query-Injektion (V1-Design): Historische Objekt-Queries werden projiziert und direkt zu den Queries des aktuellen Frames addiert, wobei ein fester Hyperparameter das Mischungsverhältnis steuert. Das in diesem Abschnitt vorgeschlagene V2-Design adressiert alle drei Einschränkungen durch Cross-Attention-Fusion, exponentielle gleitende Mittelwert (EMA) Speicheraktualisierungen und konfidenzgewichteten Speicherabruf.

---

### 4.2.2 Design Overview

**EN:** The temporal memory system consists of two cooperating components: (1) an *Object Memory Bank* that maintains a fixed-size set of learnable memory slots, each storing a historical object query feature, a confidence score, and a validity mask; and (2) a *Temporal Query Injector* that fuses memory features with current-frame queries through cross-attention with learned gating.

**Architecture Diagram (ASCII):**
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

**DE:** Das zeitliche Speichersystem besteht aus zwei kooperierenden Komponenten: (1) einer *Object Memory Bank*, die einen festen Satz von lernbaren Speicher-Slots verwaltet, und (2) einem *Temporal Query Injector*, der Speicher-Features mit aktuellen Frame-Queries durch Cross-Attention mit lernbarem Gating fusioniert.

---

### 4.2.3 Component 1: Cross-Attention Temporal Query Injector

**EN:** Let $\mathbf{Q} \in \mathbb{R}^{Q \times B \times D}$ denote the current frame's object queries and $\mathbf{M} \in \mathbb{R}^{B \times M \times D}$ denote the memory bank with $M$ slots retrieved from previous frames. The injector performs:

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

**DE:** Der Cross-Attention-Mechanismus folgt der skalierten Dot-Product-Attention-Formulierung aus der Transformer-Architektur \cite{vaswani2017attention}. Der Gating-Mechanismus ist von den gelernten Gates in Gated Recurrent Units (GRU) und Gated Feedback RNNs inspiriert \cite{chung2015gated, chung2014empirical}. Die Verwendung eines externen Speichers mit inhaltsbasierter Adressierung bezieht sich auf Memory Networks \cite{weston2015memory} und Neural Turing Machines \cite{graves2014neural}.

---

### 4.2.4 Component 2: EMA Memory Update with Matched-GT Filtering

**EN:** The memory bank is updated after each frame using an exponential moving average (EMA):

\begin{equation}
\mathbf{m}_s^{(t)} = \alpha \cdot \mathbf{m}_s^{(t-1)} + (1-\alpha) \cdot \mathbf{q}^{(t)},
\end{equation}
where $\alpha = 0.9$ is the EMA momentum and $\mathbf{q}^{(t)}$ is the new query feature. The EMA update acts as a temporal low-pass filter:
\begin{equation}
\mathbf{m}_s^{(t)} = \alpha^t \mathbf{m}_s^{(0)} + (1-\alpha) \sum_{i=1}^{t} \alpha^{t-i} \mathbf{q}^{(i)},
\end{equation}
giving recent frames higher weight with exponential decay.

A *matched-GT* filtering mechanism is applied before writing to memory: only predictions that can be associated with ground-truth objects (via IoU ≥ 0.5 and class agreement) are stored. This prevents low-quality or false-positive detections from polluting the memory bank. For each ground-truth target, only the strongest matching prediction is kept to avoid filling memory slots with duplicates.

**DE:** Die Memory Bank wird nach jedem Frame mit einem exponentiellen gleitenden Mittelwert (EMA) aktualisiert (Gleichung 4.10). Ein *Matched-GT*-Filtermechanismus wird vor dem Schreiben in den Speicher angewendet: Nur Vorhersagen, die mit Ground-Truth-Objekten assoziiert werden können, werden gespeichert.

---

### 4.2.5 Gating Constraints and Warmup Schedule

**EN:** To prevent the gate from oscillating during early training, bounded gating is employed:

\begin{equation}
\gamma_{\text{effective}} = \gamma_{\text{min}} + (\gamma_{\text{max}} - \gamma_{\text{min}}) \cdot \sigma(w_g) \cdot \lambda_{\text{warmup}}(t),
\end{equation}
where $\gamma_{\text{min}}, \gamma_{\text{max}}$ define the allowed range and $\lambda_{\text{warmup}}(t) = \min(t / T_{\text{warmup}}, 1)$ smoothly activates the gate over the first $T_{\text{warmup}}$ iterations. Separate gate parameters and bounds are maintained for object queries ($\gamma_{\text{max}} = 0.20$) and relation queries ($\gamma_{\text{max}} = 0.10$), reflecting the observation that relation predictions benefit less from entity-level temporal memory. The gate parameter receives a 5× learning rate multiplier to ensure it escapes the sigmoid saddle point at initialization.

**DE:** Um Oszillationen des Gates während des frühen Trainings zu verhindern, wird Bounded Gating eingesetzt (Gleichung 4.12). Separate Gate-Parameter und -Grenzen werden für Objekt-Queries ($\gamma_{\text{max}} = 0.20$) und Relations-Queries ($\gamma_{\text{max}} = 0.10$) verwaltet, was der Beobachtung Rechnung trägt, dass Relationsvorhersagen weniger von temporalem Entity-Speicher profitieren.

---

### 4.2.6 Video-Level State Isolation

**EN:** Each video maintains an independent memory state, keyed by `video_id` in a dictionary `MemoryStates`. When the video switches (detected via `CACHE_RESET_ON_VIDEO_SWITCH`), the memory state is automatically reset, preventing cross-video memory contamination. This design is essential for datasets like Action Genome \cite{ji2020action}, where videos are semantically independent.

**DE:** Jedes Video verwaltet einen unabhängigen Speicherzustand, der durch `video_id` in einem Dictionary `MemoryStates` indiziert ist. Beim Videowechsel wird der Speicherzustand automatisch zurückgesetzt, wodurch eine speicherübergreifende Kontamination zwischen Videos verhindert wird.

---

## 4.3 Pixel-Unshuffle Patch Merge for Information-Preserving Downsampling

## 4.3 Pixel-Unshuffle Patch Merge für informationserhaltendes Downsampling

---

### 4.3.1 Problem Statement and Motivation

**EN:** The SAM3 backbone, when operating at its standard input resolution of $1008 \times 1008$, produces feature maps of spatial size $72 \times 72$, corresponding to a stride of approximately 14 pixels. This yields $5{,}184$ visual tokens per image. DETR-series transformer encoders have $O(N^2)$ self-attention complexity in the number of input tokens $N$. With $N \approx 5{,}000$, the self-attention computation and memory consumption exceed the capacity of a single consumer GPU (24--48 GB).

The conventional solution is to apply $2 \times 2$ average pooling (stride 2) to reduce the spatial resolution from $72 \times 72$ to $36 \times 36$, cutting the token count by a factor of 4. However, average pooling irreversibly discards fine-grained spatial structure: each $2 \times 2$ window is compressed into a single scalar mean, permanently erasing the sub-pixel activation pattern. This information loss disproportionately harms small objects—whose feature activations may span only 1--3 spatial positions—and tail categories—whose sparse training samples provide weak gradient signals that are further diluted by pooling.

**DE:** Das SAM3-Backbone erzeugt bei seiner Standard-Eingabeauflösung von $1008 \times 1008$ Feature-Maps der räumlichen Größe $72 \times 72$, was einer Schrittweite von etwa 14 Pixeln entspricht. Dies ergibt $5{,}184$ visuelle Token pro Bild. Die konventionelle Lösung besteht darin, $2 \times 2$ Average Pooling (Stride 2) anzuwenden, um die räumliche Auflösung zu reduzieren. Average Pooling verwirft jedoch irreversible feinkörnige räumliche Struktur: Jedes $2 \times 2$-Fenster wird zu einem einzigen skalaren Mittelwert komprimiert, wodurch das Subpixel-Aktivierungsmuster dauerhaft gelöscht wird.

---

### 4.3.2 Proposed Method: Pixel-Unshuffle Patch Merge

**EN:** Inspired by the Pixel-Unshuffle Connector introduced in X-SAM \cite{xsam2026}, we propose replacing average pooling with a two-stage *space-to-depth* pipeline:

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

**Architecture Diagram (ASCII):**
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

**DE:** Inspiriert durch den Pixel-Unshuffle-Connector aus X-SAM \cite{xsam2026} wird Average Pooling durch eine zweistufige *Space-to-Depth*-Pipeline ersetzt. Stufe 1 (Pixel-Unshuffle) ist eine Bijektion ohne Informationsverlust. Stufe 2 ($1 \times 1$ Convolution) projiziert die erweiterte Kanaldimension zurück. Die Gewichte werden so initialisiert, dass sie zu Beginn des Trainings exakt das Average-Pooling-Verhalten nachahmen (Gleichung 4.17).

---

### 4.3.3 Integration and Configurable Behavior

**EN:** The patch merge module is integrated into the SAM3 backbone wrapper and controlled by a single boolean flag `USE_PATCH_MERGE`. When enabled, the backbone forward pass:

1. Extracts native stride-14 features from SAM3,
2. Applies an optional $1 \times 1$ projection layer to adjust channels,
3. Computes the native feature stride as $\text{round}(\text{input\_h} / \text{feature\_h})$,
4. If the native stride is smaller than `TARGET_STRIDE`, computes the downsampling factor $f = \text{TARGET\_STRIDE} / \text{native\_stride}$,
5. Applies pixel-unshuffle (if $f$ is a power of two) and the 1×1 conv; otherwise falls back to average pooling.

For the standard configuration with SAM3 native stride 14 and `TARGET_STRIDE = 32` (or equivalently 28), the factor is $f = 2$, yielding a $2 \times 2$ patch merge with $f^2 = 4\times$ channel expansion before projection.

**DE:** Das Patch-Merge-Modul ist in den SAM3-Backbone-Wrapper integriert und wird durch einen einzigen Boolean-Flag `USE_PATCH_MERGE` gesteuert. Für die Standardkonfiguration mit SAM3 nativer Schrittweite 14 und `TARGET_STRIDE = 32` beträgt der Faktor $f = 2$.

---

### 4.3.4 Relationship to Related Work

**EN:** The pixel-unshuffle operation was originally introduced as the inverse of pixel shuffle (sub-pixel convolution) for efficient image super-resolution, where it rearranges channel dimensions into spatial dimensions for upsampling \cite{shi2016real}. In this work, the operation is used in the opposite direction—rearranging spatial dimensions into channels for lossless downsampling—which is the same direction used by the Focus layer in YOLOv5 \cite{ultralytics2020yolov5} and the patch embedding in Vision Transformers \cite{dosovitskiy2021image}. The key distinction from prior work is the combination with a learnable 1×1 convolution initialized to mimic average pooling, which provides a smooth optimization landscape: the model can start from a known-good baseline and gradually discover non-uniform sub-pixel fusion patterns through gradient descent.

The most direct inspiration is the Pixel-Unshuffle Connector proposed in X-SAM (AAAI 2026) for segmentation tasks \cite{xsam2026}. While X-SAM uses this connector to bridge multi-scale features for mask prediction, this work adapts the same principle to the scene graph generation domain, where the downstream task is object detection and relation prediction rather than pixel-level segmentation. The training dynamics differ accordingly: in segmentation, the 1×1 conv receives dense pixel-level supervision; in SGG, it receives sparse object-level gradients propagated through the DETR decoder, making the learning signal weaker and the initialization strategy more critical.

**DE:** Die Pixel-Unshuffle-Operation wurde ursprünglich als Inverse von Pixel Shuffle für effiziente Bild-Super-Resolution eingeführt \cite{shi2016real}. Die direkteste Inspiration ist der in X-SAM (AAAI 2026) vorgeschlagene Pixel-Unshuffle-Connector für Segmentierungsaufgaben \cite{xsam2026}.

---

## References / Literaturverzeichnis

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

\bibitem{fu2025hybrid}
Z. Fu, Y. Liu, and R. Jin,
``Hybrid Reciprocal Transformer for Visual Relationship Detection,''
in \emph{Proc. AAAI Conference on Artificial Intelligence (AAAI)}, 2025.

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

\bibitem{rezatofighi2019giou}
H. Rezatofighi, N. Tsoi, J. Gwak, A. Sadeghian, I. Reid, and S. Savarese,
``Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression,''
in \emph{Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 2019.

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
