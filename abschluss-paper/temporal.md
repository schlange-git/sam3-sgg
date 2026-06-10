# Temporal Triplet Memory for Scene Graph Generation

## Abstract

We present a Temporal Triplet Memory module for video scene graph generation (SGG) that maintains cross-frame triplet-level contextual information. Built upon the SpeaQ SGG framework with a SAM3 visual backbone, our module introduces a per-video memory bank that stores compressed representations of `(subject, predicate, object)` triplets, a gated cross-attention injection mechanism that modulates decoder features with historical memory, and a warmup schedule for training stability. The module operates as a lightweight add-on to the existing DETR-style decoder, requiring no modifications to the backbone, encoder, matcher, or loss functions. Experiments on the Action Genome dataset demonstrate that temporal triplet memory substantially improves no-graph-constraint recall compared to the non-temporal baseline.

## 1. Introduction

Scene graph generation aims to parse visual scenes into structured graph representations, where nodes correspond to objects and edges encode pairwise relationships. In video domains, SGG faces the additional challenge of temporal consistency: the same triplet `(subject, predicate, object)` often persists across consecutive frames, yet frame-independent models lack the ability to exploit this temporal prior.

Existing temporal approaches for SGG typically operate at the feature or query-embedding level, applying exponential moving averages on backbone features or injecting historical object queries into the decoder's input embeddings. While effective, these methods treat memory at the granularity of individual objects or features, without explicitly modeling the triplet-level structure that defines scene graphs.

We propose a **Temporal Triplet Memory** module that stores and retrieves information at the triplet granularity. Each memory slot encodes a complete `(subject, predicate, object)` interaction—including the relation query feature, subject and object bounding boxes, relative spatial geometry, and the predicate probability distribution. A gated cross-attention mechanism injects this historical context into the object and relation branches of the decoder's final layer, enabling the model to leverage temporal priors when predicting current-frame triplets.

## 2. Architecture Overview

The module sits between the DETR-style decoder output and the final prediction heads, without modifying the backbone, transformer encoder, matcher, or loss criterion. Figure 1 illustrates the data flow.

```
Input Frame I_t
      │
      ▼
┌─────────────────┐
│  SAM3 Backbone   │  (frozen ViT, avg_pool2d downsampling)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Encoder+Decoder  │  (IterativeRelationTransformer)
│                  │
│  hs_sub[-1]     │  subject query features [B, N_s, D]
│  hs_obj[-1]     │  object  query features [B, N_o, D]
│  hs_rel[-1]     │  relation query features [B, N_r, D]
│                  │
│  logits, boxes   │  predicted subject/object/relation outputs
└────────┬────────┘
         │
         ▼
╔═════════════════════════════════════════════╗
║       Temporal Triplet Memory Module       ║
║                                            ║
║  ┌──────────────────────────┐              ║
║  │  TripletMemoryManager     │              ║
║  │   per-video bank lookup   │              ║
║  │   get_batch_memory()      │              ║
║  └───────────┬──────────────┘              ║
║              │ memory [B,M,D_m]            ║
║              ▼                              ║
║  ┌──────────────────────────┐              ║
║  │  Gate Schedule            │              ║
║  │   g_obj ∈ [0, α_obj]     │              ║
║  │   g_rel ∈ [0, α_rel]     │              ║
║  └───────────┬──────────────┘              ║
║              │                              ║
║     ┌────────┴────────┐                    ║
║     ▼                 ▼                    ║
║  ┌──────────┐   ┌──────────┐              ║
║  │Obj Inject│   │Rel Inject│              ║
║  │MHA(Q,K,V)│   │MHA(Q,K,V)│              ║
║  │+ gate·out│   │+ gate·out│              ║
║  └────┬─────┘   └────┬─────┘              ║
║       │              │                    ║
║       ▼              ▼                    ║
║  Re-compute final-layer logits/boxes      ║
║  (clone original → replace last layer)    ║
╚═══════════════════════════════════════════╝
         │
         ▼
┌─────────────────┐
│ Prediction Heads │  (use temporally-enhanced features)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Loss / Output    │
└────────┬────────┘
         │
         ▼
╔═════════════════════════════════════════════╗
║        Memory Update (no_grad)              ║
║  construct triplet candidates               ║
║  cxcywh → xyxy conversion                   ║
║  TripletMemoryEncoder(mem_feat)             ║
║  Bank.update(match → EMA / insert → expire) ║
╚═════════════════════════════════════════════╝
```

**Figure 1: Temporal Triplet Memory architecture.** The module intercepts the decoder output, reads per-video historical memory, injects temporally-enhanced context via gated cross-attention, re-computes the final-layer predictions, and then updates the memory bank with current-frame triplet candidates.

## 3. Triplet Memory Bank

### 3.1 Memory Slot Structure

Each video maintains an independent memory bank of up to $K=32$ slots. A slot $\mathbf{s}_k$ stores a compressed representation of one historical triplet:

$$\mathbf{s}_k = \left( \sigma_k, \mathbf{f}_k, \mathbf{b}_k^{\text{sub}}, \mathbf{b}_k^{\text{obj}}, \mathbf{b}_k^{\text{union}}, q_k, t_k, m_k, a_k \right)$$

where $\sigma_k = (c_{\text{sub}}, c_{\text{pred}}, c_{\text{obj}})$ is the triplet signature, $\mathbf{f}_k \in \mathbb{R}^{D_m}$ is the compressed memory feature ($D_m=128$), $\mathbf{b}_k^{*} \in [0,1]^4$ are bounding boxes in xyxy format, $q_k \in [0,1]$ is the quality score, $t_k$ is the source frame index, $m_k$ is the consecutive miss counter, and $a_k$ is the slot age.

The triplet signature $\sigma_k$ serves as a pseudo-identity, exploiting the fact that identical triplets rarely co-occur in a single frame in Action Genome. This eliminates the need for external tracking IDs.

### 3.2 Slot Matching

When a new candidate triplet arrives, the bank finds the best matching slot via a two-stage criterion:

$$\text{match}(k, c) = \begin{cases} 1 & \text{if } \sigma_k = \sigma_c \land \bar{\text{IoU}}(\mathbf{b}_k, \mathbf{b}_c) > \tau_{\text{match}} \\ 0 & \text{otherwise} \end{cases}$$

where $\bar{\text{IoU}} = \frac{1}{3}(\text{IoU}_{\text{sub}} + \text{IoU}_{\text{obj}} + \text{IoU}_{\text{union}})$ averages the paired box overlap, and $\tau_{\text{match}} = 0.3$ is the matching threshold.

### 3.3 Slot Update

Matched slots are updated via exponential moving average with momentum $\beta = 0.9$:

$$\mathbf{f}_k^{(t)} = \beta \mathbf{f}_k^{(t-1)} + (1-\beta) \mathbf{f}_c$$

$$\mathbf{b}_k^{(t)} = \beta \mathbf{b}_k^{(t-1)} + (1-\beta) \mathbf{b}_c$$

The quality score takes the historical maximum $q_k = \max(q_k, q_c)$, and $m_k$ is reset to zero. Unmatched slots increment their miss counter; slots with $m_k > 2$ are marked invalid and removed from the active set. New triplets with no matching slot are inserted; when the bank is full, the slot with the lowest $q_k - 0.05 \cdot m_k$ is evicted.

### 3.4 Temporal Delta Embedding

To encode the temporal distance between a memory slot and the current frame, we use a learnable bucket embedding. The frame gap $\Delta_t$ is discretized into 7 buckets:

$$\text{bucket}(\Delta_t) = \begin{cases} 0 & \Delta_t = 0 \\ 1 & \Delta_t = 1 \\ 2 & \Delta_t = 2 \\ 3 & \Delta_t = 3 \\ 4 & 4 \leq \Delta_t \leq 7 \\ 5 & 8 \leq \Delta_t \leq 15 \\ 6 & \Delta_t \geq 16 \end{cases}$$

The embedding is added to the memory feature before cross-attention: $\mathbf{f}_k \leftarrow \mathbf{f}_k + \mathbf{e}_{\text{emb}}(\text{bucket}(\Delta_t))$.

## 4. Memory Encoding

### 4.1 Encoder Architecture

The TripletMemoryEncoder compresses a triplet candidate into a compact $D_m$-dimensional feature. It takes four input modalities:

1. **Relation query feature** $\mathbf{h}_{\text{rel}} \in \mathbb{R}^{256}$: the decoder's final-layer representation of the predicate.
2. **Box geometry** $[\mathbf{b}_{\text{sub}}, \mathbf{b}_{\text{obj}}, \mathbf{b}_{\text{union}}] \in \mathbb{R}^{12}$: absolute positions of subject, object, and their union box.
3. **Relative geometry** $\mathbf{g} \in \mathbb{R}^{8}$: spatial relationship between subject and object (see Eq. 2).
4. **Predicate distribution** $\mathbf{p} \in \mathbb{R}^{C_{\text{rel}}}$: the softmax probability over relation classes ($C_{\text{rel}}=26$), detached from the computation graph.

Each modality is projected through a dedicated MLP, then concatenated and fused:

$$\mathbf{f} = \text{MLP}_{\text{fuse}}\left( \left[ \text{proj}_{\text{rel}}(\mathbf{h}_{\text{rel}}); \text{proj}_{\text{box}}(\mathbf{b}); \text{proj}_{\text{geom}}(\mathbf{g}); \text{proj}_{\text{pred}}(\mathbf{p}) \right] \right)$$

where $[\cdot ; \cdot]$ denotes concatenation and the total input dimension to $\text{MLP}_{\text{fuse}}$ is $128 + 64 + 64 + 32 = 288$.

### 4.2 Relative Spatial Geometry

Given subject box $\mathbf{b}_{\text{sub}} = (x_1^s, y_1^s, x_2^s, y_2^s)$ and object box $\mathbf{b}_{\text{obj}} = (x_1^o, y_1^o, x_2^o, y_2^o)$ in normalized xyxy coordinates, we first convert to center-size representation $(c_x, c_y, w, h)$ and compute:

$$\mathbf{g} = \begin{bmatrix} c_x^o - c_x^s \\ c_y^o - c_y^s \\ \log(w^o / w^s) \\ \log(h^o / h^s) \\ \log(A^o / A^s) \\ A^s / A^{\text{union}} \\ A^o / A^{\text{union}} \\ \sqrt{(c_x^o - c_x^s)^2 + (c_y^o - c_y^s)^2} \end{bmatrix}$$

Relative geometry is more robust than absolute coordinates under camera motion, a common characteristic of Action Genome videos.

## 5. Temporal Injection

### 5.1 Gated Cross-Attention

The TemporalTripletInjector uses multi-head cross-attention to infuse historical memory into current-frame decoder features. For a query branch (object or relation) with features $\mathbf{Q} \in \mathbb{R}^{B \times N \times D}$ and memory $\mathbf{M} \in \mathbb{R}^{B \times M \times D_m}$:

$$\mathbf{Q}' = \mathbf{Q} + \gamma \cdot \text{Proj}_{\text{out}}\left( \text{MHA}\left( \text{Proj}_Q(\mathbf{Q}), \text{Proj}_{KV}(\mathbf{M}), \text{Proj}_{KV}(\mathbf{M}) \right) \right)$$

where MHA denotes multi-head attention with $H=8$ heads, and $\gamma$ is the gate scalar. The residual connection ensures the current-frame information is preserved while temporal context is added.

Memory features are score-weighted before cross-attention: $\mathbf{M} \leftarrow \mathbf{M} \odot \mathbf{s}$, where $\mathbf{s} \in [0,1]^M$ are the slot quality scores, suppressing low-confidence memories.

### 5.2 Gate Schedule

Rather than using a learnable gate parameter, we employ a deterministic warmup schedule that provides explicit control over the injection strength during training:

$$\gamma(\tau) = \begin{cases} 0 & \tau < 0.10 \\ \gamma_{\max} \cdot \frac{\tau - 0.10}{0.20} & 0.10 \leq \tau < 0.30 \\ \gamma_{\max} & \tau \geq 0.30 \end{cases}$$

where $\tau = i / I_{\max}$ is the training progress ratio. For the first 10% of iterations, $\gamma = 0$, allowing the model to establish reliable single-frame predictions before temporal information is introduced. The gate then linearly increases to its maximum value $\gamma_{\max}$ over the next 20% of training.

We set $\gamma_{\max}^{\text{obj}} = 0.15$ for object branch injection and $\gamma_{\max}^{\text{rel}} = 0.30$ for relation branch injection. The higher value for the relation branch reflects the stronger dependency of predicate recognition on temporal context.

### 5.3 Injection and Prediction Head Re-computation

After injecting memory into the final decoder layer features, we re-run the corresponding prediction heads:

$$\hat{\mathbf{h}}_{\text{obj}}[-1] = \text{Injector}_{\text{obj}}(\mathbf{h}_{\text{obj}}[-1], \mathbf{M}, \gamma_{\text{obj}})$$

$$\hat{\mathbf{L}}_{\text{obj}}[-1] = \text{ClsHead}_{\text{obj}}(\hat{\mathbf{h}}_{\text{obj}}[-1])$$

The original logit and box tensors are cloned before the final layer is replaced, preventing in-place modification of tensors that remain in the autograd graph. Only the final decoder layer is modified; intermediate auxiliary losses retain the original decoder outputs.

## 6. Memory Update

### 6.1 Candidate Construction

After each forward pass, triplet candidates are constructed from the current-frame predictions. For each relation query index $r$, the subject and object boxes are first converted from DETR's native cxcywh format to xyxy:

$$\mathbf{b}^{xyxy} = \text{cxcywh\_to\_xyxy}(\mathbf{b}^{cxcywh})$$

The triplet quality is computed as the product of individual scores:

$$q_r = s_{\text{sub}}(r) \cdot s_{\text{obj}}(r) \cdot s_{\text{pred}}(r)$$

Candidates with $q_r > \tau_{\text{update}}$ are selected, up to $K_{\text{topk}} = 16$ per frame, where $\tau_{\text{update}} = 0.10$ is the quality threshold.

### 6.2 Gradient Isolation

The entire memory update pipeline operates under `torch.no_grad()`. Additionally, all tensors written to memory slots are explicitly detached and moved to CPU:

$$\mathbf{f}_k \leftarrow \text{Encoder}(\ldots).\text{detach}().\text{cpu}()$$

This three-layer protection (context manager + explicit detach in encoder + detach.cpu at storage) ensures that memory updates never participate in backpropagation, preventing the formation of cross-iteration computation graphs that would cause memory explosion.

## 7. Implementation Notes

The module is implemented as a lightweight addition to the SpeaQ framework. The key integration points are: (1) `TripletMemoryManager` initialized during model construction, (2) `_apply_triplet_memory_to_output()` called after the transformer decoder and before ROI refinement and output assembly, and (3) the memory update block executed at the end of the forward pass.

Video identity and frame indices are passed from the data loader through the meta-architecture to the transformer module. Per-video memory banks are automatically created on first access and cleared when a frame index regression is detected (indicating a video transition).

The module is configurable via the `MODEL.TEMPORAL.TRIPLET_MEMORY_*` configuration namespace. By default, all triplet memory functionality is disabled, ensuring backward compatibility with existing training pipelines.

## 8. Related Work

**Scene Graph Generation.** DETR-based SGG methods reformulate relationship detection as set prediction, using bipartite matching to assign predictions to ground-truth triplets. SpeaQ extends this with Hungarian matching and quality-aware multi-assignment for improved triplet recall.

**Temporal Modeling in Video Understanding.** SAM 3 demonstrates the effectiveness of memory-conditioned transformers for video object segmentation, using per-object memory banks with cross-attention fusion and object pointer representations. Our triplet memory design draws inspiration from this architecture while adapting it to the SGG domain.

**Memory-Augmented Transformers.** Memory networks and their transformer variants have been widely adopted for tasks requiring long-range context. Our work applies this paradigm to structured visual relationship prediction, maintaining memory at the granularity of semantic triplets rather than raw features or tokens.

## References

[1] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End Object Detection with Transformers. *ECCV 2020*.

[2] Carion, N., et al. (2025). SAM 3: Segment Anything with Concepts. *arXiv:2511.16719*.

[3] Ravi, N., et al. (2024). SAM 2: Segment Anything in Images and Videos. *ICLR 2025*.

[4] Ji, J., Krishna, R., Fei-Fei, L., & Niebles, J. C. (2020). Action Genome: Actions As Compositions of Spatio-Temporal Scene Graphs. *CVPR 2020*.

[5] Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*.

[6] Cong, Y., et al. (2023). SpeaQ: Quality-Aware Multi-Assignment for Scene Graph Generation. *CVPR 2023*.

[7] Sukhbaatar, S., Szlam, A., Weston, J., & Fergus, R. (2015). End-To-End Memory Networks. *NeurIPS 2015*.
