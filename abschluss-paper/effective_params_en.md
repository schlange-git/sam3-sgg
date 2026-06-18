# Effective Configuration Parameters under the ROI / Temporal-Stage4 / PatchMerge Setting

This document summarizes the configuration parameters that are **actually in effect** when only three architectural components are enabled on top of the X-SAM-pretrained baseline: (1) ROI feature refinement (`ROI_REFINE`), (2) Stage-4 temporal modeling (`TEMPORAL`, triplet-memory v3), and (3) feature-map patch merging (`SAM3.USE_PATCH_MERGE`).

Reference configuration: `configs/fulltask_roiresid_pm_clip_v3_xsam_bs24_20w.yaml`. "Effective value" denotes the value that governs runtime behaviour after merging the explicit YAML overrides with the defaults in `configs/defaults.py` and the consuming code paths. Values that are written in the config but never read by code are flagged as **inert**.

---

## 1. ROI Feature Refinement (`MODEL.ROI_REFINE`)

The ROI-refinement module re-pools a high-resolution feature region for every relation entity query (subject/object) using RoIAlign, projects it through a small MLP, and fuses it back into the query embedding through a learned per-query gate. It is designed to recover fine spatial detail lost by the coarse global feature stride, which is especially helpful for small objects.

### 1.1 Effective parameters

| Parameter | Effective value | Default | Role / what it controls |
|---|---|---|---|
| `ENABLED` | `True` | `False` | Master switch for the ROI-refinement branch. |
| `STRIDE` | `14` | `14` | Feature-map stride on which RoIAlign operates; the ROI samples the native stride-14 cached features (before patch-merge downsampling). |
| `POOL_SIZE` | `7` | `7` | RoIAlign output grid; each ROI is pooled to a `7×7` map, flattened to `H·7·7` and projected. |
| `RESNET_FPN_LEVEL` | `0` | `0` | Selects the source feature level (0 = coarsest / stride-32 / p5). |
| `APPLY_TO` | `all` | `small_only` | Which queries receive ROI refinement. `all` → every query is refined (keep-mask all-true); `small_only` → only boxes with area `< SMALL_AREA_THRESH`. |
| `SMALL_AREA_THRESH` | `0.02` | `0.02` | Normalized-area cutoff for "small" boxes. Under `APPLY_TO=all` it no longer gates *which* queries are refined; it is still used (a) to split small/large gate statistics in logging, and (b) by the ROI classification loss to select small-box queries. Threshold derived from full-dataset area statistics. |
| `DETACH_BOXES` | `True` | `True` | The boxes used to build ROIs are detached, so gradients do not flow back into box regression through the ROI sampling. |
| `USE_GATE` | `True` | `True` | Enables the learned fusion gate (see below). |
| `FUSION` | `residual` | `convex` | Fusion rule. `residual`: `refined = e + g·roi`. `convex`: `refined = (1−g)·e + g·roi`. |
| `LOSS_ENABLED` | `True` | `False` | Adds an auxiliary ROI classification loss. |
| `LOSS_WEIGHT` | `1.0` | `1.0` | Weight of the ROI classification loss in the total objective. |
| `ONLY_ROI_CLS` | `False` | `False` | If true, replaces the query embedding entirely with the ROI projection; here disabled, so the gated fusion above applies. |
| `EVAL_DUAL` | `True` | `False` | At evaluation, a single forward pass emits both the refined (`override`) and the raw (`origin`, `raw_` prefix) metrics for direct comparison. |
| `REPLACE_BEFORE_MATCHER` | `False` | `False` | Whether refined logits replace the originals before Hungarian matching. |

### 1.2 Mechanism in effect

For each kept query the module computes a region embedding `roi = roi_proj(RoIAlign(feat, box))`, where `roi_proj` is `Linear(H·7·7 → H) → ReLU → LayerNorm → Linear(H → H)`. The gate is an MLP `Linear(2H → H) → ReLU → Linear(H → 1) → Sigmoid` taking the concatenation of the original query embedding and the ROI embedding, producing a scalar `g ∈ (0,1)` per query. With the residual fusion adopted here, the refined embedding is `e + g·roi`, so the gate learns *how much* corrective ROI detail to inject without destroying the original query content. Because `DETACH_BOXES=True`, the spatial sampling locations are treated as constants, and the auxiliary ROI classification loss (computed only on small-area queries, `area < 0.02`) supervises the refinement to improve recognition of small entities.

---

## 2. Stage-4 Temporal Modeling (`MODEL.TEMPORAL`, triplet-memory v3)

Stage-4 maintains a per-video memory bank of triplet-level (subject, predicate, object) entries and injects aggregated temporal context into the object and relation queries through gated cross-attention, with a curriculum that ramps both the injection strength and the memory-writing fidelity over training.

### 2.1 Effective parameters

| Parameter | Effective value | Default | Role / what it controls |
|---|---|---|---|
| `ENABLED` / `EVAL_ENABLED` | `True` / `True` | `False` / `False` | Enables temporal modeling in training and evaluation. |
| `MODE` | `triplet_memory_v3` | `feature_ema` | Selects the triplet-memory variant (distinct from the v1 object-query memory). |
| `TRIPLET_MEMORY_ENABLED` | `True` | `False` | Activates the triplet memory bank + encoder + injector. |
| `INJECT_OBJECT` / `INJECT_RELATION` | `True` / `True` | `True` / `True` | Inject temporal memory into object queries and relation queries. |
| `INJECT_SUBJECT` | `False` (default) | `False` | Subject-query injection disabled. |
| `DETACH_MEMORY` | `True` | `True` | Memory entries are stored detached on CPU; the bank never participates in backprop. |
| `TRIPLET_MEMORY_DIM` | `128` | `128` | Dimensionality of a stored memory feature. |
| `TRIPLET_MEMORY_SIZE` | `32` | `32` | Maximum number of slots per video; weakest slot is replaced when full. |
| `TRIPLET_MEMORY_TOPK_UPDATE` | `16` | `16` | Number of top candidates written into the bank per frame. |
| `TRIPLET_MEMORY_MAX_MISS` | `2` | `2` | A slot is invalidated after this many consecutive non-matching frames. |
| `GATE_MAX_OBJECT` | `0.15` | `0.15` | Maximum injection strength for object queries. |
| `GATE_MAX_RELATION` | `0.30` | `0.30` | Maximum injection strength for relation queries (relations rely more on temporal context). |
| `GATE_ZERO_END_RATIO` | `0.10` | `0.10` | Fraction of training during which the gate is held at exactly 0 (pure warm-up, no injection). |
| `GATE_WARMUP_END_RATIO` | `0.30` | `0.30` | Fraction of training at which the gate finishes ramping to its maximum. |

### 2.2 Injection-strength schedule (the core temporal curriculum)

The gate value is a function of training progress `r = iter / MAX_ITER`, computed by `get_temporal_gate` (`modeling/temporal/triplet_memory.py`):

```
r < 0.10                 → gate = 0                       (no injection; queries returned unchanged)
0.10 ≤ r < 0.30          → gate = gate_max · (r − 0.10) / (0.30 − 0.10)   (linear ramp)
r ≥ 0.30                 → gate = gate_max                (full strength)
```

The injector applies `q' = q + gate · CrossAttn(q, memory)`. With `MAX_ITER = 200000`, the gate is exactly 0 for the first 20 000 iterations, ramps linearly from iter 20 000 to 60 000, and is held at its maximum (0.15 for object, 0.30 for relation) thereafter. This delayed warm-up lets the detector stabilize before any temporal signal is mixed in.

Two companion curricula govern *what* is written to memory:

- **Memory-update mode** (`get_memory_update_mode`): `gt_aligned` for `r < 0.30`, `mixed` for `0.30 ≤ r < 0.70`, `prediction` for `r ≥ 0.70`. Early training writes GT-aligned triplets for stability; late training writes the model's own predictions.
- **Prediction-quality threshold** (`get_prediction_threshold`): decays linearly from `0.15` to `0.05` over `r ∈ [0, 0.70]`, gradually admitting lower-confidence predictions into the bank as the model matures.

Slot matching uses a pseudo-identity signature `(sub_label, pred_label, obj_label)` plus a mean subject/object/union box IoU threshold of `0.3`, with EMA momentum `0.9` on matched slots. A bucketized temporal-delta embedding (7 buckets: 0, 1, 2, 3, 4–7, 8–15, 16+) encodes the frame gap when reading memory.

> Note (inert flags): `NON_KEY_SKIP_LOSS`, `NON_KEY_SKIP_EVAL`, and `NON_KEY_RUN_OBJECT_ONLY` are set to `True` in the config but are **only declared in `configs/defaults.py` and never consumed by any code path**; they have no runtime effect under the current implementation and should not be reported as active mechanisms.

---

## 3. Feature-Map Patch Merging (`MODEL.SAM3.USE_PATCH_MERGE`)

Patch merging reduces the spatial resolution of the frozen SAM3 feature map with a learnable downsampling operator, trading spatial granularity for a longer effective stride that matches the detection head while remaining trainable.

### 3.1 Effective parameters

| Parameter | Effective value | Default | Role / what it controls |
|---|---|---|---|
| `USE_PATCH_MERGE` | `True` | `False` | Enables the learnable patch-merge downsampling on the SAM3 feature map. |
| `TARGET_STRIDE` | `32` | `32` | Output stride after merging; features are merged from the native stride down to stride 32. |
| `IMAGE_SIZE` | `1008` | `1008` | SAM3 input resolution; sets the feature-map size before merging. |
| `FEATURE_DIM` | `256` | `256` | Channel dimension of the SAM3 feature map fed to the detection transformer. |
| `CHANNEL_REPEAT` | `1` | `1` | Channel-repeat factor applied to the SAM3 features (no repetition). |
| `FREEZE` | `True` | — | Freezes the SAM3 backbone. **Exception:** the patch-merge convolution is explicitly kept `requires_grad=True` and is trained even though the backbone is frozen. |

### 3.2 Mechanism in effect

Patch merging follows the X-SAM design: a pixel-unshuffle that folds spatial patches into the channel dimension, followed by a learnable `1×1` convolution that projects back to `FEATURE_DIM=256` at the longer `TARGET_STRIDE=32`. Although `SAM3.FREEZE=True` freezes the rest of the backbone, the patch-merge convolution remains trainable, so the merged representation is adapted to the downstream relation task. The ROI-refinement branch (Section 1) deliberately taps the native stride-14 cached features *before* this downsampling, so the two modules are complementary: patch-merge provides an efficient global stride-32 representation, while ROI-refine restores fine local detail where needed.

---

## 4. Summary

Under this three-flag setting the model (i) downsamples the frozen SAM3 features with a trainable patch-merge convolution to a stride-32 global representation, (ii) restores fine local detail for relation entities via gated, residual ROI refinement (`e + g·roi`, gate ∈ (0,1) per query) with an auxiliary small-object classification loss, and (iii) injects per-video triplet-level temporal context into object and relation queries through gated cross-attention, where the gate follows a delayed linear warm-up (zero for the first 10% of training, ramping to 0.15 / 0.30 for object / relation queries by 30%) coupled with a GT→prediction memory-writing curriculum.
