# ROI Refine 数据传递流程

## 概述

ROI Refine 分支在 DETR 最后一个 decoder layer 输出的 subject/object entity embeddings 上，利用 **原生 SAM3 stride14 高分辨率特征** 做 RoIAlign + MLP 残差修正，提升小物体分类精度。整个分支是一个**独立旁路**，不改变原有 DETR 推理路径，仅在末尾替换 logits。

## 配置

`configs/speaq_ag_roi.yaml` — ROI_REFINE 相关：

```yaml
MODEL.ROI_REFINE:
  ENABLED: True           # 总开关
  STRIDE: 14              # 使用原生 stride14 特征
  POOL_SIZE: 7            # RoIAlign 输出尺寸
  SMALL_AREA_THRESH: 1.0  # 面积阈值（归一化 w*h），小于此值才 refine
  DETACH_BOXES: True      # 是否停止 boxes 梯度
  USE_GATE: True          # 是否使用门控残差
  LOSS_ENABLED: True      # 是否计算辅助分类 loss
  LOSS_WEIGHT: 1.0        # loss 权重
  APPLY_TO: "small_only"  # "small_only" | "all"
```

## 完整数据流

```
输入图像 (像素)
     │
     ▼
┌─────────────────────────────────────────────┐
│ SAM3 backbone  (Sam3MaskedBackbone.forward)  │
│                                              │
│  1. normalize → sam3_model.backbone.forward  │
│     backbone_out = {backbone_fpn: [L0,L1,L2]}│
│                                              │
│  2. _cache_roi_refine_aux_features()         │
│     └─ _extract_native_multiscale()          │
│        提取 backbone_fpn 各层 → 投影到 256ch  │
│        → 计算实际 stride → sorted(zip)        │
│     └─ 断言 stride14 存在于 native 列表中     │
│     └─ 断言 channel==FEATURE_DIM(256)         │
│     └─ 断言 spatial==IMAGE_SIZE/14 (72)       │
│     └─ 构建 mask，存入 _last_aux_features     │
│                                              │
│  3. 返回 features 给 Joiner                  │
│     (主路径: use_backbone_fpn + last merge   │
│      返回 {fpn_14, fpn_28, fpn_7} 等         │
│      供 DETR transformer 使用)               │
└──────────────────────────────────────────────┘
         │                    │
         │ features[-1]       │ get_last_aux_features()
         │ (主路径特征)       │ (stride14 高分辨率特征)
         ▼                    ▼
┌────────────────────┐  ┌─────────────────────────┐
│ DETR Transformer   │  │ ROIRefineHead            │
│ IterativeRelation  │  │                          │
│ Decoder ×6         │  │  输入:                   │
│                    │  │    embeddings [B,Q,256]  │
│ 输出:              │  │    boxes [B,Q,4]         │
│  hs_subject_last   │  │    feature [B,256,72,72] │
│  hs_object_last    │  │    image_h/w (1008)      │
│  relation_sub...   │  │                          │
│  _coords[-1]       │  │  1. _build_rois():       │
│                    │  │     boxes_cxcywh → xyxy  │
│                    │  │     → 像素坐标           │
│                    │  │     → area = w*h < thresh│
│                    │  │     → 筛选保留的框       │
│                    │  │                          │
│                    │  │  2. roi_align() on       │
│                    │  │     stride14 feature     │
│                    │  │     → [K, 256, 7, 7]     │
│                    │  │                          │
│                    │  │  3. roi_proj MLP:        │
│                    │  │     256*7*7 → 256→256    │
│                    │  │     → roi_emb [K, 256]   │
│                    │  │                          │
│                    │  │  4. Gate（可选）:         │
│                    │  │     sigmoid(MLP(          │
│                    │  │       cat(emb, roi_emb))) │
│                    │  │     × roi_emb            │
│                    │  │     → residual [K, 256]  │
│                    │  │                          │
│                    │  │  5. 残差融合:              │
│                    │  │     refined_emb[flat_idx] │
│                    │  │       += residual        │
│                    │  │     → [B, Q, 256]        │
│                    │  │                          │
│                    │  │  返回: refined + keep_mask│
└────────────────────┘  └──────────────────────────┘
         │                       │
         ▼                       ▼
┌────────────────────────────────────────────┐
│ IterativeRelationDETR.forward (after       │
│ last decoder)                              │
│                                            │
│ 1. sub_refined, sub_mask = roi_refine_head(│
│      sub_emb, sub_boxes, roi_feat, 1008)   │
│ 2. obj_refined, obj_mask = roi_refine_head(│
│      obj_emb, obj_boxes, roi_feat, 1008)   │
│                                            │
│ 3. 重分类:                                   │
│    sub_roi_logits = object_embed(           │
│      sub_refined)  → [B,Q,36]              │
│    (或 SplitObjectClassifier)               │
│                                            │
│ 4. 存入 output dict:                        │
│    relation_subject_logits_roi             │
│    relation_object_logits_roi              │
│    roi_subject_mask                        │
│    roi_object_mask                         │
│                                            │
│ 5. 推理时 (not training):                   │
│    out['relation_subject_logits']          │
│      = out['relation_subject_logits_roi']  │
│    (替换原始 logits)                         │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│ Criterion (loss 计算)                       │
│                                            │
│ 1. forward_relation() 计算匹配 indices      │
│ 2. get_relation_losses() → 主 loss          │
│ 3. get_roi_refine_losses(outputs,           │
│        entity_targets, combined_indices)    │
│                                            │
│    └─ _compute_roi_refine_cls_loss_for_    │
│        role(logits, targets, indices):      │
│                                            │
│     遍历每个 batch:                          │
│       tgt_boxes = targets["boxes"][tgt_idx] │
│       small_mask = area(w*h) < thresh      │
│       cross_entropy(                        │
│         logits[batch, src[small_mask]],     │
│         labels[small_mask])                 │
│                                            │
│    → loss_roi_subject_cls /                │
│       loss_roi_object_cls                  │
│                                            │
│ 4. × weight_dict (LOSS_WEIGHT=1.0)         │
│    加入 total_loss                          │
└────────────────────────────────────────────┘
```

## 关键设计

### 1. 严格使用原生 stride14，禁止回退

`_cache_roi_refine_aux_features` 中：
- 只从 `backbone_out["backbone_fpn"]` 提取多尺度
- `assert self.roi_refine_stride in native` — 找不到就报错，不 resize/FPN 合成
- `assert` 检查 spatial = `IMAGE_SIZE / stride` 精确匹配

### 2. SAM3 IMAGE_SIZE 固定

SAM3 内部把所有输入 resize 到 `MODEL.SAM3.IMAGE_SIZE`（1008×1008）。ROI refine 的 `image_h, image_w` **不使用** 输入的 `samples.tensors` 的 padding 尺寸，而是直接用 `self.sam3_image_size = 1008`，与 SAM3 内部空间对齐。

特征 `[B, 256, 72, 72]` = `1008 / 14`。

### 3. 残差 + 可选门控

```
refined_emb[selected] = original_emb[selected] + residual
residual = roi_emb                                          # 无门控
residual = sigmoid(MLP(cat(emb, roi_emb))) * roi_emb        # 有门控
```

门控用一个 `Linear(512→256)→ReLU→Linear(256→1)→Sigmoid` 网络学习每组的缩放因子，控制 ROI 特征的贡献幅度。

### 4. 推理时替换 logits

```python
if not self.training:
    out['relation_subject_logits'] = out['relation_subject_logits_roi']
    out['relation_object_logits'] = out['relation_object_logits_roi']
```

推理时直接用 refined logits 替代原始 logits，无需额外后处理。

### 5. loss 只对匹配后的框计算

`_compute_roi_refine_cls_loss_for_role` 接收 `combined_indices["subject/object"]`（matcher 匹配结果），只在 `matched src_idx → tgt_idx` 对应的框上算 loss，且只有 `area = w*h < SMALL_AREA_THRESH` 的小框参与。

## 文件索引

| 文件 | 关键函数/类 | 职责 |
|------|-----------|------|
| `modeling/backbone/sam3_backbone.py` | `_cache_roi_refine_aux_features` | 缓存 SAM3 原生 stride14 特征到 `_last_aux_features` |
| `modeling/backbone/sam3_backbone.py` | `get_last_aux_features` | 暴露缓存的 aux 特征给 DETR |
| `modeling/transformer/detr.py` | `IterativeRelationDETR.forward` | 调用 ROIRefineHead，重分类，写入 output |
| `modeling/transformer/roi_refine.py` | `ROIRefineHead` | RoIAlign + MLP + 门控残差 |
| `modeling/transformer/criterion.py` | `_compute_roi_refine_cls_loss_for_role` | ROI refine 分类 CE loss |
| `modeling/transformer/criterion.py` | `get_roi_refine_losses` | 入口断言 + 调用 subject/object loss |
| `modeling/meta_arch/detr.py` | `prepare_targets` | 构建 `combined_boxes/labels`（归一化 cxcywh） |
| `modeling/meta_arch/detr.py` | weight_dict 构建 | 添加 `loss_roi_subject_cls` / `loss_roi_object_cls` 权重 |
