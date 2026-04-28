下面这版可以直接交给 Cursor。目标是实现一个 **stride14 ROI feature refinement branch**，用于增强 SpeaQ 中 **small subject / small object 的分类与 embedding 表征**，并且第一版不破坏原生 triplet 架构。

---

# 目标

在现有 SpeaQ / SAM3-SGG 中增加一个 **ROI-guided small object refinement branch**：

```text
SAM3 stride14 feature
        ↓
SpeaQ decoder 输出 subject/object boxes + embeddings
        ↓
根据预测框大小判断 small candidate
        ↓
ROIAlign 从 stride14 feature 中抽取局部高分辨率 ROI feature
        ↓
Linear / MLP 映射到 hidden_dim=256
        ↓
作为 residual 加回 hs_subject / hs_object
        ↓
重新经过原 object classification head
        ↓
得到 refined subject/object logits
```

第一版只增强：

```text
relation_subject_logits
relation_object_logits
```

不改：

```text
relation_logits
relation_boxes
matcher 主逻辑
relation decoder
subject/object/relation decoder 内部 attention
triplet index 对齐方式
```

---

# 1. 配置项新增

在 config 中新增：

```yaml
MODEL:
  ROI_REFINE:
    ENABLED: True
    STRIDE: 14
    POOL_SIZE: 7
    SMALL_AREA_THRESH: 0.02
    DETACH_BOXES: True
    USE_GATE: True
    LOSS_ENABLED: True
    LOSS_WEIGHT: 1.0
    APPLY_TO: "small_only"  # options: small_only / all
```

含义：

```text
ENABLED:
  是否启用 ROI refinement。

STRIDE:
  使用哪一级高分辨率 feature，目前使用 SAM3 stride14。

POOL_SIZE:
  ROIAlign 输出大小，默认 7×7。

SMALL_AREA_THRESH:
  判断 small object 的面积阈值。
  box 是归一化 cxcywh 格式时，area = w × h。
  area < 0.02 判定为 small。

DETACH_BOXES:
  ROIAlign 使用预测框时是否 detach。
  第一版建议 True，避免 ROI 分支反向影响 box regression，训练更稳。

USE_GATE:
  是否使用 learnable gate 控制 ROI residual 强度。

LOSS_ENABLED:
  是否对 refined logits 额外计算 small object classification auxiliary loss。

LOSS_WEIGHT:
  auxiliary loss 权重。

APPLY_TO:
  small_only 表示只对小框 query 加 ROI residual；
  all 表示所有 query 都加 ROI residual。
```

---

# 2. 数据流位置

插入位置应该在：

```text
IterativeRelationTransformer 输出 hs_subject / hs_object 后
prediction heads 之前或之后
```

推荐插入在 **head 之前**，也就是：

```text
hs_subject → ROI residual → refined_hs_subject → object_embed → refined_subject_logits
hs_object  → ROI residual → refined_hs_object  → object_embed → refined_object_logits
```

不要插入 decoder 内部第一版，因为 decoder 内部每层都要 box，训练初期 box 不稳定，容易炸。

---

# 3. 需要拿到的输入

ROI branch 需要：

```python
hs_subject: Tensor
# [num_layers, B, N, hidden_dim]

hs_object: Tensor
# [num_layers, B, N, hidden_dim]

relation_subject_boxes: Tensor
# [num_layers, B, N, 4], normalized cxcywh

relation_object_boxes: Tensor
# [num_layers, B, N, 4], normalized cxcywh

aux_feature_stride14: Tensor
# [B, 256, H14, W14], e.g. [B, 256, 72, 72]

image_sizes:
# 每张图 resize 后的尺寸，通常为 1008×1008
```

第一版只 refine 最后一层即可：

```python
sub_emb = hs_subject[-1]  # [B, N, C]
obj_emb = hs_object[-1]   # [B, N, C]

sub_boxes = relation_subject_boxes[-1]  # [B, N, 4]
obj_boxes = relation_object_boxes[-1]   # [B, N, 4]
```

后续如果要 deep supervision，再扩展到每一层。

---

# 4. 新增模块：ROIRefineHead

新建文件建议：

```text
modeling/transformer/roi_refine.py
```

代码骨架：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., 4], normalized cxcywh
    return: [..., 4], normalized xyxy
    """
    cx, cy, w, h = x.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


class ROIRefineHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        pool_size: int = 7,
        spatial_scale: float = 1.0 / 14.0,
        small_area_thresh: float = 0.02,
        detach_boxes: bool = True,
        use_gate: bool = True,
        apply_to: str = "small_only",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
        self.spatial_scale = spatial_scale
        self.small_area_thresh = small_area_thresh
        self.detach_boxes = detach_boxes
        self.use_gate = use_gate
        self.apply_to = apply_to

        in_dim = hidden_dim * pool_size * pool_size

        self.roi_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

    def _build_rois(
        self,
        boxes_cxcywh: torch.Tensor,
        image_h: int,
        image_w: int,
    ):
        """
        boxes_cxcywh: [B, N, 4], normalized cxcywh
        return:
          rois: [K, 5], format [batch_idx, x1, y1, x2, y2] in image pixel coordinates
          keep_mask: [B, N] bool
          flat_indices: [K], index into flattened [B*N]
        """
        if self.detach_boxes:
            boxes_cxcywh = boxes_cxcywh.detach()

        B, N, _ = boxes_cxcywh.shape

        area = boxes_cxcywh[..., 2].clamp(min=0) * boxes_cxcywh[..., 3].clamp(min=0)

        if self.apply_to == "small_only":
            keep_mask = area < self.small_area_thresh
        elif self.apply_to == "all":
            keep_mask = torch.ones_like(area, dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported apply_to={self.apply_to}")

        boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
        boxes_xyxy = boxes_xyxy.clamp(0.0, 1.0)

        boxes_xyxy_pixel = boxes_xyxy.clone()
        boxes_xyxy_pixel[..., [0, 2]] *= image_w
        boxes_xyxy_pixel[..., [1, 3]] *= image_h

        batch_indices = torch.arange(B, device=boxes_cxcywh.device)[:, None].expand(B, N)

        flat_keep = keep_mask.reshape(-1)
        flat_indices = torch.nonzero(flat_keep, as_tuple=False).squeeze(1)

        if flat_indices.numel() == 0:
            rois = boxes_cxcywh.new_zeros((0, 5))
            return rois, keep_mask, flat_indices

        flat_boxes = boxes_xyxy_pixel.reshape(B * N, 4)[flat_indices]
        flat_batch = batch_indices.reshape(B * N)[flat_indices].to(flat_boxes.dtype).unsqueeze(1)

        rois = torch.cat([flat_batch, flat_boxes], dim=1)
        return rois, keep_mask, flat_indices

    def forward(
        self,
        embeddings: torch.Tensor,
        boxes_cxcywh: torch.Tensor,
        feature: torch.Tensor,
        image_h: int,
        image_w: int,
    ):
        """
        embeddings: [B, N, C]
        boxes_cxcywh: [B, N, 4], normalized cxcywh
        feature: [B, C, H14, W14]

        return:
          refined_embeddings: [B, N, C]
          keep_mask: [B, N] bool
        """
        B, N, C = embeddings.shape
        assert C == self.hidden_dim

        rois, keep_mask, flat_indices = self._build_rois(
            boxes_cxcywh=boxes_cxcywh,
            image_h=image_h,
            image_w=image_w,
        )

        if rois.numel() == 0:
            return embeddings, keep_mask

        roi_feat = roi_align(
            input=feature,
            boxes=rois,
            output_size=(self.pool_size, self.pool_size),
            spatial_scale=self.spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )
        # [K, C, pool, pool]

        roi_feat = roi_feat.flatten(1)
        # [K, C*pool*pool]

        roi_emb = self.roi_proj(roi_feat)
        # [K, C]

        flat_embeddings = embeddings.reshape(B * N, C)
        selected_embeddings = flat_embeddings[flat_indices]

        if self.gate is not None:
            gate = self.gate(torch.cat([selected_embeddings, roi_emb], dim=-1))
            residual = gate * roi_emb
        else:
            residual = roi_emb

        refined_flat = flat_embeddings.clone()
        refined_flat[flat_indices] = refined_flat[flat_indices] + residual

        refined = refined_flat.reshape(B, N, C)
        return refined, keep_mask
```

---

# 5. 在 transformer / DETR forward 中接入

假设你现在在 `IterativeRelationTransformer.forward()` 或 `IterativeRelationDETR.forward()` 里已经有：

```python
hs_subject, hs_object, hs_relation = self.transformer(...)
```

以及：

```python
relation_subject_class = self.object_embed(hs_subject)
relation_subject_coords = self.object_bbox_coords(hs_subject).sigmoid()

relation_object_class = self.object_embed(hs_object)
relation_object_coords = self.object_bbox_coords(hs_object).sigmoid()

relation_class = self.relation_embed(hs_relation)
relation_coords = self.object_bbox_coords(hs_relation).sigmoid()
```

第一版建议在 **最后一层输出后** refine，然后新增输出，不直接覆盖原输出。

伪代码：

```python
outputs = {
    "relation_subject_logits": relation_subject_class[-1],
    "relation_subject_boxes": relation_subject_coords[-1],
    "relation_object_logits": relation_object_class[-1],
    "relation_object_boxes": relation_object_coords[-1],
    "relation_logits": relation_class[-1],
    "relation_boxes": relation_coords[-1],
}
```

加入：

```python
if self.roi_refine_enabled and aux_feature_stride14 is not None:
    sub_emb = hs_subject[-1].transpose(0, 1) if hs_subject[-1].shape[1] == B else hs_subject[-1]
```

注意 SpeaQ 源码中 `hs_subject` 可能已经是：

```text
[num_layers, B, N, C]
```

如果是这个格式，则直接：

```python
sub_emb = hs_subject[-1]  # [B, N, C]
obj_emb = hs_object[-1]   # [B, N, C]
```

然后：

```python
sub_boxes = relation_subject_coords[-1]  # [B, N, 4]
obj_boxes = relation_object_coords[-1]   # [B, N, 4]

sub_emb_refined, sub_roi_mask = self.roi_refine_head(
    embeddings=sub_emb,
    boxes_cxcywh=sub_boxes,
    feature=aux_feature_stride14,
    image_h=image_h,
    image_w=image_w,
)

obj_emb_refined, obj_roi_mask = self.roi_refine_head(
    embeddings=obj_emb,
    boxes_cxcywh=obj_boxes,
    feature=aux_feature_stride14,
    image_h=image_h,
    image_w=image_w,
)

relation_subject_logits_roi = self.object_embed(sub_emb_refined)
relation_object_logits_roi = self.object_embed(obj_emb_refined)

outputs["relation_subject_logits_roi"] = relation_subject_logits_roi
outputs["relation_object_logits_roi"] = relation_object_logits_roi
outputs["roi_subject_mask"] = sub_roi_mask
outputs["roi_object_mask"] = obj_roi_mask
```

推理阶段可以选择：

```python
outputs["relation_subject_logits"] = relation_subject_logits_roi
outputs["relation_object_logits"] = relation_object_logits_roi
```

但训练第一版建议保留原输出，同时加辅助 loss。

---

# 6. aux feature stride14 的传递

你的 SAM3 backbone 已经能缓存：

```text
aux stride 14 feature: [B, 256, 72, 72]
```

需要确保 forward 输出里能拿到它。

建议在 backbone 或 meta-arch 里统一命名：

```python
features, pos = self.backbone(samples)
aux_features = getattr(self.backbone, "aux_features", None)
```

或者让 backbone 返回：

```python
return features, pos, aux_features
```

推荐结构：

```python
aux_features = {
    14: feature_stride14
}
```

然后传入模型：

```python
aux_feature_stride14 = aux_features.get(14, None)
```

要求：

```text
aux_feature_stride14: [B, 256, H14, W14]
```

如果不是 256 通道，需要先用 1×1 conv 投影：

```python
self.aux_proj_14 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
```

---

# 7. image_h / image_w 怎么取

如果输入 SAM3/SpeaQ 的图已经 resize 到 1008×1008，则第一版直接：

```python
image_h = 1008
image_w = 1008
```

更稳妥写法：

```python
image_h, image_w = samples.tensors.shape[-2:]
```

如果 `samples.tensors` 是主输入图像：

```python
image_h = samples.tensors.shape[-2]
image_w = samples.tensors.shape[-1]
```

注意 ROIAlign 的 boxes 需要是 **image pixel coordinate**，不是 feature coordinate。因为你设置了：

```python
spatial_scale = 1.0 / 14.0
```

如果 boxes 已经是 feature coordinate，就不能再用 `1/14`。

本方案采用：

```text
normalized cxcywh → normalized xyxy → image pixel xyxy → ROIAlign(spatial_scale=1/14)
```

---

# 8. 训练 loss：ROI refined classification auxiliary loss

新增 criterion loss：

```text
loss_roi_subject_cls
loss_roi_object_cls
```

只对 matched relation 中的小 subject / small object 计算。

你需要在 matcher 输出中已有 relation 匹配：

```python
indices["subject"]
indices["object"]
```

通常包含：

```text
matched_pred_indices
matched_target_indices
```

伪代码：

```python
def loss_roi_refine(self, outputs, targets, indices, num_boxes):
    """
    outputs:
      relation_subject_logits_roi: [B, N, C_obj+1]
      relation_object_logits_roi:  [B, N, C_obj+1]
      roi_subject_mask: [B, N]
      roi_object_mask:  [B, N]

    targets:
      每张图包含 labels, boxes, image_relations 等

    indices:
      matcher 输出，能找到每个 relation query 对应哪个 GT relation
    """
```

更具体：

```python
def loss_roi_refine(self, outputs, targets, indices, num_boxes):
    loss_sub = outputs["relation_subject_logits_roi"].sum() * 0.0
    loss_obj = outputs["relation_object_logits_roi"].sum() * 0.0

    count_sub = 0
    count_obj = 0

    for b, target in enumerate(targets):
        # 根据你代码里 matcher 输出字段调整
        # 假设 relation match:
        src_idx, tgt_rel_idx = indices["relation"][b]

        if src_idx.numel() == 0:
            continue

        gt_relations = target["image_relations"]  # [num_rel, 3], [sub_id, obj_id, rel_id]
        gt_labels = target["labels"]              # [num_obj]
        gt_boxes = target["boxes"]                # [num_obj, 4], normalized cxcywh

        matched_rel = gt_relations[tgt_rel_idx]
        gt_sub_obj_idx = matched_rel[:, 0]
        gt_obj_obj_idx = matched_rel[:, 1]

        gt_sub_labels = gt_labels[gt_sub_obj_idx]
        gt_obj_labels = gt_labels[gt_obj_obj_idx]

        gt_sub_boxes = gt_boxes[gt_sub_obj_idx]
        gt_obj_boxes = gt_boxes[gt_obj_obj_idx]

        gt_sub_area = gt_sub_boxes[:, 2] * gt_sub_boxes[:, 3]
        gt_obj_area = gt_obj_boxes[:, 2] * gt_obj_boxes[:, 3]

        sub_small = gt_sub_area < self.roi_small_area_thresh
        obj_small = gt_obj_area < self.roi_small_area_thresh

        pred_sub_logits = outputs["relation_subject_logits_roi"][b, src_idx]
        pred_obj_logits = outputs["relation_object_logits_roi"][b, src_idx]

        if sub_small.any():
            loss_sub = loss_sub + F.cross_entropy(
                pred_sub_logits[sub_small],
                gt_sub_labels[sub_small],
            )
            count_sub += 1

        if obj_small.any():
            loss_obj = loss_obj + F.cross_entropy(
                pred_obj_logits[obj_small],
                gt_obj_labels[obj_small],
            )
            count_obj += 1

    if count_sub > 0:
        loss_sub = loss_sub / count_sub
    if count_obj > 0:
        loss_obj = loss_obj / count_obj

    return {
        "loss_roi_subject_cls": loss_sub,
        "loss_roi_object_cls": loss_obj,
    }
```

如果你的 `targets` 里 object labels / boxes 字段名不同，让 Cursor 按当前 criterion 中 `loss_labels` / `loss_boxes` 使用的字段名替换。

---

# 9. 推理阶段如何使用 refined logits

第一版建议：

```python
if self.training:
    # 原 logits 用于主 loss
    # roi logits 用于 auxiliary loss
else:
    # 推理时使用 roi refined logits 替换原 subject/object logits
```

推理替换：

```python
if "relation_subject_logits_roi" in outputs:
    outputs["relation_subject_logits"] = outputs["relation_subject_logits_roi"]

if "relation_object_logits_roi" in outputs:
    outputs["relation_object_logits"] = outputs["relation_object_logits_roi"]
```

更稳妥的融合：

```python
outputs["relation_subject_logits"] = (
    outputs["relation_subject_logits"] + alpha * outputs["relation_subject_logits_roi"]
) / (1.0 + alpha)
```

但注意 `relation_subject_logits_roi` 如果是 refined embedding 重新过同一个 head 得到的完整 logits，不是 residual logits，那么直接替换更自然。

推荐第一版用：

```text
训练：auxiliary
推理：replace
```

第二版再试：

```text
训练和推理：gated logit fusion
```

---

# 10. 小目标判断：用 GT 还是预测框？

分两处：

## ROIAlign 抽取时

使用预测框：

```text
predicted subject/object boxes
```

原因：

```text
推理时只有预测框，没有 GT。
```

第一版：

```python
DETACH_BOXES=True
```

## 训练 loss 选择 small 样本时

使用 GT 框判断 small：

```text
gt_area = gt_w × gt_h
```

原因：

```text
预测框早期不稳定，用它判断 small 会引入噪声。
```

所以：

```text
ROI 抽取：pred box
small loss mask：GT box
```

---

# 11. Deep supervision 是否要支持？

第一版不要支持 deep supervision，只 refine 最后一层：

```text
hs_subject[-1]
hs_object[-1]
```

原因：

```text
1. 代码简单
2. 训练稳定
3. 早期 decoder layer 的 box 很差，不适合 ROIAlign
```

如果后续要支持，可以对每层 aux_outputs 都加：

```text
relation_subject_logits_roi_l
relation_object_logits_roi_l
```

但不建议第一版做。

---

# 12. 与 triplet 架构的关系

这个 ROI branch 不会改变 triplet slot 对齐：

```text
第 k 个 triplet 仍然是：

subject_logits[k]
subject_box[k]
relation_logits[k]
object_logits[k]
object_box[k]
```

ROI refinement 只是把：

```text
subject_logits[k]
object_logits[k]
```

增强为：

```text
refined_subject_logits[k]
refined_object_logits[k]
```

不会改变：

```text
relation_logits[k]
query index k
Hungarian matching 的 triplet 对齐
```

所以不会破坏 SpeaQ triplet 架构。

---

# 13. Cursor 实现步骤

把下面直接交给 Cursor：

```text
请实现一个 ROI-guided small object refinement branch，用于 SpeaQ 的 subject/object 分类增强。

实现要求：

1. 新增配置：
MODEL.ROI_REFINE.ENABLED
MODEL.ROI_REFINE.STRIDE
MODEL.ROI_REFINE.POOL_SIZE
MODEL.ROI_REFINE.SMALL_AREA_THRESH
MODEL.ROI_REFINE.DETACH_BOXES
MODEL.ROI_REFINE.USE_GATE
MODEL.ROI_REFINE.LOSS_ENABLED
MODEL.ROI_REFINE.LOSS_WEIGHT
MODEL.ROI_REFINE.APPLY_TO

2. 新增文件：
modeling/transformer/roi_refine.py

实现：
- box_cxcywh_to_xyxy
- ROIRefineHead

ROIRefineHead 输入：
embeddings: [B, N, C]
boxes_cxcywh: [B, N, 4], normalized cxcywh
feature: [B, C, H14, W14]
image_h, image_w

ROIRefineHead 输出：
refined_embeddings: [B, N, C]
keep_mask: [B, N]

3. 在 SAM3 backbone 或 meta-arch 中拿到 stride14 auxiliary feature：
aux_features[14]: [B, 256, H14, W14]

4. 在 IterativeRelationDETR / transformer forward 的最后一层输出后接入：
sub_emb = hs_subject[-1]
obj_emb = hs_object[-1]
sub_boxes = relation_subject_coords[-1]
obj_boxes = relation_object_coords[-1]

调用 roi_refine_head 分别 refine subject/object embedding。

5. 使用原 object_embed head 对 refined embedding 重新分类：
relation_subject_logits_roi = self.object_embed(sub_emb_refined)
relation_object_logits_roi = self.object_embed(obj_emb_refined)

6. outputs 中新增：
relation_subject_logits_roi
relation_object_logits_roi
roi_subject_mask
roi_object_mask

7. 训练阶段：
保留原 relation_subject_logits / relation_object_logits 作为主 loss。
额外在 criterion 中加入：
loss_roi_subject_cls
loss_roi_object_cls

只对 matched relation 中 GT area < SMALL_AREA_THRESH 的 subject/object 计算 CE loss。

8. 推理阶段：
如果 ROI_REFINE.ENABLED=True，则使用 refined logits 替换原 subject/object logits：
outputs["relation_subject_logits"] = outputs["relation_subject_logits_roi"]
outputs["relation_object_logits"] = outputs["relation_object_logits_roi"]

9. 第一版只 refine 最后一层，不处理 aux_outputs deep supervision。

10. 所有新增模块需要在无 small ROI 的 batch 中返回 0 loss，不能报错。
```

---

# 14. Debug 输出建议

加一次性 debug：

```python
if debug:
    logger.info(
        f"[ROI_REFINE] feature={tuple(feature.shape)}, "
        f"sub_small={roi_subject_mask.sum().item()}, "
        f"obj_small={roi_object_mask.sum().item()}, "
        f"sub_emb={tuple(sub_emb.shape)}, "
        f"sub_emb_refined={tuple(sub_emb_refined.shape)}"
    )
```

期望看到：

```text
feature=(B,256,72,72)
sub_emb=(B,N,256)
sub_emb_refined=(B,N,256)
sub_small > 0
obj_small > 0
```

---

# 15. 最小可行版本总结

最终第一版结构：

```text
F14 high-res feature
        │
        ├── ROIAlign(subject boxes)
        │       ↓
hs_subject + ROI residual
        │       ↓
refined subject logits

F14 high-res feature
        │
        ├── ROIAlign(object boxes)
        │       ↓
hs_object + ROI residual
        │       ↓
refined object logits

relation logits 不变
relation boxes 不变
triplet slot index 不变
matcher 主逻辑不变
```

这版的优点是：

```text
1. 工程风险低
2. 不破坏 SpeaQ triplet 架构
3. 能直接利用 stride14 高分辨率特征
4. 专门针对 small object classification
5. 后续可以自然扩展到 embedding-level fusion / query group specialization
```
