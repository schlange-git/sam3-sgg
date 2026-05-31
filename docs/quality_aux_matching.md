# Quality-Aware Auxiliary CE Masking for Scene Graph Generation

## 1. 问题背景

### 1.1 SpeaQ Triplet Matching 与训练流程

每个 relation query 通过三组独立 head 输出预测：

```
query_i -> subject_head -> sub_box [4], sub_logits [C+1]
        -> object_head  -> obj_box [4], obj_logits [C+1]
        -> relation_head-> rel_logits [R+1], rel_box [4]
```

训练时 SpeaQHungarianMatcher 做 triplet 级别 Hungarian matching。代价函数是三者的联合代价：

```
C(i, triplet_j) = C_cls(sub_logits[i], gt_sub_cls_j)
                + C_box(sub_box[i], gt_sub_box_j)
                + C_cls(obj_logits[i], gt_obj_cls_j)
                + C_box(obj_box[i], gt_obj_box_j)
                + C_cls(rel_logits[i], gt_pred_j)
```

匹配结果存入 `combined_indices`：

```
combined_indices = {
    "subject":  [(query_idx, gt_obj_idx), ...],
    "object":   [(query_idx, gt_obj_idx), ...],
    "relation": [(query_idx, gt_rel_idx), ...],
}
```

未被选中的 query，其 subject/object/relation 的 CE target 全部设为 `num_classes`（no-object 类），接受背景惩罚。

### 1.2 核心矛盾：正确检测被连坐惩罚

```
query_B:
  sub_box:  IoU=0.84 with GT table    (定位正确)
  sub_cls:  softmax -> table:0.84      (分类正确)
  obj_cls:  softmax -> chair:0.70      (分类正确)
  rel_logits: 谓词预测错误            (三元组失败的唯一原因)

triplet matcher: query_B 未被选中
  -> query_B 的 sub_head 被告知 target = no-object
  -> 梯度迫使 table 0.84 的响应降低
  -> head 自己确信是 table，loss 却说"你是背景"
```

**梯度冲突**：参数更新方向在"要检测 table"和"别检测 table"之间被拉扯。

### 1.3 影响分布

小目标（doorknob, medicine, groceries）的 predicate 预测天然更难：视觉特征弱、训练样本少、关系模式复杂。因此更容易被 triplet matcher 拒绝 -> 正确的检测被反复惩罚 -> 模型学会不检测小目标 -> Recall 下降。

### 1.4 AG 数据集特性

所有 relation 的 subject 都是 person（100%）。Object 分布在 36 类。Subject head 几乎只检测 person，定位精度高；Object head 检测多样化类别。

## 2. 方案设计

### 2.1 三阶段操作

1. **识别正确的 bg query**：class 正确 + IoU >= 阈值 + score >= 阈值
2. **CE Masking**：将这些 query 的 CE target 设为 -100（ignore_index），不惩罚
3. **不加新 loss**：head 已学会检测，不需要额外正监督

### 2.2 设计决策

**Q: 为什么不加正 aux loss？**
head 的 softmax 已经输出正确的 class 高分（比如 table 0.84）。额外正 loss 会导致 score 膨胀，推理时干扰 NMS。

**Q: 为什么 sub 和 obj 分开？**
两个 head 独立输出，检测质量不同。分开判断、分开豁免，避免跨分支信号泄漏。

**Q: 为什么 START_ITER 延迟？**
训练初期检测未收敛，此时豁免大量"蒙对的"bg query 会破坏背景监督。

**Q: 为什么推理时需要 Dual Scoring / Discount？**
训练时豁免了 bg query 的背景惩罚 -> 推理时这些 query 保留中等 detection score -> 在 NMS 中存活 -> 形成假阳性。推理端扩展是对训练-推理一致性的修正。

## 3. 核心伪代码

### 3.1 主匹配函数

```
function build_quality_aware_aux_indices(outputs, targets, combined_indices, cfg):

    device = outputs["relation_subject_logits"].device
    B, N, _ = outputs["relation_subject_logits"].shape

    # 读取阈值
    iou_thresh_sub   = float(getattr(cfg, "IOU_THRESH_SUB", 0.75))
    iou_thresh_obj   = float(getattr(cfg, "IOU_THRESH_OBJ", 0.85))
    min_score_sub    = float(getattr(cfg, "MIN_SCORE_SUB", 0.0))
    min_score_obj    = float(getattr(cfg, "MIN_SCORE_OBJ", 0.0))
    apply_sub        = bool(getattr(cfg, "APPLY_SUBJECT", True))
    apply_obj        = bool(getattr(cfg, "APPLY_OBJECT", True))
    do_log           = bool(getattr(cfg, "DEBUG", False))

    subject_indices = []   # per-image list of (query_idx_tensor, gt_idx_tensor)
    object_indices  = []
    num_sub = 0; num_obj = 0

    # 收集调试统计
    all_sub_ious = []; all_sub_scores = []
    all_obj_ious = []; all_obj_scores = []

    for b in range(B):
        gt_boxes  = targets[b]["combined_boxes"]      # [G, 4] cxcywh
        gt_labels = targets[b]["combined_labels"]      # [G]
        G = gt_boxes.shape[0]
        if G == 0: continue

        # Step 1: 标记 triplet matcher 已匹配的 query
        sub_matched_q, _ = combined_indices["subject"][b]
        obj_matched_q, _ = combined_indices["object"][b]
        all_q = arange(N)

        # ========================================================
        # SUBJECT BRANCH
        # ========================================================
        if apply_sub:
            # 找背景 query
            sub_used = zeros(N, dtype=bool)
            sub_used[sub_matched_q] = True
            sub_bg = all_q[~sub_used]     # 未被 triplet 选中的 query

            # 获取 subject head 预测
            sub_prob   = softmax(outputs["relation_subject_logits"][b], dim=-1)
            sub_score  = sub_prob[:, :-1].max(dim=-1).values  # [N] 置信度
            sub_cls    = sub_prob[:, :-1].argmax(dim=-1)       # [N] 预测类别
            sub_xyxy   = box_cxcywh_to_xyxy(outputs["relation_subject_boxes"][b])
            gt_xyxy    = box_cxcywh_to_xyxy(gt_boxes)

            mq, mt = [], []
            for qi in sub_bg:
                # (a) 分类置信度过滤
                if sub_score[qi] < min_score_sub: continue

                qc = sub_cls[qi]  # 预测类别

                # (b) 找同类别 GT
                gt_match = where(gt_labels == qc)
                if len(gt_match) == 0: continue

                # (c) 计算 IoU
                ious = compute_iou(sub_xyxy[qi], gt_xyxy[gt_match])
                best_iou, best_idx = max(ious)

                # (d) IoU 过滤
                if best_iou >= iou_thresh_sub:
                    mq.append(qi)
                    mt.append(gt_match[best_idx])
                    all_sub_ious.append(best_iou)
                    all_sub_scores.append(sub_score[qi])

            subject_indices[b] = (tensor(mq), tensor(mt))
            num_sub += len(mq)
        else:
            subject_indices[b] = (empty, empty)

        # ========================================================
        # OBJECT BRANCH（结构完全相同，使用 obj 专用阈值）
        # ========================================================
        if apply_obj:
            obj_used = zeros(N, dtype=bool)
            obj_used[obj_matched_q] = True
            obj_bg = all_q[~obj_used]

            obj_prob   = softmax(outputs["relation_object_logits"][b], dim=-1)
            obj_score  = obj_prob[:, :-1].max(dim=-1).values
            obj_cls    = obj_prob[:, :-1].argmax(dim=-1)
            obj_xyxy   = box_cxcywh_to_xyxy(outputs["relation_object_boxes"][b])

            mq, mt = [], []
            for qi in obj_bg:
                if obj_score[qi] < min_score_obj: continue
                qc = obj_cls[qi]
                gt_match = where(gt_labels == qc)
                if len(gt_match) == 0: continue
                ious = compute_iou(obj_xyxy[qi], gt_xyxy[gt_match])
                best_iou, best_idx = max(ious)
                if best_iou >= iou_thresh_obj:
                    mq.append(qi)
                    mt.append(gt_match[best_idx])
                    all_obj_ious.append(best_iou)
                    all_obj_scores.append(obj_score[qi])

            object_indices[b] = (tensor(mq), tensor(mt))
            num_obj += len(mq)
        else:
            object_indices[b] = (empty, empty)

    # 调试日志
    if do_log:
        log("[QualityAux] sub=%d obj=%d", num_sub, num_obj)
        if all_sub_ious:
            log("[QualityAux] sub IoU: min=%.3f mean=%.3f max=%.3f",
                min(all_sub_ious), mean(all_sub_ious), max(all_sub_ious))

    return {
        "subject": subject_indices, "object": object_indices,
        "_cum_sub": num_sub, "_cum_obj": num_obj,
        "_sub_ious": all_sub_ious, "_sub_scores": all_sub_scores,
        "_obj_ious": all_obj_ious, "_obj_scores": all_obj_scores,
    }
```

### 3.2 CE Masking 实现

```
function loss_labels(outputs, targets, indices, num_boxes, log, **kwargs):

    src_logits = outputs["pred_logits"]   # [B, N, C+1]

    # --- Step 1: 标准 target 构建 ---
    idx = get_src_permutation_idx(indices)
    target_classes_o = concat([target["labels"][J] for target, (_, J) in zip(targets, indices)])
    target_classes = full([B, N], num_classes, dtype=int64)   # 默认 no-object
    target_classes[idx] = target_classes_o                     # 匹配的设为 GT class

    # --- Step 2: Quality aux CE masking ---
    aux_ignored = kwargs.get("aux_masked_src", None)
    if aux_ignored is not None:
        for b, q_indices in enumerate(aux_ignored):
            if len(q_indices) > 0:
                target_classes[b, q_indices] = -100  # cross_entropy ignore_index

    # --- Step 3: 计算 loss ---
    loss_ce = cross_entropy(
        src_logits.transpose(1, 2),
        target_classes,
        weight=empty_weight_obj
    )
    return {"loss_ce": loss_ce}
```

### 3.3 Criterion.forward 集成

```
function forward(outputs, targets):

    losses = {}

    # Phase 1: Triplet matching（完全不变）
    relation_outputs = {k:v for k,v in outputs.items() if "relation" in k}
    combined_indices = self.matcher.forward_relation(outputs, targets, ...)

    # Phase 2: Quality aux matching（仅 START_ITER 之后）
    aux_masked_subject = None
    aux_masked_object = None
    if self.obj_missed_aux_enabled:
        storage = get_event_storage()
        if storage.iter >= START_ITER:
            aux_indices = build_quality_aware_aux_indices(
                outputs=relation_outputs,
                targets=targets,
                combined_indices=combined_indices,
                cfg=self.obj_missed_aux_cfg,
            )
            # 提取豁免 query 列表
            aux_masked_subject = [aux_indices["subject"][b][0] for b in range(B)]
            aux_masked_object  = [aux_indices["object"][b][0] for b in range(B)]

            # 累计统计（每 100 iter 输出一次）
            self._qaux_cum_sub += aux_indices["_cum_sub"]
            self._qaux_cum_obj += aux_indices["_cum_obj"]
            if storage.iter - self._qaux_last_log_iter >= 100:
                log("[QualityAux] cum: sub=%d obj=%d (iter %d)", ...)

    # Phase 3: Primary losses（传入 aux_masked 用于 CE masking）
    losses.update(self.get_relation_losses(
        relation_outputs, entity_targets, relation_targets,
        combined_indices,
        aux_masked_subject=aux_masked_subject,   # 新增
        aux_masked_object=aux_masked_object,      # 新增
    ))
    return losses
```

### 3.4 get_relation_losses 传递 aux_masked

```
function get_relation_losses(outputs, targets, combined_indices, **kwargs):

    aux_masked_sub = kwargs.pop("aux_masked_subject", None)
    aux_masked_obj = kwargs.pop("aux_masked_object", None)

    # Subject branch
    kw_sub = {**kwargs, "aux_masked_src": aux_masked_sub}
    sub_losses = self.loss_labels(sub_outputs, targets, combined_indices["subject"], **kw_sub)

    # Object branch
    kw_obj = {**kwargs, "aux_masked_src": aux_masked_obj}
    obj_losses = self.loss_labels(obj_outputs, targets, combined_indices["object"], **kw_obj)

    # Relation branch（不受影响）
    rel_losses = self.get_relation_loss(outputs, targets, combined_indices["relation"], ...)
```

## 4. 训练 vs 推理：Score 的完整逻辑

### 4.1 训练阶段

训练时不使用 detection score。只计算 loss：

| Query 类型 | CE target | 效果 |
|-----------|-----------|------|
| triplet-matched | GT class | 正监督（正常学习） |
| quality-matched bg | -100 (ignore) | **不惩罚**（修正） |
| 其他 bg | num_classes (no-object) | 背景惩罚（正常） |

### 4.2 推理阶段（标准）

```
# 计算 detection scores
scores_s = softmax(relation_subject_logits)[:, :-1].max(-1)   # [N]
scores_o = softmax(relation_object_logits)[:, :-1].max(-1)    

# NMS
image_scores = concat([scores_s, scores_o])     # [2N]
image_boxes  = concat([sub_boxes, obj_boxes])   # [2N, 4]
keep = batched_nms(image_boxes, image_scores, labels, nms_thresh)

result.scores = image_scores[keep]     # AP 计算用这个 score
```

### 4.3 Quality Aux 对推理 Score 的影响

标准训练：bg query 被 CE 压低 -> 推理 score 低 -> NMS 淘汰。
Quality aux：bg query 未被压低 -> 推理 score 中等 -> NMS 存活 -> 假阳性 -> AP 下降。

**机制本质**：训练时"不惩罚"使得 bg query 保留了不应有的 score，破坏了 NMS 依赖的 score 排序。

### 4.4 Dual Scoring 修正

```
triplet_conf = softmax(relation_logits)[:, :-1].max(-1)    # 谓词置信度
scores_s *= (1 - alpha + alpha * triplet_conf)              # 调制检测分数
```

- triplet-matched query: 谓词好 -> triplet_conf 高 -> score 不降
- quality-matched query: 谓词差 -> triplet_conf 低 -> score 被压低 -> NMS 淘汰

### 4.5 Shadow Discount 修正

```
for each class in unique(labels):
    class_dets = detections[labels == class]
    sort by score desc
    for k in 1..len(class_dets)-1:
        max_iou = max_pairwise_iou_with_higher_scored_dets(k)
        if max_iou > DISCOUNT_IOU_THRESH:
            class_dets[k].score *= DISCOUNT_FACTOR
```

### 4.6 完整推理流程

```
function inference(image):
    output = model(image)

    # 1. 标准 detection scores
    scores_s, labels_s = softmax(sub_logits)[:,:-1].max(-1)
    scores_o, labels_o = softmax(obj_logits)[:,:-1].max(-1)

    # 2. Dual Scoring
    if TRIPLET_CONF_ALPHA > 0:
        r_conf = softmax(rel_logits)[:,:-1].max(-1)
        scores_s *= (1 - alpha + alpha * r_conf)
        scores_o *= (1 - alpha + alpha * r_conf)

    # 3. Shadow Discount
    if DISCOUNT_FACTOR > 0:
        for (scores, labels, boxes) in [(scores_s, labels_s, sub_boxes),
                                         (scores_o, labels_o, obj_boxes)]:
            for cls in unique(labels):
                apply_discount(cls, scores, labels, boxes)

    # 4. NMS + Triplet formation
    keep = nms(concat([sub_boxes, obj_boxes]), concat([scores_s, scores_o]), ...)
    result = build_result(keep, scores, labels, boxes)

    # 5. ROI cls post-placement（覆盖 pred_classes，不影响 score）
    if ROI_REFINE.ENABLED:
        result.pred_classes = roi_refined_classes[keep]

    return result
```

### 4.7 训练/推理对比总结

| 阶段 | Score 来源 | quality aux 效应 | 修正机制 |
|------|-----------|-----------------|---------|
| 训练 | 无 | CE masking | - |
| 推理-标准 | softmax(max) | bg score 未被压低 -> AP 可能下降 | - |
| 推理+dual scoring | softmax(max) × triplet_conf | triplet_conf 压低 quality-matched | TRIPLET_CONF_ALPHA |
| 推理+discount | softmax(max) × discount | 同类别低分被打折 | DISCOUNT_FACTOR |

## 5. 配置参数

```yaml
MODEL.DETR.OBJ_MISSED_AUX:
  ENABLED: False              # 全局开关（默认 OFF）
  IOU_THRESH_SUB: 0.75
  IOU_THRESH_OBJ: 0.85
  MIN_SCORE_SUB: 0.0
  MIN_SCORE_OBJ: 0.0
  APPLY_SUBJECT: True
  APPLY_OBJECT: True
  START_ITER: 0               # 延迟激活 iter 数
  LOSS_WEIGHT: 0.0            # 正 aux loss（0=仅 CE masking）
  DISCOUNT_FACTOR: 0.0        # 推理端 shadow 折扣
  DISCOUNT_IOU_THRESH: 0.5
  TRIPLET_CONF_ALPHA: 0.0     # 推理端 dual scoring
  DEBUG: False
```

## 6. 文件

| 文件 | 作用 |
|------|------|
| `modeling/transformer/obj_missed_aux.py` | 匹配逻辑 `build_quality_aware_aux_indices()` |
| `modeling/transformer/criterion.py` | CE masking + START_ITER + 累计日志 |
| `modeling/meta_arch/detr.py` | Dual scoring + discount 推理逻辑 |
| `configs/defaults.py` | OBJ_MISSED_AUX 配置（全部默认 OFF） |
