# Quality-Aware Auxiliary CE Masking

## 1. 问题背景

SpeaQ 使用 triplet-level Hungarian matching。每个 query 输出三组预测（subject/object/relation），匹配代价是三元组级别的联合代价。只有三者都好的 query 被选中获得正监督，未被选中的全部标记为 no-object。

核心矛盾：某个 query 的 subject 头检测 table 完全正确（IoU 0.84），但因为 predicate 预测错误，被 triplet matcher 拒绝，其 subject 头被惩罚为"你检测 table 是错的"。模型收到矛盾信号，梯度冲突，收敛变慢。小目标和长尾类别受害最深。

AG 数据集中所有 relation 的 subject 都是 person（100%），object 分布在 36 类。

## 2. 方案：Quality-Aware Aux CE Masking

不对正确的检测进行惩罚。对每个 bg query，检查其 subject/object 头是否"单独看是对的"：
- class 正确
- IoU 高于阈值
- 分类置信度高于阈值

满足则从背景 CE 中豁免（target = -100 ignore_index）。不增加新 loss，只移除错误的负信号。

## 3. 伪代码

```
build_quality_aware_aux_indices(outputs, targets, combined_indices, cfg):
  for b in range(B):
    sub_matched = combined_indices["subject"][b][0]
    sub_bg = all_queries - sub_matched

    sub_logits = outputs["relation_subject_logits"][b]
    sub_prob = softmax(sub_logits)
    sub_score = max(sub_prob[:, :-1])
    sub_cls = argmax(sub_prob[:, :-1])

    for qi in sub_bg:
      if sub_score[qi] < MIN_SCORE_SUB: continue
      qc = sub_cls[qi]
      gt_match = where(gt_labels == qc)
      if len(gt_match) == 0: continue
      ious = compute_iou(sub_boxes[qi], gt_boxes[gt_match])
      best_iou, best_idx = max(ious)
      if best_iou >= IOU_THRESH_SUB:
        matched_q.append(qi)
        matched_gt.append(gt_match[best_idx])

    # Object branch: same logic with IOU_THRESH_OBJ, MIN_SCORE_OBJ
  return {"subject": indices, "object": indices}
```

CE Masking:
```
loss_labels(outputs, targets, indices, **kwargs):
  target_classes = full([B,N], num_classes)   # default no-object
  target_classes[matched] = gt_classes         # triplet-matched

  aux_ignored = kwargs.get("aux_masked_src")
  if aux_ignored:
    for b, qi in enumerate(aux_ignored):
      target_classes[b, qi] = -100             # ignore
```

## 4. 推理端扩展

Dual Scoring（TRIPLET_CONF_ALPHA）：利用 relation head 置信度区分 triplet-matched 和 quality-matched query。
```
final_score = detection_score * (1 - alpha + alpha * triplet_conf)
```

Shadow Discount（DISCOUNT_FACTOR）：同类别高 IoU 低分检测施加折扣，防止干扰 NMS。

## 5. 配置（默认全部关闭）

```yaml
OBJ_MISSED_AUX:
  ENABLED: False
  IOU_THRESH_SUB: 0.75
  IOU_THRESH_OBJ: 0.85
  MIN_SCORE_SUB: 0.0
  MIN_SCORE_OBJ: 0.0
  START_ITER: 0
  DISCOUNT_FACTOR: 0.0
  TRIPLET_CONF_ALPHA: 0.0
  DEBUG: False
```

## 6. 文件

| 文件 | 作用 |
|------|------|
| modeling/transformer/obj_missed_aux.py | 匹配逻辑 |
| modeling/transformer/criterion.py | CE masking + START_ITER |
| modeling/meta_arch/detr.py | 推理端 dual scoring + discount |
| configs/defaults.py | 配置节点 |
