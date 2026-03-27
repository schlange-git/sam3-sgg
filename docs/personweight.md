Commit: e74264cdef689c3ab46e687a8bfad3c85f8d7366
Topic: 提升 person 类别分数（score-time prior）

==================================================
1) 功能目标
==================================================
在关系分支中，对 subject/object 的分类 logits 的 person 类别维度施加固定偏置：

    person_logit <- person_logit + log(PERSON_SCORE_SCALE)

作用是提升 person 类别先验，不改动训练标签和损失定义，只在输出 logits 层做偏置。

==================================================
2) 需要修改的代码位置（完整）
==================================================

[A] configs/defaults.py
函数位置：
    add_dataset_config(cfg)

修改点 1：新增 DETR 配置项
    _C.MODEL.DETR.PERSON_SCORE_SCALE = 1.0
    _C.MODEL.DETR.PERSON_CLASS_INDEX = 0

说明：
1. PERSON_SCORE_SCALE 默认 1.0（即不生效）。
2. PERSON_CLASS_INDEX 在 Action Genome 里通常是 0（person 在 object_classes.txt 的第一个）。


[B] configs/speaq_actiongenome_minimal.yaml
路径位置：
    MODEL -> DETR

修改点 2：在实验配置中显式打开该能力
    PERSON_SCORE_SCALE: 300.0

说明：
1. 这是实验值，不是必须值；可按任务调参。
2. 该值会在前向中转为 log(300.0) 加到 person 类别 logit 上。


[C] modeling/transformer/detr.py
类位置：
    class IterativeRelationDETR(DETR)

修改点 3：在 __init__ 读取配置
    self.person_score_scale = float(getattr(cfg.MODEL.DETR, "PERSON_SCORE_SCALE", 1.0))
    self.person_class_index = int(getattr(cfg.MODEL.DETR, "PERSON_CLASS_INDEX", 0))

修改点 4：在 transformer 输出后、任何下游使用前，注入 person logit 偏置

建议放置位置：
    output = self.transformer(...)
    之后；
    only_predicate_multiply 前；
    temporal memory update 前。

关键逻辑：
    if self.person_score_scale > 0 and abs(self.person_score_scale - 1.0) > 1e-8:
        log_scale = math.log(self.person_score_scale)
        for key in ("relation_subject_logits", "relation_object_logits"):
            logits = output.get(key)
            if logits is None:
                continue
            if logits.shape[-1] <= self.person_class_index:
                continue
            logits[..., self.person_class_index] = logits[..., self.person_class_index] + log_scale

说明：
1. 只改 relation_subject_logits 和 relation_object_logits。
2. 采用 log(scale) 是为了与 softmax 概率空间一致（logit 加法等价于概率乘法先验）。
3. 在“memory update/inference/eval”之前执行，保证缓存、后处理、评测看到同一套偏置后输出。

==================================================
3) 最小可运行改法（不依赖其它时序改动）
==================================================
若你只想启用 person 分数提升，不想引入额外改动，最小集合就是：

1. defaults.py 增加两个 config 字段：
   - MODEL.DETR.PERSON_SCORE_SCALE
   - MODEL.DETR.PERSON_CLASS_INDEX

2. transformer/detr.py 的 IterativeRelationDETR：
   - __init__ 读取上述两个字段
   - forward() 中 output = self.transformer(...) 后，插入 logit 偏置逻辑

3. 实验 yaml 设置：
   - MODEL.DETR.PERSON_SCORE_SCALE: 例如 300.0

==================================================
4) 验证建议
==================================================
1. 配置检查：
   训练启动日志确认 PERSON_SCORE_SCALE 非 1.0。

2. 行为检查：
   打印 bias 前后 relation_subject_logits / relation_object_logits 的 person 维度均值，
   应出现稳定上移约 log(PERSON_SCORE_SCALE)。

3. 结果检查：
   重点看 person 相关关系召回变化，避免仅整体分数波动但关系质量下降。

==================================================
5) 注意事项
==================================================
1. PERSON_CLASS_INDEX 必须和数据集类别索引一致，否则会错误提升其它类别。
2. PERSON_SCORE_SCALE 过大可能导致类别过偏置，建议从小到大网格试验（如 1, 3, 10, 30, 100, 300）。
3. 该逻辑是 score-time prior，不是 loss reweight；与 PERSON_CLASS_WEIGHT（训练损失权重）是两条独立机制。

