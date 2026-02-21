# ActionGenome 预训练权重说明

## 当前加载流程

1. **训练启动**  
   `train_iterative_model.py` 中 `JointTransformerTrainer(cfg)` 会调用 detectron2 的 `DefaultTrainer.resume_or_load(resume=args.resume)`：
   - **resume=True** 且 `OUTPUT_DIR` 下存在 checkpoint → 加载该 checkpoint（含 optimizer/scheduler），不加载 `MODEL.WEIGHTS`。
   - **resume=False** 或没有 checkpoint → 调用 `checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)`，即加载 **MODEL.WEIGHTS** 指定的权重。

2. **评测 / 可视化**  
   `--eval-only` 或单独跑 `visualize_actiongenome_by_video.py` 时，会显式执行：
   `DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)`，因此会加载你在命令行或配置里传入的权重路径。

结论：**预训练权重的入口就是 `MODEL.WEIGHTS`，由 DetectionCheckpointer 统一加载，调用链是正确的。**

## 为何“预训练对训练帮助不大”？

- 当前 **speaq_actiongenome_minimal.yaml** 里默认是：
  `MODEL.WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-101.pkl"`
- 该权重是 **ImageNet 分类预训练的 ResNet-101**，只包含 backbone 的卷积层。
- DETR 的 **decoder、object/relation 的 class head、bbox head** 等都不在该 pkl 里，因此会保持 **随机初始化**。
- 所以训练前期主要是在学关系头和检测头，backbone 虽已预训练，整体 loss 下降仍会显得较慢，这是预期现象。

## 若希望关系头也有预训练

若你有 **VG 上训好的 DETR 权重**（例如 `vg_objectdetector_pretrained.pth`），可以只加载 DETR 部分（含关系头），而不覆盖 backbone 的 ImageNet 权重（或先加载 R-101 再叠 DETR head）：

1. 在配置或命令行中设置：
   - `MODEL.DETR.HEAD_WEIGHTS` = 指向 `vg_objectdetector_pretrained.pth`
   - `MODEL.DETR.LOAD_HEAD_ONLY` = True  
2. 在 `train_iterative_model.py` 的 `setup()` 里，当 `LOAD_HEAD_ONLY` 且 `HEAD_WEIGHTS` 非空时，会把 `MODEL.WEIGHTS` 清空，避免整图加载 VG 权重覆盖 backbone。
3. 在 `modeling/meta_arch/detr.py` 的 `__init__` 末尾会调用 `_load_detr_head_only(cfg.MODEL.DETR.HEAD_WEIGHTS)`，只把名字前缀为 `detr.` 且 shape 匹配的参数加载进当前模型。

这样 backbone 仍可用 ImageNet R-101（或你指定的 MODEL.WEIGHTS），关系与检测头则用 VG 预训练，通常收敛会更快。注意 VG 与 ActionGenome 的类别数/关系数若不一致，对应 head 的最后一层会因 shape 不匹配被跳过，其余层仍会加载。
