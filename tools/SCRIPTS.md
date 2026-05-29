# Tools Script Index

本文件用于快速区分 `tools/` 下各个 `.sh` 脚本的实验目的、关键开关和默认训练规模。路径中的输出目录均可通过脚本第一个参数覆盖；多数脚本还支持通过 `OPTS="KEY VALUE ..."` 追加配置。

## 重要脚本总览

| 脚本 | 主要用途 | 默认输出目录 | 数据/评测 | 关键模型开关 | 默认训练规模 | 备注 |
|---|---|---|---|---|---|---|
| `overfit_xsam_no_roi_bs12_16000.sh` | 单独验证 X-SAM patch merge 是否能在 overfit temporal 子集上收敛 | `z_outputs/overfit_xsam_no_roi_bs12_16000` | `dataset_overfit_temporal`；train/test 都是 `AG_train` | `SAM3=True`，`USE_PATCH_MERGE=True`，`TEMPORAL=False`，`ROI_REFINE=False` | 单卡，BS=12，16000 iter，LR=1e-4，steps=(4000,12000)，eval=2000 | 当前本地只运行这个作为第一项诊断 |
| `overfit_temporal_v2_relation_no_roi_bs12_16000.sh` | 复现原始 temporal v2 relation memory 的过拟合问题 | `z_outputs/overfit_temporal_v2_relation_no_roi_bs12_16000` | `dataset_overfit_temporal`；train/test 都是 `AG_train` | `SAM3=True`，`USE_PATCH_MERGE=False`，`TEMPORAL=True`，`RELATION_MEMORY=True`，`ROI_REFINE=False` | 单卡，BS=12，16000 iter，LR=1e-4，gate LR x5，eval=2000 | 用于对比修复前 v2 relation memory 行为 |
| `overfit_temporal_v2_relation_fixed_all_no_roi_bs12_16000.sh` | 开启全部 temporal relation 修复后的 overfit probe | `z_outputs/overfit_temporal_v2_relation_fixed_all_no_roi_bs12_16000` | `dataset_overfit_temporal`；train/test 都是 `AG_train` | `SAM3=True`，`USE_PATCH_MERGE=False`，`TEMPORAL=True`，独立 relation memory，matched-GT memory update，object gate max=0.20/warmup=2000，relation gate max=0.10/warmup=3000，`ROI_REFINE=False` | 单卡，BS=12，16000 iter，LR=1e-4，gate LR x5，eval=2000 | 推荐作为修复后第一轮 v2 debug 脚本 |
| `run_overfit_xsam_then_temporal_v2_no_roi_bs12_16000.sh` | 顺序运行 X-SAM probe 和原始 temporal v2 relation probe | 无独立输出目录 | 依次调用两个 overfit 脚本 | 第一步 X-SAM only；第二步 temporal v2 relation only | 单卡顺序运行 | 为避免抢显存而串行；当前已停止父脚本，不会自动接 v2 |
| `reproduce_sam3_roi_eiou_cornerloss_dist.sh` | 尽量复现旧的稳定 SAM3 + ROI14 + eIoU/corner-loss BS24 实验 | `z_outputs/repro_sam3_roi_eiou_cornerloss_bs24_160000_dist` | 正常 AG train/val | `SAM3=True`，`ROI_REFINE=True`，`SMALL_AREA_THRESH=1.0`，`BOX_LOSS_TYPE=eiou`，`USE_CORNER_LOSS=True`，`PERSON_SCORE_SCALE=200`，`TEMPORAL=False`，`USE_PATCH_MERGE=False` | 4 GPU，BS=24，160000 iter，LR=1e-4，steps=(40000,120000)，eval=20000 | 会写 `roi_gate_log.csv`，用于 ROI gate 融合比例追踪 |
| `reproduce_sam3_roi_eiou_cornerloss_dist_bs48.sh` | BS48 等效复现旧稳定 ROI 实验，样本数与 BS24/160k 对齐 | `z_outputs/repro_sam3_roi_eiou_cornerloss_bs48_80000_dist` | 正常 AG train/val | 同 BS24 复现脚本，`TEMPORAL=False`，`USE_PATCH_MERGE=False` | 4 GPU，BS=48，80000 iter，LR=2e-4，steps=(20000,60000)，eval=10000 | 推荐用于大显存环境下复现 ROI baseline |
| `fulltask_pretrain_roi14_dist.sh` | full task + ROI refine 的分布式训练入口 | `z_outputs/full_roi14_only_bs48_from99999pth_iter50000` | 正常 AG train | `SAM3=True`，`ROI_REFINE=True`，`TEMPORAL=False`，当前脚本里 `USE_PATCH_MERGE=False` | 4 GPU，BS=48，50000 iter，LR=4e-4，steps=(7500,22500)，eval=5000 | 名称里历史写 X-SAM + ROI14，但当前关键开关是 patch merge off、ROI on |
| `detectiononly_pretrain_dist.sh` | detection-only 预训练，验证检测分支和 X-SAM | `z_outputs/xsam_pretrained_bs48_iter50000_from_99999pth` | 正常 AG train | `DETECTION_ONLY=True`，`SAM3=True`，`USE_PATCH_MERGE=True`，`TEMPORAL=False` | 4 GPU，BS=48，50000 iter，LR=4e-4，steps=(7500,22500)，eval=5000 | 用于先训练/检查检测分支，不训练 relation loss |
| `run_actiongenome_train_eval.sh` | 通用 Action Genome 训练、评测和可视化入口 | `z_outputs/sam3_res101_from_pretrained_160000iters_bs8` | 默认正常 AG train/val；可通过 `NUM_VIDEOS_TRAIN` 做小视频 overfit | 依赖 config 和传入 `OPTS`；支持 DETR head/full 权重加载 | 默认单卡；训练规模主要由 config/OPTS 控制 | 功能最全，但参数较多，适合正式训练管线 |
| `run_actiongenome_eval.sh` | 对已有 checkpoint 做 eval-only，可选 overfit train eval | 传入训练输出目录 | 默认 AG_val；`EVAL_OVERFIT_TRAIN=1` 可追加 train split eval | 只加载 checkpoint，不训练 | 默认 2 GPU eval | 自动选择 `CHECKPOINT`、`model_final.pth` 或 `last_checkpoint` |
| `run_actiongenome_roi_refine.sh` | ROI refine 训练/小规模 overfit 调试入口 | `z_outputs/sam3_roi_eiou_cornerloss_bs8_localdebug` | 正常 AG；第三参数可切 overfit | 使用 `configs/speaq_ag_roi.yaml`，可加载 DETR full/head-only 权重，默认 ROI 配置在 config 中 | 默认自动/指定 GPU；overfit 模式可设帧数和 iter | 老的 ROI 调试入口，和两个 `reproduce_*` 脚本相比可控性更泛化 |
| `run_actiongenome_temporal_nonkey.sh` | temporal non-key 训练实验入口 | `z_outputs/temporal_nonkey_160000iters_bs16` | 正常 AG，可指定视频数 | `TEMPORAL=True`，`NON_KEY_SKIP_LOSS=True`，`NON_KEY_SKIP_EVAL=True`，`NON_KEY_RUN_OBJECT_ONLY=True` | 默认单卡，BS=16 | 用于非关键帧 temporal 缓存/跳过逻辑实验 |
| `run_actiongenome_obj_missed_aux.sh` | object missed auxiliary matching 训练入口 | `z_outputs/sam3_objmissedaux_160000iters_bs16` | 正常 AG，可指定视频数 | 使用 `configs/speaq_actiongenome_obj_missed_aux.yaml`，默认开启 `MODEL.DETR.OBJ_MISSED_AUX.*` | 默认 2 GPU；训练规模由脚本和 config 控制 | 可用 `AUX_*` 环境变量调辅助匹配 IoU、loss、small/tail 策略 |
| `visual.sh` | checkpoint 可视化，必要时附带 eval-only | `<CKPT_DIR>/<CKPT_NAME>_vis` | 默认 `AG_val`，可通过参数切数据集 | 不训练；可设置 `PERSON_SCORE_BOOST`、box/rel score threshold | 默认可视化 1000 张；eval 默认单卡 | 适合快速检查预测框和关系可视化 |
| `prepare_actiongenome_frames.sh` | 从 Action Genome/Charades 视频抽帧 | `dataset-full/frames` | 读取 `dataset-full/Charades_v1` 和 `dataset-full/annotations` | 不涉及模型 | 非训练脚本 | 可设置 `KEEP_ALL_FRAMES=1` 或 `EXTRA_FRAMES_PER_INTERVAL` |

## 使用建议

| 场景 | 推荐脚本 |
|---|---|
| 判断 X-SAM patch merge 是否破坏 overfit 收敛 | `overfit_xsam_no_roi_bs12_16000.sh` |
| 判断原始 v2 relation memory 是否是低 recall 来源 | `overfit_temporal_v2_relation_no_roi_bs12_16000.sh` |
| 验证已修复 temporal relation memory 是否可过拟合 | `overfit_temporal_v2_relation_fixed_all_no_roi_bs12_16000.sh` |
| 复现旧稳定 ROI baseline | `reproduce_sam3_roi_eiou_cornerloss_dist_bs48.sh` 或 `reproduce_sam3_roi_eiou_cornerloss_dist.sh` |
| 只跑评测不训练 | `run_actiongenome_eval.sh` |
| 看预测图和关系图 | `visual.sh` |

## 关于 `.gitignore`

当前 `.gitignore` 已包含 `docs/`、`z_outputs/`、`dataset/`、`detectron2/` 等规则；这些规则只影响未被 Git 跟踪的新文件。已经被 Git 跟踪过的文件，即使路径命中 ignore 规则，后续修改或删除仍会出现在 `git status` 中。若要让某个已跟踪目录以后真正不再进入 changes，需要显式从索引移除，例如 `git rm --cached -r docs`，再提交这次索引变更。
