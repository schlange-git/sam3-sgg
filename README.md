# SAM3-SGG with Temporal Triplet Memory

基于 [SpeaQ](https://arxiv.org/abs/2403.17709)（CVPR 2024）的视频场景图生成（Video SGG）实现。  
本仓库在 Action Genome 上接入 **SAM3 backbone**、**X-SAM patch_merge**、**ROI Refine**，并加入 **Temporal Triplet Memory (v3)**：以三元组粒度维护跨帧记忆，经门控 cross-attention 注入当前帧的 object / relation query。

当前主分支：`main`。

## 核心思想

Action Genome 中同一 `(person, predicate, object)` 往往会在相邻帧持续出现，但单帧 SGG 无法利用这种时序先验。v3 做法：

1. **Read**：从当前视频的 memory bank 取出有效 slot，加 \(\Delta t\) 编码与置信度加权  
2. **Gated-Inject**：用 cross-attention 把记忆注入 object / relation query（蓝本固定门控 `gate_obj=0.30`，`gate_rel=0.45`），再重算最后一层 logits / boxes  
3. **Write**（`no_grad`）：按 `quality = obj_score × pred_score` 取 top-k 三元组写入 bank（命中 EMA / 未命中插入替换 / miss 超限淘汰）

约束：**读+注入可训练，写路径全程 detach**；时序新模块用更高学习率（如 `TEMPORAL_LR_MULTIPLIER=100`）。

## 主要模块

| 模块 | 说明 | 关键配置 / 代码 |
|------|------|----------------|
| SAM3 Backbone | 冻结 ViT，输出统一特征图 | `MODEL.SAM3.*`，`modeling/backbone/sam3_backbone.py` |
| X-SAM patch_merge | pixel-unshuffle + 可学习 1×1 conv 下采样（需在 `__init__` 预建以便加载预训练） | `MODEL.SAM3.USE_PATCH_MERGE` |
| ROI Refine | stride-14 ROI 特征 refine，支持 residual 融合 | `MODEL.ROI_REFINE.*` |
| Triplet Memory v3 | 每视频 ≤32 slot 的三元组记忆库 | `MODEL.TEMPORAL.MODE=triplet_memory_v3`，`modeling/temporal/triplet_memory.py` |
| Clip sampler | 按视频连续帧采样，便于时序更新 | `DATASETS.ACTION_GENOME.SAMPLER_MODE=clip` |

## 环境安装

```bash
conda create -n speaq python=3.9
conda activate speaq

pip install -r requirements.txt

# Detectron2（按本机 CUDA / PyTorch 版本选择对应 wheel）
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/index.html

# 本地 SAM3
cd sam3 && pip install -e . && cd ..
```

实测常用组合：PyTorch 2.x + 对应 CUDA + Detectron2。`detectron2/`、权重 `*.pth`、数据集目录等已在 `.gitignore` 中忽略。

## 数据准备（Action Genome）

期望目录结构（可用软链）：

```text
dataset/
├── annotations/          # AG 标注
└── frames/               # 抽帧结果
```

抽帧（关键帧之间可均匀插帧，默认每区间额外 1 帧）：

```bash
# VIDEO_DIR / FRAME_DIR / ANNOTATION_DIR 可按需覆盖
bash tools/prepare_actiongenome_frames.sh
```

损坏图片列表见 `broken_images.txt`（`DATASETS.ACTION_GENOME.BROKEN_IMAGES_LIST`）。

过拟合子集标注：

- `dataset_overfit/`
- `dataset_overfit_temporal/`
- `dataset_overfit_temporal_20v/`

## 训练

统一入口：`train_iterative_model.py`。常用脚本在 `tools/`。

### 1. 全量训练（SAM3 + patch_merge + ROI + triplet_memory_v3）

生产配置：`configs/fulltask_roiresid_pm_clip_v3_xsam_bs24_20w.yaml`  
（clip 采样、patch_merge、ROI 残差、时序 v3；起始权重为 X-SAM 预训练）

```bash
# 8 卡 · bs96 · 80k iter
bash tools/train_fulltask_v3_8gpu_bs96_80k.sh [OUTPUT_DIR] [NUM_GPUS]

# 2 卡小规模
bash tools/train_fulltask_v3_2gpu_bs24_20w.sh
```

### 2. 时序微调（固定门控，推荐蓝本）

在已收敛的全量权重（如 `model_0079999.pth`）上，固定 `gate_obj=0.30` / `gate_rel=0.45`，提高时序模块 LR：

```bash
# 8 卡 · bs16 · 20k iter
bash tools/run_temporal_finetune_80k_fixed_gate030_045_8gpu_ml_bs16.sh

# 单卡 / 2 卡变体
bash tools/run_temporal_finetune_79999_bs8_20k_fixed_gate030_045_single.sh
bash tools/run_temporal_finetune_79999_bs16_20k_fixed_gate030_045_2gpu.sh
```

可学习门控实验：

```bash
bash tools/run_temporal_finetune_79999_bs8_20k_learnable_gate_single.sh
bash tools/run_temporal_finetune_79999_bs8_20k_learnable_gate100x_single.sh
```

### 3. Overfit / 诊断

```bash
# clip_sample 过拟合（20v）
bash tools/overfit_clipsample_1gpu_bs16_20v_pscale300.sh

# patch_merge 相关探针
bash tools/overfit_patchmerge_globalavg_2gpu_20v.sh
bash tools/full_patchmerge_globalavg_overfit5999_2gpu.sh
```

### 4. 权重加载注意

- 整模加载：`MODEL.WEIGHTS` + `MODEL.DETR.LOAD_FULL_WEIGHTS True`
- 仅 DETR head：`MODEL.DETR.HEAD_WEIGHTS` + `LOAD_HEAD_ONLY True`
- `USE_PATCH_MERGE=True` 时必须预建 patch_merge conv，否则预训练权重会被丢弃并回退 avg_pool

## 评测

```bash
python train_iterative_model.py --eval-only --num-gpus 1 \
  --config-file <CONFIG> \
  OUTPUT_DIR <OUTPUT_DIR> \
  MODEL.WEIGHTS <CHECKPOINT> \
  DATASETS.ACTION_GENOME.ANNOTATIONS dataset/annotations \
  DATASETS.ACTION_GENOME.FRAMES dataset/frames \
  MODEL.TEMPORAL.ENABLED True \
  MODEL.TEMPORAL.EVAL_ENABLED True
```

有序 / 打乱帧评测脚本示例：

```bash
bash tools/run_temporal_ordered_eval_m3.sh
bash tools/run_temporal_shuffle_eval_m3.sh
```

结果提取与对比：

```bash
python save_eval_results.py <OUTPUT_DIR> --name "exp"
python compare_results.py <dir1> <dir2>
```

详见 [`EVAL_TOOLS_README.md`](EVAL_TOOLS_README.md)。早期 VG 全链路脚本说明见 [`PIPELINE_README.md`](PIPELINE_README.md)。

## 可视化

```bash
python visualize_actiongenome_by_video.py   # 按视频可视化预测
python visualize_predictions.py
```

## 仓库结构（简）

```text
├── train_iterative_model.py     # 训练 / 评测入口
├── configs/                     # yaml + defaults.py
├── modeling/
│   ├── backbone/                # SAM3、patch_merge
│   ├── temporal/                # triplet_memory v3
│   └── transformer/             # DETR decoder、注入点、ROI refine
├── data/datasets/               # Action Genome / VG
└── tools/                       # 训练、评测、抽帧脚本
```

## 引用

若使用本仓库的 SpeaQ 基线部分，请引用：

```bibtex
@inproceedings{kim2024groupwise,
  title={Groupwise Query Specialization and Quality-Aware Multi-Assignment for Transformer-based Visual Relationship Detection},
  author={Kim, Jongha and Park, Jihwan and Park, Jinyoung and Kim, Jinyoung and Kim, Sehyung and Kim, Hyunwoo J},
  booktitle={CVPR},
  year={2024}
}
```

## Acknowledgements

- [SpeaQ](https://github.com/) / [Iterative Scene Graph Generation](https://github.com/ubc-vision/IterativeSG)
- [SAM 3](https://github.com/facebookresearch/sam3)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- Action Genome / Charades 数据集提供方
