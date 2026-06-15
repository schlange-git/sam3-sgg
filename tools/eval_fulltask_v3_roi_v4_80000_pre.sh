#!/usr/bin/env bash
# 用 ROI_EVAL_RAW=1 (pre-refine: 跳过 roi 覆盖, eval 用 raw/原始 logits) 评测
# 目的: 对 fulltask_v3_roi_v4 (convex 训练) 的 model_0079999 做"不覆盖(取代前)"评测,
#       与其训练期 override-ON(取代后) 结果 det_R@50=62.76 对比, 看 raw 效果
# 说明: 该 ckpt 为 convex 训练, 当前远端代码也是 convex -> fusion 一致, 无 train/eval 失配;
#       本次唯一变量是 override(ROI_EVAL_RAW=1 跳过替换), 用于观察 override 的 recall/AP 取舍
# checkpoint = fulltask_v3_roi_v4_eiou_corner_bs48_iter80000 / model_0079999.pth
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate base
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
EXP=z_outputs/fulltask_v3_roi_v4_eiou_corner_bs48_iter80000
WEIGHT="${EXP}/model_0079999.pth"
CFG="${EXP}/config.yaml"
OUTPUT_DIR="z_outputs/eval_fulltask_v3_roi_v4_80000_pre"
mkdir -p "${OUTPUT_DIR}"
export ROI_EVAL_RAW=1
echo "[eval] ckpt=model_0079999 logic=pre(raw) ROI_EVAL_RAW=${ROI_EVAL_RAW} OUT=${OUTPUT_DIR}"
python3 train_iterative_model.py --num-gpus 1 --eval-only --config-file "${CFG}" \
    MODEL.WEIGHTS "${WEIGHT}" OUTPUT_DIR "${OUTPUT_DIR}"
echo "[eval] Done exit=$?"
