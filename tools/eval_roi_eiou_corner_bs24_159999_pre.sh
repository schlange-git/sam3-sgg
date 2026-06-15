#!/usr/bin/env bash
# 用 ROI_EVAL_RAW=1 (pre-refine: 跳过 roi 覆盖, eval 用 raw/原始 logits) 评测
# 目的: 验证 model_0159999 重评测回退(66.36->61.27)的根因是"默认开启 roi 覆盖", 预期回到 ~66.36/0.3665
# checkpoint = sam3_roi_eiou_cornerloss_bs24_160000_dist / model_0159999.pth
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate base
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
EXP=z_outputs/sam3_roi_eiou_cornerloss_bs24_160000_dist
WEIGHT="${EXP}/model_0159999.pth"
CFG="${EXP}/config.yaml"
OUTPUT_DIR="z_outputs/eval_roi_eiou_corner_bs24_159999_pre"
mkdir -p "${OUTPUT_DIR}"
export ROI_EVAL_RAW=1
echo "[eval] ckpt=model_0159999 logic=pre(raw) ROI_EVAL_RAW=${ROI_EVAL_RAW} OUT=${OUTPUT_DIR}"
python3 train_iterative_model.py --num-gpus 1 --eval-only --config-file "${CFG}" \
    MODEL.WEIGHTS "${WEIGHT}" OUTPUT_DIR "${OUTPUT_DIR}"
echo "[eval] Done exit=$?"
