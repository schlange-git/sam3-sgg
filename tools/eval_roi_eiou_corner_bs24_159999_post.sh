#!/usr/bin/env bash
# 用现在的逻辑(post-refine: ROI_EVAL_RAW 未设 -> eval 用 refined/roi logits)评测
# checkpoint = sam3_roi_eiou_cornerloss_bs24_160000_dist / model_0159999.pth
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate base
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
EXP=z_outputs/sam3_roi_eiou_cornerloss_bs24_160000_dist
WEIGHT="${EXP}/model_0159999.pth"
CFG="${EXP}/config.yaml"
OUTPUT_DIR="z_outputs/eval_roi_eiou_corner_bs24_159999_post"
mkdir -p "${OUTPUT_DIR}"
unset ROI_EVAL_RAW || true
echo "[eval] ckpt=model_0159999 logic=post(refined) ROI_EVAL_RAW=${ROI_EVAL_RAW:-unset} OUT=${OUTPUT_DIR}"
python3 train_iterative_model.py --num-gpus 1 --eval-only --config-file "${CFG}" \
    MODEL.WEIGHTS "${WEIGHT}" OUTPUT_DIR "${OUTPUT_DIR}"
echo "[eval] Done exit=$?"
