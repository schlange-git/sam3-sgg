#!/usr/bin/env bash
# =============================================================================
# roi_refine cls 替换 对照评测 (单卡 10901)
#   同 checkpoint(best_model_final.pth.pth) + 同 config + 同 AG_val，唯一变量 =
#   detr.py:601 的替换段 开/关（由 ROI_EVAL_RAW 门控）：
#     post  -> ROI_EVAL_RAW 未设 -> eval 用 refined(roi) logits（线上现状）
#     pre   -> ROI_EVAL_RAW=1   -> eval 用原始 pre-refine logits
#   用法: bash eval_roirefine_compare_1gpu.sh post | pre
# =============================================================================
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate base

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"

MODE="${1:?需指定 post 或 pre}"
EXP=z_outputs/fulltask_v3_roi_v4_eiou_corner_bs48_iter80000
WEIGHT="${EXP}/best_model_final.pth.pth"
CFG="${EXP}/config.yaml"
OUTPUT_DIR="z_outputs/eval_roirefine_compare/${MODE}"
mkdir -p "${OUTPUT_DIR}"

if [ "$MODE" = "pre" ]; then
    export ROI_EVAL_RAW=1
elif [ "$MODE" = "post" ]; then
    unset ROI_EVAL_RAW || true
else
    echo "[ERR] MODE 必须是 post 或 pre, got=${MODE}"; exit 2
fi

echo "[eval roirefine compare] MODE=${MODE} ROI_EVAL_RAW=${ROI_EVAL_RAW:-unset} OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py \
    --num-gpus 1 \
    --eval-only \
    --config-file "${CFG}" \
    MODEL.WEIGHTS "${WEIGHT}" \
    OUTPUT_DIR "${OUTPUT_DIR}"

echo "[eval roirefine compare] MODE=${MODE} Done exit=$?"
