#!/usr/bin/env bash
set -euo pipefail

cd /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching

OUT_DIR="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/z_outputs/temporal_ordered_eval_update001_0079999"
WEIGHTS="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/z_outputs/train_fulltask_v3_8gpu_bs96_80k/model_0079999.pth"
CONFIG="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/z_outputs/train_fulltask_v3_8gpu_bs96_80k/config.yaml"

mkdir -p "${OUT_DIR}"

PYTHONUNBUFFERED=1 python3 tools/eval_temporal_ordered_full.py \
  --config-file "${CONFIG}" \
  --model-weights "${WEIGHTS}" \
  --output-dir "${OUT_DIR}" \
  --dataset-name AG_val \
  --num-workers 0 \
  DATASETS.ACTION_GENOME.ANNOTATIONS /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/annotations \
  DATASETS.ACTION_GENOME.FRAMES /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/frames \
  MODEL.WEIGHTS "${WEIGHTS}" \
  MODEL.TEMPORAL.UPDATE_SCORE_THRESH 0.01 \
  2>&1 | tee "${OUT_DIR}/ordered_eval_update001.log"
