#!/usr/bin/env bash
set -euo pipefail

cd /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching

OUT_DIR="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/z_outputs/temporal_ordered_eval_roi0001_0079999_run2"
WEIGHTS="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/z_outputs/train_fulltask_v3_8gpu_bs96_80k/model_0079999.pth"
CONFIG="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/z_outputs/train_fulltask_v3_8gpu_bs96_80k/config.yaml"

mkdir -p "${OUT_DIR}"

PYTHONUNBUFFERED=1 python3 tools/stat_ag_val_temporal.py \
  --config-file "${CONFIG}" \
  DATASETS.ACTION_GENOME.ANNOTATIONS /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/annotations \
  DATASETS.ACTION_GENOME.FRAMES /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/frames \
  2>&1 | tee "${OUT_DIR}/ag_val_stats.log"

PYTHONUNBUFFERED=1 python3 tools/eval_temporal_ordered.py \
  --config-file "${CONFIG}" \
  --model-weights "${WEIGHTS}" \
  --output-dir "${OUT_DIR}" \
  --dataset-name AG_val \
  --num-workers 0 \
  DATASETS.ACTION_GENOME.ANNOTATIONS /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/annotations \
  DATASETS.ACTION_GENOME.FRAMES /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/frames \
  MODEL.WEIGHTS "${WEIGHTS}" \
  MODEL.ROI_REFINE.SMALL_AREA_THRESH 0.001 \
  2>&1 | tee "${OUT_DIR}/ordered_eval_roi0001.log"
