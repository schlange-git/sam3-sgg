#!/usr/bin/env bash
set -euo pipefail

cd /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching

OUT_DIR="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/z_outputs/temporal_shuffle_nomemory_gate1e-6_0079999"
WEIGHTS="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/z_outputs/train_fulltask_v3_8gpu_bs96_80k/model_0079999.pth"

mkdir -p "${OUT_DIR}"

CONFIG="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/z_outputs/train_fulltask_v3_8gpu_bs96_80k/config.yaml"

PYTHONUNBUFFERED=1 python3 tools/stat_ag_val_temporal.py \
  --config-file "${CONFIG}" \
  DATASETS.ACTION_GENOME.ANNOTATIONS /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/annotations \
  DATASETS.ACTION_GENOME.FRAMES /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/frames \
  2>&1 | tee "${OUT_DIR}/ag_val_stats.log"

PYTHONUNBUFFERED=1 python3 tools/eval_temporal_shuffle.py \
  --config-file "${CONFIG}" \
  --model-weights "${WEIGHTS}" \
  --output-dir "${OUT_DIR}" \
  --dataset-name AG_val \
  --seed 20260630 \
  --num-workers 0 \
  --reset-temporal-each-batch \
  DATASETS.ACTION_GENOME.ANNOTATIONS /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/annotations \
  DATASETS.ACTION_GENOME.FRAMES /home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/frames \
  MODEL.WEIGHTS "${WEIGHTS}" \
  MODEL.TEMPORAL.GATE_MAX_OBJECT 1e-6 \
  MODEL.TEMPORAL.GATE_MAX_RELATION 1e-6 \
  2>&1 | tee "${OUT_DIR}/shuffle_eval.log"
