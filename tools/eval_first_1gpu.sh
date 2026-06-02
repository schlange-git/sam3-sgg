#!/usr/bin/env bash
# Eval-only on 1 GPU for checkpoint validation (M3)
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
CKPT="${1:-res101_pretrained.pth}"
OUTPUT_DIR="${2:-z_outputs/eval_first_1gpu}"

echo "=========================================="
echo "[Eval-Only 1GPU] CKPT=${CKPT}"
echo "[Eval-Only 1GPU] OUTPUT=${OUTPUT_DIR}"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"

python3 train_iterative_model.py \
    --eval-only \
    --num-gpus 1 \
    --config-file configs/speaq_actiongenome_minimal.yaml \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    MODEL.SAM3.ENABLED False \
    MODEL.TEMPORAL.ENABLED False \
    MODEL.ROI_REFINE.ENABLED True \
    MODEL.ROI_REFINE.LOSS_ENABLED True \
    MODEL.ROI_REFINE.RESNET_FPN_LEVEL 1 \
    MODEL.ROI_REFINE.STRIDE 16 \
    MODEL.ROI_REFINE.APPLY_TO all \
    MODEL.WEIGHTS "${CKPT}" \
    MODEL.DETR.HEAD_WEIGHTS "${CKPT}" \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    SOLVER.IMS_PER_BATCH 8 \
    DATALOADER.NUM_WORKERS 4

RET=$?
echo "[Eval-Only 1GPU] Done (exit ${RET})"
