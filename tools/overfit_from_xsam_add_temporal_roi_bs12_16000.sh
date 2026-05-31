#!/usr/bin/env bash
# Start from overfit X-SAM checkpoint, then add temporal+ROI training.
# Tests if X-SAM convergence issues are from initialization gradient conflicts.
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
XSAM_CKPT="${1:-z_outputs/overfit_xsam_no_roi_bs12_16000/model_final.pth}"
OUTPUT_DIR="${2:-z_outputs/overfit_from_xsam_add_temporal_roi_bs12_16000}"
mkdir -p "${OUTPUT_DIR}"
echo "[From-XSAM + Temporal + ROI] XSAM_CKPT=${XSAM_CKPT}"
echo "[From-XSAM + Temporal + ROI] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py \
    --num-gpus 1 --config-file configs/speaq_ag_roi.yaml \
    --dist-url tcp://127.0.0.1:29550 \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.TRAIN "('AG_train',)" \
    DATASETS.TEST "('AG_train',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0 \
    MODEL.SAM3.ENABLED True \
    MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt \
    MODEL.SAM3.FREEZE True \
    MODEL.SAM3.USE_PATCH_MERGE True \
    MODEL.WEIGHTS "${XSAM_CKPT}" \
    MODEL.DETR.HEAD_WEIGHTS "${XSAM_CKPT}" \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    MODEL.TEMPORAL.ENABLED True \
    MODEL.TEMPORAL.EVAL_ENABLED True \
    MODEL.TEMPORAL.MODE object_query_memory_v1 \
    MODEL.TEMPORAL.RELATION_MEMORY_ENABLED True \
    MODEL.TEMPORAL.RELATION_MEMORY_SOURCE relation \
    MODEL.TEMPORAL.MEMORY_UPDATE_MODE matched_gt \
    MODEL.TEMPORAL.RELATION_MEMORY_UPDATE_MODE matched_gt \
    MODEL.TEMPORAL.GATE_MIN 0.0 \
    MODEL.TEMPORAL.GATE_MAX 0.20 \
    MODEL.TEMPORAL.GATE_WARMUP_ITERS 500 \
    MODEL.TEMPORAL.RELATION_GATE_MIN 0.0 \
    MODEL.TEMPORAL.RELATION_GATE_MAX 0.10 \
    MODEL.TEMPORAL.RELATION_GATE_WARMUP_ITERS 750 \
    MODEL.ROI_REFINE.ENABLED True \
    MODEL.ROI_REFINE.LOSS_ENABLED True \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.STEPS '(4000,12000)' \
    SOLVER.WARMUP_ITERS 500 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    SOLVER.GATE_LR_MULTIPLIER 5.0 \
    TEST.EVAL_PERIOD 2000 \
    DATALOADER.NUM_WORKERS
RET=$?
echo "[From-XSAM + Temporal + ROI] Done exit=${RET}"
