#!/usr/bin/env bash
# Overfit with ALL flags: X-SAM + temporal v2 + ROI14, 4 GPU, BS48
# Uses giou + no corner loss + person_score_scale=1 to isolate X-SAM effect.
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/overfit_all_xsam_temporal_roi_bs12_16000}"
mkdir -p "${OUTPUT_DIR}"
echo "[Overfit ALL] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py \
    --num-gpus 4 \
    --config-file configs/speaq_ag_roi.yaml \
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
    MODEL.DETR.BOX_LOSS_TYPE giou \
    MODEL.DETR.USE_CORNER_LOSS False \
    MODEL.DETR.PERSON_SCORE_SCALE 1.0 \
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
    MODEL.WEIGHTS model_0099999.pth \
    MODEL.DETR.HEAD_WEIGHTS model_0099999.pth \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    SOLVER.IMS_PER_BATCH 48 \
    SOLVER.BASE_LR 0.0004 \
    SOLVER.MAX_ITER 4000 \
    SOLVER.STEPS "'(1000,3000)'" \
    SOLVER.WARMUP_ITERS 125 \
    SOLVER.CHECKPOINT_PERIOD 500 \
    SOLVER.GATE_LR_MULTIPLIER 5.0 \
    TEST.EVAL_PERIOD 500 \
    DATALOADER.NUM_WORKERS 4
RET=$?
echo "[Overfit ALL] Done exit=${RET}"
