#!/usr/bin/env bash
# Single-GPU overfit probe with all temporal relation fixes enabled.
# Fixes enabled:
#   - relation attention softmax over memory slots
#   - bounded/warmed object and relation gates
#   - dedicated relation memory bank
#   - matched-GT-only memory updates
# ROI refine and X-SAM patch merge are disabled to isolate temporal behavior.
export PYTHONPATH="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/detectron2:${PYTHONPATH:-}"
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
set -euo pipefail

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"

OUTPUT_DIR="${1:-z_outputs/overfit_temporal_v2_relation_fixed_all_no_roi_bs12_16000}"
CONFIG="configs/speaq_actiongenome_minimal.yaml"
PORT="${PORT:-29563}"

mkdir -p "${OUTPUT_DIR}"

echo "==========================================================="
echo "[Overfit Temporal V2 Relation Fixed-All No ROI] OUTPUT=${OUTPUT_DIR}"
echo "[Overfit Temporal V2 Relation Fixed-All No ROI] single GPU, BS=12, MAX_ITER=16000"
echo "[Overfit Temporal V2 Relation Fixed-All No ROI] train/test = dataset_overfit_temporal as AG_train"
echo "==========================================================="

TRAIN_OPTS=(
    OUTPUT_DIR "${OUTPUT_DIR}"

    DATASETS.TRAIN "('AG_train',)"
    DATASETS.TEST "('AG_train',)"
    DATASETS.ACTION_GENOME.ANNOTATIONS "dataset_overfit_temporal/annotations"
    DATASETS.ACTION_GENOME.FRAMES "dataset/frames"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "-1"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL "0"

    MODEL.SAM3.ENABLED "True"
    MODEL.SAM3.CHECKPOINT_PATH "sam3/weights/sam3.pt"
    MODEL.SAM3.FREEZE "True"
    MODEL.SAM3.USE_PATCH_MERGE "False"

    MODEL.ROI_REFINE.ENABLED "False"
    MODEL.ROI_REFINE.LOSS_ENABLED "False"

    MODEL.TEMPORAL.ENABLED "True"
    MODEL.TEMPORAL.EVAL_ENABLED "True"
    MODEL.TEMPORAL.MODE "object_query_memory_v1"
    MODEL.TEMPORAL.RELATION_MEMORY_ENABLED "True"
    MODEL.TEMPORAL.RELATION_MEMORY_SOURCE "relation"
    MODEL.TEMPORAL.MEMORY_UPDATE_MODE "matched_gt"
    MODEL.TEMPORAL.RELATION_MEMORY_UPDATE_MODE "matched_gt"
    MODEL.TEMPORAL.MEMORY_MATCH_IOU_THRESH "0.5"
    MODEL.TEMPORAL.MEMORY_MATCH_REQUIRE_CLASS "True"
    MODEL.TEMPORAL.MEMORY_STORE_GT_BOXES "True"
    MODEL.TEMPORAL.DETACH_MEMORY "True"

    MODEL.TEMPORAL.GATE_MIN "0.0"
    MODEL.TEMPORAL.GATE_MAX "0.20"
    MODEL.TEMPORAL.GATE_WARMUP_ITERS "2000"
    MODEL.TEMPORAL.RELATION_GATE_MIN "0.0"
    MODEL.TEMPORAL.RELATION_GATE_MAX "0.10"
    MODEL.TEMPORAL.RELATION_GATE_WARMUP_ITERS "3000"

    MODEL.WEIGHTS "model_0099999.pth"
    MODEL.DETR.HEAD_WEIGHTS "model_0099999.pth"
    MODEL.DETR.LOAD_HEAD_ONLY "False"
    MODEL.DETR.LOAD_FULL_WEIGHTS "True"
    MODEL.DETR.LOAD_CLASS_HEAD "True"

    SOLVER.IMS_PER_BATCH "12"
    SOLVER.BASE_LR "0.0001"
    SOLVER.MAX_ITER "16000"
    SOLVER.STEPS "(4000,12000)"
    SOLVER.WARMUP_ITERS "500"
    SOLVER.CHECKPOINT_PERIOD "2000"
    SOLVER.GATE_LR_MULTIPLIER "5.0"

    TEST.EVAL_PERIOD "2000"
    DATALOADER.NUM_WORKERS "2"
)

echo "[Train] Starting..."
python3 train_iterative_model.py \
    --num-gpus 1 \
    --config-file "${CONFIG}" \
    --dist-url "tcp://127.0.0.1:${PORT}" \
    "${TRAIN_OPTS[@]}" \
    ${OPTS:-}

echo "[Train] Done: ${OUTPUT_DIR}"
