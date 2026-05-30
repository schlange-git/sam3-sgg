#!/usr/bin/env bash
# Four-GPU pairwise overfit probe: fixed temporal v2 relation memory + X-SAM patch merge.
# ROI refine is disabled.
export PYTHONPATH="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/detectron2:${PYTHONPATH:-}"
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
set -euo pipefail

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"

OUTPUT_DIR="${1:-z_outputs/overfit_temporal_xsam_no_roi_bs48_4000}"
NUM_GPUS="${2:-4}"
CONFIG="configs/speaq_ag_roi.yaml"
PORT="${PORT:-29574}"

mkdir -p "${OUTPUT_DIR}"
export LEARNABLE_DIAGNOSTICS_LOG_PATH="${LEARNABLE_DIAGNOSTICS_LOG_PATH:-${OUTPUT_DIR}/learnable_diagnostics.csv}"
export LEARNABLE_DIAGNOSTICS_LOG_PERIOD="${LEARNABLE_DIAGNOSTICS_LOG_PERIOD:-50}"

echo "==========================================================="
echo "[Overfit Temporal + X-SAM No ROI] OUTPUT=${OUTPUT_DIR}"
echo "[Overfit Temporal + X-SAM No ROI] GPU=${NUM_GPUS}, BS=48, MAX_ITER=4000"
echo "[Overfit Temporal + X-SAM No ROI] LR=4e-4 STEPS=(1000,3000) WARMUP=125"
echo "[Diagnostics] learnable=${LEARNABLE_DIAGNOSTICS_LOG_PATH} period=${LEARNABLE_DIAGNOSTICS_LOG_PERIOD}"
echo "==========================================================="

TRAIN_OPTS=(
    OUTPUT_DIR "${OUTPUT_DIR}"

    DATASETS.TRAIN "('AG_train',)"
    DATASETS.TEST "('AG_train',)"
    DATASETS.ACTION_GENOME.ANNOTATIONS "dataset_overfit_temporal/annotations"
    DATASETS.ACTION_GENOME.FRAMES "dataset/frames"
    DATASETS.ACTION_GENOME.VIDEOS "dataset/videos"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "-1"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL "0"

    MODEL.SAM3.ENABLED "True"
    MODEL.SAM3.CHECKPOINT_PATH "sam3/weights/sam3.pt"
    MODEL.SAM3.FREEZE "True"
    MODEL.SAM3.USE_PATCH_MERGE "True"
    MODEL.SAM3.TARGET_STRIDE "28"
    MODEL.SAM3.FPN_STRIDES "[7, 14, 28]"
    MODEL.SAM3.USE_FPN "False"
    MODEL.SAM3.USE_BACKBONE_FPN "True"
    MODEL.SAM3.MULTISCALE_MERGE "last"

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
    MODEL.TEMPORAL.GATE_WARMUP_ITERS "500"
    MODEL.TEMPORAL.RELATION_GATE_MIN "0.0"
    MODEL.TEMPORAL.RELATION_GATE_MAX "0.10"
    MODEL.TEMPORAL.RELATION_GATE_WARMUP_ITERS "750"

    MODEL.DETR.BOX_LOSS_TYPE "eiou"
    MODEL.DETR.USE_CORNER_LOSS "True"
    MODEL.DETR.CORNER_LOSS_WEIGHT "0.5"
    MODEL.DETR.PERSON_SCORE_SCALE "200.0"
    MODEL.WEIGHTS "model_0099999.pth"
    MODEL.DETR.HEAD_WEIGHTS "model_0099999.pth"
    MODEL.DETR.LOAD_HEAD_ONLY "False"
    MODEL.DETR.LOAD_FULL_WEIGHTS "True"
    MODEL.DETR.LOAD_CLASS_HEAD "True"

    SOLVER.IMS_PER_BATCH "48"
    SOLVER.BASE_LR "0.0004"
    SOLVER.MAX_ITER "4000"
    SOLVER.STEPS "(1000,3000)"
    SOLVER.WARMUP_ITERS "125"
    SOLVER.CHECKPOINT_PERIOD "500"
    SOLVER.GATE_LR_MULTIPLIER "5.0"

    TEST.EVAL_PERIOD "500"
    DATALOADER.NUM_WORKERS "4"
)

echo "[Train] Starting..."
python3 train_iterative_model.py \
    --num-gpus "${NUM_GPUS}" \
    --config-file "${CONFIG}" \
    --dist-url "tcp://127.0.0.1:${PORT}" \
    "${TRAIN_OPTS[@]}" \
    ${OPTS:-}

echo "[Train] Done: ${OUTPUT_DIR}"
