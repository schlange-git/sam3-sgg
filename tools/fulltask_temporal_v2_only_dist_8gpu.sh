#!/usr/bin/env bash
# Full task temporal v2 only (no X-SAM, no ROI), 8 GPU
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/fulltask_temporal_v2_only_bs96_iter40000}"
NUM_GPUS="${2:-8}"
CONFIG="configs/speaq_ag_roi.yaml"
PORT="${PORT:-29500}"

echo "=============================================="
echo "[Full Temporal V2 Only] OUTPUT=${OUTPUT_DIR}"
echo "[Full Temporal V2 Only] GPU=${NUM_GPUS} BS=12/gpu (total=96)"
echo "[Full Temporal V2 Only] LR=8e-4 MAX_ITER=40K STEPS=(6K,18K)"
echo "=============================================="

TRAIN_OPTS=(
    OUTPUT_DIR "${OUTPUT_DIR}"
    DATASETS.ACTION_GENOME.ANNOTATIONS "dataset/annotations"
    DATASETS.ACTION_GENOME.FRAMES "dataset/frames"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "-1"

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
    MODEL.TEMPORAL.GATE_MIN "0.0"
    MODEL.TEMPORAL.GATE_MAX "0.20"
    MODEL.TEMPORAL.GATE_WARMUP_ITERS "600"
    MODEL.TEMPORAL.RELATION_GATE_MIN "0.0"
    MODEL.TEMPORAL.RELATION_GATE_MAX "0.10"
    MODEL.TEMPORAL.RELATION_GATE_WARMUP_ITERS "1000"

    MODEL.DETR.BOX_LOSS_TYPE "giou"
    MODEL.DETR.USE_CORNER_LOSS "False"
    MODEL.DETR.PERSON_SCORE_SCALE "1.0"

    SOLVER.IMS_PER_BATCH "96"
    SOLVER.BASE_LR "0.0008"
    SOLVER.MAX_ITER "40000"
    SOLVER.STEPS "(6000,18000)"
    SOLVER.WARMUP_ITERS "1200"
    SOLVER.WARMUP_METHOD "linear"
    SOLVER.CHECKPOINT_PERIOD "5000"
    SOLVER.GATE_LR_MULTIPLIER "5.0"

    TEST.EVAL_PERIOD "5000"

    MODEL.WEIGHTS "model_0099999.pth"
    MODEL.DETR.HEAD_WEIGHTS "model_0099999.pth"
    MODEL.DETR.LOAD_HEAD_ONLY "False"
    MODEL.DETR.LOAD_FULL_WEIGHTS "True"
)

RESUME_ARGS=()
if [[ -d "${OUTPUT_DIR}" ]]; then
    shopt -s nullglob
    CKPTS=("${OUTPUT_DIR}"/*.pth)
    shopt -u nullglob
    if (( ${#CKPTS[@]} > 0 )); then
        LATEST=$(ls -1t "${CKPTS[@]}" 2>/dev/null | head -1)
        if [[ -n "${LATEST}" ]]; then
            echo "[Resume] ${LATEST}"
            RESUME_ARGS=(--resume)
        fi
    fi
fi

echo "[CUDA] Clearing cache..."
python3 -c 'import torch; torch.cuda.empty_cache()' 2>/dev/null || true

echo "[Train] Starting..."
python3 train_iterative_model.py     "${RESUME_ARGS[@]}"     --num-gpus "${NUM_GPUS}"     --config-file "${CONFIG}"     --dist-url "tcp://127.0.0.1:${PORT}"     "${TRAIN_OPTS[@]}"     ${OPTS:-}
RET=$?
if [ "${RET}" -ne 0 ]; then echo "[Train] FAILED (exit ${RET})"; exit "${RET}"; fi
echo "[Train] Done: ${OUTPUT_DIR}"
