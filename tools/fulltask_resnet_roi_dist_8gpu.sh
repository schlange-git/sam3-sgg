#!/usr/bin/env bash
# Full task ResNet-101 + ROI14, 8 GPU
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/fulltask_resnet_roi_bs96_iter40000}"
NUM_GPUS="${2:-8}"
CONFIG="configs/speaq_actiongenome_minimal.yaml"
PORT="${PORT:-29500}"

echo "=============================================="
echo "[Full ResNet+ROI] OUTPUT=${OUTPUT_DIR}"
echo "[Full ResNet+ROI] GPU=${NUM_GPUS} BS=12/gpu (total=96)"
echo "[Full ResNet+ROI] LR=8e-4 MAX_ITER=40K"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"

TRAIN_OPTS=(
    OUTPUT_DIR "${OUTPUT_DIR}"
    DATASETS.ACTION_GENOME.ANNOTATIONS "dataset/annotations"
    DATASETS.ACTION_GENOME.FRAMES "dataset/frames"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "-1"

    MODEL.SAM3.ENABLED "False"
    MODEL.ROI_REFINE.ENABLED "True"
    MODEL.ROI_REFINE.LOSS_ENABLED "True"
    MODEL.ROI_REFINE.RESNET_FPN_LEVEL "1"
    MODEL.ROI_REFINE.STRIDE "16"
    MODEL.TEMPORAL.ENABLED "False"
    MODEL.TEMPORAL.EVAL_ENABLED "False"

    MODEL.DETR.BOX_LOSS_TYPE "giou"
    MODEL.DETR.USE_CORNER_LOSS "False"
    MODEL.DETR.PERSON_SCORE_SCALE "1.0"

    MODEL.WEIGHTS "res101_pretrained.pth"
    MODEL.DETR.HEAD_WEIGHTS "res101_pretrained.pth"
    MODEL.DETR.LOAD_HEAD_ONLY "False"
    MODEL.DETR.LOAD_FULL_WEIGHTS "True"

    SOLVER.IMS_PER_BATCH "96"
    SOLVER.BASE_LR "0.0008"
    SOLVER.MAX_ITER "40000"
    SOLVER.STEPS "(6000,18000)"
    SOLVER.WARMUP_ITERS "1200"
    SOLVER.WARMUP_METHOD "linear"
    SOLVER.CHECKPOINT_PERIOD "5000"

    TEST.EVAL_PERIOD "2000"
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
python3 train_iterative_model.py "${RESUME_ARGS[@]}" --num-gpus "${NUM_GPUS}" --config-file "${CONFIG}" --dist-url "tcp://127.0.0.1:${PORT}" "${TRAIN_OPTS[@]}" ${OPTS:-}
RET=$?
if [ "${RET}" -ne 0 ]; then echo "[Train] FAILED (exit ${RET})"; exit "${RET}"; fi
echo "[Train] Done: ${OUTPUT_DIR}"
