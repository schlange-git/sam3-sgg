#!/usr/bin/env bash
# Full task X-SAM only, 4 GPU, loading model_xsam_pretrained.pth
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/fulltask_xsam_pretrained_bs48_iter50000_reduceLR}"
NUM_GPUS="${2:-4}"
CONFIG="configs/speaq_actiongenome_minimal.yaml"
PORT="${PORT:-29500}"

mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"

echo "=============================================="
echo "[Full X-SAM Pretrained] OUTPUT=${OUTPUT_DIR}"
echo "[Full X-SAM Pretrained] GPU=${NUM_GPUS} BS=12/gpu (total=48)"
echo "[Full X-SAM Pretrained] LR=4e-4 MAX_ITER=50K EVAL=2000"
echo "[Full X-SAM Pretrained] Weights=model_xsam_pretrained.pth"
echo "=============================================="

TRAIN_OPTS=(
    OUTPUT_DIR "${OUTPUT_DIR}"
    DATASETS.ACTION_GENOME.ANNOTATIONS "dataset/annotations"
    DATASETS.ACTION_GENOME.FRAMES "dataset/frames"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "-1"

    MODEL.SAM3.ENABLED "True"
    MODEL.SAM3.CHECKPOINT_PATH "sam3/weights/sam3.pt"
    MODEL.SAM3.FREEZE "True"
    MODEL.SAM3.USE_PATCH_MERGE "True"

    MODEL.TEMPORAL.ENABLED "False"
    MODEL.TEMPORAL.EVAL_ENABLED "False"
    MODEL.ROI_REFINE.ENABLED "False"
    MODEL.ROI_REFINE.LOSS_ENABLED "False"

    MODEL.WEIGHTS "model_xsam_pretrained.pth"
    MODEL.DETR.HEAD_WEIGHTS "model_xsam_pretrained.pth"
    MODEL.DETR.LOAD_HEAD_ONLY "False"
    MODEL.DETR.LOAD_FULL_WEIGHTS "True"
    MODEL.DETR.LOAD_CLASS_HEAD "True"

    SOLVER.IMS_PER_BATCH "48"
    SOLVER.BASE_LR "0.00004"
    SOLVER.MAX_ITER "50000"
    SOLVER.STEPS "(7500,42500)"
    SOLVER.WARMUP_ITERS "1500"
    SOLVER.WARMUP_METHOD "linear"
    SOLVER.CHECKPOINT_PERIOD "5000"

    TEST.EVAL_PERIOD "5000"
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
