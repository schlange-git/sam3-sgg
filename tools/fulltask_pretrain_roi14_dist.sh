#!/usr/bin/env bash
# Full task pretrain with X-SAM + ROI14, 4 GPU
export PYTHONPATH="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/detectron2:${PYTHONPATH:-}"
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
set -euo pipefail

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"

OUTPUT_DIR="${1:-z_outputs/full_pretrain_roi14_bs48_iter50000}"
NUM_GPUS="${2:-4}"

CONFIG="configs/speaq_actiongenome_minimal.yaml"
PORT="${PORT:-29500}"

echo "=============================================="
echo "[Full Pretrain + ROI14] OUTPUT=${OUTPUT_DIR}"
echo "[Full Pretrain + ROI14] GPU=${NUM_GPUS} BS=12/gpu (total=48)"
echo "[Full Pretrain + ROI14] LR=4e-4 MAX_ITER=50K STEPS=(7.5K,22.5K)"
echo "[Full Pretrain + ROI14] Flags: X-SAM + ROI14 (full task, not detection_only)"
echo "=============================================="

TRAIN_OPTS=(
    OUTPUT_DIR "${OUTPUT_DIR}"
    DATASETS.ACTION_GENOME.ANNOTATIONS "/home/tione/output/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/annotations"
    DATASETS.ACTION_GENOME.FRAMES "/home/tione/output/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/frames"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "-1"

    # Full task (NOT detection_only)
    MODEL.TEMPORAL.ENABLED "False"
    MODEL.TEMPORAL.EVAL_ENABLED "False"

    # X-SAM + ROI14
    MODEL.SAM3.USE_PATCH_MERGE "True"
    MODEL.ROI_REFINE.ENABLED "True"
    MODEL.ROI_REFINE.LOSS_ENABLED "True"

    # Solver: scaled for BS=48 (4GPU×12)
    SOLVER.IMS_PER_BATCH "48"
    SOLVER.BASE_LR "0.0004"
    SOLVER.MAX_ITER "50000"
    SOLVER.STEPS "(7500,22500)"
    SOLVER.WARMUP_ITERS "1500"
    SOLVER.WARMUP_METHOD "linear"
    SOLVER.CHECKPOINT_PERIOD "5000"

    # Eval
    TEST.EVAL_PERIOD "5000"

    # Full SAM3 pretrained weights
    MODEL.WEIGHTS "model_0099999.pth"
    MODEL.DETR.HEAD_WEIGHTS "model_0099999.pth"
    MODEL.DETR.LOAD_HEAD_ONLY "False"
    MODEL.DETR.LOAD_FULL_WEIGHTS "True"
    MODEL.SAM3.CHECKPOINT_PATH "sam3/weights/sam3.pt"
    MODEL.SAM3.ENABLED "True"
    MODEL.SAM3.FREEZE "True"
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
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true
python3 -c '
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    if hasattr(torch.cuda,"ipc_collect"): torch.cuda.ipc_collect()
    if hasattr(torch.cuda,"synchronize"): torch.cuda.synchronize()
' || true

echo "[Train] Starting..."
python3 train_iterative_model.py     "${RESUME_ARGS[@]}"     --num-gpus "${NUM_GPUS}"     --config-file "${CONFIG}"     --dist-url "tcp://127.0.0.1:${PORT}"     "${TRAIN_OPTS[@]}"     ${OPTS:-}
RET=$?
if [ "${RET}" -ne 0 ]; then echo "[Train] FAILED (exit ${RET})"; exit "${RET}"; fi
echo "[Train] Done: ${OUTPUT_DIR}"
