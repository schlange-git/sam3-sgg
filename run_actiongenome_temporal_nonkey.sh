#!/usr/bin/env bash
# ActionGenome temporal non-key training script
set -euo pipefail

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"

OUTPUT_DIR="${1:-z_outputs/temporal_nonkey_160000iters_bs16}"
NUM_GPUS="${2:-1}"
NUM_VIDEOS_TRAIN="${3:--1}"

CONFIG="configs/speaq_actiongenome_minimal.yaml"
PORT="${PORT:-29500}"

AG_ANNOTATIONS="dataset/annotations"
AG_FRAMES="dataset/frames"
DETR_HEAD_WEIGHTS="model_0099999.pth"
SAM3_CHECKPOINT_PATH="sam3/weights/sam3.pt"

TEMPORAL_ENABLED="True"
TEMPORAL_EVAL_ENABLED="True"
NON_KEY_SKIP_LOSS="True"
NON_KEY_SKIP_EVAL="True"
NON_KEY_RUN_OBJECT_ONLY="True"
CACHE_RESET_ON_VIDEO_SWITCH="True"

echo "=============================================="
echo "[Temporal] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[Temporal] GPU=${NUM_GPUS} Videos=${NUM_VIDEOS_TRAIN}"
echo "[Temporal] Weight=${DETR_HEAD_WEIGHTS}"
echo "=============================================="

TRAIN_OPTS=(
    OUTPUT_DIR "${OUTPUT_DIR}"
    DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}"
    DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "${NUM_VIDEOS_TRAIN}"
    SOLVER.IMS_PER_BATCH "16"
    MODEL.WEIGHTS "${DETR_HEAD_WEIGHTS}"
    MODEL.SAM3.CHECKPOINT_PATH "${SAM3_CHECKPOINT_PATH}"
    MODEL.DETR.HEAD_WEIGHTS "${DETR_HEAD_WEIGHTS}"
    MODEL.DETR.LOAD_HEAD_ONLY "False"
    MODEL.DETR.LOAD_FULL_WEIGHTS "True"
    MODEL.DETR.LOAD_CLASS_HEAD "True"
    MODEL.TEMPORAL.ENABLED "${TEMPORAL_ENABLED}"
    MODEL.TEMPORAL.EVAL_ENABLED "${TEMPORAL_EVAL_ENABLED}"
    MODEL.TEMPORAL.NON_KEY_SKIP_LOSS "${NON_KEY_SKIP_LOSS}"
    MODEL.TEMPORAL.NON_KEY_SKIP_EVAL "${NON_KEY_SKIP_EVAL}"
    MODEL.TEMPORAL.NON_KEY_RUN_OBJECT_ONLY "${NON_KEY_RUN_OBJECT_ONLY}"
    MODEL.TEMPORAL.CACHE_RESET_ON_VIDEO_SWITCH "${CACHE_RESET_ON_VIDEO_SWITCH}"
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

monitor_memory() {
    local pid=$1 threshold=93 interval=5
    echo "[Memory] Watching PID ${pid}"
    while kill -0 "${pid}" 2>/dev/null; do
        local info=$(free | grep Mem)
        local used=$(echo "${info}" | awk '{print $3}')
        local total=$(echo "${info}" | awk '{print $2}')
        local pct=$((used * 100 / total))
        if [ "${pct}" -ge "${threshold}" ]; then
            echo "[Memory] ${pct}% >= ${threshold}% killing ${pid}"
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 2; kill -KILL "${pid}" 2>/dev/null || true
            exit 1
        fi
        sleep "${interval}"
    done
}

echo "[CUDA] Clearing cache..."
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true
python3 -c '
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    if hasattr(torch.cuda,"ipc_collect"): torch.cuda.ipc_collect()
    if hasattr(torch.cuda,"synchronize"): torch.cuda.synchronize()
' || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true

echo "[Train] Starting..."
python3 train_iterative_model.py     "${RESUME_ARGS[@]}"     --num-gpus "${NUM_GPUS}"     --config-file "${CONFIG}"     --dist-url "tcp://127.0.0.1:${PORT}"     "${TRAIN_OPTS[@]}"     ${OPTS:-} &
TRAIN_PID=$!
monitor_memory "${TRAIN_PID}" &
wait "${TRAIN_PID}"
RET=$?
kill %1 2>/dev/null || true
if [ "${RET}" -ne 0 ]; then echo "[Train] FAILED (exit ${RET})"; exit "${RET}"; fi
echo "[Train] Done: ${OUTPUT_DIR}"
