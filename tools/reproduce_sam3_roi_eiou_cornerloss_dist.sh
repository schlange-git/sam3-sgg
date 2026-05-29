#!/usr/bin/env bash
# Reproduce the previous stable SAM3 + ROI14 + eIoU/corner-loss experiment.
#
# Reference run:
#   z_outputs/sam3_roi_eiou_cornerloss_bs24_160000_dist
#
# This script intentionally uses configs/speaq_ag_roi.yaml instead of the newer
# minimal config, and pins the key options that changed in later commits.
export PYTHONPATH="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/detectron2:${PYTHONPATH:-}"
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
set -euo pipefail

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"

OUTPUT_DIR="${1:-z_outputs/repro_sam3_roi_eiou_cornerloss_bs24_160000_dist}"
NUM_GPUS="${2:-4}"

CONFIG="configs/speaq_ag_roi.yaml"
PORT="${PORT:-29500}"

echo "=============================================="
echo "[Repro SAM3 ROI14 eIoU Corner] OUTPUT=${OUTPUT_DIR}"
echo "[Repro SAM3 ROI14 eIoU Corner] GPU=${NUM_GPUS} BS=24 MAX_ITER=160K"
echo "[Repro SAM3 ROI14 eIoU Corner] Config=${CONFIG}"
echo "[Repro SAM3 ROI14 eIoU Corner] Temporal=off, PatchMerge=off, ROI small_thresh=1.0"
echo "=============================================="

mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${ROI_GATE_LOG_PATH:-${OUTPUT_DIR}/roi_gate_log.csv}"
export ROI_GATE_LOG_PERIOD="${ROI_GATE_LOG_PERIOD:-20}"
echo "[ROI gate log] ${ROI_GATE_LOG_PATH} (period=${ROI_GATE_LOG_PERIOD} iter)"

TRAIN_OPTS=(
    OUTPUT_DIR "${OUTPUT_DIR}"

    # Dataset paths used by the previous successful run.
    DATASETS.ACTION_GENOME.ANNOTATIONS "dataset/annotations"
    DATASETS.ACTION_GENOME.FRAMES "dataset/frames"
    DATASETS.ACTION_GENOME.VIDEOS "dataset/videos"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "-1"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL "400"

    # Keep temporal logic fully disabled for this ROI-only reproduction.
    MODEL.TEMPORAL.ENABLED "False"
    MODEL.TEMPORAL.EVAL_ENABLED "False"
    MODEL.TEMPORAL.RELATION_MEMORY_ENABLED "False"

    # Match the previous stable SAM3/ROI14 feature setup.
    MODEL.SAM3.ENABLED "True"
    MODEL.SAM3.CHECKPOINT_PATH "sam3/weights/sam3.pt"
    MODEL.SAM3.FREEZE "True"
    MODEL.SAM3.USE_PATCH_MERGE "False"
    MODEL.SAM3.TARGET_STRIDE "28"
    MODEL.SAM3.FPN_STRIDES "[7, 14, 28]"
    MODEL.SAM3.USE_FPN "False"
    MODEL.SAM3.USE_BACKBONE_FPN "True"
    MODEL.SAM3.MULTISCALE_MERGE "last"

    # Match the previous ROI14 classification refinement behavior.
    MODEL.ROI_REFINE.ENABLED "True"
    MODEL.ROI_REFINE.STRIDE "14"
    MODEL.ROI_REFINE.POOL_SIZE "7"
    MODEL.ROI_REFINE.SMALL_AREA_THRESH "1.0"
    MODEL.ROI_REFINE.DETACH_BOXES "True"
    MODEL.ROI_REFINE.USE_GATE "True"
    MODEL.ROI_REFINE.LOSS_ENABLED "True"
    MODEL.ROI_REFINE.LOSS_WEIGHT "1.0"
    MODEL.ROI_REFINE.APPLY_TO "small_only"
    MODEL.ROI_REFINE.ONLY_ROI_CLS "False"

    # Match the previous detection/relation loss and person prior settings.
    MODEL.DETR.BOX_LOSS_TYPE "eiou"
    MODEL.DETR.USE_CORNER_LOSS "True"
    MODEL.DETR.CORNER_LOSS_WEIGHT "0.5"
    MODEL.DETR.PERSON_SCORE_SCALE "200.0"
    MODEL.DETR.LOAD_HEAD_ONLY "False"
    MODEL.DETR.LOAD_FULL_WEIGHTS "True"
    MODEL.DETR.LOAD_CLASS_HEAD "True"
    MODEL.WEIGHTS "model_0099999.pth"
    MODEL.DETR.HEAD_WEIGHTS "model_0099999.pth"

    # Match the previous 160k schedule. GATE_LR_MULTIPLIER is pinned to 1.0
    # so the newer optimizer rule does not boost roi_refine_head.gate.
    SOLVER.IMS_PER_BATCH "24"
    SOLVER.BASE_LR "0.0001"
    SOLVER.MAX_ITER "160000"
    SOLVER.STEPS "(40000,120000)"
    SOLVER.WARMUP_ITERS "3000"
    SOLVER.CHECKPOINT_PERIOD "5000"
    SOLVER.GATE_LR_MULTIPLIER "1.0"

    TEST.EVAL_PERIOD "20000"
)

RESUME_ARGS=()
if [[ -d "${OUTPUT_DIR}" ]]; then
    shopt -s nullglob
    CKPTS=("${OUTPUT_DIR}"/*.pth)
    shopt -u nullglob
    if (( ${#CKPTS[@]} > 0 )); then
        LATEST=$(ls -1t "${CKPTS[@]}" 2>/dev/null | sed -n '1p')
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
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    if hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize()
' || true

echo "[Train] Starting..."
python3 train_iterative_model.py \
    "${RESUME_ARGS[@]}" \
    --num-gpus "${NUM_GPUS}" \
    --config-file "${CONFIG}" \
    --dist-url "tcp://127.0.0.1:${PORT}" \
    "${TRAIN_OPTS[@]}" \
    ${OPTS:-}

RET=$?
if [ "${RET}" -ne 0 ]; then
    echo "[Train] FAILED (exit ${RET})"
    exit "${RET}"
fi
echo "[Train] Done: ${OUTPUT_DIR}"
