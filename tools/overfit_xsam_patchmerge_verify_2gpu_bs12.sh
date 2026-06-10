#!/usr/bin/env bash
# X-SAM patch-merge convergence verification, 2 GPU local run.
# Total batch size remains 12; with 2 GPUs this is 6 images/GPU.
# DATALOADER.NUM_WORKERS=0 avoids eval-time multiprocessing BrokenPipe on this SSH/nohup setup.
set -euo pipefail

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"

OUTPUT_DIR="${1:-z_outputs/overfit_xsam_patchmerge_verify_bs12_16000}"
PORT="${PORT:-29561}"
mkdir -p "${OUTPUT_DIR}"

export LEARNABLE_DIAGNOSTICS_LOG_PATH="${OUTPUT_DIR}/diagnostics_log.csv"
export LEARNABLE_DIAGNOSTICS_LOG_PERIOD=50

RESUME_ARGS=()
if [[ "${FRESH:-0}" != "1" && -f "${OUTPUT_DIR}/last_checkpoint" ]]; then
    RESUME_ARGS=(--resume)
fi

echo "[X-SAM patch-merge verify 2GPU] OUTPUT=${OUTPUT_DIR} PORT=${PORT} RESUME=${RESUME_ARGS[*]:-no}"
echo "[X-SAM patch-merge verify 2GPU] total batch size remains SOLVER.IMS_PER_BATCH=12"

python3 train_iterative_model.py \
    "${RESUME_ARGS[@]}" \
    --num-gpus 2 --config-file configs/speaq_ag_roi.yaml \
    --dist-url "tcp://127.0.0.1:${PORT}" \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.TRAIN "('AG_train',)" \
    DATASETS.TEST "('AG_train',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0 \
    MODEL.WEIGHTS model_0099999.pth \
    MODEL.DETR.HEAD_WEIGHTS model_0099999.pth \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    MODEL.SAM3.ENABLED True \
    MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt \
    MODEL.SAM3.FREEZE True \
    MODEL.SAM3.TARGET_STRIDE 32 \
    MODEL.SAM3.USE_PATCH_MERGE True \
    MODEL.SAM3.PATCH_MERGE_INIT_NOISE_STD 0.01 \
    MODEL.TEMPORAL.ENABLED False \
    MODEL.TEMPORAL.EVAL_ENABLED False \
    MODEL.ROI_REFINE.ENABLED False \
    MODEL.ROI_REFINE.LOSS_ENABLED False \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.BACKBONE_MULTIPLIER 1.0 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.STEPS '(4000,12000)' \
    SOLVER.WARMUP_ITERS 500 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    TEST.EVAL_PERIOD 2000 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 0

RET=$?
echo "[X-SAM patch-merge verify 2GPU] Done exit=${RET}"
