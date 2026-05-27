#!/usr/bin/env bash
# Detection-only overfit pretrain (1 GPU, X-SAM patch merge, SAM3 frozen)
# Model in the detection_only code, successfully output loss, training progresses > 100 iters
set -euo pipefail

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"

OUTPUT_DIR="${1:-z_outputs/detectiononly_overfit_v5_16000}"
CONFIG="configs/speaq_actiongenome_minimal.yaml"

echo "=============================================="
echo "[Overfit-Det] OUTPUT=${OUTPUT_DIR}"
echo "[Overfit-Det] GPU=1 BS=8"
echo "[Overfit-Det] LR=1e-4 MAX_ITER=16000"
echo "[Overfit-Det] Flags: detection_only + X-SAM patch merge"
echo "=============================================="

TRAIN_OPTS=(
    OUTPUT_DIR "${OUTPUT_DIR}"
    DATASETS.ACTION_GENOME.ANNOTATIONS "dataset_overfit/annotations"
    DATASETS.ACTION_GENOME.FRAMES "dataset/frames"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "-1"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL "100"

    MODEL.DETR.DETECTION_ONLY "True"
    MODEL.TEMPORAL.ENABLED "False"
    MODEL.TEMPORAL.EVAL_ENABLED "False"

    MODEL.SAM3.ENABLED "True"
    MODEL.SAM3.USE_PATCH_MERGE "True"
    MODEL.SAM3.CHECKPOINT_PATH "sam3/weights/sam3.pt"
    MODEL.SAM3.FREEZE "True"

    MODEL.WEIGHTS "model_0099999.pth"
    MODEL.DETR.HEAD_WEIGHTS "model_0099999.pth"
    MODEL.DETR.LOAD_HEAD_ONLY "False"
    MODEL.DETR.LOAD_FULL_WEIGHTS "True"

    SOLVER.IMS_PER_BATCH "8"
    SOLVER.BASE_LR "0.0001"
    SOLVER.MAX_ITER "16000"
    SOLVER.STEPS "(4000,12000)"
    SOLVER.WARMUP_ITERS "500"
    SOLVER.CHECKPOINT_PERIOD "2000"
    TEST.EVAL_PERIOD "2000"
    DATALOADER.NUM_WORKERS "2"
)

echo "[CUDA] Clearing cache..."
python3 -c 'import torch; torch.cuda.empty_cache()' 2>/dev/null || true

echo "[Train] Starting at $(date)"
python3 train_iterative_model.py     --num-gpus 1     --config-file "${CONFIG}"     --dist-url "tcp://127.0.0.1:29550"     "${TRAIN_OPTS[@]}" ${OPTS:-}
RET=$?
if [ "${RET}" -ne 0 ]; then echo "[Train] FAILED (exit ${RET})"; exit "${RET}"; fi
echo "[Train] Done at $(date): ${OUTPUT_DIR}"
