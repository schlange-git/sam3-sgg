#!/usr/bin/env bash
# Eval-first for fulltask_resnet_roi: checks checkpoint weights, then eval, then train
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/fulltask_resnet_roi_bs96_iter40000}"

echo "=============================================="
echo "[Eval-First] Step 1: eval-only with weight check"
echo "=============================================="

# Find latest checkpoint
CKPT=""
if [[ -d "${OUTPUT_DIR}" ]]; then
    CKPT=$(ls -1t "${OUTPUT_DIR}"/model_*.pth 2>/dev/null | head -1)
    if [[ -z "${CKPT}" ]]; then
        CKPT=$(ls -1t "${OUTPUT_DIR}"/model_final.pth 2>/dev/null | head -1)
    fi
fi

# Fallback to pretrained weight
if [[ -z "${CKPT}" ]]; then
    CKPT="res101_pretrained.pth"
    echo "[Eval-First] No checkpoint found, using ${CKPT}"
else
    echo "[Eval-First] Using checkpoint: ${CKPT}"
fi

# Step 1: Eval-only
python3 train_iterative_model.py     --eval-only     --num-gpus 8     --config-file configs/speaq_actiongenome_minimal.yaml     --dist-url tcp://127.0.0.1:29500     OUTPUT_DIR "${OUTPUT_DIR}/eval_first"     DATASETS.ACTION_GENOME.ANNOTATIONS dataset/annotations     DATASETS.ACTION_GENOME.FRAMES dataset/frames     DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1     MODEL.SAM3.ENABLED False     MODEL.TEMPORAL.ENABLED False     MODEL.ROI_REFINE.ENABLED True     MODEL.ROI_REFINE.LOSS_ENABLED True     MODEL.ROI_REFINE.RESNET_FPN_LEVEL 1     MODEL.ROI_REFINE.STRIDE 16     MODEL.ROI_REFINE.APPLY_TO all     MODEL.WEIGHTS "${CKPT}"     MODEL.DETR.HEAD_WEIGHTS "${CKPT}"     MODEL.DETR.LOAD_HEAD_ONLY False     MODEL.DETR.LOAD_FULL_WEIGHTS True     SOLVER.IMS_PER_BATCH 96     DATALOADER.NUM_WORKERS 4
RET=$?
echo "[Eval-First] Eval done (exit ${RET})"

echo ""
echo "=============================================="
echo "[Eval-First] Step 2: training from ${CKPT}"
echo "=============================================="
bash tools/fulltask_resnet_roi_dist_8gpu.sh "${OUTPUT_DIR}" 8
