#!/usr/bin/env bash
# Overfit X-SAM only (no temporal, no ROI), 8 GPU, BS96
# Uses giou + no corner loss + person_score_scale=1 to isolate X-SAM effect.
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/overfit_xsam_only_8gpu_bs96_4000}"
mkdir -p "${OUTPUT_DIR}"
echo "[Overfit X-SAM Only 8GPU] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py \
    --num-gpus 8 \
    --config-file configs/speaq_ag_roi.yaml \
    --dist-url tcp://127.0.0.1:29550 \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.TRAIN "('AG_train',)" \
    DATASETS.TEST "('AG_train',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0 \
    MODEL.SAM3.ENABLED True \
    MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt \
    MODEL.SAM3.FREEZE True \
    MODEL.SAM3.USE_PATCH_MERGE True \
    MODEL.DETR.BOX_LOSS_TYPE giou \
    MODEL.DETR.USE_CORNER_LOSS False \
    MODEL.DETR.PERSON_SCORE_SCALE 1.0 \
    MODEL.TEMPORAL.ENABLED False \
    MODEL.TEMPORAL.EVAL_ENABLED False \
    MODEL.ROI_REFINE.ENABLED False \
    MODEL.ROI_REFINE.LOSS_ENABLED False \
    MODEL.WEIGHTS model_0099999.pth \
    MODEL.DETR.HEAD_WEIGHTS model_0099999.pth \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    SOLVER.IMS_PER_BATCH 96 \
    SOLVER.BASE_LR 0.0008 \
    SOLVER.MAX_ITER 4000 \
    SOLVER.STEPS '(1000,3000)'  \
    SOLVER.WARMUP_ITERS 125 \
    SOLVER.CHECKPOINT_PERIOD 500 \
    TEST.EVAL_PERIOD 500 \
    DATALOADER.NUM_WORKERS 8
RET=$?
echo "[Overfit X-SAM Only 8GPU] Done exit=${RET}"
