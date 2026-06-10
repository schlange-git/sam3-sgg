#!/usr/bin/env bash
# Full task v3-only + ROI, 4 GPU DDP, 80000 iter, eval 5000.
# v4 alignment: PERSON_SCORE_SCALE=300 (consistent person scoring so triplet
#   memory bank fills and the injector receives gradient).
# ROI box regression: eiou + corner loss (proven recipe from
#   z_outputs/repro_sam3_roi_eiou_cornerloss_bs48_80000_dist).
# LR: 3-segment WarmupMultiStepLR, BASE_LR unchanged, GAMMA=0.1 -> final ~4e-6.
#   seg1 0-12000: 4e-4 | seg2 12000-70000: 4e-5 | seg3 70000-80000: 4e-6 (near-keep tail).
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/fulltask_v3_roi_v4_eiou_corner_bs48_iter80000}"
NUM_GPUS="${2:-4}"
mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"
echo "[Full v3+ROI v4] OUTPUT=${OUTPUT_DIR} GPU=${NUM_GPUS}"

python3 train_iterative_model.py \
    --num-gpus "${NUM_GPUS}" \
    --config-file configs/speaq_actiongenome_minimal.yaml \
    --dist-url tcp://127.0.0.1:29701 \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    MODEL.WEIGHTS model_0099999.pth \
    MODEL.DETR.HEAD_WEIGHTS model_0099999.pth \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    MODEL.DETR.LOAD_CLASS_HEAD True \
    MODEL.DETR.PERSON_SCORE_SCALE 300.0 \
    MODEL.DETR.BOX_LOSS_TYPE eiou \
    MODEL.DETR.USE_CORNER_LOSS True \
    MODEL.DETR.CORNER_LOSS_WEIGHT 0.5 \
    MODEL.SAM3.ENABLED True \
    MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt \
    MODEL.SAM3.FREEZE True \
    MODEL.SAM3.USE_PATCH_MERGE False \
    MODEL.TEMPORAL.ENABLED True \
    MODEL.TEMPORAL.EVAL_ENABLED True \
    MODEL.TEMPORAL.MODE triplet_memory_v3 \
    MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED True \
    MODEL.TEMPORAL.INJECT_OBJECT True \
    MODEL.TEMPORAL.INJECT_RELATION True \
    MODEL.TEMPORAL.NON_KEY_SKIP_LOSS True \
    MODEL.TEMPORAL.NON_KEY_SKIP_EVAL True \
    MODEL.TEMPORAL.NON_KEY_RUN_OBJECT_ONLY True \
    MODEL.ROI_REFINE.ENABLED True \
    MODEL.ROI_REFINE.LOSS_ENABLED True \
    MODEL.ROI_REFINE.RESNET_FPN_LEVEL 0 \
    MODEL.ROI_REFINE.STRIDE 14 \
    MODEL.ROI_REFINE.APPLY_TO all \
    MODEL.ROI_REFINE.SMALL_AREA_THRESH 0.02 \
    SOLVER.IMS_PER_BATCH 48 \
    SOLVER.BASE_LR 0.0004 \
    SOLVER.MAX_ITER 80000 \
    SOLVER.GAMMA 0.1 \
    SOLVER.NUM_DECAYS 2 \
    SOLVER.STEPS '(12000,70000)' \
    SOLVER.WARMUP_ITERS 2000 \
    SOLVER.GATE_LR_MULTIPLIER 5.0 \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    TEST.EVAL_PERIOD 5000 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 4

RET=$?
echo "[Full v3+ROI v4] Done exit=${RET}"
