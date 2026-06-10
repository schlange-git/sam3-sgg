#!/usr/bin/env bash
# Overfit v3-only stage4: single GPU, 1000-frame, 16000 iter, eval 2000
# Includes injector gradient check (removed after first verification)
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/overfit_v3_stage4}"
mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"
echo "[v3 Stage4 Overfit] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py     --num-gpus 1     --config-file configs/speaq_actiongenome_minimal.yaml     --dist-url tcp://127.0.0.1:29800     OUTPUT_DIR "${OUTPUT_DIR}"     DATASETS.TRAIN "('AG_train',)"     DATASETS.TEST "('AG_train',)"     DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal/annotations     DATASETS.ACTION_GENOME.FRAMES dataset/frames     DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1     DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0     DATASETS.AG_TEMPORAL.ENABLED True     DATASETS.AG_TEMPORAL.CLIP_MODE between_keyframes     DATASETS.AG_TEMPORAL.NUM_INTERMEDIATE_FRAMES 0     MODEL.WEIGHTS model_0099999.pth     MODEL.DETR.HEAD_WEIGHTS model_0099999.pth     MODEL.DETR.LOAD_HEAD_ONLY False     MODEL.DETR.LOAD_FULL_WEIGHTS True     MODEL.DETR.LOAD_CLASS_HEAD True     MODEL.SAM3.ENABLED True     MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt     MODEL.SAM3.FREEZE True     MODEL.SAM3.USE_PATCH_MERGE False     MODEL.TEMPORAL.ENABLED True     MODEL.TEMPORAL.EVAL_ENABLED True     MODEL.TEMPORAL.MODE triplet_memory_v3     MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED True     MODEL.TEMPORAL.INJECT_OBJECT True     MODEL.TEMPORAL.INJECT_RELATION True     MODEL.TEMPORAL.NON_KEY_SKIP_LOSS True     MODEL.TEMPORAL.NON_KEY_SKIP_EVAL True     MODEL.TEMPORAL.NON_KEY_RUN_OBJECT_ONLY True     MODEL.ROI_REFINE.ENABLED False     MODEL.ROI_REFINE.LOSS_ENABLED False     SOLVER.IMS_PER_BATCH 8     SOLVER.BASE_LR 0.0001     SOLVER.MAX_ITER 16000     SOLVER.STEPS '(4000,12000)'     SOLVER.WARMUP_ITERS 500     SOLVER.CHECKPOINT_PERIOD 2000     TEST.EVAL_PERIOD 2000     SOLVER.EVAL_FIRST True     DATALOADER.NUM_WORKERS 0

RET=$?
echo "[v3 Stage4 Overfit] Done exit=${RET}"
