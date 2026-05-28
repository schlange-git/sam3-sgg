#!/usr/bin/env bash
# Temporal Memory v2 overfit: 10 videos, 1090 consecutive frames, 1 GPU
# Full task, no X-SAM, no ROI14, gate starts at 0.5 (free learning)
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/temporal_v2_overfit_16000}"
echo "[temporal-v2-overfit] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py     --num-gpus 1 --config-file configs/speaq_actiongenome_minimal.yaml     --dist-url tcp://127.0.0.1:29550     OUTPUT_DIR "${OUTPUT_DIR}"     DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal/annotations     DATASETS.ACTION_GENOME.FRAMES dataset/frames     DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1     DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0     MODEL.TEMPORAL.ENABLED True     MODEL.TEMPORAL.EVAL_ENABLED True     MODEL.TEMPORAL.MODE object_query_memory_v1     MODEL.TEMPORAL.RELATION_MEMORY_ENABLED False     MODEL.SAM3.ENABLED True     MODEL.SAM3.USE_PATCH_MERGE False     MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt     MODEL.SAM3.FREEZE True     MODEL.WEIGHTS model_0099999.pth     MODEL.DETR.HEAD_WEIGHTS model_0099999.pth     MODEL.DETR.LOAD_HEAD_ONLY False     MODEL.DETR.LOAD_FULL_WEIGHTS True     SOLVER.IMS_PER_BATCH 12     SOLVER.BASE_LR 0.0001     SOLVER.MAX_ITER 16000     SOLVER.STEPS '(4000,12000)'     SOLVER.WARMUP_ITERS 500     SOLVER.CHECKPOINT_PERIOD 2000     TEST.EVAL_PERIOD 2000     DATALOADER.NUM_WORKERS 2
RET=$?
echo "[temporal-v2-overfit] Done exit=${RET}"
