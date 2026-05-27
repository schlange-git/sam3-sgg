#!/usr/bin/env bash
# Full task overfit (1 GPU, X-SAM patch merge, SAM3 frozen, patch_merge_proj trainable)
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/fulltask_overfit_16000}"
echo "[fulltask-overfit] OUTPUT=${OUTPUT_DIR}"
python3 train_iterative_model.py     --num-gpus 1 --config-file configs/speaq_actiongenome_minimal.yaml     --dist-url tcp://127.0.0.1:29550     OUTPUT_DIR "${OUTPUT_DIR}"     DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit/annotations     DATASETS.ACTION_GENOME.FRAMES dataset/frames     DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1     DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 100     MODEL.TEMPORAL.ENABLED False     MODEL.SAM3.ENABLED True     MODEL.SAM3.USE_PATCH_MERGE True     MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt     MODEL.SAM3.FREEZE True     MODEL.WEIGHTS model_0099999.pth     MODEL.DETR.HEAD_WEIGHTS model_0099999.pth     MODEL.DETR.LOAD_HEAD_ONLY False     MODEL.DETR.LOAD_FULL_WEIGHTS True     SOLVER.IMS_PER_BATCH 8     SOLVER.BASE_LR 0.0001     SOLVER.MAX_ITER 16000     SOLVER.STEPS '(4000,12000)'     SOLVER.WARMUP_ITERS 500     SOLVER.CHECKPOINT_PERIOD 2000     TEST.EVAL_PERIOD 2000     DATALOADER.NUM_WORKERS 2
RET=$?
echo "[fulltask-overfit] Done exit=${RET}"
