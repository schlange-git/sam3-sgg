#!/usr/bin/env bash
# Same as overfit_xsam_no_roi but 2 GPU + model_xsam_pretrained.pth checkpoint.
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/overfit_xsam_pretrained_2gpu_bs24_8000}"
mkdir -p "${OUTPUT_DIR}"
echo "[Overfit X-SAM Pretrained 2GPU] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py     --num-gpus 2     --config-file configs/speaq_actiongenome_minimal.yaml     --dist-url tcp://127.0.0.1:29563     OUTPUT_DIR "${OUTPUT_DIR}"     DATASETS.TRAIN "('AG_train',)"     DATASETS.TEST "('AG_train',)"     DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal/annotations     DATASETS.ACTION_GENOME.FRAMES dataset/frames     DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1     DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0     MODEL.SAM3.ENABLED True     MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt     MODEL.SAM3.FREEZE True     MODEL.SAM3.USE_PATCH_MERGE True     MODEL.ROI_REFINE.ENABLED False     MODEL.ROI_REFINE.LOSS_ENABLED False     MODEL.TEMPORAL.ENABLED False     MODEL.TEMPORAL.EVAL_ENABLED False     MODEL.TEMPORAL.RELATION_MEMORY_ENABLED False     MODEL.WEIGHTS model_xsam_pretrained.pth     MODEL.DETR.HEAD_WEIGHTS model_xsam_pretrained.pth     MODEL.DETR.LOAD_HEAD_ONLY False     MODEL.DETR.LOAD_FULL_WEIGHTS True     MODEL.DETR.LOAD_CLASS_HEAD True     SOLVER.IMS_PER_BATCH 24     SOLVER.BASE_LR 0.0002     SOLVER.MAX_ITER 8000     SOLVER.STEPS '(2000,6000)'     SOLVER.WARMUP_ITERS 250     SOLVER.CHECKPOINT_PERIOD 2000     SOLVER.GATE_LR_MULTIPLIER 1.0     TEST.EVAL_PERIOD 2000     DATALOADER.NUM_WORKERS 2
RET=$?
echo "[Overfit X-SAM Pretrained 2GPU] Done exit=${RET}"
