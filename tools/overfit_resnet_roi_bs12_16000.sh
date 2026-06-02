#!/usr/bin/env bash
# Overfit ResNet-101 + ROI14 only (pure SpeaQ baseline, no SAM3/temporal/X-SAM)
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/overfit_resnet_roi_bs12_16000}"
mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"
echo "[Overfit ResNet+ROI] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py     --num-gpus 1     --config-file configs/speaq_actiongenome_minimal.yaml     --dist-url tcp://127.0.0.1:29550     OUTPUT_DIR "${OUTPUT_DIR}"     DATASETS.TRAIN "('AG_train',)"     DATASETS.TEST "('AG_train',)"     DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal/annotations     DATASETS.ACTION_GENOME.FRAMES dataset/frames     DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1     DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0     MODEL.SAM3.ENABLED False     MODEL.TEMPORAL.ENABLED False     MODEL.TEMPORAL.EVAL_ENABLED False     MODEL.ROI_REFINE.ENABLED True     MODEL.ROI_REFINE.LOSS_ENABLED True     MODEL.ROI_REFINE.RESNET_FPN_LEVEL 1     MODEL.ROI_REFINE.STRIDE 16     MODEL.WEIGHTS res101_pretrained.pth     MODEL.DETR.HEAD_WEIGHTS res101_pretrained.pth     MODEL.DETR.LOAD_HEAD_ONLY False     MODEL.DETR.LOAD_FULL_WEIGHTS True     SOLVER.IMS_PER_BATCH 12     SOLVER.BASE_LR 0.0001     SOLVER.MAX_ITER 16000     SOLVER.STEPS '(4000,12000)'     SOLVER.WARMUP_ITERS 500     SOLVER.CHECKPOINT_PERIOD 2000     TEST.EVAL_PERIOD 2000     DATALOADER.NUM_WORKERS 2
    MODEL.ROI_REFINE.APPLY_TO "all"
    MODEL.ROI_REFINE.SMALL_AREA_THRESH "0.001"
RET=$?
echo "[Overfit ResNet+ROI] Done exit=${RET}"
