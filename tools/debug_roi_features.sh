#!/usr/bin/env bash
# Eval-only with overfit checkpoint to check ROI predictions
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUT_DIR=z_outputs/overfit_resnet_roi_bs12_16000/debug_roi
mkdir -p "${OUT_DIR}"

python3 train_iterative_model.py     --eval-only     --num-gpus 1     --config-file configs/speaq_actiongenome_minimal.yaml     --dist-url tcp://127.0.0.1:29599     OUTPUT_DIR "${OUT_DIR}"     DATASETS.TRAIN "('AG_train',)"     DATASETS.TEST "('AG_train',)"     DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal/annotations     DATASETS.ACTION_GENOME.FRAMES dataset/frames     DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1     DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0     MODEL.SAM3.ENABLED False     MODEL.TEMPORAL.ENABLED False     MODEL.ROI_REFINE.ENABLED True     MODEL.ROI_REFINE.LOSS_ENABLED True     MODEL.ROI_REFINE.RESNET_FPN_LEVEL 1     MODEL.ROI_REFINE.STRIDE 16     MODEL.ROI_REFINE.APPLY_TO all     MODEL.WEIGHTS z_outputs/overfit_resnet_roi_bs12_16000/model_0015999.pth     MODEL.DETR.HEAD_WEIGHTS z_outputs/overfit_resnet_roi_bs12_16000/model_0015999.pth     MODEL.DETR.LOAD_HEAD_ONLY False     MODEL.DETR.LOAD_FULL_WEIGHTS True     SOLVER.IMS_PER_BATCH 4     DATALOADER.NUM_WORKERS 2
echo "Eval done: ${OUT_DIR}"
