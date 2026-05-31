#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p z_outputs/test_qualityaux
python3 train_iterative_model.py     --num-gpus 1 --num-machines 1     --config-file configs/speaq_actiongenome_minimal.yaml     --dist-url "tcp://127.0.0.1:29599"     OUTPUT_DIR z_outputs/test_qualityaux     MODEL.WEIGHTS model_0099999.pth     MODEL.DETR.LOAD_FULL_WEIGHTS True     MODEL.SAM3.ENABLED True     MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt     MODEL.SAM3.FREEZE True     MODEL.ROI_REFINE.ENABLED True     MODEL.ROI_REFINE.LOSS_ENABLED True     DATASETS.ACTION_GENOME.ANNOTATIONS dataset/annotations     DATASETS.ACTION_GENOME.FRAMES dataset/frames     SOLVER.IMS_PER_BATCH 2     SOLVER.MAX_ITER 50     MODEL.DETR.OBJ_MISSED_AUX.ENABLED False     2>&1 | tee z_outputs/test_qualityaux/log.txt
