#!/usr/bin/env bash
# Continue training with --resume to restore optimizer state, MAX_ITER=40000
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
CKPT_DIR=z_outputs/fulltask_temporal_roi_no_xsam_bs96_iter40000
echo "[Continue] Resuming from ${CKPT_DIR} to MAX_ITER=40000"
python3 train_iterative_model.py     --resume     --num-gpus 8     --config-file configs/speaq_ag_roi.yaml     --dist-url tcp://127.0.0.1:29500     OUTPUT_DIR "${CKPT_DIR}"     SOLVER.MAX_ITER 40000     SOLVER.STEPS '(10000,30000)'
RET=$?
echo "[Continue] Done exit=${RET}"
