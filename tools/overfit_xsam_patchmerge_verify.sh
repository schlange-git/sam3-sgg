#!/usr/bin/env bash
# X-SAM patch-merge convergence VERIFICATION (overfit, default reproduction profile).
# Goal: test whether the patch-merge 1x1 conv can actually learn a better-than-avgpool
# downsampling once we remove the three things that pin it at the avg-pool stationary point:
#   (1) starved LR     -> SOLVER.BACKBONE_MULTIPLIER 1.0  (full base LR; only patch-merge
#                          is a trainable "backbone" param since SAM3 ViT is frozen).
#   (2) stationary init -> MODEL.SAM3.PATCH_MERGE_INIT_NOISE_STD 0.01 (Gaussian offset on
#                          the avg-pool init so iter0 gradient is non-zero; not zeroed,
#                          not fully random -- avg-pool skeleton + sizable perturbation).
#   (3) weight_decay pull -> patch-merge param group now uses weight_decay=0 (code change in
#                          engine/trainer.py _ensure_late_patch_merge_optimizer_params).
# Verification metric: xsam_delta_norm column in the diagnostics CSV
#   (LEARNABLE_DIAGNOSTICS_LOG_PATH), logged every 50 iters -- should now move far beyond
#   the ~7% baseline drift if the conv is genuinely learning.
# Base config / data / steps mirror the reproduction run
#   z_outputs/overfit_xsam_no_roi_bs12_16000_reporduct (bs12, BASE_LR 1e-4, 16000 iter).
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/overfit_xsam_patchmerge_verify_bs12_16000}"
mkdir -p "${OUTPUT_DIR}"
export LEARNABLE_DIAGNOSTICS_LOG_PATH="${OUTPUT_DIR}/diagnostics_log.csv"
export LEARNABLE_DIAGNOSTICS_LOG_PERIOD=50
echo "[X-SAM patch-merge verify] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py \
    --num-gpus 1 --config-file configs/speaq_ag_roi.yaml \
    --dist-url tcp://127.0.0.1:29560 \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.TRAIN "('AG_train',)" \
    DATASETS.TEST "('AG_train',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0 \
    MODEL.WEIGHTS model_0099999.pth \
    MODEL.DETR.HEAD_WEIGHTS model_0099999.pth \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    MODEL.SAM3.ENABLED True \
    MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt \
    MODEL.SAM3.FREEZE True \
    MODEL.SAM3.USE_PATCH_MERGE True \
    MODEL.SAM3.PATCH_MERGE_INIT_NOISE_STD 0.01 \
    MODEL.TEMPORAL.ENABLED False \
    MODEL.TEMPORAL.EVAL_ENABLED False \
    MODEL.ROI_REFINE.ENABLED False \
    MODEL.ROI_REFINE.LOSS_ENABLED False \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.BACKBONE_MULTIPLIER 1.0 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.STEPS '(4000,12000)' \
    SOLVER.WARMUP_ITERS 500 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    TEST.EVAL_PERIOD 2000 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 4

RET=$?
echo "[X-SAM patch-merge verify] Done exit=${RET}"
