#!/usr/bin/env bash
# Overfit triplet memory v3 + consistent person scoring.
# Same as overfit_triplet_v3_16000.sh but enables PERSON_SCORE_SCALE=300 so the
# person prior feeds BOTH eval output and the triplet-memory update (single source
# of truth). Without it the relation-subject head under-scores person natively,
# triplet quality collapses to ~0, the memory bank stays empty, and the injector
# receives zero gradient. Base config is left untouched (fulltask relies on scale=1.0).
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate base
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/overfit_triplet_v3_16000_pscale_repro1000}"
mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"
echo "[Triplet v3 16K pscale] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py \
    --num-gpus 1 \
    --config-file configs/speaq_actiongenome_minimal.yaml \
    --dist-url tcp://127.0.0.1:29833 \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.TRAIN "('AG_train',)" \
    DATASETS.TEST "('AG_train',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0 \
    DATASETS.AG_TEMPORAL.ENABLED True \
    DATASETS.AG_TEMPORAL.CLIP_MODE between_keyframes \
    DATASETS.AG_TEMPORAL.NUM_INTERMEDIATE_FRAMES 0 \
    MODEL.WEIGHTS model_0099999.pth \
    MODEL.DETR.HEAD_WEIGHTS model_0099999.pth \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    MODEL.DETR.LOAD_CLASS_HEAD True \
    MODEL.DETR.PERSON_SCORE_SCALE 300.0 \
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
    MODEL.ROI_REFINE.ENABLED False \
    MODEL.ROI_REFINE.LOSS_ENABLED False \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.STEPS '(4000,12000)' \
    SOLVER.WARMUP_ITERS 500 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    TEST.EVAL_PERIOD 2000 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 4

RET=$?
echo "[Triplet v3 16K pscale] Done exit=${RET}"
