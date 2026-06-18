#!/usr/bin/env bash
# =============================================================================
# stage5 · clip-as-sample + sequential forward · 单卡 bs16 · 20v 过拟合
#   修复: CLIP_SEQUENTIAL_FORWARD=True，同 clip 内逐帧 read→inject→update，与 eval 一致
#   对照 stage3 成功过拟合: PERSON_SCORE_SCALE=1.0, MAX_ITER=16000, 确定性 clip phase
#   go/no-go: AP50 40+, Recall@50 75+, SGG R@20 0.45+
# =============================================================================
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate speaq 2>/dev/null || true

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/overfit_stage5_clipsample_1gpu_bs16_20v}"
mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"
echo "[stage5 clipsample sequential 1gpu bs16] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py \
    --num-gpus 1 \
    --config-file configs/speaq_actiongenome_minimal.yaml \
    --dist-url tcp://127.0.0.1:29836 \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.TRAIN "('AG_train',)" \
    DATASETS.TEST "('AG_train',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal_20v/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0 \
    DATASETS.ACTION_GENOME.SAMPLER_MODE clip_sample \
    DATASETS.ACTION_GENOME.CLIP_SAMPLE_LEN 4 \
    DATASETS.ACTION_GENOME.CLIP_SAMPLE_RANDOM_PHASE False \
    DATASETS.AG_TEMPORAL.ENABLED True \
    DATASETS.AG_TEMPORAL.CLIP_MODE between_keyframes \
    DATASETS.AG_TEMPORAL.NUM_INTERMEDIATE_FRAMES 0 \
    MODEL.WEIGHTS model_0099999.pth \
    MODEL.DETR.HEAD_WEIGHTS model_0099999.pth \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    MODEL.DETR.LOAD_CLASS_HEAD True \
    MODEL.DETR.PERSON_SCORE_SCALE 1.0 \
    MODEL.SAM3.ENABLED True \
    MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt \
    MODEL.SAM3.FREEZE True \
    MODEL.SAM3.USE_PATCH_MERGE False \
    MODEL.TEMPORAL.ENABLED True \
    MODEL.TEMPORAL.EVAL_ENABLED True \
    MODEL.TEMPORAL.MODE triplet_memory_v3 \
    MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED True \
    MODEL.TEMPORAL.CLIP_SEQUENTIAL_FORWARD True \
    MODEL.TEMPORAL.INJECT_OBJECT True \
    MODEL.TEMPORAL.INJECT_RELATION True \
    MODEL.TEMPORAL.NON_KEY_SKIP_LOSS True \
    MODEL.TEMPORAL.NON_KEY_SKIP_EVAL True \
    MODEL.TEMPORAL.NON_KEY_RUN_OBJECT_ONLY True \
    MODEL.ROI_REFINE.ENABLED False \
    MODEL.ROI_REFINE.LOSS_ENABLED False \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.STEPS '(8000,14000)' \
    SOLVER.WARMUP_ITERS 500 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    TEST.EVAL_PERIOD 1000 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 4

RET=$?
echo "[stage5 clipsample sequential 1gpu bs16] Done exit=${RET}"
