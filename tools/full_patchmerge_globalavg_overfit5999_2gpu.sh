#!/usr/bin/env bash
# =============================================================================
# Full Action Genome patch_merge training from 20v global_avg overfit checkpoint.
#   Warm-start: z_outputs/overfit_patchmerge_globalavg_2gpu_20v/model_0005999.pth
#   Aligns with reference full patch_merge run xsam_local_pretrain_bs16_iter160000:
#   TEMPORAL/ROI/OBJ_SPLIT off, PERSON_SCORE_SCALE=1.0, BASE_LR=1e-4,
#   BACKBONE_MULTIPLIER=1.0, USE_PATCH_MERGE=True, TARGET_STRIDE=32.
# =============================================================================
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate speaq 2>/dev/null || true

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"

OUTPUT_DIR="${1:-z_outputs/full_patchmerge_globalavg_overfit5999_2gpu}"
NUM_GPUS="${2:-2}"
PORT="${PORT:-29839}"
WARM_START="z_outputs/overfit_patchmerge_globalavg_2gpu_20v/model_0005999.pth"

assert_file() {
    if [[ ! -f "$1" ]]; then
        echo "[full patchmerge globalavg overfit5999] missing file: $1" >&2
        exit 1
    fi
}

assert_file "$WARM_START"
assert_file "configs/speaq_actiongenome_minimal.yaml"
assert_file "sam3/weights/sam3.pt"
mkdir -p "$OUTPUT_DIR"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"

echo "=============================================="
echo "[full patchmerge globalavg overfit5999] OUTPUT=${OUTPUT_DIR} PORT=${PORT} GPUS=${NUM_GPUS}"
echo "[full patchmerge globalavg overfit5999] WARM_START=${WARM_START}"
echo "[full patchmerge globalavg overfit5999] full AG dataset, pure patch_merge flags aligned to xsam_local_pretrain_bs16_iter160000"
echo "[full patchmerge globalavg overfit5999] heatmaps -> ${OUTPUT_DIR}/patch_merge_heatmaps"
echo "=============================================="

python3 train_iterative_model.py \
    --num-gpus "${NUM_GPUS}" \
    --config-file configs/speaq_actiongenome_minimal.yaml \
    --dist-url tcp://127.0.0.1:${PORT} \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.TRAIN "('AG_train',)" \
    DATASETS.TEST "('AG_val',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 400 \
    DATASETS.ACTION_GENOME.SAMPLER_MODE clip \
    DATASETS.AG_TEMPORAL.ENABLED False \
    DATASETS.AG_TEMPORAL.CLIP_MODE between_keyframes \
    DATASETS.AG_TEMPORAL.NUM_INTERMEDIATE_FRAMES 1 \
    MODEL.META_ARCHITECTURE IterativeRelationDetr \
    MODEL.WEIGHTS "${WARM_START}" \
    MODEL.DETR.HEAD_WEIGHTS "${WARM_START}" \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    MODEL.DETR.LOAD_CLASS_HEAD True \
    MODEL.DETR.PERSON_SCORE_SCALE 1.0 \
    MODEL.DETR.BOX_LOSS_TYPE giou \
    MODEL.OBJ_SPLIT.ENABLED False \
    MODEL.SAM3.ENABLED True \
    MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt \
    MODEL.SAM3.FREEZE True \
    MODEL.SAM3.USE_PATCH_MERGE True \
    MODEL.SAM3.TARGET_STRIDE 32 \
    MODEL.SAM3.PATCH_MERGE_INIT_MODE global_avg \
    MODEL.SAM3.PATCH_MERGE_INIT_NOISE_STD 0.0 \
    MODEL.TEMPORAL.ENABLED False \
    MODEL.TEMPORAL.EVAL_ENABLED False \
    MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED False \
    MODEL.ROI_REFINE.ENABLED False \
    MODEL.ROI_REFINE.LOSS_ENABLED False \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.BACKBONE_MULTIPLIER 1.0 \
    SOLVER.GATE_LR_MULTIPLIER 5.0 \
    SOLVER.MAX_ITER 160000 \
    SOLVER.STEPS '(40000,144000)' \
    SOLVER.WARMUP_ITERS 1000 \
    SOLVER.WARMUP_METHOD linear \
    SOLVER.CHECKPOINT_PERIOD 10000 \
    TEST.EVAL_PERIOD 20000 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 4

RET=$?
echo "[full patchmerge globalavg overfit5999] Done exit=${RET}"
