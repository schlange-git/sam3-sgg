#!/usr/bin/env bash
# =============================================================================
# 纯 patch_merge 20v 过拟合：全局平均初始化范式
#   对齐参考训练 z_outputs/xsam_local_pretrain_bs16_iter160000：
#   TEMPORAL/ROI/OBJ_SPLIT 全关，PERSON_SCORE_SCALE=1.0，BASE_LR=1e-4，
#   BACKBONE_MULTIPLIER=1.0，USE_PATCH_MERGE=True，TARGET_STRIDE=32。
#
#   与旧 avg-init run 的关键差异：
#   PATCH_MERGE_INIT_MODE=global_avg，初始化 W[256,1024] 全部为 1/1024，
#   即每个输出通道从所有 pixel_unshuffle 后输入通道全局平均开始。
#
#   PatchMergeHeatmapHook 每次输出：
#   1) 全局 256x1024 heatmap；
#   2) 固定中部 16x64 局部 heatmap (rows 120:136, cols 480:544)。
# =============================================================================
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate speaq 2>/dev/null || true

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/overfit_patchmerge_globalavg_2gpu_20v}"
PORT="${PORT:-29838}"
mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"

echo "=============================================="
echo "[overfit patchmerge globalavg 2gpu 20v] OUTPUT=${OUTPUT_DIR} PORT=${PORT}"
echo "[overfit patchmerge globalavg 2gpu 20v] W[256,1024] global init = 1/1024, no noise"
echo "[overfit patchmerge globalavg 2gpu 20v] heatmaps -> ${OUTPUT_DIR}/patch_merge_heatmaps"
echo "=============================================="

python3 train_iterative_model.py \
    --num-gpus 2 \
    --config-file configs/speaq_actiongenome_minimal.yaml \
    --dist-url tcp://127.0.0.1:${PORT} \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.TRAIN "('AG_train',)" \
    DATASETS.TEST "('AG_train',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal_20v/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0 \
    DATASETS.ACTION_GENOME.SAMPLER_MODE clip \
    DATASETS.AG_TEMPORAL.ENABLED True \
    DATASETS.AG_TEMPORAL.CLIP_MODE between_keyframes \
    DATASETS.AG_TEMPORAL.NUM_INTERMEDIATE_FRAMES 1 \
    MODEL.META_ARCHITECTURE IterativeRelationDetr \
    MODEL.WEIGHTS model_0099999.pth \
    MODEL.DETR.HEAD_WEIGHTS model_0099999.pth \
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
    SOLVER.MAX_ITER 6000 \
    SOLVER.STEPS '(4500,5500)' \
    SOLVER.WARMUP_ITERS 250 \
    SOLVER.CHECKPOINT_PERIOD 500 \
    TEST.EVAL_PERIOD 500 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 4

RET=$?
echo "[overfit patchmerge globalavg 2gpu 20v] Done exit=${RET}"
