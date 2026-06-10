#!/usr/bin/env bash
# 全量训练（修复版）—— 为"开启全部 flag 的最终实验"提供一个收敛良好的 X-SAM
# patch-merge 模块起点。
# 在 USE_PATCH_MERGE 基础上叠加 patch-merge 收敛三杠杆：
#   (1) 偏置初始化  MODEL.SAM3.PATCH_MERGE_INIT_NOISE_STD 0.01 (打破 avg_pool 驻点)
#   (2) 满额梯度倍率 SOLVER.BACKBONE_MULTIPLIER 1.0 (冻结 SAM3 下仅作用于 patch-merge)
#   (3) patch-merge 参数组 weight_decay=0 (代码层固定，防 L2 拉零)
# 其余 flag 不开（时序 / ROI / eiou / corner / person_score 均关），保持纯净，
# 只让 patch-merge 在全量数据上学到一个比 avg_pool 更好的下采样后做权重起点。
# 规格：bs16，MAX_ITER 20000。诊断 xsam_delta_norm 写入 diagnostics_log.csv（每 50 iter）。
set -euo pipefail
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/xsam_local_pretrain_bs16_iter20000}"
NUM_GPUS="${2:-1}"
mkdir -p "${OUTPUT_DIR}"
export LEARNABLE_DIAGNOSTICS_LOG_PATH="${OUTPUT_DIR}/diagnostics_log.csv"
export LEARNABLE_DIAGNOSTICS_LOG_PERIOD=50
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"
echo "[X-SAM local pretrain · FIXED] OUTPUT=${OUTPUT_DIR} GPU=${NUM_GPUS} bs=16 iter=20000"

python3 train_iterative_model.py \
    --num-gpus "${NUM_GPUS}" \
    --config-file configs/speaq_actiongenome_minimal.yaml \
    --dist-url tcp://127.0.0.1:29610 \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    MODEL.WEIGHTS model_0099999.pth \
    MODEL.DETR.HEAD_WEIGHTS model_0099999.pth \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    MODEL.DETR.LOAD_CLASS_HEAD True \
    MODEL.SAM3.ENABLED True \
    MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt \
    MODEL.SAM3.FREEZE True \
    MODEL.SAM3.USE_PATCH_MERGE True \
    MODEL.SAM3.PATCH_MERGE_INIT_NOISE_STD 0.01 \
    MODEL.TEMPORAL.ENABLED False \
    MODEL.TEMPORAL.EVAL_ENABLED False \
    MODEL.ROI_REFINE.ENABLED False \
    MODEL.ROI_REFINE.LOSS_ENABLED False \
    SOLVER.IMS_PER_BATCH 16 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.BACKBONE_MULTIPLIER 1.0 \
    SOLVER.MAX_ITER 20000 \
    SOLVER.GAMMA 0.1 \
    SOLVER.STEPS '(12000,18000)' \
    SOLVER.WARMUP_ITERS 1000 \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    TEST.EVAL_PERIOD 5000 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 4

RET=$?
echo "[X-SAM local pretrain · FIXED] Done exit=${RET}"
