#!/usr/bin/env bash
# 20v 过拟合验证 (scale=1.0): 直接复用 20w 生产配置 fulltask_roiresid_pm_clip_v3_xsam_bs24_20w.yaml,
#   仅 opts 切到 20v 数据集 + 单卡小 batch + TRIPLET_DEBUG=1。
#   目的: 验证 PERSON_SCORE_SCALE=1.0 下 triplet bank 是否填充(三元素分数能否过 0.10 阈值)、
#         注入是否真正发生(mem 是否 None)。权重/patch_merge/scale/ROI残差/时序模块 全与 20w 一致。
#   起始权重沿用 yaml 内 xsam model_0159999.pth (patch_merge True / scale 1.0)。
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate speaq 2>/dev/null || true

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
CONFIG="configs/fulltask_roiresid_pm_clip_v3_xsam_bs24_20w.yaml"
OUTPUT_DIR="${1:-z_outputs/overfit_fulltask_v3_20v_scale1_dbg}"
NUM_GPUS="${2:-1}"
PORT="${PORT:-29820}"
mkdir -p "${OUTPUT_DIR}"
export TRIPLET_DEBUG=1
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"

echo "=============================================="
echo "[overfit v3 20v scale1 dbg] OUTPUT=${OUTPUT_DIR} GPU=${NUM_GPUS}"
echo "[overfit v3 20v scale1 dbg] CONFIG=${CONFIG} (复用 20w 生产配置)"
echo "[overfit v3 20v scale1 dbg] TRIPLET_DEBUG=1 -> 观察 bank 填充/注入; PERSON_SCORE_SCALE=1.0"
echo "=============================================="

python3 train_iterative_model.py \
    --num-gpus "${NUM_GPUS}" \
    --config-file "${CONFIG}" \
    --dist-url "tcp://127.0.0.1:${PORT}" \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.TRAIN "('AG_train',)" \
    DATASETS.TEST "('AG_train',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal_20v/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0 \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.STEPS "(4000,12000)" \
    SOLVER.WARMUP_ITERS 500 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    TEST.EVAL_PERIOD 2000 \
    SOLVER.EVAL_FIRST False \
    DATALOADER.NUM_WORKERS 4
RET=$?
echo "[overfit v3 20v scale1 dbg] Done exit=${RET}"
