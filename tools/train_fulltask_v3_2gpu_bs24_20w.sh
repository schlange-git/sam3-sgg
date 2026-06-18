#!/usr/bin/env bash
# 本地双卡训练 (fulltask v3, x300 gate-only): bs24 / 12 每卡 (RTX5880 Ada 46G, 约 23G/卡, 放得下)。
# 复用生产配置 configs/fulltask_roiresid_pm_clip_v3_xsam_bs24_20w.yaml (与八卡脚本同一 yaml)。
#
# 与八卡脚本 tools/train_fulltask_v3_8gpu_bs96_80k.sh 的关系:
#   - 八卡: bs96 / BASE_LR 2e-4 / 80000 iters  (大 batch, 另一套 LR 与 iter 数)。
#   - 本脚本: bs24 / BASE_LR 1e-4 / 200000 iters (sqrt 缩放: bs24 即基准 -> LR 不变 1e-4)。
#   - 两者 IMS_PER_BATCH 不同 => 不是同一次可 resume 续训的训练 (见下方 RESUME 说明)。
#
# 【关于 resume 到八卡】(已确认 SOLVER.REFERENCE_WORLD_SIZE=0, auto_scale_workers 为 no-op,
#  故 IMS_PER_BATCH 是字面"总 batch", 不随 --num-gpus 自动缩放; 每卡 = IMS_PER_BATCH / num_gpus):
#   - 干净 resume 充要条件 = IMS_PER_BATCH / OUTPUT_DIR / LR 调度 全部不变, 仅改 --num-gpus。
#     即八卡空出后: NUM_GPUS=8 RESUME=1 bash 本脚本 <同一OUTPUT_DIR>
#     d2 读 OUTPUT_DIR/last_checkpoint 续 iter+optimizer+scheduler; 八卡此时 3/卡, 仅加速,数值完全一致。
#   - 若想换成 bs96 大 batch (八卡脚本那套): 那是"换 batch", 不能当 resume;
#     正确做法 = 用本次 checkpoint 作 MODEL.WEIGHTS 暖启动八卡 bs96 新 run (LR 调度重新计)。
#
# LR: BASE_LR=1e-4 (与参考实验 z_outputs/sam3_roi_eiou_cornerloss_bs24_160000_dist 同 bs24 同 LR);
#     GATE_LR_MULTIPLIER=2.0 (比参考 ×1 稍激进, 与八卡脚本一致, 远低于 yaml 生产 ×5)。
# PERSON_SCORE_SCALE=300: gate-only (只进 triplet 注入质量门 sub_score 连乘, 不进 criterion/eval)。
# STEPS/WARMUP/MAX_ITER 沿用生产 yaml 的 bs24 调度 (200000)。正式训练关闭 TRIPLET_DEBUG。
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate speaq 2>/dev/null || true

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
CONFIG="configs/fulltask_roiresid_pm_clip_v3_xsam_bs24_20w.yaml"
OUTPUT_DIR="${1:-z_outputs/train_fulltask_v3_2gpu_bs24_20w}"
NUM_GPUS="${2:-2}"
PORT="${PORT:-29832}"
RESUME="${RESUME:-0}"        # RESUME=1 续训 (读 OUTPUT_DIR/last_checkpoint); 默认 0 全新训练
mkdir -p "${OUTPUT_DIR}"
export TRIPLET_DEBUG=0
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"

RESUME_ARG=""
if [ "${RESUME}" = "1" ]; then RESUME_ARG="--resume"; fi

echo "=============================================="
echo "[train v3 2gpu bs24 20w] OUTPUT=${OUTPUT_DIR} GPU=${NUM_GPUS} PORT=${PORT} RESUME=${RESUME}"
echo "[train v3 2gpu bs24 20w] CONFIG=${CONFIG} (生产配置, 全量数据集)"
echo "[train v3 2gpu bs24 20w] bs=24 (12/卡) | BASE_LR=1e-4 (bs24 基准, sqrt 不变) | GATE_LR_MULT=2.0 (gate 绝对 2e-4)"
echo "[train v3 2gpu bs24 20w] MAX_ITER=200000 STEPS=(30000,175000) WARMUP=2000 | PERSON_SCORE_SCALE=300 (gate-only)"
echo "=============================================="

python3 train_iterative_model.py \
    --num-gpus "${NUM_GPUS}" \
    --config-file "${CONFIG}" \
    --dist-url "tcp://127.0.0.1:${PORT}" \
    ${RESUME_ARG} \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    MODEL.DETR.PERSON_SCORE_SCALE 300.0 \
    SOLVER.IMS_PER_BATCH 24 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.GATE_LR_MULTIPLIER 2.0 \
    SOLVER.MAX_ITER 200000 \
    SOLVER.STEPS "(30000,175000)" \
    SOLVER.WARMUP_ITERS 2000 \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    TEST.EVAL_PERIOD 5000 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 8
RET=$?
echo "[train v3 2gpu bs24 20w] Done exit=${RET}"
