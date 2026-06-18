#!/usr/bin/env bash
# 八卡正式训练 (fulltask v3, x300 gate-only): 相当于 bs96 / 80000 iters。
# 复用生产配置 configs/fulltask_roiresid_pm_clip_v3_xsam_bs24_20w.yaml
#   (SAMPLER_MODE=clip + patch_merge Fix A 已生效 + roi 残差融合 + triplet_memory_v3)。
#
# 学习率随 bs 缩放 (基准 = z_outputs/sam3_roi_eiou_cornerloss_bs24_160000_dist, 该实验 bs24/BASE_LR 1e-4/gate ×1):
#   - BASE_LR: bs 24->96 (4x), 取 sqrt 缩放 2x => 2e-4 (不用激进的线性 4x=4e-4, 也即不用 yaml 生产的 4e-4)。
#   - GATE_LR_MULTIPLIER: 参考实验 gate 无加速(×1); 这里给 ×2, 比参考稍激进, 远低于 yaml 生产的 ×5。
#     => gate 绝对 LR = 2e-4 * 2 = 4e-4 (yaml 生产是 4e-4*5=2e-3, 过激)。
#
# 其余沿用 yaml 生产默认: 全量数据集 (dataset/annotations + AG_val 400)、EVAL_FIRST=True、
#   起始权重 xsam model_0159999.pth (patch_merge True, Fix A 预建保证加载)。
# PERSON_SCORE_SCALE=300: 沿用 overfit 已验证的 gate-only 方案 (只进 triplet 注入质量门 sub_score 连乘,
#   不进 criterion / eval)。如需改回 1.0, 改下面这一行即可。
# batch: IMS_PER_BATCH=96 / 8卡 = 12/卡 (96 可被 world_size=8 整除; clip 模式无 clip_len 整除约束),
#        每卡负载与 yaml bs24/2卡=12 一致。
# STEPS/WARMUP/周期按 80000 等比 yaml(200000): STEPS(30000,175000)->(12000,70000), 周期 5000。
# 正式训练关闭 TRIPLET_DEBUG (不刷调试日志)。
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate speaq 2>/dev/null || true

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
CONFIG="configs/fulltask_roiresid_pm_clip_v3_xsam_bs24_20w.yaml"
OUTPUT_DIR="${1:-z_outputs/train_fulltask_v3_8gpu_bs96_80k}"
NUM_GPUS="${2:-8}"
PORT="${PORT:-29831}"
mkdir -p "${OUTPUT_DIR}"
export TRIPLET_DEBUG=0
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"

echo "=============================================="
echo "[train v3 8gpu bs96 80k] OUTPUT=${OUTPUT_DIR} GPU=${NUM_GPUS} PORT=${PORT}"
echo "[train v3 8gpu bs96 80k] CONFIG=${CONFIG} (生产配置, 全量数据集)"
echo "[train v3 8gpu bs96 80k] bs=96 (12/卡) | BASE_LR=2e-4 (bs sqrt 缩放) | GATE_LR_MULT=2.0 (gate 绝对 4e-4)"
echo "[train v3 8gpu bs96 80k] MAX_ITER=80000 STEPS=(12000,70000) WARMUP=2000 | PERSON_SCORE_SCALE=300 (gate-only)"
echo "=============================================="

python3 train_iterative_model.py \
    --num-gpus "${NUM_GPUS}" \
    --config-file "${CONFIG}" \
    --dist-url "tcp://127.0.0.1:${PORT}" \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    MODEL.DETR.PERSON_SCORE_SCALE 300.0 \
    SOLVER.IMS_PER_BATCH 96 \
    SOLVER.BASE_LR 0.0002 \
    SOLVER.GATE_LR_MULTIPLIER 2.0 \
    SOLVER.MAX_ITER 80000 \
    SOLVER.STEPS "(12000,70000)" \
    SOLVER.WARMUP_ITERS 2000 \
    SOLVER.CHECKPOINT_PERIOD 5000 \
    TEST.EVAL_PERIOD 5000 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 16
RET=$?
echo "[train v3 8gpu bs96 80k] Done exit=${RET}"
