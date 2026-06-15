#!/usr/bin/env bash
# 全内容训练: patch_merge + ROI(残差融合) + stage4 时序(clip 模式, triplet_memory_v3) · 从 X-SAM 预训练微调
#   IMS_PER_BATCH=24 / world_size 2 (12/卡) · MAX_ITER 20w · eval&ckpt 周期 5000
#   ROI 残差融合 (FUSION=residual, USE_GATE) + EVAL_DUAL (单次前向同出 override/raw 两套指标)
# 全部实验配置自包含在 yaml 中, 本脚本仅负责运行期参数 (OUTPUT_DIR / num-gpus / dist-url / resume)。
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate speaq 2>/dev/null || true

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
CONFIG="configs/fulltask_roiresid_pm_clip_v3_xsam_bs24_20w.yaml"
OUTPUT_DIR="${1:-z_outputs/fulltask_roiresid_pm_clip_v3_xsam_bs24_20w}"
NUM_GPUS="${2:-2}"
PORT="${PORT:-29711}"
mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"

echo "=============================================="
echo "[fulltask roiresid+pm+clip-v3 xsam] OUTPUT=${OUTPUT_DIR}"
echo "[fulltask roiresid+pm+clip-v3 xsam] CONFIG=${CONFIG} GPU=${NUM_GPUS} (bs24 -> 12/卡)"
echo "[fulltask roiresid+pm+clip-v3 xsam] init=z_outputs/xsam_local_pretrain_bs16_iter160000/model_0159999.pth"
echo "=============================================="

# 仅当 OUTPUT_DIR 已有 ckpt 时启用续训; 全新运行则从 MODEL.WEIGHTS (X-SAM) 初始化
RESUME_ARGS=()
if [[ -d "${OUTPUT_DIR}" ]]; then
    shopt -s nullglob
    CKPTS=("${OUTPUT_DIR}"/*.pth)
    shopt -u nullglob
    if (( ${#CKPTS[@]} > 0 )); then
        echo "[Resume] 发现 ${#CKPTS[@]} 个 ckpt, 启用 --resume"
        RESUME_ARGS=(--resume)
    fi
fi

python3 train_iterative_model.py \
    "${RESUME_ARGS[@]}" \
    --num-gpus "${NUM_GPUS}" \
    --config-file "${CONFIG}" \
    --dist-url "tcp://127.0.0.1:${PORT}" \
    OUTPUT_DIR "${OUTPUT_DIR}"
RET=$?
echo "[fulltask roiresid+pm+clip-v3 xsam] Done exit=${RET}"
