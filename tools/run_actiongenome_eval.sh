#!/usr/bin/env bash
# ActionGenome 评测 + 可视化（与训练解耦）
# 用法:
#   ./run_actiongenome_eval.sh [OUTPUT_DIR] [NUM_GPUS]
#   - OUTPUT_DIR: 训练阶段的输出目录（里面要有 model_final.pth 或 last_checkpoint）
#   - NUM_GPUS  : 评测使用的 GPU 数（默认 2）
#
# 说明:
#   - 本脚本只做 eval-only ，不再发起训练。
#   - 路径、数据集配置与 run_actiongenome_train_eval.sh 保持一致，方便无缝切换。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"/..

# 修正无效的 OMP_NUM_THREADS
if [[ -n "${OMP_NUM_THREADS:-}" ]] && [[ "${OMP_NUM_THREADS}" == "0" || ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ ]]; then
  unset OMP_NUM_THREADS
fi

OUTPUT_DIR="${1:-z_outputs/res101_from_pretrained_160000iters_bs8}"
NUM_GPUS="${2:-2}"

CONFIG="configs/speaq_actiongenome_minimal.yaml"
PORT="${PORT:-29500}"

# 数据路径（与训练脚本保持一致）
AG_ANNOTATIONS="${AG_ANNOTATIONS:-dataset/annotations}"
AG_FRAMES="${AG_FRAMES:-dataset/frames}"
AG_VIDEOS="${AG_VIDEOS:-dataset/videos}"

# 是否额外在 overfit 训练集上评测（1/0）
EVAL_OVERFIT_TRAIN="${EVAL_OVERFIT_TRAIN:-0}"
# 可视化最大图片数（-1 = 全部）
VIS_MAX_IMAGES="${VIS_MAX_IMAGES:-1000}"

# -----------------------------
# 内存监控函数：当内存占用超过93%时自动kill评测进程
# -----------------------------
monitor_memory() {
    local pid=$1
    local threshold=93  # 内存使用率阈值（%）
    local check_interval=5  # 检查间隔（秒）
    
    echo "[内存监控] 开始监控进程 ${pid}，阈值: ${threshold}%"
    
    while kill -0 "${pid}" 2>/dev/null; do
        local mem_info
        mem_info=$(free | grep Mem)
        local mem_total
        mem_total=$(echo "${mem_info}" | awk '{print $2}')
        local mem_used
        mem_used=$(echo "${mem_info}" | awk '{print $3}')
        local mem_usage=$((mem_used * 100 / mem_total))
        
        if [ "${mem_usage}" -ge "${threshold}" ]; then
            echo "[内存监控] ⚠️  警告: 内存使用率 ${mem_usage}% 超过阈值 ${threshold}%"
            echo "[内存监控] 正在终止评测进程 ${pid}..."
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 2
            if kill -0 "${pid}" 2>/dev/null; then
                echo "[内存监控] 强制终止进程 ${pid}..."
                kill -KILL "${pid}" 2>/dev/null || true
            fi
            echo "[内存监控] 评测进程已终止"
            exit 1
        fi
        
        sleep "${check_interval}"
    done
    
    echo "[内存监控] 评测进程已结束，停止监控"
}

# -----------------------------
# 1) 解析要评测的 checkpoint
#    优先顺序:
#      1) 环境变量 CHECKPOINT
#      2) OUTPUT_DIR/model_final.pth
#      3) OUTPUT_DIR/last_checkpoint 中记录的权重
# -----------------------------
if [[ -z "${CHECKPOINT:-}" ]]; then
  if [[ -f "${OUTPUT_DIR}/model_final.pth" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/model_final.pth"
  elif [[ -f "${OUTPUT_DIR}/last_checkpoint" ]]; then
    ck_rel=$(head -n 1 "${OUTPUT_DIR}/last_checkpoint" | tr -d ' \r\n')
    CHECKPOINT="${OUTPUT_DIR}/${ck_rel}"
  else
    CHECKPOINT=""
  fi
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "[EVAL-ONLY] Checkpoint not found. Tried:"
  echo "  - CHECKPOINT=${CHECKPOINT:-<empty>}"
  echo "  - ${OUTPUT_DIR}/model_final.pth"
  echo "  - ${OUTPUT_DIR}/last_checkpoint"
  exit 1
fi

ck_name="$(basename "${CHECKPOINT}")"
ck_tag="${ck_name%.*}"

echo "=============================================="
echo "[EVAL-ONLY] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[EVAL-ONLY] NUM_GPUS=${NUM_GPUS}"
echo "[EVAL-ONLY] CHECKPOINT=${CHECKPOINT}"
echo "[EVAL-ONLY] CK_TAG=${ck_tag}"
echo "[EVAL-ONLY] AG_ANNOTATIONS=${AG_ANNOTATIONS}"
echo "[EVAL-ONLY] AG_FRAMES=${AG_FRAMES}"
echo "[EVAL-ONLY] AG_VIDEOS=${AG_VIDEOS}"
echo "[EVAL-ONLY] EVAL_OVERFIT_TRAIN=${EVAL_OVERFIT_TRAIN}"
echo "[EVAL-ONLY] VIS_MAX_IMAGES=${VIS_MAX_IMAGES}"
echo "=============================================="

# -----------------------------
# 2) 主评测：AG_val (eval-only)
#    按 checkpoint 名字后缀单独建目录，避免多次评测互相覆盖
# -----------------------------
VAL_EVAL_DIR="${OUTPUT_DIR}/eval_AG_val_${ck_tag}"
mkdir -p "${VAL_EVAL_DIR}"
echo "[EVAL-ONLY] Running evaluation on AG_val (with memory monitor) ..."

python train_iterative_model.py \
  --eval-only \
  --num-gpus "${NUM_GPUS}" \
  --config-file "${CONFIG}" \
  --dist-url "tcp://127.0.0.1:${PORT}" \
  OUTPUT_DIR "${VAL_EVAL_DIR}" \
  DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}" \
  DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}" \
  DATASETS.ACTION_GENOME.VIDEOS "${AG_VIDEOS}" \
  MODEL.WEIGHTS "${CHECKPOINT}" \
  ${OPTS:-} &
EVAL_PID=$!

monitor_memory "${EVAL_PID}" &
MONITOR_PID=$!

wait "${EVAL_PID}"
EVAL_EXIT_CODE=$?
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true

if [ "${EVAL_EXIT_CODE}" -ne 0 ]; then
  echo "[EVAL-ONLY] Evaluation on AG_val failed with exit code: ${EVAL_EXIT_CODE}"
  exit "${EVAL_EXIT_CODE}"
fi
echo "[EVAL-ONLY] Evaluation on AG_val done. Output: ${VAL_EVAL_DIR}"

# -----------------------------
# 3) 可选：在 overfit 训练集上再评测一次
# -----------------------------
if [[ "${EVAL_OVERFIT_TRAIN}" == "1" ]]; then
  TRAIN_EVAL_DIR="${OUTPUT_DIR}/eval_AG_train_overfit_${ck_tag}"
  mkdir -p "${TRAIN_EVAL_DIR}"
  echo "[EVAL-ONLY] Running evaluation on AG_train overfit split (with memory monitor) ..."

  python train_iterative_model.py \
    --eval-only \
    --num-gpus "${NUM_GPUS}" \
    --config-file "${CONFIG}" \
    --dist-url "tcp://127.0.0.1:${PORT}" \
    OUTPUT_DIR "${TRAIN_EVAL_DIR}" \
    DATASETS.TEST "('AG_train',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}" \
    DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}" \
    DATASETS.ACTION_GENOME.VIDEOS "${AG_VIDEOS}" \
    MODEL.WEIGHTS "${CHECKPOINT}" \
    ${OPTS:-} &
  EVAL_TRAIN_PID=$!

  monitor_memory "${EVAL_TRAIN_PID}" &
  MONITOR_PID=$!

  wait "${EVAL_TRAIN_PID}"
  EVAL_TRAIN_EXIT_CODE=$?
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true

  if [ "${EVAL_TRAIN_EXIT_CODE}" -ne 0 ]; then
    echo "[EVAL-ONLY] Overfit-train evaluation failed with exit code: ${EVAL_TRAIN_EXIT_CODE}"
    exit "${EVAL_TRAIN_EXIT_CODE}"
  fi
  echo "[EVAL-ONLY] Evaluation on AG_train overfit split done. Output: ${TRAIN_EVAL_DIR}"
else
  echo "[EVAL-ONLY] Skip AG_train overfit evaluation (EVAL_OVERFIT_TRAIN=${EVAL_OVERFIT_TRAIN})"
fi
