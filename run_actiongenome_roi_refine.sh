#!/usr/bin/env bash
# ROI refine 训练（SAM3 + Stride14 ROI refinement）
# 用法:
#   ./run_actiongenome_roi_refine.sh [OUTPUT_DIR] [NUM_GPUS] [OVERFIT]
#
#   第3个参数 OVERFIT:
#     0 或 off  -> 正常训练（默认）
#     1 或 on   -> 过拟合模式（500 帧，500 次迭代）
#     N         -> 过拟合模式（N 帧，N 次迭代）
#
# 环境变量:
#   SAM3_CHECKPOINT_PATH  (默认: sam3/weights/sam3.pt)
#   AG_ANNOTATIONS        (默认: dataset/annotations)
#   AG_FRAMES             (默认: dataset/frames)
#   AG_VIDEOS             (默认: dataset/videos)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Fix invalid OMP_NUM_THREADS
if [[ -n "${OMP_NUM_THREADS:-}" ]] && [[ "${OMP_NUM_THREADS}" == "0" || ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ ]]; then
  unset OMP_NUM_THREADS
fi

OUTPUT_DIR="${1:-z_outputs/sam3_roi_refine}"
NUM_GPUS="${2:-1}"
OVERFIT="${3:-1}"

CONFIG="configs/speaq_ag_roi.yaml"
PORT="${PORT:-29500}"

# 数据路径
AG_ANNOTATIONS="${AG_ANNOTATIONS:-dataset/annotations}"
AG_FRAMES="${AG_FRAMES:-dataset/frames}"
AG_VIDEOS="${AG_VIDEOS:-dataset/videos}"

# SAM3 权重路径
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-sam3/weights/sam3.pt}"
# DETR 预训练 head 权重（可选，从 VG 等 checkpoint 加载 transformer 权重）
DETR_HEAD_WEIGHTS="${DETR_HEAD_WEIGHTS:-model_0099999.pth}"
# 是否全量加载 DETR 权重（1=全量，0=head-only）
DETR_LOAD_FULL_WEIGHTS="${DETR_LOAD_FULL_WEIGHTS:-1}"

echo "=============================================="
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "OVERFIT=${OVERFIT}"
echo "SAM3_CHECKPOINT_PATH=${SAM3_CHECKPOINT_PATH}"
echo "DETR_HEAD_WEIGHTS=${DETR_HEAD_WEIGHTS}"
echo "=============================================="

# ---------- Overfitting 模式 ----------
OVERFIT_OPTS=()
if [[ "${OVERFIT}" != "0" && "${OVERFIT}" != "off" && "${OVERFIT}" != "" ]]; then
  if [[ "${OVERFIT}" == "1" || "${OVERFIT}" == "on" || "${OVERFIT}" == "true" ]]; then
    OVERFIT_ITERS=500
  else
    OVERFIT_ITERS="${OVERFIT}"
  fi
  echo ">>> 启用过拟合模式: ${OVERFIT_ITERS} 帧，${OVERFIT_ITERS} 次迭代"

  # 复用固定的 500 帧抽样注释，保证训练和评测使用同一份数据
  OVERFIT_ANNOT_DIR="z_outputs/speaq_real_multihead_overfitting3/overfit_annotations_500"
  if [[ ! -d "${OVERFIT_ANNOT_DIR}" ]]; then
    echo "ERROR: overfit annotations directory not found: ${OVERFIT_ANNOT_DIR}"
    exit 1
  fi

  OVERFIT_OPTS=(
    DATALOADER.NUM_WORKERS 0
    SOLVER.IMS_PER_BATCH 1
    SOLVER.MAX_ITER "${OVERFIT_ITERS}"
    DATASETS.ACTION_GENOME.ANNOTATIONS "${OVERFIT_ANNOT_DIR}"
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0
    DATASETS.TRAIN "('AG_train',)"
    DATASETS.TEST "('AG_train',)"
    SOLVER.STEPS "(999999,)"
    TEST.EVAL_PERIOD 0
    SOLVER.CHECKPOINT_PERIOD 5000
  )
fi

# 构建训练参数
TRAIN_OPTS=(
  OUTPUT_DIR "${OUTPUT_DIR}"
  DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}"
  DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}"
  DATASETS.ACTION_GENOME.VIDEOS "${AG_VIDEOS}"
  MODEL.SAM3.CHECKPOINT_PATH "${SAM3_CHECKPOINT_PATH}"
)

# DETR head 预训练加载
if [[ -n "${DETR_HEAD_WEIGHTS}" && "${DETR_HEAD_WEIGHTS}" != "none" ]]; then
  if [[ "${DETR_LOAD_FULL_WEIGHTS}" == "1" ]]; then
    echo "Will load DETR full weights from: ${DETR_HEAD_WEIGHTS}"
    TRAIN_OPTS+=(
      MODEL.WEIGHTS "${DETR_HEAD_WEIGHTS}"
      MODEL.DETR.HEAD_WEIGHTS "${DETR_HEAD_WEIGHTS}"
      MODEL.DETR.LOAD_HEAD_ONLY "False"
      MODEL.DETR.LOAD_FULL_WEIGHTS "True"
      MODEL.DETR.LOAD_CLASS_HEAD "True"
    )
  else
    echo "Will load DETR head-only weights from: ${DETR_HEAD_WEIGHTS}"
    TRAIN_OPTS+=(
      MODEL.DETR.HEAD_WEIGHTS "${DETR_HEAD_WEIGHTS}"
      MODEL.DETR.LOAD_HEAD_ONLY "True"
      MODEL.DETR.LOAD_FULL_WEIGHTS "False"
      MODEL.DETR.LOAD_CLASS_HEAD "False"
    )
  fi
else
  echo "No DETR head pretrained weights specified."
fi

# 自动恢复（overfitting 模式下跳过）
LATEST_CKPT=""
if [[ -n "${OVERFIT_OPTS[@]}" ]]; then
  echo "Overfitting mode: skipping auto-resume."
else
  if [[ -d "${OUTPUT_DIR}" ]]; then
    shopt -s nullglob
    CKPT_CANDIDATES=("${OUTPUT_DIR}"/*.pth)
    shopt -u nullglob
    if (( ${#CKPT_CANDIDATES[@]} > 0 )); then
      IFS=$'\n' SORTED_CKPTS=($(ls -1t "${CKPT_CANDIDATES[@]}"))
      unset IFS
      LATEST_CKPT="${SORTED_CKPTS[0]}"
    fi
  fi
fi

RESUME_ARGS=()
if [[ -n "${LATEST_CKPT}" ]]; then
  echo "Auto-resume enabled. Latest checkpoint: ${LATEST_CKPT}"
  RESUME_ARGS=(--resume)
  TRAIN_OPTS+=(MODEL.WEIGHTS "${LATEST_CKPT}")
fi

# -----------------------------
# 内存监控
# -----------------------------
monitor_memory() {
    local pid=$1
    local threshold=93
    local check_interval=5
    while kill -0 "${pid}" 2>/dev/null; do
        local mem_info=$(free | grep Mem)
        local mem_total=$(echo "${mem_info}" | awk '{print $2}')
        local mem_used=$(echo "${mem_info}" | awk '{print $3}')
        local mem_usage=$((mem_used * 100 / mem_total))
        if [ "${mem_usage}" -ge "${threshold}" ]; then
            echo "[内存监控] 警告: 内存使用率 ${mem_usage}% 超过阈值 ${threshold}%"
            echo "[内存监控] 正在终止训练进程 ${pid}..."
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 2
            if kill -0 "${pid}" 2>/dev/null; then
                kill -KILL "${pid}" 2>/dev/null || true
            fi
            echo "[内存监控] 训练进程已终止"
            exit 1
        fi
        sleep "${check_interval}"
    done
}

# -----------------------------
# 训练
# -----------------------------
echo "Starting training..."
python train_iterative_model.py \
  "${RESUME_ARGS[@]}" \
  --num-gpus "${NUM_GPUS}" \
  --config-file "${CONFIG}" \
  --dist-url "tcp://127.0.0.1:${PORT}" \
  "${TRAIN_OPTS[@]}" \
  "${OVERFIT_OPTS[@]}" \
  ${OPTS:-} &
PID=$!
monitor_memory "${PID}" &
MON_PID=$!
wait "${PID}"
RET=$?
kill "${MON_PID}" 2>/dev/null || true
wait "${MON_PID}" 2>/dev/null || true

# Overfitting 模式：直接退出，跳过 after_train 触发的 final eval
if [[ -n "${OVERFIT_OPTS[@]}" ]]; then
  echo "过拟合模式完成，跳过 final eval。"
  exit "${RET}"
fi

if [[ "${RET}" -ne 0 ]]; then
  echo "训练异常退出 (exit code: ${RET})"
  exit "${RET}"
fi

echo "训练完成. Output: ${OUTPUT_DIR}"
