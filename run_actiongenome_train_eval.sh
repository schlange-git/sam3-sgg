#!/usr/bin/env bash
# ActionGenome 训练 + 评测 + 可视化
# 用法:
#   ./run_actiongenome_train_eval.sh [OUTPUT_DIR] [NUM_GPUS] [NUM_VIDEOS_TRAIN]
#   NUM_VIDEOS_TRAIN: -1 = 使用全部训练视频; N = 仅用前 N 个视频（过拟合实验）
#
# 预训练权重说明:
#   1. BACKBONE_WEIGHTS (默认: ImageNet R-101)
#      - 由 MODEL.WEIGHTS 传入，通过 DetectionCheckpointer.resume_or_load() 加载
#      - 只包含 ResNet-101 backbone（ImageNet 分类预训练）
#   2. DETR_HEAD_WEIGHTS (默认: vg_objectdetector_pretrained.pth)
#      - 若设置，会通过 MODEL.DETR.HEAD_WEIGHTS + MODEL.DETR.LOAD_HEAD_ONLY=True 加载
#      - 只加载 DETR 部分（transformer + 检测头 + 关系头），不覆盖 backbone
#      - 若类别数不匹配，最后一层 class_embed 会被跳过，其余层仍会加载
#   3. 加载流程:
#      - setup() 检测到 LOAD_HEAD_ONLY + HEAD_WEIGHTS，会清空 MODEL.WEIGHTS（避免整图加载）
#      - detr.py 的 __init__ 末尾调用 _load_detr_head_only()，只加载名字前缀为 "detr." 且 shape 匹配的参数
#   4. 禁用 DETR head 预训练:
#      设置环境变量: DETR_HEAD_WEIGHTS="" 或 DETR_HEAD_WEIGHTS="none"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Fix invalid OMP_NUM_THREADS value (if set to 0 or invalid)
if [[ -n "${OMP_NUM_THREADS:-}" ]] && [[ "${OMP_NUM_THREADS}" == "0" || ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ ]]; then
  unset OMP_NUM_THREADS
fi

OUTPUT_DIR="${1:-z_outputs/sam3_temporal_debug}"
NUM_GPUS="${2:-2}"
NUM_VIDEOS_TRAIN="${3:-10}"

CONFIG="configs/speaq_actiongenome_minimal.yaml"
PORT="${PORT:-29500}"

# SAM3 多卡提示：不再强制改单卡，由用户自行决定 NUM_GPUS。
# 若担心加载峰值内存，可设置环境变量 SAM3_LOAD_STAGGER_SEC（见 sam3_backbone.py）。
if [[ -f "${CONFIG}" ]]; then
  if grep -q "SAM3:" "${CONFIG}" 2>/dev/null && grep -A5 "SAM3:" "${CONFIG}" 2>/dev/null | grep -qi "ENABLED: *True"; then
    if [[ "${NUM_GPUS}" -gt 1 ]]; then
      echo "[SAM3] 检测到 SAM3.ENABLED=True，当前保留 NUM_GPUS=${NUM_GPUS}（不再强制降到1）。"
    fi
  fi
fi

# 数据路径（可按需修改）
AG_ANNOTATIONS="${AG_ANNOTATIONS:-dataset/annotations}"
AG_FRAMES="${AG_FRAMES:-dataset/frames}"
AG_VIDEOS="${AG_VIDEOS:-dataset/videos}"

# 预训练权重配置
# SAM3权重路径（仅在MODEL.SAM3.ENABLED=True时使用）
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-sam3.pt}"
# BACKBONE_WEIGHTS: ImageNet R-101（仅在MODEL.SAM3.ENABLED=False时使用）
# 如果使用SAM3作为backbone，此值会被忽略（train_iterative_model.py会自动清空MODEL.WEIGHTS）
BACKBONE_WEIGHTS="${BACKBONE_WEIGHTS:-detectron2://ImageNetPretrained/MSRA/R-101.pkl}"
# DETR_HEAD_WEIGHTS: VG 上训好的 DETR 权重（包含 transformer + 检测头 + 关系头）
# 设置为空字符串则只使用 backbone，不加载 DETR head 预训练
DETR_HEAD_WEIGHTS="${DETR_HEAD_WEIGHTS:-/root/result/sam3_predtrain_detr_detection_from_vg_100Kx12bs/model_0099999.pth}"
# DETR_HEAD_WEIGHTS="${DETR_HEAD_WEIGHTS:-vg_objectdetector_pretrained.pth}"
# 是否将 DETR_HEAD_WEIGHTS 作为 MODEL.WEIGHTS 全量加载（true/1=全量；false/0=仅 HEAD_WEIGHTS+LOAD_HEAD_ONLY）
# 兼容旧变量 LOAD_FULL_DETR_WEIGHTS；默认 1（完整加载）
DETR_LOAD_FULL_WEIGHTS="${DETR_LOAD_FULL_WEIGHTS:-${LOAD_FULL_DETR_WEIGHTS:-1}}"
# 是否额外在 overfit 训练集上评测（1/0）
EVAL_OVERFIT_TRAIN="${EVAL_OVERFIT_TRAIN:-1}"
# 可视化最大图片数（-1 = 全部）
VIS_MAX_IMAGES="${VIS_MAX_IMAGES:-100}"

echo "=============================================="
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "NUM_VIDEOS_TRAIN=${NUM_VIDEOS_TRAIN} (use -1 for all videos)"
echo "SAM3_CHECKPOINT_PATH=${SAM3_CHECKPOINT_PATH}"
echo "BACKBONE_WEIGHTS=${BACKBONE_WEIGHTS} (ignored if SAM3.ENABLED=True)"
echo "DETR_HEAD_WEIGHTS=${DETR_HEAD_WEIGHTS}"
echo "DETR_LOAD_FULL_WEIGHTS=${DETR_LOAD_FULL_WEIGHTS}"
echo "EVAL_OVERFIT_TRAIN=${EVAL_OVERFIT_TRAIN}"
echo "VIS_MAX_IMAGES=${VIS_MAX_IMAGES}"
echo "=============================================="

# 解析 DETR_LOAD_FULL_WEIGHTS（兼容 true/false/True/False 及 1/0）
LOAD_FULL_DETR_VAL="0"
if [[ "$(echo "${DETR_LOAD_FULL_WEIGHTS}" | tr '[:upper:]' '[:lower:]')" == "true" || "${DETR_LOAD_FULL_WEIGHTS}" == "1" ]]; then
  LOAD_FULL_DETR_VAL="1"
fi

# 确定 MODEL.WEIGHTS 初值：全量加载时用 DETR_HEAD_WEIGHTS，否则用 BACKBONE_WEIGHTS
if [[ -n "${DETR_HEAD_WEIGHTS}" && "${DETR_HEAD_WEIGHTS}" != "none" && "${DETR_HEAD_WEIGHTS}" != "" && "${LOAD_FULL_DETR_VAL}" == "1" ]]; then
  INITIAL_WEIGHTS="${DETR_HEAD_WEIGHTS}"
else
  INITIAL_WEIGHTS="${BACKBONE_WEIGHTS}"
fi

# 构建训练命令的参数
TRAIN_OPTS=(
  OUTPUT_DIR "${OUTPUT_DIR}"
  DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}"
  DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}"
  DATASETS.ACTION_GENOME.VIDEOS "${AG_VIDEOS}"
  DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "${NUM_VIDEOS_TRAIN}"
  MODEL.WEIGHTS "${INITIAL_WEIGHTS}"
  MODEL.SAM3.CHECKPOINT_PATH "${SAM3_CHECKPOINT_PATH}"
)

# 如果设置了 DETR_HEAD_WEIGHTS，则启用 DETR head 预训练加载
if [[ -n "${DETR_HEAD_WEIGHTS}" && "${DETR_HEAD_WEIGHTS}" != "none" && "${DETR_HEAD_WEIGHTS}" != "" ]]; then
  if [[ "${LOAD_FULL_DETR_VAL}" == "1" ]]; then
    echo "Will load DETR full weights from: ${DETR_HEAD_WEIGHTS}"
    TRAIN_OPTS+=(
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
  echo "DETR_HEAD_WEIGHTS not set, will only use backbone pretrained weights."
fi

# 自动恢复：检测 OUTPUT_DIR 下最新 .pth 并 resume
RESUME_ARGS=()
LATEST_CKPT=""
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

if [[ -n "${LATEST_CKPT}" ]]; then
  echo "Auto-resume enabled. Latest checkpoint: ${LATEST_CKPT}"
  RESUME_ARGS=(--resume)
  # 当 OUTPUT_DIR 下不存在 last_checkpoint 时，--resume 会回退到 MODEL.WEIGHTS。
  TRAIN_OPTS+=(MODEL.WEIGHTS "${LATEST_CKPT}")
else
  echo "Auto-resume: no checkpoint found under ${OUTPUT_DIR}, start from scratch."
fi

# -----------------------------
# 内存监控函数：当内存占用超过93%时自动kill训练进程
# -----------------------------
monitor_memory() {
    local pid=$1
    local threshold=93  # 内存使用率阈值（%）
    local check_interval=5  # 检查间隔（秒）
    
    echo "[内存监控] 开始监控进程 ${pid}，阈值: ${threshold}%"
    
    while kill -0 "${pid}" 2>/dev/null; do
        # 获取系统内存使用率（使用awk计算，不依赖bc）
        local mem_info=$(free | grep Mem)
        local mem_total=$(echo "${mem_info}" | awk '{print $2}')
        local mem_used=$(echo "${mem_info}" | awk '{print $3}')
        # 计算使用率（整数部分）
        local mem_usage=$((mem_used * 100 / mem_total))
        
        # 如果内存使用率超过阈值
        if [ "${mem_usage}" -ge "${threshold}" ]; then
            echo "[内存监控] ⚠️  警告: 内存使用率 ${mem_usage}% 超过阈值 ${threshold}%"
            echo "[内存监控] 正在终止训练进程 ${pid}..."
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 2
            # 如果进程还在运行，强制kill
            if kill -0 "${pid}" 2>/dev/null; then
                echo "[内存监控] 强制终止进程 ${pid}..."
                kill -KILL "${pid}" 2>/dev/null || true
            fi
            echo "[内存监控] 训练进程已终止"
            exit 1
        fi
        
        sleep "${check_interval}"
    done
    
    echo "[内存监控] 训练进程已结束，停止监控"
}

# -----------------------------
# 训练阶段封装：复用内存监控
# run_train_phase <resume_args...> <extra_opts...>
# 例: run_train_phase --resume MODEL.WEIGHTS /path/to/model.pth SOLVER.MAX_ITER 10000
# -----------------------------
run_train_phase() {
    local phase_resume=()
    local phase_opts=("${TRAIN_OPTS[@]}")
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--resume" ]]; then
            phase_resume=(--resume)
            shift
        elif [[ $# -ge 2 ]]; then
            phase_opts+=("$1" "$2")
            shift 2
        else
            shift
        fi
    done

    python train_iterative_model.py \
      "${phase_resume[@]}" \
      --num-gpus "${NUM_GPUS}" \
      --config-file "${CONFIG}" \
      --dist-url "tcp://127.0.0.1:${PORT}" \
      "${phase_opts[@]}" \
      ${OPTS:-} &
    local pid=$!
    monitor_memory "${pid}" &
    local mon_pid=$!
    wait "${pid}"
    local ret=$?
    kill "${mon_pid}" 2>/dev/null || true
    wait "${mon_pid}" 2>/dev/null || true
    return "${ret}"
}

# -----------------------------
# 1) 训练（单阶段，带内存监控）
# -----------------------------
FINAL_OUTPUT_DIR="${OUTPUT_DIR}"
if run_train_phase "${RESUME_ARGS[@]}"; then
  :
else
  echo "训练进程异常退出"
  exit 1
fi

echo "Training finished. Output: ${FINAL_OUTPUT_DIR}"

# -----------------------------
# 2) 评测 (eval-only，使用训练好的 checkpoint)
# -----------------------------
CHECKPOINT="${FINAL_OUTPUT_DIR}/model_final.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}, skip eval."
else
  # 等待系统释放训练阶段占用的内存/显存
  echo "Waiting 10s for memory release before evaluation ..."
  sleep 10

  VAL_EVAL_DIR="${FINAL_OUTPUT_DIR}/eval_AG_val"
  mkdir -p "${VAL_EVAL_DIR}"
  echo "Running evaluation on AG_val (with memory monitor) ..."
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
    echo "Evaluation failed with exit code: ${EVAL_EXIT_CODE}"
    exit "${EVAL_EXIT_CODE}"
  fi
  echo "Evaluation on AG_val done. Output: ${VAL_EVAL_DIR}"

  # 可选：在 overfit 训练集上再评测一次，验证是否真的学到训练样本
  if [[ "${EVAL_OVERFIT_TRAIN}" == "1" && "${NUM_VIDEOS_TRAIN}" -gt 0 ]]; then
    TRAIN_EVAL_DIR="${FINAL_OUTPUT_DIR}/eval_AG_train_overfit"
    mkdir -p "${TRAIN_EVAL_DIR}"
    echo "Running evaluation on AG_train overfit split (with memory monitor) ..."
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
      echo "Overfit-train evaluation failed with exit code: ${EVAL_TRAIN_EXIT_CODE}"
      exit "${EVAL_TRAIN_EXIT_CODE}"
    fi
    echo "Evaluation on AG_train overfit split done. Output: ${TRAIN_EVAL_DIR}"
  else
    echo "Skip AG_train overfit evaluation (EVAL_OVERFIT_TRAIN=${EVAL_OVERFIT_TRAIN}, NUM_VIDEOS_TRAIN=${NUM_VIDEOS_TRAIN})"
  fi
fi

# -----------------------------
# 3) 可视化：在 OUTPUT_DIR/vis 下按 视频/帧 保存推理结果
# -----------------------------
VIS_SCRIPT="visualize_actiongenome_by_video.py"
if [[ -f "${CHECKPOINT}" && -f "${VIS_SCRIPT}" ]]; then
  VIS_DIR="${FINAL_OUTPUT_DIR}/vis"
  echo "Saving per-video, per-frame visualizations to ${VIS_DIR} ..."
  python "${VIS_SCRIPT}" \
    --config-file "${CONFIG}" \
    --model-weights "${CHECKPOINT}" \
    --output-dir "${VIS_DIR}" \
    --dataset-name "AG_val" \
    --num-images "${VIS_MAX_IMAGES}" \
    DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}" \
    DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}" \
    DATASETS.ACTION_GENOME.VIDEOS "${AG_VIDEOS}" \
    ${OPTS:-}
  echo "Visualization done: ${VIS_DIR}"
else
  if [[ ! -f "${VIS_SCRIPT}" ]]; then
    echo "Script ${VIS_SCRIPT} not found, skip visualization."
  fi
fi

echo "All done. OUTPUT_DIR=${FINAL_OUTPUT_DIR}"
