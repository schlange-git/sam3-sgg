#!/usr/bin/env bash
# ActionGenome Obj-Missed-Auxiliary Matching 训练入口（独立脚本）
# 默认开启：
#   - MODEL.DETR.OBJ_MISSED_AUX.ENABLED=True
#   - 默认辅助匹配参数见下方 DEFAULT_OPTS
#
# 用法:
#   ./run_actiongenome_obj_missed_aux.sh [OUTPUT_DIR] [NUM_GPUS] [NUM_VIDEOS_TRAIN]
#
# 你仍可通过环境变量覆盖默认值，例如：
#   AUX_IOU_THRESH=0.6 AUX_LOSS_WEIGHT=0.5 ./run_actiongenome_obj_missed_aux.sh
#   OPTS="SOLVER.MAX_ITER 120000" ./run_actiongenome_obj_missed_aux.sh
#
# 常用调优:
#   AUX_LOSS_WEIGHT=0.2        辅助 loss 权重（越大 => 背景 query 越趋向精确回归）
#   AUX_IOU_THRESH=0.5         匹配最小 IoU 阈值（太高可能没有匹配，太低引入噪声）
#   AUX_MIN_SCORE=0.3          候选 query 最小前景分数门槛（过滤低质量候选）
#   AUX_SMALL_ONLY=True        仅对小物体做辅助匹配（小物体易被漏检）
#   AUX_TAIL_ONLY=True         仅对长尾类别做辅助匹配
#   AUX_TAIL_CLASS_IDS="[0,1]" 长尾类别 ID 列表（需配合 TAIL_ONLY）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"/..

BASE_SCRIPT="${SCRIPT_DIR}/run_actiongenome_train_eval.sh"
if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Base script not found: ${BASE_SCRIPT}"
  exit 1
fi

OUTPUT_DIR="${1:-z_outputs/sam3_objmissedaux_160000iters_bs16}"
NUM_GPUS="${2:-2}"
NUM_VIDEOS_TRAIN="${3:--1}"

# ===== Obj-Missed-Aux 参数（可通过环境变量覆盖）=====
AUX_ENABLED="${AUX_ENABLED:-True}"
AUX_IOU_THRESH="${AUX_IOU_THRESH:-0.5}"
AUX_HIGH_COST="${AUX_HIGH_COST:-1000.0}"
AUX_CLS_COST="${AUX_CLS_COST:-1.0}"
AUX_L1_COST="${AUX_L1_COST:-5.0}"
AUX_GIOU_COST="${AUX_GIOU_COST:-2.0}"
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-0.2}"
AUX_MIN_SCORE="${AUX_MIN_SCORE:-0.0}"
AUX_REQUIRE_CLASS_MATCH="${AUX_REQUIRE_CLASS_MATCH:-False}"
AUX_SMALL_ONLY="${AUX_SMALL_ONLY:-False}"
AUX_SMALL_AREA_THRESH="${AUX_SMALL_AREA_THRESH:-0.02}"
AUX_TAIL_ONLY="${AUX_TAIL_ONLY:-False}"
AUX_TAIL_CLASS_IDS="${AUX_TAIL_CLASS_IDS:-[]}"
AUX_APPLY_SUBJECT="${AUX_APPLY_SUBJECT:-True}"
AUX_APPLY_OBJECT="${AUX_APPLY_OBJECT:-True}"
AUX_DEBUG="${AUX_DEBUG:-False}"

CUSTOM_CONFIG="configs/speaq_actiongenome_obj_missed_aux.yaml"

DEFAULT_OPTS=(
  MODEL.DETR.OBJ_MISSED_AUX.ENABLED "${AUX_ENABLED}"
  MODEL.DETR.OBJ_MISSED_AUX.IOU_THRESH "${AUX_IOU_THRESH}"
  MODEL.DETR.OBJ_MISSED_AUX.HIGH_COST "${AUX_HIGH_COST}"
  MODEL.DETR.OBJ_MISSED_AUX.CLS_COST "${AUX_CLS_COST}"
  MODEL.DETR.OBJ_MISSED_AUX.L1_COST "${AUX_L1_COST}"
  MODEL.DETR.OBJ_MISSED_AUX.GIOU_COST "${AUX_GIOU_COST}"
  MODEL.DETR.OBJ_MISSED_AUX.LOSS_WEIGHT "${AUX_LOSS_WEIGHT}"
  MODEL.DETR.OBJ_MISSED_AUX.MIN_SCORE "${AUX_MIN_SCORE}"
  MODEL.DETR.OBJ_MISSED_AUX.REQUIRE_CLASS_MATCH "${AUX_REQUIRE_CLASS_MATCH}"
  MODEL.DETR.OBJ_MISSED_AUX.SMALL_ONLY "${AUX_SMALL_ONLY}"
  MODEL.DETR.OBJ_MISSED_AUX.SMALL_AREA_THRESH "${AUX_SMALL_AREA_THRESH}"
  MODEL.DETR.OBJ_MISSED_AUX.TAIL_ONLY "${AUX_TAIL_ONLY}"
  MODEL.DETR.OBJ_MISSED_AUX.TAIL_CLASS_IDS "${AUX_TAIL_CLASS_IDS}"
  MODEL.DETR.OBJ_MISSED_AUX.APPLY_SUBJECT "${AUX_APPLY_SUBJECT}"
  MODEL.DETR.OBJ_MISSED_AUX.APPLY_OBJECT "${AUX_APPLY_OBJECT}"
  MODEL.DETR.OBJ_MISSED_AUX.DEBUG "${AUX_DEBUG}"
)

echo "=============================================="
echo "[Obj-Missed-Aux] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[Obj-Missed-Aux] NUM_GPUS=${NUM_GPUS}"
echo "[Obj-Missed-Aux] NUM_VIDEOS_TRAIN=${NUM_VIDEOS_TRAIN}"
echo "[Obj-Missed-Aux] Config: ${CUSTOM_CONFIG}"
echo "--- AUX 参数 ---"
echo "  ENABLED=${AUX_ENABLED}"
echo "  IOU_THRESH=${AUX_IOU_THRESH}"
echo "  LOSS_WEIGHT=${AUX_LOSS_WEIGHT}"
echo "  MIN_SCORE=${AUX_MIN_SCORE}"
echo "  SMALL_ONLY=${AUX_SMALL_ONLY}"
echo "  TAIL_ONLY=${AUX_TAIL_ONLY}"
echo "  TAIL_CLASS_IDS=${AUX_TAIL_CLASS_IDS}"
echo "  APPLY_SUBJECT=${AUX_APPLY_SUBJECT}"
echo "  APPLY_OBJECT=${AUX_APPLY_OBJECT}"
echo "  DEBUG=${AUX_DEBUG}"
echo "=============================================="

# ===== 注意：===
# 我们使用自定义 config 文件，其中已包含 OBJ_MISSED_AUX 的默认值。
# 此处再次通过 OPTS 传入 env override 参数，确保覆盖生效。
# 由于 run_actiongenome_train_eval.sh 内部使用 ${OPTS:-} 追加到命令行，
# 所以这里把 DEFAULT_OPTS（含 MODEL.DETR.OBJ_MISSED_AUX.*）加进去。
# 同时还要让 base 脚本知道使用自定义 config 而非默认的 minimal yaml。
#
# 通过 OPTS 合并后传给 base 脚本，
# 且让 base 脚本的 --config-file 指向我们的自定义 config。

if [[ -n "${OPTS:-}" ]]; then
  COMBINED_OPTS="${DEFAULT_OPTS[*]} ${OPTS}"
else
  COMBINED_OPTS="${DEFAULT_OPTS[*]}"
fi

# 额外添加 --config-file 重定向（run_actiongenome_train_eval.sh 内 CONFIG 是定死的 speaq_actiongenome_minimal.yaml，
# 所以这里通过 OPTS 无法改 CONFIG。办法：我们在此脚本中直接调用 train_iterative_model.py，而不是走 base 脚本。
#
# 下面采用直接调用的方式，复用 base 脚本中的逻辑，但使用自己的 config。

# ===== 以下直接调用 train_iterative_model.py，绕开 base 脚本的 CONFIG 限制 =====

# 数据路径（可按需修改）
AG_ANNOTATIONS="${AG_ANNOTATIONS:-dataset/annotations}"
AG_FRAMES="${AG_FRAMES:-dataset/frames}"
AG_VIDEOS="${AG_VIDEOS:-dataset/videos}"

SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-sam3.pt}"
BACKBONE_WEIGHTS="${BACKBONE_WEIGHTS:-detectron2://ImageNetPretrained/MSRA/R-101.pkl}"
DETR_HEAD_WEIGHTS="${DETR_HEAD_WEIGHTS:-/root/result/sam3_predtrain_detr_detection_from_vg_100Kx12bs/model_0099999.pth}"
DETR_LOAD_FULL_WEIGHTS="${DETR_LOAD_FULL_WEIGHTS:-1}"
EVAL_OVERFIT_TRAIN="${EVAL_OVERFIT_TRAIN:-0}"
RUN_VAL_EVAL="${RUN_VAL_EVAL:-1}"
VIS_MAX_IMAGES="${VIS_MAX_IMAGES:-1000}"
VIS_BOX_SCORE_THRESH="${VIS_BOX_SCORE_THRESH:-0.2}"
PORT="${PORT:-29500}"

echo "=== 参数汇总 ==="
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "NUM_VIDEOS_TRAIN=${NUM_VIDEOS_TRAIN}"
echo "SAM3_CHECKPOINT_PATH=${SAM3_CHECKPOINT_PATH}"
echo "DETR_HEAD_WEIGHTS=${DETR_HEAD_WEIGHTS}"
echo "DETR_LOAD_FULL_WEIGHTS=${DETR_LOAD_FULL_WEIGHTS}"
echo "=============================================="

# 解析 LOAD_FULL
LOAD_FULL_DETR_VAL="0"
if [[ "$(echo "${DETR_LOAD_FULL_WEIGHTS}" | tr '[:upper:]' '[:lower:]')" == "true" || "${DETR_LOAD_FULL_WEIGHTS}" == "1" ]]; then
  LOAD_FULL_DETR_VAL="1"
fi

if [[ -n "${DETR_HEAD_WEIGHTS}" && "${DETR_HEAD_WEIGHTS}" != "none" && "${DETR_HEAD_WEIGHTS}" != "" && "${LOAD_FULL_DETR_VAL}" == "1" ]]; then
  INITIAL_WEIGHTS="${DETR_HEAD_WEIGHTS}"
else
  INITIAL_WEIGHTS="${BACKBONE_WEIGHTS}"
fi

# 构建训练命令参数
TRAIN_OPTS=(
  OUTPUT_DIR "${OUTPUT_DIR}"
  DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}"
  DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}"
  DATASETS.ACTION_GENOME.VIDEOS "${AG_VIDEOS}"
  DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "${NUM_VIDEOS_TRAIN}"
  MODEL.WEIGHTS "${INITIAL_WEIGHTS}"
  MODEL.SAM3.CHECKPOINT_PATH "${SAM3_CHECKPOINT_PATH}"
)

if [[ "${RUN_VAL_EVAL}" == "0" ]]; then
  echo "RUN_VAL_EVAL=0: Disable in-training validation evaluation."
  TRAIN_OPTS+=(
    TEST.EVAL_PERIOD "0"
  )
fi

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
fi

# 将 Obj-Missed-Aux 参数加入 TRAIN_OPTS
TRAIN_OPTS+=( "${DEFAULT_OPTS[@]}" )

# 自动恢复
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
  TRAIN_OPTS+=(MODEL.WEIGHTS "${LATEST_CKPT}")
else
  echo "Auto-resume: no checkpoint found under ${OUTPUT_DIR}, start from scratch."
fi

# ===================== 内存监控 =====================
monitor_memory() {
    local pid=$1
    local threshold=93
    local check_interval=5

    echo "[内存监控] 开始监控进程 ${pid}，阈值: ${threshold}%"

    while kill -0 "${pid}" 2>/dev/null; do
        local mem_info=$(free | grep Mem)
        local mem_total=$(echo "${mem_info}" | awk '{print $2}')
        local mem_used=$(echo "${mem_info}" | awk '{print $3}')
        local mem_usage=$((mem_used * 100 / mem_total))

        if [ "${mem_usage}" -ge "${threshold}" ]; then
            echo "[内存监控] ⚠️  警告: 内存使用率 ${mem_usage}% 超过阈值 ${threshold}%"
            echo "[内存监控] 正在终止训练进程 ${pid}..."
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 2
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

# ===================== 训练阶段 =====================
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
      --config-file "${CUSTOM_CONFIG}" \
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

# ===================== 1) 训练 =====================
FINAL_OUTPUT_DIR="${OUTPUT_DIR}"
if run_train_phase "${RESUME_ARGS[@]}"; then
  :
else
  echo "训练进程异常退出"
  exit 1
fi

echo "Training finished. Output: ${FINAL_OUTPUT_DIR}"

# ===================== 2) 评测 =====================
CHECKPOINT="${FINAL_OUTPUT_DIR}/model_final.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}, skip eval."
else
  echo "Waiting 10s for memory release before evaluation ..."
  sleep 10

  # 2.1 正常验证集 AG_val
  if [[ "${RUN_VAL_EVAL}" == "1" ]]; then
    VAL_EVAL_DIR="${FINAL_OUTPUT_DIR}/eval_AG_val"
    mkdir -p "${VAL_EVAL_DIR}"
    echo "Running evaluation on AG_val (with memory monitor) ..."
    python train_iterative_model.py \
      --eval-only \
      --num-gpus "${NUM_GPUS}" \
      --config-file "${CUSTOM_CONFIG}" \
      --dist-url "tcp://127.0.0.1:${PORT}" \
      OUTPUT_DIR "${VAL_EVAL_DIR}" \
      DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}" \
      DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}" \
      DATASETS.ACTION_GENOME.VIDEOS "${AG_VIDEOS}" \
      MODEL.WEIGHTS "${CHECKPOINT}" \
      MODEL.DETR.OBJ_MISSED_AUX.ENABLED "False" \
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
  else
    echo "Skip AG_val evaluation (RUN_VAL_EVAL=${RUN_VAL_EVAL})"
  fi

  # 2.2 可选：overfit 训练集评测
  if [[ "${EVAL_OVERFIT_TRAIN}" == "1" && "${NUM_VIDEOS_TRAIN}" -gt 0 ]]; then
    TRAIN_EVAL_DIR="${FINAL_OUTPUT_DIR}/eval_AG_train_overfit"
    mkdir -p "${TRAIN_EVAL_DIR}"
    echo "Running evaluation on AG_train overfit split (with memory monitor) ..."
    python train_iterative_model.py \
      --eval-only \
      --num-gpus "${NUM_GPUS}" \
      --config-file "${CUSTOM_CONFIG}" \
      --dist-url "tcp://127.0.0.1:${PORT}" \
      OUTPUT_DIR "${TRAIN_EVAL_DIR}" \
      DATASETS.TEST "('AG_train',)" \
      DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}" \
      DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}" \
      DATASETS.ACTION_GENOME.VIDEOS "${AG_VIDEOS}" \
      DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "${NUM_VIDEOS_TRAIN}" \
      MODEL.WEIGHTS "${CHECKPOINT}" \
      MODEL.DETR.OBJ_MISSED_AUX.ENABLED "False" \
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

# ===================== 3) 可视化 =====================
VIS_SCRIPT="visualize_actiongenome_by_video.py"
if [[ -f "${CHECKPOINT}" && -f "${VIS_SCRIPT}" ]]; then
  VIS_DIR_VAL="${FINAL_OUTPUT_DIR}/vis_AG_val"
  echo "Saving per-video, per-frame visualizations on AG_val to ${VIS_DIR_VAL} ..."
  python "${VIS_SCRIPT}" \
    --config-file "${CUSTOM_CONFIG}" \
    --model-weights "${CHECKPOINT}" \
    --output-dir "${VIS_DIR_VAL}" \
    --dataset-name "AG_val" \
    --num-images "${VIS_MAX_IMAGES}" \
    --box-score-thresh "${VIS_BOX_SCORE_THRESH}" \
    DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}" \
    DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}" \
    DATASETS.ACTION_GENOME.VIDEOS "${AG_VIDEOS}" \
    MODEL.DETR.OBJ_MISSED_AUX.ENABLED "False" \
    ${OPTS:-}
  echo "Visualization on AG_val done: ${VIS_DIR_VAL}"

  if [[ "${EVAL_OVERFIT_TRAIN}" == "1" && "${NUM_VIDEOS_TRAIN}" -gt 0 ]]; then
    VIS_DIR_TRAIN="${FINAL_OUTPUT_DIR}/vis_AG_train_overfit"
    echo "Saving per-video, per-frame visualizations on AG_train overfit subset to ${VIS_DIR_TRAIN} ..."
    python "${VIS_SCRIPT}" \
      --config-file "${CUSTOM_CONFIG}" \
      --model-weights "${CHECKPOINT}" \
      --output-dir "${VIS_DIR_TRAIN}" \
      --dataset-name "AG_train" \
      --num-images "${VIS_MAX_IMAGES}" \
      --box-score-thresh "${VIS_BOX_SCORE_THRESH}" \
      DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}" \
      DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}" \
      DATASETS.ACTION_GENOME.VIDEOS "${AG_VIDEOS}" \
      DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN "${NUM_VIDEOS_TRAIN}" \
      MODEL.DETR.OBJ_MISSED_AUX.ENABLED "False" \
      ${OPTS:-}
    echo "Visualization on AG_train overfit subset done: ${VIS_DIR_TRAIN}"
  fi
else
  if [[ ! -f "${VIS_SCRIPT}" ]]; then
    echo "Script ${VIS_SCRIPT} not found, skip visualization."
  fi
fi

echo "All done. OUTPUT_DIR=${FINAL_OUTPUT_DIR}"
