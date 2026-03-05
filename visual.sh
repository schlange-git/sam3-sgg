#!/usr/bin/env bash
# ActionGenome 可视化独立脚本（可选可视化后跑评测）
# 用法:
#   ./visual.sh [/path/to/checkpoint.pth] [NUM_IMAGES] [DATASET_NAME] [RUN_EVAL] [NUM_GPUS]
#   - NUM_IMAGES: 可视化的最大图片数（默认: 1000，-1 表示全部）
#   - DATASET_NAME: Detectron2 数据集名称（默认: AG_val）
#   - RUN_EVAL: 1/true/yes 时在可视化后对 AG_val 做 eval-only（默认: 0）
#   - NUM_GPUS: 评测时使用的 GPU 数（默认: 1，可设 2 双卡）
#
# 说明:
#   - 可视化结果: <CKPT_DIR>/<CKPT_NAME>_vis
#   - 若 RUN_EVAL=1，评测结果: <CKPT_DIR>/eval_AG_val
#   - 也可用环境变量: RUN_EVAL=1 NUM_GPUS=2 ./visual.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 参数: checkpoint（可选）；NUM_IMAGES；DATASET_NAME；RUN_EVAL；NUM_GPUS（评测用）
DEFAULT_CKPT="z_outputs/sam3_80000iters_from_pretrained_full_train/model_0079999.pth"
CHECKPOINT="${1:-${CHECKPOINT:-${DEFAULT_CKPT}}}"
NUM_IMAGES="${2:-${NUM_IMAGES:-1000}}"
DATASET_NAME="${3:-${DATASET_NAME:-AG_val}}"
RUN_EVAL="${4:-${RUN_EVAL:-true}}"
NUM_GPUS="${5:-${NUM_GPUS:-2}}"

# 归一化 RUN_EVAL
RUN_EVAL_VAL="0"
case "$(echo "${RUN_EVAL}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y) RUN_EVAL_VAL="1" ;;
esac

CONFIG="configs/speaq_actiongenome_minimal.yaml"
VIS_SCRIPT="visualize_actiongenome_by_video.py"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint 不存在: ${CHECKPOINT}"
  echo "用法: $0 [/path/to/checkpoint.pth] [NUM_IMAGES] [DATASET_NAME] [RUN_EVAL] [NUM_GPUS]"
  exit 1
fi

if [[ ! -f "${VIS_SCRIPT}" ]]; then
  echo "可视化脚本不存在: ${VIS_SCRIPT}，请确认该文件在项目根目录下。"
  exit 1
fi

CKPT_DIR="$(cd "$(dirname "${CHECKPOINT}")" && pwd)"
CKPT_BASENAME="$(basename "${CHECKPOINT}")"
CKPT_NAME="${CKPT_BASENAME%.*}"

VIS_DIR="${CKPT_DIR}/${CKPT_NAME}_vis"
mkdir -p "${VIS_DIR}"

echo "=============================================="
echo "Checkpoint      : ${CHECKPOINT}"
echo "Config          : ${CONFIG}"
echo "Dataset         : ${DATASET_NAME}"
echo "Num images      : ${NUM_IMAGES}"
echo "Run eval after vis: ${RUN_EVAL_VAL}"
echo "Num GPUs (eval) : ${NUM_GPUS}"
echo "Output (visual) : ${VIS_DIR}"
echo "=============================================="

# 数据路径（与训练脚本保持一致，必要时可通过环境变量覆盖）
AG_ANNOTATIONS="${AG_ANNOTATIONS:-dataset/annotations}"
AG_FRAMES="${AG_FRAMES:-dataset/frames}"
AG_VIDEOS="${AG_VIDEOS:-dataset/videos}"
PORT="${PORT:-29500}"
REL_SCORE_THRESH="${REL_SCORE_THRESH:-0.05}"
BOX_SCORE_THRESH="${BOX_SCORE_THRESH:-0.1}"
CLASSWISE_MINIOU_THRESH="${CLASSWISE_MINIOU_THRESH:-0.6}"
FORCE_KEEP_PERSON=0
PERSON_SCORE_BOOST="${PERSON_SCORE_BOOST:-true}"   # 1/true/yes/y -> person 分数放大
PERSON_SCORE_SCALE="1.0"
case "$(echo "${PERSON_SCORE_BOOST}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y) PERSON_SCORE_SCALE="500.0" ;;
esac

python "${VIS_SCRIPT}" \
  --config-file "${CONFIG}" \
  --model-weights "${CHECKPOINT}" \
  --output-dir "${VIS_DIR}" \
  --dataset-name "${DATASET_NAME}" \
  --num-images "${NUM_IMAGES}" \
  --rel-score-thresh "${REL_SCORE_THRESH}" \
  --box-score-thresh "${BOX_SCORE_THRESH}" \
  --classwise-miniou-thresh "${CLASSWISE_MINIOU_THRESH}" \
  --force-keep-person "${FORCE_KEEP_PERSON}" \
  --person-score-scale "${PERSON_SCORE_SCALE}" \
  DATASETS.ACTION_GENOME.ANNOTATIONS "${AG_ANNOTATIONS}" \
  DATASETS.ACTION_GENOME.FRAMES "${AG_FRAMES}" \
  DATASETS.ACTION_GENOME.VIDEOS "${AG_VIDEOS}" \
  ${OPTS:-}

echo "可视化完成，结果保存在: ${VIS_DIR}"

# 可选：可视化后再跑评测（eval-only），结果写到 <CKPT_DIR>/eval_AG_val
if [[ "${RUN_EVAL_VAL}" == "1" ]]; then
  VAL_EVAL_DIR="${CKPT_DIR}/eval_AG_val"
  mkdir -p "${VAL_EVAL_DIR}"
  echo "[评测] OUTPUT_DIR=${VAL_EVAL_DIR} ..."
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
    ${OPTS:-}
  echo "[评测] 完成: ${VAL_EVAL_DIR}"
fi
