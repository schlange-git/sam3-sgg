#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate speaq

PROJ="/home/tione/code/shizekun1_v/sam3-sgg-Auxiliary-Matching"
cd "${PROJ}"

OUT_DIR="${PROJ}/z_outputs/temporal_finetune_79999_bs16_20k_fixed_gate_obj030_rel045_update001_2gpu"
CONFIG="${PROJ}/z_outputs/train_fulltask_v3_8gpu_bs96_80k/config.yaml"
WEIGHTS="${PROJ}/z_outputs/train_fulltask_v3_8gpu_bs96_80k/model_0079999.pth"
PORT="${PORT:-29665}"

assert_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing file: $1" >&2
    exit 1
  fi
}

assert_file "${CONFIG}"
assert_file "${WEIGHTS}"
assert_file "${OUT_DIR}/last_checkpoint"
grep -q "quality = obj_score \* pred_score" "${PROJ}/modeling/transformer/detr.py"
grep -q "TEMPORAL_LR_MULTIPLIER" "${PROJ}/engine/trainer.py"

PYTHONUNBUFFERED=1 python3 train_iterative_model.py \
  --resume \
  --num-gpus 1 \
  --config-file "${CONFIG}" \
  --dist-url "tcp://127.0.0.1:${PORT}" \
  OUTPUT_DIR "${OUT_DIR}" \
  DATASETS.ACTION_GENOME.ANNOTATIONS "${PROJ}/dataset/annotations" \
  DATASETS.ACTION_GENOME.FRAMES "${PROJ}/dataset/frames" \
  MODEL.WEIGHTS "${WEIGHTS}" \
  MODEL.DETR.HEAD_WEIGHTS "${WEIGHTS}" \
  MODEL.DETR.LOAD_HEAD_ONLY False \
  MODEL.DETR.LOAD_FULL_WEIGHTS True \
  MODEL.DETR.LOAD_CLASS_HEAD True \
  MODEL.TEMPORAL.ENABLED True \
  MODEL.TEMPORAL.EVAL_ENABLED True \
  MODEL.TEMPORAL.LEARNABLE_GATE False \
  MODEL.TEMPORAL.GATE_MAX_OBJECT 0.30 \
  MODEL.TEMPORAL.GATE_MAX_RELATION 0.45 \
  MODEL.TEMPORAL.GATE_ZERO_END_RATIO 0.0 \
  MODEL.TEMPORAL.GATE_WARMUP_END_RATIO 0.0 \
  MODEL.TEMPORAL.UPDATE_SCORE_THRESH 0.01 \
  MODEL.TEMPORAL.PRED_UPDATE_THRESH_START 0.01 \
  MODEL.TEMPORAL.PRED_UPDATE_THRESH_END 0.01 \
  SOLVER.EVAL_FIRST False \
  SOLVER.IMS_PER_BATCH 16 \
  SOLVER.BASE_LR 0.00004 \
  SOLVER.MAX_ITER 20000 \
  SOLVER.WARMUP_ITERS 500 \
  SOLVER.STEPS "(3000, 17500)" \
  SOLVER.CHECKPOINT_PERIOD 4000 \
  SOLVER.TEMPORAL_LR_MULTIPLIER 100.0 \
  TEST.EVAL_PERIOD 4000 \
  2>&1 | tee -a "${OUT_DIR}/train_resume_1gpu_bs16.log"
