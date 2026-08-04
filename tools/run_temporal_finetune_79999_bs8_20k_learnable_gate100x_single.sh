#!/usr/bin/env bash
set -euo pipefail

PROJ="/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching"
cd "${PROJ}"

OUT_DIR="${PROJ}/z_outputs/temporal_finetune_79999_bs8_20k_learnable_gate100x_update001_single"
CONFIG="${PROJ}/z_outputs/train_fulltask_v3_8gpu_bs96_80k/config.yaml"
WEIGHTS="${PROJ}/z_outputs/train_fulltask_v3_8gpu_bs96_80k/model_0079999.pth"
PORT="${PORT:-29693}"
export TRIPLET_TRAIN_GATE_CSV="${OUT_DIR}/triplet_train_gate.csv"

assert_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing file: $1" >&2
    exit 1
  fi
}

assert_file "${CONFIG}"
assert_file "${WEIGHTS}"
assert_file "${PROJ}/configs/defaults.py"
assert_file "${PROJ}/modeling/transformer/detr.py"
assert_file "${PROJ}/engine/trainer.py"

grep -q "LEARNABLE_GATE" "${PROJ}/configs/defaults.py"
grep -q "TRIPLET_TRAIN_GATE_CSV" "${PROJ}/modeling/transformer/detr.py"
grep -q "triplet_obj_inject_logit" "${PROJ}/modeling/transformer/detr.py"
grep -q "TEMPORAL_LR_MULTIPLIER" "${PROJ}/engine/trainer.py"
grep -q "quality = obj_score \* pred_score" "${PROJ}/modeling/transformer/detr.py"

mkdir -p "${OUT_DIR}"
rm -f "${TRIPLET_TRAIN_GATE_CSV}"

PYTHONUNBUFFERED=1 python3 train_iterative_model.py \
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
  MODEL.TEMPORAL.LEARNABLE_GATE True \
  MODEL.TEMPORAL.LEARNABLE_GATE_INIT_OBJECT 0.15 \
  MODEL.TEMPORAL.LEARNABLE_GATE_INIT_RELATION 0.30 \
  MODEL.TEMPORAL.UPDATE_SCORE_THRESH 0.01 \
  MODEL.TEMPORAL.PRED_UPDATE_THRESH_START 0.01 \
  MODEL.TEMPORAL.PRED_UPDATE_THRESH_END 0.01 \
  SOLVER.EVAL_FIRST False \
  SOLVER.IMS_PER_BATCH 8 \
  SOLVER.BASE_LR 0.00004 \
  SOLVER.MAX_ITER 20000 \
  SOLVER.WARMUP_ITERS 500 \
  SOLVER.STEPS "(3000, 17500)" \
  SOLVER.CHECKPOINT_PERIOD 4000 \
  SOLVER.TEMPORAL_LR_MULTIPLIER 100.0 \
  SOLVER.GATE_LR_MULTIPLIER 500.0 \
  TEST.EVAL_PERIOD 4000 \
  2>&1 | tee "${OUT_DIR}/train.log"
