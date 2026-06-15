#!/usr/bin/env bash
# =============================================================================
# 控制实验：参考分支(Auxiliary-Matching)原始代码 + overfit_triplet_v3_16000_pscale
# 的完全相同配置，仅换成我们的 20v/2011帧数据集，看 400/800 是否正常收敛上升。
#   目的：若上升 => 我们 20v 数据 OK、参考代码 OK => bug 在 stage5 代码侧
#         (数据传输/训练loop/评测)；若下降 => 20v 数据本身有问题。
#   与 overfit_triplet_v3_16000_pscale.sh 的唯一差异：
#     ANNOTATIONS dataset_overfit_temporal -> dataset_overfit_temporal_20v (软链到 stage5)
#     EVAL_PERIOD 2000 -> 400  (为看 400/800 早期信号)
#     dist-url 29801 -> 29803, OUTPUT_DIR 新名(不覆盖旧结果)
#   其余(num-gpus 1, bs8, scale300, 时序全开, MAX_ITER 16000, STEPS, workers 4)全一致。
# =============================================================================
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate speaq
PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/overfit_triplet_v3_16000_pscale_20v_eval400}"
mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"
echo "[Triplet v3 16K pscale 20v eval400] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py \
    --num-gpus 1 \
    --config-file configs/speaq_actiongenome_minimal.yaml \
    --dist-url tcp://127.0.0.1:29803 \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.TRAIN "('AG_train',)" \
    DATASETS.TEST "('AG_train',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal_20v/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0 \
    DATASETS.AG_TEMPORAL.ENABLED True \
    DATASETS.AG_TEMPORAL.CLIP_MODE between_keyframes \
    DATASETS.AG_TEMPORAL.NUM_INTERMEDIATE_FRAMES 0 \
    MODEL.WEIGHTS model_0099999.pth \
    MODEL.DETR.HEAD_WEIGHTS model_0099999.pth \
    MODEL.DETR.LOAD_HEAD_ONLY False \
    MODEL.DETR.LOAD_FULL_WEIGHTS True \
    MODEL.DETR.LOAD_CLASS_HEAD True \
    MODEL.DETR.PERSON_SCORE_SCALE 300.0 \
    MODEL.SAM3.ENABLED True \
    MODEL.SAM3.CHECKPOINT_PATH sam3/weights/sam3.pt \
    MODEL.SAM3.FREEZE True \
    MODEL.SAM3.USE_PATCH_MERGE False \
    MODEL.TEMPORAL.ENABLED True \
    MODEL.TEMPORAL.EVAL_ENABLED True \
    MODEL.TEMPORAL.MODE triplet_memory_v3 \
    MODEL.TEMPORAL.TRIPLET_MEMORY_ENABLED True \
    MODEL.TEMPORAL.INJECT_OBJECT True \
    MODEL.TEMPORAL.INJECT_RELATION True \
    MODEL.TEMPORAL.NON_KEY_SKIP_LOSS True \
    MODEL.TEMPORAL.NON_KEY_SKIP_EVAL True \
    MODEL.TEMPORAL.NON_KEY_RUN_OBJECT_ONLY True \
    MODEL.ROI_REFINE.ENABLED False \
    MODEL.ROI_REFINE.LOSS_ENABLED False \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.STEPS "(4000,12000)" \
    SOLVER.WARMUP_ITERS 500 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    TEST.EVAL_PERIOD 400 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 4

RET=$?
echo "[Triplet v3 16K pscale 20v eval400] Done exit=${RET}"
