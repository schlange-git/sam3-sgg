#!/usr/bin/env bash
# =============================================================================
# 原库(Auxiliary-Matching) · clip-as-sample 时序采样 · 双卡 bs32 · 20v 过拟合
#   基底 = 已验证可正常过拟合的 overfit_triplet_v3_16000_pscale_20v(clip 模式, 单卡 bs8)。
#   唯一【范式变量】= SAMPLER_MODE clip -> clip_sample (CLIP_SAMPLE_LEN=4)：
#     双卡 per_worker=16 / clip4 -> 4 slot/卡 -> 全局 8 视频 x 4 连续关键帧/batch；
#     memory 桶在紧凑窗口累积, 贴近健康 clip、远离 lane 的逐帧时效漂移。
#   bs8->bs32 温和缩放(非范式变量, 避免发散混淆): LR 1e-4->2e-4, ITER 16000->6000,
#     STEPS (4000,12000)->(2000,4500), WARMUP 500->250。
#   go/no-go: det_Recall@50 能否从 iter0(~59) 恢复并爬向 70+ —— 是则范式有效、可上全量。
#   若每卡 16 图 OOM, 降 IMS_PER_BATCH 16(每卡 8, 2 slot)。
# =============================================================================
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate speaq

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"
OUTPUT_DIR="${1:-z_outputs/overfit_clipsample_2gpu_bs32_20v_pscale300}"
mkdir -p "${OUTPUT_DIR}"
export ROI_GATE_LOG_PATH="${OUTPUT_DIR}/roi_gate_log.csv"
echo "[clipsample 2gpu bs32 pscale300] OUTPUT=${OUTPUT_DIR}"

python3 train_iterative_model.py \
    --num-gpus 2 \
    --config-file configs/speaq_actiongenome_minimal.yaml \
    --dist-url tcp://127.0.0.1:29835 \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    DATASETS.TRAIN "('AG_train',)" \
    DATASETS.TEST "('AG_train',)" \
    DATASETS.ACTION_GENOME.ANNOTATIONS dataset_overfit_temporal_20v/annotations \
    DATASETS.ACTION_GENOME.FRAMES dataset/frames \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_TRAIN -1 \
    DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL 0 \
    DATASETS.ACTION_GENOME.SAMPLER_MODE clip_sample \
    DATASETS.ACTION_GENOME.CLIP_SAMPLE_LEN 4 \
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
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.BASE_LR 0.0002 \
    SOLVER.MAX_ITER 6000 \
    SOLVER.STEPS '(2000,4500)' \
    SOLVER.WARMUP_ITERS 250 \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    TEST.EVAL_PERIOD 500 \
    SOLVER.EVAL_FIRST True \
    DATALOADER.NUM_WORKERS 4

RET=$?
echo "[clipsample 2gpu bs32 pscale300] Done exit=${RET}"
