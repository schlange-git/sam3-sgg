#!/bin/bash
#
# SpeaQ 全链路训练与评测脚本
# 功能：预训练评测 → Finetune → 评测 → 结果对比
#
# 用法: ./run_full_pipeline.sh [--iters ITERS] [--gpus NUM_GPUS] [--overfit NUM_IMAGES]
#

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数 (可通过命令行参数覆盖)
# ============================================================================

# 默认参数
NUM_GPUS=1
FINETUNE_ITERS=200
OVERFIT_NUM_IMAGES=2
OVERFIT_SEED=42
BATCH_SIZE=1
NUM_WORKERS=2
CHECKPOINT_PERIOD=50
# SAM3 backbone settings
SAM3_ENABLED=True
SAM3_CHECKPOINT_PATH="sam3/weights/sam3.pt"
SAM3_IMAGE_SIZE=1008
SAM3_FEATURE_DIM=256
SAM3_CHANNEL_REPEAT=1
SAM3_FREEZE=True
SAM3_EVAL_ENABLED=${SAM3_ENABLED}
SAM3_DEVICE="cuda"
SAM3_TARGET_STRIDE=32
SAM3_PRECOMPUTE=True
SAM3_USE_PRECOMPUTED=True
SAM3_ENABLED_PRETRAINED=False          # 预训练评测固定使用 ResNet，不走 SAM3
SAM3_USE_PRECOMPUTED_PRETRAINED=False  # 预训练评测不使用预计算特征
SAM3_FEATUREMAP_DIR="data/featuremaps"
# Visualization
VIS_NUM_IMAGES=5
VIS_DATASET_NAME="VG_train"
# DETR head-only loading
DETR_HEAD_ONLY=False
DETR_HEAD_WEIGHTS="vg_objectdetector_pretrained.pth"
# DETR query settings
DETR_NUM_OBJECT_QUERIES=10
DETR_NUM_RELATION_QUERIES=10

# 数据集路径
DATASET_ROOT="/home/shi/abschluss/dataset/vg150"
VG_IMAGES="${DATASET_ROOT}/images/VG_100K"
VG_MAPPING="${DATASET_ROOT}/VG-SGG-dicts-with-attri.json"
VG_IMAGE_DATA="${DATASET_ROOT}/image_data.json"
VG_ATTRIBUTE_H5="${DATASET_ROOT}/VG-SGG-with-attri.h5"

# 预训练权重路径
PRETRAINED_WEIGHTS="vg_objectdetector_pretrained.pth"

# 输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="z_outputs/output_pipeline_${TIMESTAMP}"
OUTPUT_PRETRAINED="${OUTPUT_BASE}/pretrained_eval"
OUTPUT_FINETUNE="${OUTPUT_BASE}/finetune_${FINETUNE_ITERS}iter"

# 端口配置
PORT_PRETRAINED=29500
PORT_FINETUNE=29501

# ============================================================================
# 解析命令行参数
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --iters)
            FINETUNE_ITERS="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --overfit)
            OVERFIT_NUM_IMAGES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --sam3)
            SAM3_ENABLED=True
            shift 1
            ;;
        --sam3-checkpoint)
            SAM3_CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --sam3-image-size)
            SAM3_IMAGE_SIZE="$2"
            shift 2
            ;;
        --sam3-feature-dim)
            SAM3_FEATURE_DIM="$2"
            shift 2
            ;;
        --sam3-channel-repeat)
            SAM3_CHANNEL_REPEAT="$2"
            shift 2
            ;;
        --sam3-device)
            SAM3_DEVICE="$2"
            shift 2
            ;;
        --sam3-precompute)
            SAM3_PRECOMPUTE=True
            SAM3_USE_PRECOMPUTED=True
            shift 1
            ;;
        --sam3-use-precomputed)
            SAM3_USE_PRECOMPUTED=True
            shift 1
            ;;
        --sam3-eval-use-precomputed)
            SAM3_USE_PRECOMPUTED_PRETRAINED=True
            shift 1
            ;;
        --sam3-eval-no-precomputed)
            SAM3_USE_PRECOMPUTED_PRETRAINED=False
            shift 1
            ;;
        --sam3-featuremaps-dir)
            SAM3_FEATUREMAP_DIR="$2"
            shift 2
            ;;
        --detr-head-only)
            DETR_HEAD_ONLY=True
            shift 1
            ;;
        --detr-head-weights)
            DETR_HEAD_WEIGHTS="$2"
            shift 2
            ;;
        --obj-queries)
            DETR_NUM_OBJECT_QUERIES="$2"
            shift 2
            ;;
        --rel-queries)
            DETR_NUM_RELATION_QUERIES="$2"
            shift 2
            ;;
        --sam3-freeze)
            SAM3_FREEZE=True
            shift 1
            ;;
        --sam3-unfreeze)
            SAM3_FREEZE=False
            shift 1
            ;;
        --sam3-eval)
            SAM3_EVAL_ENABLED=True
            shift 1
            ;;
        --sam3-eval-off)
            SAM3_EVAL_ENABLED=False
            shift 1
            ;;
        --pretrained)
            PRETRAINED_WEIGHTS="$2"
            shift 2
            ;;
        --dataset)
            DATASET_ROOT="$2"
            VG_IMAGES="${DATASET_ROOT}/images/VG_100K"
            VG_MAPPING="${DATASET_ROOT}/VG-SGG-dicts-with-attri.json"
            VG_IMAGE_DATA="${DATASET_ROOT}/image_data.json"
            VG_ATTRIBUTE_H5="${DATASET_ROOT}/VG-SGG-with-attri.h5"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --iters ITERS           Finetune 迭代次数 (默认: 1000)"
            echo "  --gpus NUM_GPUS         使用的 GPU 数量 (默认: 1)"
            echo "  --overfit NUM_IMAGES    过拟合数据集大小 (默认: 10)"
            echo "  --batch-size SIZE       批次大小 (默认: 2)"
            echo "  --sam3                 启用 SAM3 backbone"
            echo "  --sam3-checkpoint PATH SAM3 checkpoint 路径（可选）"
            echo "  --sam3-image-size SIZE SAM3 输入分辨率 (默认: 1008)"
            echo "  --sam3-feature-dim DIM SAM3 输出特征维度 (默认: 256)"
            echo "  --sam3-channel-repeat N SAM3 通道重复次数 (默认: 1)"
            echo "  --sam3-device DEV      SAM3 设备 (cuda 或 cpu, 默认: cuda)"
            echo "  --sam3-precompute      预先生成 SAM3 特征并在训练中使用"
            echo "  --sam3-use-precomputed 使用已有 SAM3 特征"
            echo "  --sam3-featuremaps-dir DIR SAM3 特征保存/读取目录"
            echo "  --sam3-freeze          冻结 SAM3 参数（默认）"
            echo "  --sam3-unfreeze        解冻 SAM3 参数"
            echo "  --sam3-eval            评测阶段启用 SAM3（默认跟随 --sam3）"
            echo "  --sam3-eval-off        评测阶段禁用 SAM3（用于评测ResNet预训练权重）"
            echo "  --detr-head-only       仅加载 DETR head 权重"
            echo "  --detr-head-weights    DETR head 权重路径"
            echo "  --obj-queries N        DETR 对象 query 数 (默认: 300)"
            echo "  --rel-queries N        DETR 关系 query 数 (默认: 300)"
            echo "  --pretrained PATH       预训练权重路径 (默认: vg_objectdetector_pretrained.pth)"
            echo "  --dataset PATH          数据集根目录 (默认: /home/shi/abschluss/dataset/vg150)"
            echo "  -h, --help              显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --iters 2000 --gpus 2"
            echo "  $0 --overfit 20 --batch-size 4"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================================
# 环境检查
# ============================================================================

echo "════════════════════════════════════════════════════════════════"
echo "  SpeaQ 全链路训练与评测流程"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "配置信息:"
echo "  预训练权重: ${PRETRAINED_WEIGHTS}"
echo "  Finetune 迭代: ${FINETUNE_ITERS}"
echo "  过拟合数据集: ${OVERFIT_NUM_IMAGES} 张图片"
echo "  GPU 数量: ${NUM_GPUS}"
echo "  批次大小: ${BATCH_SIZE}"
echo "  SAM3 启用: ${SAM3_ENABLED}"
echo "  SAM3 评测启用: ${SAM3_EVAL_ENABLED}"
echo "  SAM3 评测是否用预计算: ${SAM3_USE_PRECOMPUTED_PRETRAINED}"
echo "  SAM3 Checkpoint: ${SAM3_CHECKPOINT_PATH}"
echo "  SAM3 Image Size: ${SAM3_IMAGE_SIZE}"
echo "  SAM3 Feature Dim: ${SAM3_FEATURE_DIM}"
echo "  SAM3 Channel Repeat: ${SAM3_CHANNEL_REPEAT}"
echo "  SAM3 Device: ${SAM3_DEVICE}"
echo "  SAM3 Precompute: ${SAM3_PRECOMPUTE}"
echo "  SAM3 Use Precomputed: ${SAM3_USE_PRECOMPUTED}"
echo "  SAM3 Featuremaps Dir: ${SAM3_FEATUREMAP_DIR}"
echo "  SAM3 Freeze: ${SAM3_FREEZE}"
echo "  DETR Head-only: ${DETR_HEAD_ONLY}"
echo "  DETR Head weights: ${DETR_HEAD_WEIGHTS}"
echo "  DETR Obj Queries: ${DETR_NUM_OBJECT_QUERIES}"
echo "  DETR Rel Queries: ${DETR_NUM_RELATION_QUERIES}"
echo "  Vis Num Images: ${VIS_NUM_IMAGES}"
echo "  Vis Dataset: ${VIS_DATASET_NAME}"
echo "  输出目录: ${OUTPUT_BASE}/"
echo ""

# 检查预训练权重
if [ ! -f "${PRETRAINED_WEIGHTS}" ]; then
    echo "❌ 错误: 找不到预训练权重文件: ${PRETRAINED_WEIGHTS}"
    exit 1
fi

# 检查数据集
if [ ! -d "${VG_IMAGES}" ]; then
    echo "❌ 错误: 找不到数据集图片目录: ${VG_IMAGES}"
    exit 1
fi

# 检查 conda 环境
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 找不到 conda 命令"
    exit 1
fi

# 激活 conda 环境
echo "激活 conda 环境: speaq"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate speaq

# Ensure SAM3 package is importable
export PYTHONPATH="$(pwd)/sam3:$(pwd):${PYTHONPATH}"
# CUDA allocator hint to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "✓ 环境检查通过"
echo ""

# ============================================================================
# Step 0: 预生成 SAM3 特征 (可选)
# ============================================================================

if [ "${SAM3_PRECOMPUTE}" = "True" ]; then
    echo "════════════════════════════════════════════════════════════════"
    echo "  Step 0/4: 预生成 SAM3 特征"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    python precompute_sam3_featuremaps.py --config-file configs/speaq.yaml \
        --output-dir "${SAM3_FEATUREMAP_DIR}" \
        DATASETS.VISUAL_GENOME.IMAGES "${VG_IMAGES}" \
        DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY "${VG_MAPPING}" \
        DATASETS.VISUAL_GENOME.IMAGE_DATA "${VG_IMAGE_DATA}" \
        DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 "${VG_ATTRIBUTE_H5}" \
        DATASETS.VISUAL_GENOME.OVERFIT_NUM_IMAGES ${OVERFIT_NUM_IMAGES} \
        DATASETS.VISUAL_GENOME.OVERFIT_SEED ${OVERFIT_SEED} \
        DATASETS.VISUAL_GENOME.OVERFIT_SOURCE_SPLIT train \
        MODEL.SAM3.CHECKPOINT_PATH "${SAM3_CHECKPOINT_PATH}" \
        MODEL.SAM3.IMAGE_SIZE ${SAM3_IMAGE_SIZE} \
        MODEL.SAM3.FEATURE_DIM ${SAM3_FEATURE_DIM} \
        MODEL.SAM3.CHANNEL_REPEAT ${SAM3_CHANNEL_REPEAT} \
        MODEL.SAM3.TARGET_STRIDE ${SAM3_TARGET_STRIDE} \
        MODEL.SAM3.DEVICE ${SAM3_DEVICE} \
        MODEL.SAM3.USE_PRECOMPUTED False \
        MODEL.DEVICE ${SAM3_DEVICE}
    echo ""
    echo "✓ Step 0 完成: SAM3 特征已保存到 ${SAM3_FEATUREMAP_DIR}"
    echo ""
fi

# ============================================================================
# Step 1: 评测预训练权重
# ============================================================================

echo "════════════════════════════════════════════════════════════════"
echo "  Step 1/4: 评测预训练权重"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Verify precomputed features for pretrained eval if required
if [ "${SAM3_USE_PRECOMPUTED_PRETRAINED}" = "True" ]; then
    if [ ! -d "${SAM3_FEATUREMAP_DIR}" ]; then
        echo "❌ 错误: 预计算特征目录不存在: ${SAM3_FEATUREMAP_DIR}"
        echo "   请先运行预计算步骤 (--sam3-precompute) 或设置 SAM3_USE_PRECOMPUTED=False"
        exit 1
    fi
    feature_count=$(find "${SAM3_FEATUREMAP_DIR}" -name "*.pt" 2>/dev/null | wc -l)
    if [ "${feature_count}" -eq 0 ]; then
        echo "❌ 错误: 预计算特征目录为空: ${SAM3_FEATUREMAP_DIR}"
        echo "   请先运行预计算步骤 (--sam3-precompute)"
        exit 1
    fi
    echo "✓ 预计算特征目录检查通过: ${SAM3_FEATUREMAP_DIR} (${feature_count} 个 .pt 文件)"
fi

python train_iterative_model.py --eval-only --num-gpus ${NUM_GPUS} \
    --config-file configs/speaq.yaml --dist-url ${PORT_PRETRAINED} \
    OUTPUT_DIR "${OUTPUT_PRETRAINED}" \
    MODEL.WEIGHTS "${PRETRAINED_WEIGHTS}" \
    MODEL.DETR.LOAD_HEAD_ONLY ${DETR_HEAD_ONLY} \
    MODEL.DETR.HEAD_WEIGHTS "${DETR_HEAD_WEIGHTS}" \
    MODEL.DETR.NUM_OBJECT_QUERIES ${DETR_NUM_OBJECT_QUERIES} \
    MODEL.DETR.NUM_RELATION_QUERIES ${DETR_NUM_RELATION_QUERIES} \
    MODEL.SAM3.ENABLED ${SAM3_ENABLED_PRETRAINED} \
    MODEL.SAM3.CHECKPOINT_PATH "${SAM3_CHECKPOINT_PATH}" \
    MODEL.SAM3.IMAGE_SIZE ${SAM3_IMAGE_SIZE} \
    MODEL.SAM3.FEATURE_DIM ${SAM3_FEATURE_DIM} \
    MODEL.SAM3.CHANNEL_REPEAT ${SAM3_CHANNEL_REPEAT} \
    MODEL.SAM3.TARGET_STRIDE ${SAM3_TARGET_STRIDE} \
    MODEL.SAM3.USE_PRECOMPUTED ${SAM3_USE_PRECOMPUTED_PRETRAINED} \
    MODEL.SAM3.FEATUREMAP_DIR "${SAM3_FEATUREMAP_DIR}" \
    MODEL.SAM3.DEVICE ${SAM3_DEVICE} \
    MODEL.SAM3.FREEZE ${SAM3_FREEZE} \
    DATASETS.VISUAL_GENOME.IMAGES "${VG_IMAGES}" \
    DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY "${VG_MAPPING}" \
    DATASETS.VISUAL_GENOME.IMAGE_DATA "${VG_IMAGE_DATA}" \
    DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 "${VG_ATTRIBUTE_H5}" \
    DATASETS.VISUAL_GENOME.OVERFIT_NUM_IMAGES ${OVERFIT_NUM_IMAGES} \
    DATASETS.VISUAL_GENOME.OVERFIT_SEED ${OVERFIT_SEED} \
    DATASETS.VISUAL_GENOME.OVERFIT_SOURCE_SPLIT train \
    SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
    DATALOADER.NUM_WORKERS ${NUM_WORKERS}

# 保存预训练评测结果
python save_eval_results.py "${OUTPUT_PRETRAINED}" \
    --name "Pretrained Weights (${OVERFIT_NUM_IMAGES}-clip)" \
    --output eval_results.txt

echo ""
echo "✓ Step 1 完成: 预训练权重评测结果已保存"
echo "  结果文件: ${OUTPUT_PRETRAINED}/eval_results.txt"
echo ""

# 预训练可视化（使用 ResNet，SAM3 关闭，默认不用预计算）
PRETRAINED_VIS_DIR="${OUTPUT_PRETRAINED}/vis"
python visualize_predictions.py --config-file configs/speaq.yaml \
    --model-weights "${PRETRAINED_WEIGHTS}" \
    --output-dir "${PRETRAINED_VIS_DIR}" \
    --dataset-name "${VIS_DATASET_NAME}" \
    --num-images ${VIS_NUM_IMAGES} \
    MODEL.DETR.LOAD_HEAD_ONLY ${DETR_HEAD_ONLY} \
    MODEL.DETR.HEAD_WEIGHTS "${DETR_HEAD_WEIGHTS}" \
    MODEL.DETR.NUM_OBJECT_QUERIES ${DETR_NUM_OBJECT_QUERIES} \
    MODEL.DETR.NUM_RELATION_QUERIES ${DETR_NUM_RELATION_QUERIES} \
    MODEL.SAM3.ENABLED ${SAM3_ENABLED_PRETRAINED} \
    MODEL.SAM3.USE_PRECOMPUTED ${SAM3_USE_PRECOMPUTED_PRETRAINED} \
    MODEL.SAM3.DEVICE ${SAM3_DEVICE} \
    DATASETS.VISUAL_GENOME.IMAGES "${VG_IMAGES}" \
    DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY "${VG_MAPPING}" \
    DATASETS.VISUAL_GENOME.IMAGE_DATA "${VG_IMAGE_DATA}" \
    DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 "${VG_ATTRIBUTE_H5}" \
    DATASETS.VISUAL_GENOME.OVERFIT_NUM_IMAGES ${OVERFIT_NUM_IMAGES} \
    DATASETS.VISUAL_GENOME.OVERFIT_SEED ${OVERFIT_SEED} \
    DATASETS.VISUAL_GENOME.OVERFIT_SOURCE_SPLIT train \
    SOLVER.IMS_PER_BATCH 1 \
    DATALOADER.NUM_WORKERS 0

# ============================================================================
# Step 2: Finetune 训练
# ============================================================================

echo "════════════════════════════════════════════════════════════════"
echo "  Step 2/4: Finetune 训练 (${FINETUNE_ITERS} 次迭代)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Verify precomputed features if required (same check as Step 1)
if [ "${SAM3_USE_PRECOMPUTED}" = "True" ]; then
    if [ ! -d "${SAM3_FEATUREMAP_DIR}" ]; then
        echo "❌ 错误: 预计算特征目录不存在: ${SAM3_FEATUREMAP_DIR}"
        echo "   请先运行预计算步骤 (--sam3-precompute) 或设置 SAM3_USE_PRECOMPUTED=False"
        exit 1
    fi
    feature_count=$(find "${SAM3_FEATUREMAP_DIR}" -name "*.pt" 2>/dev/null | wc -l)
    if [ "${feature_count}" -eq 0 ]; then
        echo "❌ 错误: 预计算特征目录为空: ${SAM3_FEATUREMAP_DIR}"
        echo "   请先运行预计算步骤 (--sam3-precompute)"
        exit 1
    fi
fi

python train_iterative_model.py --num-gpus ${NUM_GPUS} \
    --config-file configs/speaq.yaml --dist-url ${PORT_FINETUNE} \
    OUTPUT_DIR "${OUTPUT_FINETUNE}" \
    MODEL.WEIGHTS "${PRETRAINED_WEIGHTS}" \
    MODEL.DETR.LOAD_HEAD_ONLY ${DETR_HEAD_ONLY} \
    MODEL.DETR.HEAD_WEIGHTS "${DETR_HEAD_WEIGHTS}" \
    MODEL.DETR.NUM_OBJECT_QUERIES ${DETR_NUM_OBJECT_QUERIES} \
    MODEL.DETR.NUM_RELATION_QUERIES ${DETR_NUM_RELATION_QUERIES} \
    MODEL.SAM3.ENABLED ${SAM3_ENABLED} \
    MODEL.SAM3.CHECKPOINT_PATH "${SAM3_CHECKPOINT_PATH}" \
    MODEL.SAM3.IMAGE_SIZE ${SAM3_IMAGE_SIZE} \
    MODEL.SAM3.FEATURE_DIM ${SAM3_FEATURE_DIM} \
    MODEL.SAM3.CHANNEL_REPEAT ${SAM3_CHANNEL_REPEAT} \
    MODEL.SAM3.TARGET_STRIDE ${SAM3_TARGET_STRIDE} \
    MODEL.SAM3.USE_PRECOMPUTED ${SAM3_USE_PRECOMPUTED} \
    MODEL.SAM3.FEATUREMAP_DIR "${SAM3_FEATUREMAP_DIR}" \
    MODEL.SAM3.DEVICE ${SAM3_DEVICE} \
    MODEL.SAM3.FREEZE ${SAM3_FREEZE} \
    DATASETS.VISUAL_GENOME.IMAGES "${VG_IMAGES}" \
    DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY "${VG_MAPPING}" \
    DATASETS.VISUAL_GENOME.IMAGE_DATA "${VG_IMAGE_DATA}" \
    DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 "${VG_ATTRIBUTE_H5}" \
    DATASETS.VISUAL_GENOME.OVERFIT_NUM_IMAGES ${OVERFIT_NUM_IMAGES} \
    DATASETS.VISUAL_GENOME.OVERFIT_SEED ${OVERFIT_SEED} \
    DATASETS.VISUAL_GENOME.OVERFIT_SOURCE_SPLIT train \
    SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
    SOLVER.MAX_ITER ${FINETUNE_ITERS} \
    SOLVER.CHECKPOINT_PERIOD ${CHECKPOINT_PERIOD} \
    DATALOADER.NUM_WORKERS ${NUM_WORKERS} \
    TEST.EVAL_PERIOD 0

echo ""
echo "✓ Step 2 完成: Finetune 训练完成"
echo "  模型文件: ${OUTPUT_FINETUNE}/model_final.pth"
echo ""

# Finetune 可视化（遵循当前 SAM3 配置）
FINETUNE_VIS_DIR="${OUTPUT_FINETUNE}/vis"
python visualize_predictions.py --config-file configs/speaq.yaml \
    --model-weights "${OUTPUT_FINETUNE}/model_final.pth" \
    --output-dir "${FINETUNE_VIS_DIR}" \
    --dataset-name "${VIS_DATASET_NAME}" \
    --num-images ${VIS_NUM_IMAGES} \
    MODEL.DETR.LOAD_HEAD_ONLY ${DETR_HEAD_ONLY} \
    MODEL.DETR.HEAD_WEIGHTS "${DETR_HEAD_WEIGHTS}" \
    MODEL.DETR.NUM_OBJECT_QUERIES ${DETR_NUM_OBJECT_QUERIES} \
    MODEL.DETR.NUM_RELATION_QUERIES ${DETR_NUM_RELATION_QUERIES} \
    MODEL.SAM3.ENABLED ${SAM3_ENABLED} \
    MODEL.SAM3.CHECKPOINT_PATH "${SAM3_CHECKPOINT_PATH}" \
    MODEL.SAM3.IMAGE_SIZE ${SAM3_IMAGE_SIZE} \
    MODEL.SAM3.FEATURE_DIM ${SAM3_FEATURE_DIM} \
    MODEL.SAM3.CHANNEL_REPEAT ${SAM3_CHANNEL_REPEAT} \
    MODEL.SAM3.TARGET_STRIDE ${SAM3_TARGET_STRIDE} \
    MODEL.SAM3.USE_PRECOMPUTED ${SAM3_USE_PRECOMPUTED} \
    MODEL.SAM3.FEATUREMAP_DIR "${SAM3_FEATUREMAP_DIR}" \
    MODEL.SAM3.DEVICE ${SAM3_DEVICE} \
    MODEL.SAM3.FREEZE ${SAM3_FREEZE} \
    DATASETS.VISUAL_GENOME.IMAGES "${VG_IMAGES}" \
    DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY "${VG_MAPPING}" \
    DATASETS.VISUAL_GENOME.IMAGE_DATA "${VG_IMAGE_DATA}" \
    DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 "${VG_ATTRIBUTE_H5}" \
    DATASETS.VISUAL_GENOME.OVERFIT_NUM_IMAGES ${OVERFIT_NUM_IMAGES} \
    DATASETS.VISUAL_GENOME.OVERFIT_SEED ${OVERFIT_SEED} \
    DATASETS.VISUAL_GENOME.OVERFIT_SOURCE_SPLIT train \
    SOLVER.IMS_PER_BATCH 1 \
    DATALOADER.NUM_WORKERS 0

# ============================================================================
# Step 3: 评测 Finetune 后的模型
# ============================================================================

echo "════════════════════════════════════════════════════════════════"
echo "  Step 3/4: 评测 Finetune 后的模型"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 训练过程中已经包含了最终评测，结果已在 log.txt 和 metrics.json 中

# 保存 Finetune 评测结果
python save_eval_results.py "${OUTPUT_FINETUNE}" \
    --name "Finetuned ${FINETUNE_ITERS} iter (${OVERFIT_NUM_IMAGES}-clip)" \
    --output eval_results.txt

echo ""
echo "✓ Step 3 完成: Finetune 模型评测结果已保存"
echo "  结果文件: ${OUTPUT_FINETUNE}/eval_results.txt"
echo ""

# ============================================================================
# Step 4: 生成结果对比
# ============================================================================

echo "════════════════════════════════════════════════════════════════"
echo "  Step 4/4: 生成结果对比"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 生成对比报告
python compare_results.py "${OUTPUT_PRETRAINED}" "${OUTPUT_FINETUNE}"

# 将对比结果复制到输出目录
cp results_comparison.txt "${OUTPUT_BASE}/results_comparison.txt"

echo ""
echo "✓ Step 4 完成: 结果对比已生成"
echo "  对比文件: ${OUTPUT_BASE}/results_comparison.txt"
echo ""

# ============================================================================
# 生成总结报告
# ============================================================================

SUMMARY_FILE="${OUTPUT_BASE}/pipeline_summary.txt"

cat > "${SUMMARY_FILE}" << EOF
════════════════════════════════════════════════════════════════
SpeaQ 全链路训练与评测流程 - 执行总结
════════════════════════════════════════════════════════════════

执行时间: $(date)

配置参数:
────────────────────────────────────────────────────────────────
  预训练权重:     ${PRETRAINED_WEIGHTS}
  Finetune 迭代:  ${FINETUNE_ITERS}
  过拟合数据集:   ${OVERFIT_NUM_IMAGES} 张图片 (seed=${OVERFIT_SEED})
  GPU 数量:       ${NUM_GPUS}
  批次大小:       ${BATCH_SIZE}
  检查点周期:     每 ${CHECKPOINT_PERIOD} 次迭代
  SAM3 启用:      ${SAM3_ENABLED}
  SAM3 评测启用:  ${SAM3_EVAL_ENABLED}
  SAM3 Checkpoint:${SAM3_CHECKPOINT_PATH}
  SAM3 Image Size:${SAM3_IMAGE_SIZE}
  SAM3 Feature Dim:${SAM3_FEATURE_DIM}
  SAM3 Channel Repeat:${SAM3_CHANNEL_REPEAT}
  SAM3 Device:    ${SAM3_DEVICE}
  SAM3 Freeze:    ${SAM3_FREEZE}
  SAM3 Precompute:${SAM3_PRECOMPUTE}
  SAM3 Use Precomputed:${SAM3_USE_PRECOMPUTED}
  SAM3 Featuremaps Dir:${SAM3_FEATUREMAP_DIR}
  DETR Head-only: ${DETR_HEAD_ONLY}
  DETR Head weights:${DETR_HEAD_WEIGHTS}
  DETR Obj Queries:${DETR_NUM_OBJECT_QUERIES}
  DETR Rel Queries:${DETR_NUM_RELATION_QUERIES}

生成的文件:
────────────────────────────────────────────────────────────────
  1. 预训练评测结果:
     ${OUTPUT_PRETRAINED}/eval_results.txt
     ${OUTPUT_PRETRAINED}/metrics.json
     ${OUTPUT_PRETRAINED}/log.txt

  2. Finetune 训练结果:
     ${OUTPUT_FINETUNE}/model_final.pth       (最终模型)
     ${OUTPUT_FINETUNE}/eval_results.txt      (评测结果)
     ${OUTPUT_FINETUNE}/metrics.json
     ${OUTPUT_FINETUNE}/log.txt

  3. 对比结果:
     ${OUTPUT_BASE}/results_comparison.txt

快速查看命令:
────────────────────────────────────────────────────────────────
  # 查看预训练结果
  cat ${OUTPUT_PRETRAINED}/eval_results.txt

  # 查看 Finetune 结果
  cat ${OUTPUT_FINETUNE}/eval_results.txt

  # 查看对比结果
  cat ${OUTPUT_BASE}/results_comparison.txt

  # 使用 Finetune 后的模型
  MODEL.WEIGHTS ${OUTPUT_FINETUNE}/model_final.pth

下一步建议:
────────────────────────────────────────────────────────────────
EOF

# 添加评测结果摘要
echo "  预训练结果摘要:" >> "${SUMMARY_FILE}"
grep "AP" "${OUTPUT_PRETRAINED}/eval_results.txt" | head -6 | sed 's/^/    /' >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

echo "  Finetune 结果摘要:" >> "${SUMMARY_FILE}"
grep "AP" "${OUTPUT_FINETUNE}/eval_results.txt" | head -6 | sed 's/^/    /' >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# 添加建议
cat >> "${SUMMARY_FILE}" << EOF
  1. 如果场景图指标仍为 0，建议增加训练迭代次数:
     ./run_full_pipeline.sh --iters 5000

  2. 可视化训练曲线:
     tensorboard --logdir=${OUTPUT_FINETUNE}

  3. 继续在此模型基础上训练:
     使用 MODEL.WEIGHTS ${OUTPUT_FINETUNE}/model_final.pth

════════════════════════════════════════════════════════════════
流程执行完成！所有结果保存在: ${OUTPUT_BASE}/
════════════════════════════════════════════════════════════════
EOF

# ============================================================================
# 显示最终总结
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  🎉 全链路流程执行完成！"
echo "════════════════════════════════════════════════════════════════"
echo ""
cat "${SUMMARY_FILE}"
echo ""
echo "总结报告已保存: ${SUMMARY_FILE}"
echo ""
