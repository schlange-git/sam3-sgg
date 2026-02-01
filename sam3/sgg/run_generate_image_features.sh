#!/bin/bash
# Script to generate SAM3 image features (whole image features, not box-specific)
# 使用 SAM3 作为 backbone 提取整图的 feature（区别于针对具体 box 的 mask 生成）

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
#conda activate sam3

# Navigate to project root
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

# Configuration
# 使用相对路径，尝试多个可能的数据集路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# 尝试多个可能的数据集路径
if [ -d "$PROJECT_ROOT/../../dataset/vg150" ]; then
    VG_ROOT="$(cd "$PROJECT_ROOT/../../dataset/vg150" && pwd)"
elif [ -d "$PROJECT_ROOT/../../../dataset/vg150" ]; then
    VG_ROOT="$(cd "$PROJECT_ROOT/../../../dataset/vg150" && pwd)"
elif [ -d "$HOME/桌面/abschluss/sgg/dataset/vg150" ]; then
    VG_ROOT="$HOME/桌面/abschluss/sgg/dataset/vg150"
else
    # 如果都找不到，使用相对路径（用户需要自己修改）
    VG_ROOT="$PROJECT_ROOT/../../dataset/vg150"
    echo "⚠️ 警告: 使用默认路径 $VG_ROOT，如果不存在请修改脚本"
fi
OUT_DIR="sgg/cache/image_features"
SPLIT="train"  # "train" or "val"
FEATURE_DIM=256
IMAGE_SIZE=1008  # SAM3 默认分辨率为 1008（不是 1024）
DEVICE="cuda"
NUM_GPUS=1  # 使用双 GPU (两张 2080ti 22GB)
BATCH_SIZE=1  # 批处理大小（避免一次加载过多导致爆内存）
NUM_WORKERS=0  # DataLoader 工作线程数（不要设置太高，因为内存有限）
LIMIT=8 # -1 or 0 = no limit (all images), or set to N for quick validation (first N images)
START_IDX=0 # 从数据集的哪个样本开始处理，用于人工分段跑（例如前半/后半）
SAVE_FP16=0  # 暂不启用 FP16 保存
CHECKPOINT_PATH="weights/sam3.pt"  # 留空表示自动查找，或指定相对路径如 "sam3.pt" 或 "checkpoints/sam3.pt"

# 检测可用 GPU 数量
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected $AVAILABLE_GPUS GPU(s)"
    if [ $AVAILABLE_GPUS -lt $NUM_GPUS ]; then
        echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available. Using $AVAILABLE_GPUS GPUs."
        NUM_GPUS=$AVAILABLE_GPUS
    fi
else
    echo "Warning: nvidia-smi not found, assuming $NUM_GPUS GPUs available"
fi

# Create output directory
mkdir -p "$OUT_DIR"

# 构建命令
CMD="python sgg/precompute/image_feature_generator.py \
    --data_root \"$VG_ROOT\" \
    --split \"$SPLIT\" \
    --output_dir \"$OUT_DIR\" \
    --device \"$DEVICE\" \
    --feature_dim $FEATURE_DIM \
    --image_size $IMAGE_SIZE \
    --max_samples $LIMIT \
    --start_idx $START_IDX \
    --num_gpus $NUM_GPUS \
    --use_multi_gpu \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS"

# 如果指定了 checkpoint 路径，添加到命令中
if [ -n "$CHECKPOINT_PATH" ]; then
    CMD="$CMD --checkpoint_path \"$CHECKPOINT_PATH\""
fi
if [ "${SAVE_FP16}" = "1" ]; then
    CMD="$CMD --save_fp16"
fi

# Run image feature generation
echo "Running with $NUM_GPUS GPU(s)..."
eval $CMD

echo "Image feature generation completed!"

