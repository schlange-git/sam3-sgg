#!/bin/bash
# 快速测试脚本：只处理少量图像，验证修复是否有效

source $(conda info --base)/etc/profile.d/conda.sh
conda activate sam3

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

VG_ROOT="/home/shi/桌面/abschluss/sgg/dataset/vg150"
OUT_DIR="sgg/cache/image_features_test"
SPLIT="train"
FEATURE_DIM=256
IMAGE_SIZE=1008
DEVICE="cuda"
NUM_GPUS=2  # 先用单GPU测试，避免内存溢出（2个GPU会加载2倍的模型和数据集）
BATCH_SIZE=1  # 较小的 batch size 用于测试（减少内存占用）
NUM_WORKERS=0  # 禁用DataLoader的num_workers（在multiprocessing中已禁用）
LIMIT=100  # 0 或 -1 表示处理全部图像；设置为正数（如 50）表示只处理指定数量的图像用于测试
CHECKPOINT_PATH="weights/sam3.pt"

mkdir -p "$OUT_DIR"

if [ "$LIMIT" -eq 0 ] || [ "$LIMIT" -eq -1 ]; then
    echo "=== 测试模式：处理全部图像 ==="
else
    echo "=== 测试模式：处理 $LIMIT 张图像 ==="
fi
echo ""

python sgg/precompute/image_feature_generator.py \
    --data_root "$VG_ROOT" \
    --split "$SPLIT" \
    --output_dir "$OUT_DIR" \
    --device "$DEVICE" \
    --feature_dim $FEATURE_DIM \
    --image_size $IMAGE_SIZE \
    --max_samples $LIMIT \
    --num_gpus $NUM_GPUS \
    --use_multi_gpu \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --checkpoint_path "$CHECKPOINT_PATH"

echo ""
echo "=== 测试完成 ==="
echo "检查输出目录: $OUT_DIR/$SPLIT"

