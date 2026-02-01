#!/bin/bash
# Script to run fast testing with cached pairs

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sam3

# Navigate to project root
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

# Configuration
# 注意：CACHE_DIR 需要与 SPLIT 匹配（val 对应 sgg/cache/val，train 对应 sgg/cache/train）
# 如果还没有 val 缓存，需要先运行 build_cache.py 生成
CACHE_DIR="sgg/cache/train"  # 或 train，根据评测集
CHECKPOINT_PATH="checkpoints/sgghead/rel_head_ep300.pt"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# 尝试多个可能的数据集路径
if [ -d "$PROJECT_ROOT/../../dataset/vg150" ]; then
    DATA_ROOT="$(cd "$PROJECT_ROOT/../../dataset/vg150" && pwd)"
elif [ -d "$PROJECT_ROOT/../../../dataset/vg150" ]; then
    DATA_ROOT="$(cd "$PROJECT_ROOT/../../../dataset/vg150" && pwd)"
elif [ -d "$HOME/桌面/abschluss/sgg/dataset/vg150" ]; then
    DATA_ROOT="$HOME/桌面/abschluss/sgg/dataset/vg150"
else
    # 如果都找不到，使用相对路径（用户需要自己修改）
    DATA_ROOT="$PROJECT_ROOT/../../dataset/vg150"
    echo "⚠️ 警告: 使用默认路径 $DATA_ROOT，如果不存在请修改脚本"
fi
SPLIT="train"  # 或 train，需要与 CACHE_DIR 匹配
BATCH_SIZE=32
NUM_WORKERS=4
AMP=true
ENABLE_VIS=true  # 启用可视化
VIS_DIR="$PROJECT_ROOT/checkpoints/sgghead/visualizations"
VIS_NUM_SAMPLES=20  # 可视化样本数
K_LIST="20 50 100"  # R@K 和 mR@K 的 K 值
SEED=42

# Create visualization directory
if [ "$ENABLE_VIS" = "true" ]; then
    mkdir -p "$VIS_DIR"
fi

# Run testing
python sgg/train/test_fast.py \
    --cache_dir "$CACHE_DIR" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_root "$DATA_ROOT" \
    --split "$SPLIT" \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --vis_num_samples $VIS_NUM_SAMPLES \
    --k_list $K_LIST \
    --seed $SEED \
    $([ "$AMP" = "true" ] && echo "--amp" || echo "--no_amp") \
    $([ "$ENABLE_VIS" = "true" ] && echo "--enable_vis --vis_dir $VIS_DIR" || echo "")

echo "Testing completed!"

