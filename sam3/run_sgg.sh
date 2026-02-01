#!/bin/bash
# 运行SGG训练脚本

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate sam3

# ==================== 训练配置 ====================
# 数据集路径 - 使用相对路径查找
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# 尝试多个可能的数据集路径
if [ -d "$PROJECT_ROOT/../dataset/visual_genome" ]; then
    DATA_ROOT="$(cd "$PROJECT_ROOT/../dataset/visual_genome" && pwd)"
elif [ -d "$PROJECT_ROOT/../../dataset/vg150" ]; then
    DATA_ROOT="$(cd "$PROJECT_ROOT/../../dataset/vg150" && pwd)"
elif [ -d "$HOME/桌面/abschluss/sgg/dataset/vg150" ]; then
    DATA_ROOT="$HOME/桌面/abschluss/sgg/dataset/vg150"
else
    # 如果都找不到，使用相对路径（用户需要自己修改）
    DATA_ROOT="$PROJECT_ROOT/../dataset/visual_genome"
    echo "⚠️ 警告: 使用默认路径 $DATA_ROOT，如果不存在请修改脚本"
fi

# 训练参数
BATCH_SIZE=4      
NUM_EPOCHS=3         
LR=1e-4              
DEVICE="cuda"        
NUM_WORKERS=4         
SAVE_DIR="./checkpoints"  

# 损失函数配置
NEG_RATIO=3           # 负样本比例
USE_CLASS_WEIGHTS=true   # 使用类别权重 (默认启用)
USE_FOCAL_LOSS=false     # 使用Focal Loss (设置为true则使用Focal Loss)
# ==================================================

echo "=========================================="
echo "SGG Training Configuration:"
echo "  Data root: $DATA_ROOT"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning rate: $LR"
echo "  Device: $DEVICE"
echo "  Num workers: $NUM_WORKERS"
echo "  Save directory: $SAVE_DIR"
echo "  Negative ratio: $NEG_RATIO"
echo "  Use class weights: $USE_CLASS_WEIGHTS"
echo "  Use focal loss: $USE_FOCAL_LOSS"
echo "  Output interval: Every 10 iterations"
echo "=========================================="
echo ""

# 构建命令
CMD="python train_sgg.py \
    --data_root \"$DATA_ROOT\" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --device $DEVICE \
    --num_workers $NUM_WORKERS \
    --save_dir \"$SAVE_DIR\" \
    --neg_ratio $NEG_RATIO"

# 添加可选标志
if [ "$USE_CLASS_WEIGHTS" = true ]; then
    CMD="$CMD --use_class_weights"
fi

if [ "$USE_FOCAL_LOSS" = true ]; then
    CMD="$CMD --use_focal_loss"
fi

# 执行训练
echo "Starting training..."
echo ""
eval $CMD

echo ""
echo "Training completed!"

