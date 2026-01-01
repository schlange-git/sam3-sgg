#!/bin/bash
# Training script for SGG PredCls
# 使用 conda sam3 环境

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sam3

# 检查环境
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# 训练配置
DATA_ROOT="/home/shi/abschluss/dataset/vg150"  # VG150 数据集路径
BATCH_SIZE=4
NUM_EPOCHS=3
LR=1e-4
NUM_WORKERS=4
DEVICE="cuda"
NEG_RATIO=3
MAX_NEGS=50
BG_WEIGHT=0.1
SAVE_DIR="sgg/checkpoints"
LOG_DIR="sgg/logfiles"
LOG_INTERVAL=10

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 设置 PYTHONPATH 以确保能找到 sgg 模块
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练
python sgg/train_predcls.py \
    --data_root "$DATA_ROOT" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --neg_ratio $NEG_RATIO \
    --max_negs $MAX_NEGS \
    --bg_weight $BG_WEIGHT \
    --save_dir $SAVE_DIR \
    --log_dir $LOG_DIR \
    --log_interval $LOG_INTERVAL

echo "Training completed!"
