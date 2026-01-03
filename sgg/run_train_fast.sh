#!/bin/bash
# Script to run fast training with cached pairs

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sam3

# Navigate to project root
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

# Configuration
CACHE_DIR="sgg/cache/train"
OUT_DIR="/home/shi/abschluss/sam3/checkpoints/sgghead"
NUM_PREDICATES=51  # 50 predicates + 1 background
EPOCHS=300
BATCH_SIZE=32
LR=1e-3
WEIGHT_DECAY=1e-4
NUM_WORKERS=4
AMP=true
GRAD_CLIP=5.0
BG_WEIGHT=0.2
LOG_EVERY=50
SAVE_EVERY=100
SEED=42

# Create output directory
mkdir -p "$OUT_DIR"

# Run training
python sgg/train/train_fast.py \
    --cache_dir "$CACHE_DIR" \
    --out_dir "$OUT_DIR" \
    --num_predicates $NUM_PREDICATES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --num_workers $NUM_WORKERS \
    --grad_clip $GRAD_CLIP \
    --bg_weight $BG_WEIGHT \
    --log_every $LOG_EVERY \
    --save_every $SAVE_EVERY \
    --seed $SEED \
    $([ "$AMP" = "true" ] && echo "--amp" || echo "--no_amp")

echo "Training completed!"

