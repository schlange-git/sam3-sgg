#!/bin/bash
# Script to build cache for offline preprocessing
# Set LIMIT=1000 for quick validation, or LIMIT=-1/0 for all images

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sam3

# Navigate to project root
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

# Configuration
VG_ROOT="/home/shi/abschluss/dataset/vg150"
OUT_DIR="sgg/cache/train"
SPLIT="train"
P=128
NEG_RATIO=3
MASK_SIZE=256
SEED=42
LIMIT=10  # -1 or 0 = no limit (all images), or set to N for quick validation (first N images)
SAM3_IMPL="real"  # "real" or "dummy"
DEVICE="cuda"
SAVE_EVERY=10  # Save metadata every N images

# Create output directory
mkdir -p "$OUT_DIR"

# Run build cache
python sgg/precompute/build_cache.py \
    --vg_root "$VG_ROOT" \
    --out_dir "$OUT_DIR" \
    --split "$SPLIT" \
    --P $P \
    --neg_ratio $NEG_RATIO \
    --mask_size $MASK_SIZE \
    --seed $SEED \
    --limit $LIMIT \
    --sam3_impl "$SAM3_IMPL" \
    --device "$DEVICE" \
    --save_every $SAVE_EVERY

echo "Cache building completed!"

