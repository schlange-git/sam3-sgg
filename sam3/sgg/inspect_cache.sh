#!/bin/bash
# Quick script to inspect cache files

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sam3

# Navigate to project root
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

# Configuration
CACHE_DIR="sgg/cache/train"
NUM_SAMPLES=5
OUTPUT_JSON="sgg/cache/inspection.json"

# Run inspection
python sgg/utils/inspect_cache.py \
    --cache_dir "$CACHE_DIR" \
    --num_samples $NUM_SAMPLES \
    --output_json "$OUTPUT_JSON"

echo ""
echo "To inspect a single file:"
echo "  python sgg/utils/inspect_cache.py --cache_file sgg/cache/train/00000001.pt --output_json single_file.json"

