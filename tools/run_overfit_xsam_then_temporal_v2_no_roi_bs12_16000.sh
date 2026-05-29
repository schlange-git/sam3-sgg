#!/usr/bin/env bash
# Run the two local single-GPU overfit probes sequentially to avoid GPU memory contention.
set -euo pipefail

PROJ=/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching
cd "$PROJ"

echo "[Sequence] Starting X-SAM no-ROI overfit probe..."
PORT="${XSAM_PORT:-29562}" bash tools/overfit_xsam_no_roi_bs12_16000.sh

echo "[Sequence] Starting temporal-v2 relation no-ROI overfit probe..."
PORT="${TEMPORAL_PORT:-29561}" bash tools/overfit_temporal_v2_relation_no_roi_bs12_16000.sh

echo "[Sequence] All probes completed."
