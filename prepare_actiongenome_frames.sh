#!/usr/bin/env bash
set -euo pipefail

# Standalone frame extraction for Action Genome.
# This script is intentionally independent from training for easier reuse/debug.

# 所有输出同时写到终端和日志文件
LOG_FILE="${LOG_FILE:-prepare_actiongenome_frames_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "${LOG_FILE}")" || true
exec > >(tee -a "${LOG_FILE}") 2>&1

VIDEO_DIR="${VIDEO_DIR:-dataset-full/Charades_v1}"
FRAME_DIR="${FRAME_DIR:-dataset-full/frames}"
ANNOTATION_DIR="${ANNOTATION_DIR:-dataset-full/annotations}"

KEEP_ALL_FRAMES="${KEEP_ALL_FRAMES:-0}"  # 1 -> pass --all_frames
EXTRA_FRAMES_PER_INTERVAL="${EXTRA_FRAMES_PER_INTERVAL:-1}"  # used when KEEP_ALL_FRAMES=0

echo "VIDEO_DIR=${VIDEO_DIR}"
echo "FRAME_DIR=${FRAME_DIR}"
echo "ANNOTATION_DIR=${ANNOTATION_DIR}"
echo "KEEP_ALL_FRAMES=${KEEP_ALL_FRAMES}"
echo "EXTRA_FRAMES_PER_INTERVAL=${EXTRA_FRAMES_PER_INTERVAL}"
echo "LOG_FILE=${LOG_FILE}"

if [[ ! -d "${VIDEO_DIR}" ]]; then
  echo "ERROR: video dir not found: ${VIDEO_DIR}"
  exit 1
fi
if [[ ! -d "${ANNOTATION_DIR}" ]]; then
  echo "ERROR: annotation dir not found: ${ANNOTATION_DIR}"
  exit 1
fi

mkdir -p "${FRAME_DIR}"

ALL_FRAMES_ARG=""
if [[ "${KEEP_ALL_FRAMES}" == "1" ]]; then
  ALL_FRAMES_ARG="--all_frames"
fi

python data/ActionGenome/tools/dump_frames.py \
  --video_dir "${VIDEO_DIR}" \
  --frame_dir "${FRAME_DIR}" \
  --annotation_dir "${ANNOTATION_DIR}" \
  --extra_frames_per_interval "${EXTRA_FRAMES_PER_INTERVAL}" \
  ${ALL_FRAMES_ARG}

echo "ActionGenome frame extraction done."
