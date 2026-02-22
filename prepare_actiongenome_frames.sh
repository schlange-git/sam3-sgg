#!/usr/bin/env bash
set -euo pipefail

# Standalone frame extraction for Action Genome.
# This script is intentionally independent from training for easier reuse/debug.

VIDEO_DIR="${VIDEO_DIR:-dataset/Charades_v1_480}"
FRAME_DIR="${FRAME_DIR:-dataset/frames}"
ANNOTATION_DIR="${ANNOTATION_DIR:-dataset/annotations}"

KEEP_ALL_FRAMES="${KEEP_ALL_FRAMES:-0}"  # 1 -> pass --all_frames

echo "VIDEO_DIR=${VIDEO_DIR}"
echo "FRAME_DIR=${FRAME_DIR}"
echo "ANNOTATION_DIR=${ANNOTATION_DIR}"
echo "KEEP_ALL_FRAMES=${KEEP_ALL_FRAMES}"

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
  ${ALL_FRAMES_ARG}

echo "ActionGenome frame extraction done."
