#!/usr/bin/env python3
import argparse
import os
import sys
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from detectron2.config import get_cfg

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.data.datasets.action_genome import ActionGenomeTrainData


def buildCfg(args):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def summarizeSplit(dataset):
    records = dataset.dataset_dicts
    frame_counts = Counter(str(r["video_id"]) for r in records)
    total_frames = len(records)
    total_videos = len(frame_counts)
    max_count = max(frame_counts.values()) if frame_counts else 0
    min_count = min(frame_counts.values()) if frame_counts else 0
    max_videos = sorted([v for v, c in frame_counts.items() if c == max_count])
    min_videos = sorted([v for v, c in frame_counts.items() if c == min_count])
    return records, frame_counts, total_frames, total_videos, max_count, min_count, max_videos, min_videos


def main():
    parser = argparse.ArgumentParser(description="统计 Action Genome 评测集视频/帧分布")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = buildCfg(args)
    val_data = ActionGenomeTrainData(cfg, split="val")
    records, frame_counts, total_frames, total_videos, max_count, min_count, max_videos, min_videos = summarizeSplit(val_data)

    all_annotated_counts = Counter()
    for frame_key in val_data.frame_list:
        if "/" not in frame_key:
            continue
        video, _ = frame_key.split("/", 1)
        all_annotated_counts[video] += 1

    videos = sorted(val_data.video_to_frames.keys())
    if cfg.DATASETS.ACTION_GENOME.VAL_SET_RANDOMIZED:
        import random
        rng = random.Random(cfg.DATASETS.VISUAL_GENOME.OVERFIT_SEED)
        rng.shuffle(videos)
    n_val = max(0, min(len(videos), int(cfg.DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL)))
    val_videos = set(videos[-n_val:] if n_val > 0 else videos)
    train_videos = set(videos[:-n_val] if n_val > 0 else videos)

    print("===== AG_val temporal statistics =====")
    print(f"Eval frames: {total_frames}")
    print(f"Eval videos: {total_videos}")
    print(f"Train videos: {len(train_videos)}")
    print(f"Val split videos configured: {len(val_videos)}")
    print(f"Train/val video overlap: {len(train_videos & val_videos)}")
    print(f"Max frames per eval video: {max_count}")
    print(f"Min frames per eval video: {min_count}")
    print("")

    print("Top-10 eval videos by frame count:")
    for video, count in frame_counts.most_common(10):
        split_only = video in val_videos and video not in train_videos
        all_count = all_annotated_counts.get(video, 0)
        print(f"  {video}: eval_frames={count}, all_annotated_frames={all_count}, val_only_video={split_only}")

    print("")
    print("Max-frame video(s):")
    for video in max_videos:
        split_only = video in val_videos and video not in train_videos
        all_count = all_annotated_counts.get(video, 0)
        print(f"  {video}: eval_frames={frame_counts[video]}, all_annotated_frames={all_count}, val_only_video={split_only}")

    print("")
    print("Min-frame video(s):")
    for video in min_videos[:20]:
        split_only = video in val_videos and video not in train_videos
        all_count = all_annotated_counts.get(video, 0)
        print(f"  {video}: eval_frames={frame_counts[video]}, all_annotated_frames={all_count}, val_only_video={split_only}")
    if len(min_videos) > 20:
        print(f"  ... ({len(min_videos) - 20} more videos with {min_count} frame)")


if __name__ == "__main__":
    main()
