#!/usr/bin/env python3
"""Compute per-class average box area and find small-object threshold."""
import sys, os, pickle, json, math
from collections import defaultdict
import numpy as np

ANNO_DIR = '/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching/dataset/annotations'

# Load annotations
with open(os.path.join(ANNO_DIR, 'object_bbox_and_relationship.pkl'), 'rb') as f:
    obj_anno = pickle.load(f)
with open(os.path.join(ANNO_DIR, 'object_classes.txt')) as f:
    class_names = [l.strip() for l in f if l.strip()]
with open(os.path.join(ANNO_DIR, 'frame_list.txt')) as f:
    frame_list = [l.strip() for l in f if l.strip()]
with open(os.path.join(ANNO_DIR, 'person_bbox.pkl'), 'rb') as f:
    person_anno = pickle.load(f)

# Load frame dimensions cache
cache_path = os.path.join(ANNO_DIR, 'frame_dimensions.json')
if os.path.exists(cache_path):
    with open(cache_path) as f:
        frame_dims = json.load(f)
else:
    frame_dims = {}

print(f'Classes: {len(class_names)}')
print(f'Frames: {len(frame_list)}')

# Collect per-class areas (normalized by image area)
class_areas = defaultdict(list)

def parse_bbox_xyxy(bbox_data):
    if isinstance(bbox_data, dict):
        return [bbox_data.get('x1', 0), bbox_data.get('y1', 0),
                bbox_data.get('x2', 0), bbox_data.get('y2', 0)]
    return None

def get_annotation_size(frame_key, person_item):
    if isinstance(person_item, dict):
        bs = person_item.get('bbox_size')
        if isinstance(bs, (list, tuple)) and len(bs) == 2:
            try:
                w, h = float(bs[0]), float(bs[1])
                if w > 1 and h > 1: return w, h
            except: pass
    return None

processed = 0
for frame_key in frame_list:
    if '/' not in frame_key: continue
    video, fname = frame_key.split('/', 1)
    
    # Get frame dimensions
    if frame_key in frame_dims:
        width, height = frame_dims[frame_key]
    else:
        width, height = 360, 480  # default annotation size
    
    anno_size = get_annotation_size(frame_key, person_anno.get(frame_key, {}))
    if anno_size:
        anno_w, anno_h = anno_size
        scale_x = width / anno_w
        scale_y = height / anno_h
    else:
        scale_x, scale_y = 1.0, 1.0
    
    img_area = width * height
    
    objects = obj_anno.get(frame_key, [])
    for obj in objects:
        bbox = parse_bbox_xyxy(obj.get('bbox'))
        if bbox is None: continue
        cls_name = str(obj.get('class', 'object'))
        if cls_name not in class_names: continue
        
        # Scale to image coordinates
        x1 = min(max(bbox[0] * scale_x, 0), width)
        x2 = min(max(bbox[2] * scale_x, 0), width)
        y1 = min(max(bbox[1] * scale_y, 0), height)
        y2 = min(max(bbox[3] * scale_y, 0), height)
        
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0: continue
        
        # Normalized area (fraction of image)
        norm_area = (box_w * box_h) / img_area
        class_areas[cls_name].append(norm_area)
    
    processed += 1
    if processed % 50000 == 0:
        print(f'  Processed {processed}/{len(frame_list)} frames...')

print(f'\nProcessed {processed} frames')

# Compute per-class statistics
print(f'\n{"Class":<20s} {"Count":>8s} {"Mean Area":>12s} {"Median":>12s} {"Norm Mean":>12s}')
print('-' * 70)
stats = []
for cls_name in class_names:
    areas = class_areas.get(cls_name, [])
    if not areas:
        stats.append((cls_name, 0, 0.0, 0.0, 0.0))
        continue
    areas_np = np.array(areas)
    mean_area = areas_np.mean()
    median_area = np.median(areas_np)
    stats.append((cls_name, len(areas_np), mean_area, median_area, mean_area))

# Sort by mean area (ascending)
stats.sort(key=lambda x: x[2])

print(f'\n=== Sorted by mean area (smallest first) ===')
for i, (name, count, mean, median, _) in enumerate(stats):
    marker = ' <<<' if i >= len(stats) - 10 else ''
    print(f'{i+1:3d}. {name:<20s} count={count:6d}  mean_area={mean:.8f}  median={median:.8f}{marker}')

# Bottom 10 classes
bottom10 = stats[:10]
bottom10_mean = np.mean([s[2] for s in bottom10])
print(f'\n=== Bottom 10 classes (smallest) ===')
for name, count, mean, median, _ in bottom10:
    print(f'  {name}: count={count}, mean_area={mean:.8f}, median={median:.8f}')
print(f'\nBottom-10 average mean area: {bottom10_mean:.8f}')
print(f'Suggested SMALL_AREA_THRESH: {bottom10_mean:.8f}')
