#!/usr/bin/env python3
"""快速测试数据集加载"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing dataset loading...")
try:
    from sgg.datasets.vg150_dataset import VG150Dataset
    
    data_root = "/home/shi/abschluss/dataset/vg150"
    print(f"Loading dataset from {data_root}...")
    
    dataset = VG150Dataset(
        data_root=data_root,
        split="train",
        max_objects=50,
        max_relations=200,
    )
    
    print(f"✅ Dataset loaded! Total images: {len(dataset)}")
    
    if len(dataset) > 0:
        print("Loading first sample...")
        sample = dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   Image shape: {sample['image'].shape}")
        print(f"   GT boxes: {sample['gt_boxes'].shape}")
        print(f"   GT rels: {sample['gt_rels'].shape}")
        print(f"   Image ID: {sample['image_id']}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
