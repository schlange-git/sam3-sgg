#!/usr/bin/env python3
"""快速测试数据集加载"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing dataset loading...")
try:
    from sgg.datasets.vg150_dataset import VG150Dataset
    
    # 使用相对路径，从当前脚本位置查找数据集
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))  # 回到项目根目录
    # 尝试多个可能的数据集路径
    possible_paths = [
        os.path.join(project_root, "..", "..", "dataset", "vg150"),
        os.path.join(project_root, "..", "..", "..", "dataset", "vg150"),
        os.path.join(os.path.expanduser("~"), "桌面", "abschluss", "sgg", "dataset", "vg150"),
    ]
    data_root = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            data_root = abs_path
            break
    if data_root is None:
        # 如果都找不到，使用第一个路径（用户需要自己修改）
        data_root = os.path.abspath(possible_paths[0])
        print(f"⚠️ 警告: 使用默认路径 {data_root}，如果不存在请修改脚本")
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
