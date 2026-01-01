"""
测试 VG150 数据集加载
"""
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from sgg.datasets.vg150_dataset import VG150Dataset
    print("✅ Dataset import successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# 测试数据集路径
data_root = "/home/shi/abschluss/dataset/vg150"

if not os.path.exists(data_root):
    print(f"❌ Data root not found: {data_root}")
    sys.exit(1)

# 检查必要文件
required_files = [
    "VG-SGG-dicts-with-attri.json",
    "VG-SGG-with-attri.h5",
    "image_data.json",
]

print("\n检查必要文件:")
for fname in required_files:
    fpath = os.path.join(data_root, fname)
    if os.path.exists(fpath):
        print(f"  ✅ {fname}")
    else:
        print(f"  ❌ {fname} (not found)")

# 检查图像目录
image_dirs = ["images", "images2"]
print("\n检查图像目录:")
for img_dir in image_dirs:
    img_path = os.path.join(data_root, img_dir)
    if os.path.exists(img_path):
        num_files = len([f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        print(f"  ✅ {img_dir}/ ({num_files} images)")
    else:
        print(f"  ❌ {img_dir}/ (not found)")

# 尝试加载数据集
print("\n尝试加载数据集...")
try:
    dataset = VG150Dataset(
        data_root=data_root,
        split="train",
        max_objects=50,
        max_relations=200,
    )
    print(f"✅ Dataset loaded successfully!")
    print(f"   - Total images: {len(dataset)}")
    print(f"   - Number of predicates: {dataset.num_predicates}")
    
    # 尝试加载一个样本
    if len(dataset) > 0:
        print("\n尝试加载第一个样本...")
        sample = dataset[0]
        print(f"  ✅ Sample loaded:")
        print(f"     - Image shape: {sample['image'].shape}")
        print(f"     - GT boxes: {sample['gt_boxes'].shape}")
        print(f"     - GT relations: {sample['gt_rels'].shape}")
        print(f"     - Image ID: {sample['image_id']}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All tests passed!")
