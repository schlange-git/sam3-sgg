"""
测试真实的 boxes_512 转换流程
模拟 VG150 数据集的实际转换过程
参考官方demo的可视化方式
"""
import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import h5py

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sgg.datasets.vg150_dataset import VG150Dataset
from sgg.datasets.vg150_reader import VG150Reader


def draw_single_box(pic, box, color=(255, 0, 255, 128)):
    """
    参考官方demo：在PIL图像上绘制单个边界框
    Args:
        pic: PIL Image对象
        box: [x1, y1, x2, y2] 格式的边界框
        color: 颜色，RGBA格式
    """
    draw = ImageDraw.Draw(pic)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)


def draw_boxes(image, boxes):
    """
    参考官方demo：在图像上绘制所有边界框
    Args:
        image: PIL Image对象或图像路径
        boxes: [N, 4] 格式的边界框数组，xyxy格式
    Returns:
        PIL Image对象
    """
    if isinstance(image, str):
        pic = Image.open(image)
    else:
        pic = image.copy()
    
    num_obj = boxes.shape[0]
    for i in range(num_obj):
        draw_single_box(pic, boxes[i])
    return pic


def convert_box_1024_to_xyxy(box_1024, height, width, USE_BOX_SIZE=1024):
    """
    参考官方demo：将boxes_1024格式（中心坐标+宽高）转换为xyxy格式（像素坐标）
    
    Args:
        box_1024: [cx, cy, w, h] 格式，基于USE_BOX_SIZE坐标系
        height: 原始图像高度
        width: 原始图像宽度
        USE_BOX_SIZE: 使用的box尺寸（官方demo使用1024）
    
    Returns:
        [x1, y1, x2, y2] 格式的边界框（像素坐标）
    """
    box = box_1024.copy()
    # 转换为x1,y1,x2,y2格式（仍在USE_BOX_SIZE坐标系中）
    box[:2] = box[:2] - box[2:] / 2
    box[2:] = box[:2] + box[2:]
    # 转换回原始图像尺寸
    box = box.astype(np.float32) / USE_BOX_SIZE * max(height, width)
    return box


def test_real_image_conversion():
    """
    测试真实图像的边界框转换
    """
    print("=" * 60)
    print("Testing Real Image Box Conversion")
    print("=" * 60)
    
    data_root = "/home/shi/abschluss/dataset/vg150"
    
    # 1. 从数据集加载一个样本
    print("\n1. Loading sample from VG150Dataset...")
    ds = VG150Dataset(data_root, split="train", image_size=1008, max_objects=50, max_relations=200)
    
    if len(ds) == 0:
        print("   Error: No samples in dataset!")
        return
    
    sample = ds[0]
    image_id = sample["image_id"]
    gt_boxes_norm = sample["gt_boxes"].numpy()  # 归一化坐标
    image_pil = sample["image_pil"]
    orig_w, orig_h = image_pil.size
    
    print(f"   Image ID: {image_id}")
    print(f"   Image size: {orig_w} x {orig_h}")
    print(f"   Number of boxes: {len(gt_boxes_norm)}")
    
    # 2. 从 VG150Reader 获取边界框（像素坐标）
    print("\n2. Loading sample from VG150Reader...")
    reader = VG150Reader(data_root, split="train")
    
    # 找到对应的样本
    reader_sample = None
    for s in reader.iter_samples():
        if s.image_id == image_id:
            reader_sample = s
            break
    
    if reader_sample is None:
        print("   Error: Could not find sample in VG150Reader!")
        return
    
    boxes_pixel_from_reader = reader_sample.boxes_xyxy
    print(f"   Boxes from VG150Reader (pixel coords):")
    print(f"     Shape: {boxes_pixel_from_reader.shape}")
    print(f"     First 3 boxes:")
    for i, box in enumerate(boxes_pixel_from_reader[:3]):
        print(f"       Box {i}: {box}")
    
    # 3. 从归一化坐标转换回像素坐标
    print("\n3. Converting normalized -> pixel...")
    boxes_pixel_from_norm = []
    for box_norm in gt_boxes_norm:
        x1_norm, y1_norm, x2_norm, y2_norm = box_norm
        x1_px = x1_norm * orig_w
        y1_px = y1_norm * orig_h
        x2_px = x2_norm * orig_w
        y2_px = y2_norm * orig_h
        boxes_pixel_from_norm.append([x1_px, y1_px, x2_px, y2_px])
    
    boxes_pixel_from_norm = np.array(boxes_pixel_from_norm)
    print(f"   Converted boxes (pixel coords):")
    print(f"     First 3 boxes:")
    for i, box in enumerate(boxes_pixel_from_norm[:3]):
        print(f"       Box {i}: {box}")
    
    # 4. 比较两种方式得到的像素坐标
    print("\n4. Comparing pixel coordinates...")
    if len(boxes_pixel_from_reader) != len(boxes_pixel_from_norm):
        print(f"   Warning: Different number of boxes!")
        print(f"     Reader: {len(boxes_pixel_from_reader)}")
        print(f"     Norm->Pixel: {len(boxes_pixel_from_norm)}")
        min_len = min(len(boxes_pixel_from_reader), len(boxes_pixel_from_norm))
        boxes_pixel_from_reader = boxes_pixel_from_reader[:min_len]
        boxes_pixel_from_norm = boxes_pixel_from_norm[:min_len]
    
    errors = np.abs(boxes_pixel_from_reader - boxes_pixel_from_norm)
    max_error = errors.max()
    mean_error = errors.mean()
    
    print(f"   Max error: {max_error:.2f} pixels")
    print(f"   Mean error: {mean_error:.2f} pixels")
    
    if max_error < 1.0:
        print("   ✓ Coordinates match!")
    else:
        print("   ✗ Coordinates don't match!")
        print(f"\n   First 3 boxes comparison:")
        for i in range(min(3, len(boxes_pixel_from_reader))):
            print(f"     Box {i}:")
            print(f"       Reader:    {boxes_pixel_from_reader[i]}")
            print(f"       Norm->Px:  {boxes_pixel_from_norm[i]}")
            print(f"       Error:     {errors[i]}")
    
    # 5. 可视化（参考官方demo的方式）
    print("\n5. Creating visualization...")
    test_dir = "sgg/test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # 方法1：使用PIL绘制（参考官方demo）
    print("   5.1 Using PIL visualization (official demo style)...")
    pic_with_boxes = draw_boxes(image_pil, boxes_pixel_from_reader[:10])  # 只显示前10个
    pil_vis_path = os.path.join(test_dir, f"real_box_comparison_pil_{image_id}.png")
    pic_with_boxes.save(pil_vis_path)
    print(f"      Saved PIL visualization to: {pil_vis_path}")
    
    # 方法2：使用matplotlib绘制（对比）
    print("   5.2 Using matplotlib visualization (comparison)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左侧：从 Reader 获取的像素坐标
    axes[0].imshow(image_pil)
    axes[0].set_title("Boxes from VG150Reader (Pixel Coords)", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    colors = plt.cm.tab20(np.linspace(0, 1, len(boxes_pixel_from_reader)))
    for i, (x1, y1, x2, y2) in enumerate(boxes_pixel_from_reader[:10]):  # 只显示前10个
        color = colors[i]
        rect = FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="round,pad=2",
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 5, f"O{i}", fontsize=8, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # 右侧：从归一化坐标转换的像素坐标
    axes[1].imshow(image_pil)
    axes[1].set_title("Boxes from Normalized Coords (Converted)", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    for i, (x1, y1, x2, y2) in enumerate(boxes_pixel_from_norm[:10]):  # 只显示前10个
        color = colors[i]
        rect = FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="round,pad=2",
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            linestyle="--",  # 虚线
        )
        axes[1].add_patch(rect)
        axes[1].text(x1, y1 - 5, f"O{i}", fontsize=8, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    vis_path = os.path.join(test_dir, f"real_box_comparison_{image_id}.png")
    plt.tight_layout()
    plt.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved matplotlib visualization to: {vis_path}")
    
    # 6. 检查原始 boxes_512 的值（参考官方demo的转换方式）
    print("\n6. Checking raw boxes_512 values and conversion...")
    try:
        h5_path = os.path.join(data_root, "VG-SGG-with-attri.h5")
        with h5py.File(h5_path, 'r') as f:
            img_to_first_box = f['img_to_first_box']
            img_to_last_box = f['img_to_last_box']
            
            # 找到图像索引（假设 image_id 就是索引）
            img_idx = image_id
            if img_idx < len(img_to_first_box):
                first_box = int(img_to_first_box[img_idx])
                last_box = int(img_to_last_box[img_idx])
                
                if last_box >= first_box:
                    # 检查是否有boxes_512或boxes_1024
                    if 'boxes_512' in f:
                        boxes_raw = f['boxes_512'][first_box:last_box+1]
                        box_key = 'boxes_512'
                        USE_BOX_SIZE = 512
                    elif 'boxes_1024' in f:
                        boxes_raw = f['boxes_1024'][first_box:last_box+1]
                        box_key = 'boxes_1024'
                        USE_BOX_SIZE = 1024
                    else:
                        print("   Could not find boxes_512 or boxes_1024 in h5 file")
                        return
                    
                    print(f"   Raw {box_key} (first 3):")
                    for i, box in enumerate(boxes_raw[:3]):
                        print(f"     Box {i}: {box.tolist()} (format: [cx, cy, w, h])")
                    
                    print(f"   Raw {box_key} value range:")
                    print(f"     Min: {boxes_raw.min()}, Max: {boxes_raw.max()}")
                    
                    # 使用官方demo的方式转换第一个box
                    if len(boxes_raw) > 0:
                        print(f"\n   Converting first box using official demo method:")
                        box_1024 = boxes_raw[0].copy()
                        print(f"     Original (cx,cy,w,h): {box_1024.tolist()}")
                        
                        # 官方demo的转换方式
                        box_xyxy = box_1024.copy()
                        box_xyxy[:2] = box_xyxy[:2] - box_xyxy[2:] / 2
                        box_xyxy[2:] = box_xyxy[:2] + box_xyxy[2:]
                        print(f"     After conversion to xyxy (in {USE_BOX_SIZE} coord): {box_xyxy.tolist()}")
                        
                        # 转换回原始图像尺寸
                        box_pixel = box_xyxy.astype(np.float32) / USE_BOX_SIZE * max(orig_h, orig_w)
                        print(f"     After scaling to pixel coords: {box_pixel.tolist()}")
                        print(f"     Image size: {orig_w} x {orig_h}, max={max(orig_h, orig_w)}")
    except Exception as e:
        print(f"   Could not check raw boxes: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_real_image_conversion()

