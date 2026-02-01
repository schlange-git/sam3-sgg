import torch
import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

# Load the model
# 代码会自动使用 ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/main/ 中的本地文件
# 无需网络连接，无需指定路径
print("正在加载模型（使用本地缓存文件）...")
model = build_sam3_image_model()
print("模型加载完成！")

# 降低confidence_threshold以检测更多对象
# 默认值是0.5，如果检测不到对象，可以降低这个值
processor = Sam3Processor(model, confidence_threshold=0.3)
# Load an image
# 使用相对路径，从当前脚本位置查找数据集
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # sam3 目录的父目录（项目根目录）
# 尝试多个可能的数据集路径
possible_paths = [
    os.path.join(project_root, "..", "..", "dataset", "visual genome", "images", "VG_100K", "2.jpg"),
    os.path.join(project_root, "..", "..", "..", "dataset", "vg150", "images", "VG_100K", "2.jpg"),
    os.path.join(os.path.expanduser("~"), "桌面", "abschluss", "sgg", "dataset", "vg150", "images", "VG_100K", "2.jpg"),
]
image_path = None
for path in possible_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        image_path = abs_path
        break
if image_path is None:
    # 如果都找不到，使用第一个路径（用户需要自己修改）
    image_path = possible_paths[0]
    print(f"⚠️ 警告: 使用默认路径 {image_path}，如果不存在请修改脚本")
image = Image.open(image_path)
inference_state = processor.set_image(image)
# 保存结果到 results 文件夹（相对于项目根目录）
results_dir = os.path.join(project_root, "results", "visual_genome")
os.makedirs(results_dir, exist_ok=True)




# Prompt the model with text
text_prompt = "person"  # 修改为您想要的提示词
print(f"文本提示: {text_prompt}")
print(f"置信度阈值: {processor.confidence_threshold}")
output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(f"检测到 {len(scores)} 个对象")

# 如果检测不到对象，尝试降低置信度阈值
if len(scores) == 0:
    print("\n⚠️ 未检测到对象，尝试降低置信度阈值...")
    processor.confidence_threshold = 0.05
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    print(f"降低阈值后检测到 {len(scores)} 个对象")
    if len(scores) > 0:
        print(f"置信度分数: {[f'{s.item():.3f}' for s in scores]}")



if len(scores) > 0:
    # 生成时间戳文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    prompt_safe = text_prompt.replace(" ", "_").replace("/", "_")
    
    # 1. 保存可视化图像
    output_image_path = os.path.join(results_dir, f"{image_name}_{prompt_safe}_{timestamp}.png")
    plt.figure(figsize=(12, 8))
    plot_results(image, output)
    plt.title(f"文本提示: {text_prompt} | 检测到 {len(scores)} 个对象")
    plt.savefig(output_image_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ 可视化结果已保存到: {output_image_path}")
    
    # 2. 保存JSON数据
    output_json_path = os.path.join(results_dir, f"{image_name}_{prompt_safe}_{timestamp}.json")
    result_data = {
        "text_prompt": text_prompt,
        "image_path": image_path,
        "confidence_threshold": processor.confidence_threshold,
        "num_objects": len(scores),
        "objects": []
    }
    
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        result_data["objects"].append({
            "id": i,
            "score": score.item(),
            "box": box.cpu().tolist(),
            "mask_shape": list(mask.shape),
        })
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 结果数据已保存到: {output_json_path}")
    
    # 3. 保存单独的mask图像（可选）
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask_image_path = os.path.join(results_dir, f"{image_name}_{prompt_safe}_{timestamp}_mask_{i+1}_score_{score.item():.3f}.png")
        # 将mask转换为PIL图像
        # mask可能是3D的 [1, H, W] 或 [H, W]，需要先squeeze掉多余的维度
        mask_np = mask.cpu().numpy()
        # 移除所有大小为1的维度
        while len(mask_np.shape) > 2:
            mask_np = np.squeeze(mask_np, axis=0)
        # 确保是2D的 [H, W]
        if len(mask_np.shape) != 2:
            raise ValueError(f"Mask shape should be 2D after squeezing, got {mask_np.shape}")
        # 转换为uint8并缩放到0-255
        mask_np = (mask_np.astype(np.float32) * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_np, mode='L')
        mask_img.save(mask_image_path)
    
    print(f"✅ 所有结果已保存到: {results_dir}")
else:
    print(f"\n⚠️ 未检测到对象，未保存结果")