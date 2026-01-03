"""
Fast Testing Script for Cached Pairs
支持评测和可视化
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sgg.datasets.cached_pairs import CachedPairDataset, collate_cached
from sgg.datasets.vg150_reader import VG150Reader
from sgg.models.relation_head_geom import RelationHeadMLP
from sgg.eval.sgg_metrics import evaluate_predcls_batch
from sgg.utils.seed import seed_everything
from sgg.utils.io import ensure_dir


@dataclass
class TestConfig:
    cache_dir: str
    checkpoint_path: str
    data_root: str
    split: str = "val"
    batch_size: int = 32
    num_workers: int = 4
    amp: bool = True
    enable_vis: bool = False
    vis_dir: Optional[str] = None
    vis_num_samples: int = 10
    k_list: List[int] = None
    seed: int = 42


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[RelationHeadMLP, int, int]:
    """
    加载模型
    
    Returns:
        (model, in_dim, num_predicates)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    in_dim = ckpt.get("in_dim", 11)  # 默认 11 (6 box + 5 mask)
    num_predicates = ckpt.get("num_predicates", 51)  # 默认 51 (50 + bg)
    
    model = RelationHeadMLP(in_dim=in_dim, num_classes=num_predicates).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  in_dim: {in_dim}")
    print(f"  num_predicates: {num_predicates}")
    
    return model, in_dim, num_predicates


def predict_batch(
    model: RelationHeadMLP,
    geom: torch.Tensor,
    mask: torch.Tensor,
    pair_idx: torch.Tensor,
    num_obj: int,
    top_k: int = 100,
    amp: bool = True,
) -> Tuple[List[Tuple[int, int, int]], List[float]]:
    """
    对 batch 进行预测
    
    Args:
        model: 模型
        geom: [B, P, Gd] 几何特征
        mask: [B, P] 有效对掩码
        pair_idx: [B, P, 2] 物体对索引
        num_obj: 每个图像的物体数量（batch 内可能不同，这里假设相同）
        top_k: 返回前 K 个预测
        amp: 是否使用 AMP
        
    Returns:
        (pred_triplets_list, pred_scores_list) 每个图像的预测列表
    """
    B, P, Gd = geom.shape
    device = geom.device
    
    # Flatten
    x = geom.view(B * P, Gd)
    
    with torch.no_grad(), autocast(device_type=device.type, enabled=amp):
        logits = model(x)  # [B*P, C]
        probs = torch.softmax(logits, dim=-1)  # [B*P, C]
    
    # 重塑回 batch 形状
    logits = logits.view(B, P, -1)  # [B, P, C]
    probs = probs.view(B, P, -1)    # [B, P, C]
    
    pred_triplets_list = []
    pred_scores_list = []
    
    for b in range(B):
        # 获取该图像的预测
        logits_b = logits[b]  # [P, C]
        probs_b = probs[b]   # [P, C]
        mask_b = mask[b]      # [P]
        pair_idx_b = pair_idx[b]  # [P, 2]
        
        # 只考虑有效对
        valid_mask = mask_b.bool()
        if not valid_mask.any():
            pred_triplets_list.append([])
            pred_scores_list.append([])
            continue
        
        # 获取有效对的预测
        valid_logits = logits_b[valid_mask]  # [V, C]
        valid_probs = probs_b[valid_mask]   # [V, C]
        valid_pair_idx = pair_idx_b[valid_mask]  # [V, 2]
        
        # 获取每个对的最大概率和对应的 predicate
        max_probs, pred_ids = torch.max(valid_probs, dim=-1)  # [V]
        
        # 构建三元组列表
        triplets = []
        scores = []
        for i in range(len(valid_pair_idx)):
            s_idx = int(valid_pair_idx[i, 0])
            o_idx = int(valid_pair_idx[i, 1])
            pred_id = int(pred_ids[i])
            score = float(max_probs[i])
            
            # 跳过 background (pred_id=0) 和无效索引
            if pred_id > 0 and s_idx >= 0 and o_idx >= 0:
                triplets.append((s_idx, o_idx, pred_id))
                scores.append(score)
        
        # 按分数排序，取前 top_k
        if len(triplets) > 0:
            combined = list(zip(triplets, scores))
            combined.sort(key=lambda x: x[1], reverse=True)
            triplets, scores = zip(*combined[:top_k])
            pred_triplets_list.append(list(triplets))
            pred_scores_list.append(list(scores))
        else:
            pred_triplets_list.append([])
            pred_scores_list.append([])
    
    return pred_triplets_list, pred_scores_list


def get_gt_triplets_from_h5(
    reader,
    image_id: int,
) -> List[Tuple[int, int, int]]:
    """
    直接从h5文件获取GT三元组（完全复用官方demo的方式）
    使用与boxes相同的索引映射，确保对齐
    
    Args:
        reader: VG150Reader实例
        image_id: 图像ID
        
    Returns:
        GT三元组列表 [(s_local, o_local, p), ...]，使用local索引（0到N-1）
    """
    # 直接从reader获取sample，使用与可视化相同的逻辑
    sample = reader.get_sample_by_image_id(image_id)
    if sample is None:
        return []
    
    # sample.rels已经是local索引了（在get_sample_by_image_id中已转换）
    # 格式：[R, 3] (s_local, o_local, pred_id)
    gt_triplets = []
    num_boxes = len(sample.boxes_xyxy)
    for rel in sample.rels:
        if len(rel) >= 3:
            s_local = int(rel[0])
            o_local = int(rel[1])
            pred_id = int(rel[2])
            # 验证索引范围
            if s_local >= 0 and s_local < num_boxes and o_local >= 0 and o_local < num_boxes and pred_id > 0:
                gt_triplets.append((s_local, o_local, pred_id))
    
    return gt_triplets


def get_gt_triplets_from_cache(
    cache_file: str,
    image_id: int,
) -> List[Tuple[int, int, int]]:
    """
    从缓存文件中获取 GT 三元组（已废弃，保留用于兼容）
    现在应该使用get_gt_triplets_from_h5直接从h5文件读取
    
    Args:
        cache_file: 缓存文件路径
        image_id: 图像 ID（用于验证）
        
    Returns:
        GT 三元组列表 [(s, o, p), ...]
    """
    pack = torch.load(cache_file, map_location="cpu", weights_only=False)
    
    # 验证 image_id
    assert pack["image_id"] == image_id, f"Image ID mismatch: {pack['image_id']} != {image_id}"
    
    pair_idx = pack["pair_idx"]  # [P, 2]
    pair_label = pack["pair_label"]  # [P]
    pair_mask = pack["pair_mask"]  # [P]
    
    # 构建 GT 三元组（只取正样本，label > 0）
    gt_triplets = []
    for i in range(len(pair_mask)):
        if pair_mask[i] and pair_label[i] > 0:
            s_idx = int(pair_idx[i, 0])
            o_idx = int(pair_idx[i, 1])
            pred_id = int(pair_label[i])
            if s_idx >= 0 and o_idx >= 0:
                gt_triplets.append((s_idx, o_idx, pred_id))
    
    return gt_triplets


def calculate_box_iou(box1, box2):
    """计算两个box的IOU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def find_non_overlapping_label_positions(boxes_xyxy, label_positions, min_distance=30):
    """
    找到不重叠的标签位置
    Args:
        boxes_xyxy: 边界框数组
        label_positions: 初始标签位置列表 [(x, y), ...]
        min_distance: 最小距离阈值
    Returns:
        调整后的标签位置列表
    """
    adjusted_positions = []
    label_boxes = []  # 每个标签的虚拟box（用于IOU计算）
    
    for i, (x, y) in enumerate(label_positions):
        # 创建标签的虚拟box（假设标签大小约为40x20）
        label_w, label_h = 40, 20
        label_box = [x - label_w/2, y - label_h/2, x + label_w/2, y + label_h/2]
        
        # 检查与物体box的IOU
        overlaps = False
        for box in boxes_xyxy:
            iou = calculate_box_iou(label_box, box)
            if iou > 0.1:  # 如果IOU > 0.1，认为重叠
                overlaps = True
                break
        
        # 检查与其他标签的距离
        if not overlaps:
            for other_pos, other_box in zip(adjusted_positions, label_boxes):
                dist = np.sqrt((x - other_pos[0])**2 + (y - other_pos[1])**2)
                if dist < min_distance:
                    # 计算两个标签box的IOU
                    iou = calculate_box_iou(label_box, other_box)
                    if iou > 0.1:
                        overlaps = True
                        break
        
        # 如果重叠，尝试调整位置
        if overlaps:
            # 尝试向上移动
            for offset_y in range(10, 50, 5):
                new_y = y - offset_y
                new_label_box = [x - label_w/2, new_y - label_h/2, x + label_w/2, new_y + label_h/2]
                new_overlaps = False
                for box in boxes_xyxy:
                    if calculate_box_iou(new_label_box, box) > 0.1:
                        new_overlaps = True
                        break
                if not new_overlaps:
                    adjusted_positions.append((x, new_y))
                    label_boxes.append(new_label_box)
                    break
            else:
                # 如果向上移动不行，尝试向右移动
                for offset_x in range(10, 50, 5):
                    new_x = x + offset_x
                    new_label_box = [new_x - label_w/2, y - label_h/2, new_x + label_w/2, y + label_h/2]
                    new_overlaps = False
                    for box in boxes_xyxy:
                        if calculate_box_iou(new_label_box, box) > 0.1:
                            new_overlaps = True
                            break
                    if not new_overlaps:
                        adjusted_positions.append((new_x, y))
                        label_boxes.append(new_label_box)
                        break
                else:
                    # 如果都不行，使用原位置
                    adjusted_positions.append((x, y))
                    label_boxes.append(label_box)
        else:
            adjusted_positions.append((x, y))
            label_boxes.append(label_box)
    
    return adjusted_positions


def visualize_scene_graph(
    image_path: str,
    boxes_xyxy: np.ndarray,  # [N, 4]
    obj_labels: np.ndarray,   # [N] (可选，用于显示)
    pred_triplets: List[Tuple[int, int, int]],
    gt_triplets: List[Tuple[int, int, int]],
    idx_to_predicate: Dict[int, str],
    save_path: str,
    title: str = "Scene Graph",
):
    """
    可视化场景图
    参考官方demo的可视化方式，使用PIL绘制边界框
    优化：避免标签重叠，relation链接标签，On标签和框颜色一致
    
    Args:
        image_path: 图像路径
        boxes_xyxy: 边界框 [N, 4] (x1, y1, x2, y2) in pixel coords
        obj_labels: 物体标签 [N] (可选)
        pred_triplets: 预测的三元组 [(s, o, p), ...]，s和o是local索引（0到N-1）
        gt_triplets: GT 三元组 [(s, o, p), ...]，s和o是local索引（0到N-1）
        idx_to_predicate: Predicate ID 到名称的映射
        save_path: 保存路径
        title: 标题
    """
    # 验证索引范围
    num_boxes = len(boxes_xyxy)
    for s, o, p in pred_triplets:
        if s >= num_boxes or o >= num_boxes or s < 0 or o < 0:
            print(f"Warning: Invalid pred_triplet index: ({s}, {o}, {p}), num_boxes={num_boxes}")
    for s, o, p in gt_triplets:
        if s >= num_boxes or o >= num_boxes or s < 0 or o < 0:
            print(f"Warning: Invalid gt_triplet index: ({s}, {o}, {p}), num_boxes={num_boxes}")
    
    # 加载图像（完全复用官方demo的方式）
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    
    # 参考官方demo：使用PIL绘制边界框
    from PIL import ImageDraw
    pic = img.copy()
    draw = ImageDraw.Draw(pic)
    
    # 绘制边界框（完全复用官方demo的draw_single_box方式）
    # 使用与标签相同的颜色
    colors_rgb = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 洋红
        (0, 255, 255),  # 青色
        (255, 128, 0),  # 橙色
        (128, 0, 255),  # 紫色
        (255, 192, 203), # 粉色
        (128, 128, 128), # 灰色
    ]
    
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # 完全复用官方demo的坐标验证逻辑
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        
        # 确保x1 < x2, y1 < y2
        if x1 >= x2:
            x2 = x1 + 1
        if y1 >= y2:
            y2 = y1 + 1
        
        color = colors_rgb[i % len(colors_rgb)]
        # 降低线宽到一半（从3到1.5，取整为1）
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=1)
    
    # 转换为numpy数组用于matplotlib显示
    img_array = np.array(pic)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img_array)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    # 计算物体标签位置（避免重叠）
    label_positions = []
    label_colors = []
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        label_positions.append((x1, y1 - 5))
        # 使用与框相同的颜色
        color_rgb = colors_rgb[i % len(colors_rgb)]
        # 转换为matplotlib颜色格式（0-1范围）
        color = tuple(c/255.0 for c in color_rgb)
        label_colors.append(color)
    
    # 找到不重叠的标签位置（全局IOU检查，包括物体和关系标签）
    adjusted_label_positions = find_non_overlapping_label_positions(
        boxes_xyxy, label_positions, min_distance=30
    )
    
    # 存储标签位置用于后续IOU检查
    all_label_boxes = []
    for (x, y) in adjusted_label_positions:
        label_w, label_h = 40, 20
        all_label_boxes.append([x - label_w/2, y - label_h/2, x + label_w/2, y + label_h/2])
    
    # 添加物体索引标签（使用matplotlib，避免重叠，颜色与框一致）
    for i, ((x, y), color) in enumerate(zip(adjusted_label_positions, label_colors)):
        # 添加物体索引标签，颜色与框一致
        ax.text(x, y, f"O{i}", fontsize=10, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 绘制关系（预测和 GT）
    # 预测关系用实线，GT 用虚线
    pred_set = set(pred_triplets)
    gt_set = set(gt_triplets)
    correct_set = pred_set & gt_set
    false_pos_set = pred_set - gt_set
    false_neg_set = gt_set - pred_set
    
    # 收集所有关系标签位置，避免重叠
    rel_label_positions = []
    rel_labels = []
    
    def get_label_position(box_idx):
        """获取标签位置（用于relation链接）"""
        if box_idx < len(adjusted_label_positions):
            return adjusted_label_positions[box_idx]
        else:
            # 如果索引超出范围，使用box中心
            x1, y1, x2, y2 = boxes_xyxy[box_idx]
            return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # 绘制正确预测的关系（绿色实线）- 链接标签
    for s, o, p in correct_set:
        if s < len(adjusted_label_positions) and o < len(adjusted_label_positions):
            # 获取标签位置
            x_s, y_s = get_label_position(s)
            x_o, y_o = get_label_position(o)
            
            pred_name = idx_to_predicate.get(p, f"P{p}")
            ax.plot([x_s, x_o], [y_s, y_o], "g-", linewidth=2, alpha=0.7)
            
            # 标签位置（中点）
            label_x, label_y = (x_s + x_o) / 2, (y_s + y_o) / 2
            rel_label_positions.append((label_x, label_y))
            rel_labels.append((pred_name, "green", 9))
    
    # 绘制误报（FP，使用黄色虚线，不那么显眼）
    for s, o, p in false_pos_set:
        if s < len(adjusted_label_positions) and o < len(adjusted_label_positions):
            x_s, y_s = get_label_position(s)
            x_o, y_o = get_label_position(o)
            
            pred_name = idx_to_predicate.get(p, f"P{p}")
            ax.plot([x_s, x_o], [y_s, y_o], "y--", linewidth=1.5, alpha=0.5)
            
            label_x, label_y = (x_s + x_o) / 2, (y_s + y_o) / 2
            rel_label_positions.append((label_x, label_y))
            rel_labels.append((pred_name, "orange", 8))
    
    # 绘制漏报（FN，使用红色实线，更显眼，这个指标更重要）
    for s, o, p in false_neg_set:
        if s < len(adjusted_label_positions) and o < len(adjusted_label_positions):
            x_s, y_s = get_label_position(s)
            x_o, y_o = get_label_position(o)
            
            pred_name = idx_to_predicate.get(p, f"P{p}")
            ax.plot([x_s, x_o], [y_s, y_o], "r-", linewidth=2, alpha=0.8)
            
            label_x, label_y = (x_s + x_o) / 2, (y_s + y_o) / 2
            rel_label_positions.append((label_x, label_y))
            rel_labels.append((pred_name, "red", 9))
    
    # 绘制关系标签（全局IOU检查，包括物体标签）
    if rel_label_positions:
        # 将物体标签box也加入检查
        all_boxes_for_iou = list(boxes_xyxy) + all_label_boxes
        adjusted_rel_positions = find_non_overlapping_label_positions(
            all_boxes_for_iou, rel_label_positions, min_distance=40
        )
        # 更新all_label_boxes以包含关系标签
        for (x, y) in adjusted_rel_positions:
            label_w, label_h = 50, 20  # 关系标签可能更宽
            all_label_boxes.append([x - label_w/2, y - label_h/2, x + label_w/2, y + label_h/2])
        
        for (x, y), (pred_name, color, fontsize) in zip(adjusted_rel_positions, rel_labels):
            ax.text(x, y, pred_name, fontsize=fontsize, color=color, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 添加图例（FN和FP颜色已交换）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", linewidth=2, label="Correct (TP)"),
        Line2D([0], [0], color="orange", linewidth=1.5, linestyle="--", label="False Positive"),
        Line2D([0], [0], color="red", linewidth=2, linestyle="-", label="False Negative (重要)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def test(cfg: TestConfig) -> None:
    """测试主函数"""
    seed_everything(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    model, in_dim, num_predicates = load_model(cfg.checkpoint_path, device)
    
    # 加载数据集
    ds = CachedPairDataset(cfg.cache_dir)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_cached,
    )
    
    print(f"\nDataset: {len(ds)} samples, {len(dl)} batches")
    
    # 如果需要可视化，加载原始数据
    reader = None
    idx_to_predicate = None
    if cfg.enable_vis:
        reader = VG150Reader(cfg.data_root, split=cfg.split)
        # 获取 predicate 映射
        idx_to_predicate = reader.dataset.idx_to_predicate
        ensure_dir(cfg.vis_dir)
        print(f"Visualization enabled, output to: {cfg.vis_dir}")
    
    # 评测
    all_pred_triplets = []
    all_pred_scores = []
    all_gt_triplets = []
    
    print("\nRunning evaluation...")
    vis_count = 0
    
    for batch_idx, batch in enumerate(dl):
        geom = batch["geom_feat"].to(device, non_blocking=True)   # [B,P,Gd]
        labels = batch["pair_label"].to(device, non_blocking=True) # [B,P]
        mask = batch["pair_mask"].to(device, non_blocking=True)    # [B,P]
        pair_idx = batch.get("pair_idx")  # [B,P,2] (如果缓存中有)
        
        # 如果没有 pair_idx，需要从缓存文件读取
        if pair_idx is None:
            # 从 batch 索引推断缓存文件
            start_idx = batch_idx * cfg.batch_size
            pair_idx_list = []
            for i in range(len(geom)):
                cache_file = ds.files[start_idx + i]
                pack = torch.load(cache_file, map_location="cpu", weights_only=False)
                if "pair_idx" in pack:
                    pair_idx_list.append(torch.from_numpy(pack["pair_idx"]))
                else:
                    # 如果没有 pair_idx，创建一个占位符（不应该发生）
                    P = geom.shape[1]
                    pair_idx_list.append(torch.full((P, 2), -1, dtype=torch.long))
            pair_idx = torch.stack(pair_idx_list, dim=0).to(device)
        else:
            # 确保 pair_idx 在正确的设备上
            pair_idx = pair_idx.to(device, non_blocking=True)
        
        # 预测
        pred_triplets_list, pred_scores_list = predict_batch(
            model, geom, mask, pair_idx, num_obj=None, top_k=100, amp=cfg.amp
        )
        
        # 获取 GT
        start_idx = batch_idx * cfg.batch_size
        for i in range(len(geom)):
            cache_file = ds.files[start_idx + i]
            pack = torch.load(cache_file, map_location="cpu", weights_only=False)
            image_id = pack["image_id"]
            
            # 获取 GT 三元组（直接从h5文件读取，使用与boxes相同的索引映射）
            gt_triplets = get_gt_triplets_from_h5(reader, image_id)
            
            all_pred_triplets.append(pred_triplets_list[i])
            all_pred_scores.append(pred_scores_list[i])
            all_gt_triplets.append(gt_triplets)
            
            # 可视化
            if cfg.enable_vis and vis_count < cfg.vis_num_samples:
                # 直接从reader中通过image_id获取样本（避免迭代查找）
                sample = reader.get_sample_by_image_id(image_id)
                
                if sample is not None:
                    # 验证boxes和image的匹配
                    if len(sample.boxes_xyxy) == 0:
                        print(f"  Warning: Image {image_id} has no boxes, skipping visualization")
                    else:
                        # 验证sample的image_id是否匹配
                        if sample.image_id != image_id:
                            print(f"  Error: Sample image_id mismatch! Expected {image_id}, got {sample.image_id}")
                        else:
                            # 验证索引对齐：检查GT triplets的索引是否在有效范围内
                            num_boxes = len(sample.boxes_xyxy)
                            invalid_gt = [t for t in gt_triplets if t[0] >= num_boxes or t[1] >= num_boxes or t[0] < 0 or t[1] < 0]
                            invalid_pred = [t for t in pred_triplets_list[i] if t[0] >= num_boxes or t[1] >= num_boxes or t[0] < 0 or t[1] < 0]
                            
                            if invalid_gt:
                                print(f"  Warning: Image {image_id} has {len(invalid_gt)} invalid GT triplets (out of {len(gt_triplets)} total)")
                                print(f"    Invalid GT triplets: {invalid_gt[:5]}")  # 只显示前5个
                            if invalid_pred:
                                print(f"  Warning: Image {image_id} has {len(invalid_pred)} invalid pred triplets (out of {len(pred_triplets_list[i])} total)")
                                print(f"    Invalid pred triplets: {invalid_pred[:5]}")
                            
                            # 验证：检查cache文件中的num_obj是否匹配
                            cache_num_obj = pack.get("num_obj", -1)
                            if cache_num_obj != num_boxes:
                                print(f"  Warning: Image {image_id} num_obj mismatch: cache={cache_num_obj}, actual={num_boxes}")
                            
                            vis_path = os.path.join(cfg.vis_dir, f"vis_{image_id:08d}.png")
                            visualize_scene_graph(
                                sample.image_path,
                                sample.boxes_xyxy,
                                sample.obj_labels,
                                pred_triplets_list[i],
                                gt_triplets,
                                idx_to_predicate,
                                vis_path,
                                title=f"Image {image_id} (PredCls)",
                            )
                            vis_count += 1
                            print(f"  Visualized image {image_id} (boxes: {len(sample.boxes_xyxy)}, objects: {len(sample.obj_labels)}, "
                                  f"GT rels: {len(gt_triplets)}, Pred rels: {len(pred_triplets_list[i])})")
                else:
                    print(f"  Warning: Could not find sample for image_id {image_id} (may not exist in dataset or index mismatch)")
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{len(dl)} batches")
    
    # 计算评测指标
    print("\nComputing metrics...")
    k_list = cfg.k_list if cfg.k_list else [20, 50, 100]
    results = evaluate_predcls_batch(
        all_pred_triplets,
        all_pred_scores,
        all_gt_triplets,
        num_predicates,
        k_list=k_list,
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Evaluation Results (PredCls)")
    print("=" * 60)
    for k in k_list:
        print(f"  R@{k}:  {results[f'R@{k}']:.2f}%")
        print(f"  mR@{k}: {results[f'mR@{k}']:.2f}%")
    print("=" * 60)
    
    # 保存结果
    result_path = os.path.join(os.path.dirname(cfg.checkpoint_path), "eval_results.txt")
    with open(result_path, "w") as f:
        f.write("Evaluation Results (PredCls)\n")
        f.write("=" * 60 + "\n")
        for k in k_list:
            f.write(f"R@{k}:  {results[f'R@{k}']:.2f}%\n")
            f.write(f"mR@{k}: {results[f'mR@{k}']:.2f}%\n")
        f.write("=" * 60 + "\n")
    print(f"\nResults saved to: {result_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast testing with cached pairs")
    p.add_argument("--cache_dir", type=str, required=True, help="Cache directory")
    p.add_argument("--checkpoint_path", type=str, required=True, help="Model checkpoint path")
    p.add_argument("--data_root", type=str, required=True, help="VG150 data root")
    p.add_argument("--split", type=str, default="val", help="Split: train or val")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    p.add_argument("--amp", action="store_true", help="Use AMP")
    p.add_argument("--no_amp", action="store_true", help="Disable AMP")
    p.add_argument("--enable_vis", action="store_true", help="Enable visualization")
    p.add_argument("--vis_dir", type=str, default=None, help="Visualization output directory")
    p.add_argument("--vis_num_samples", type=int, default=10, help="Number of samples to visualize")
    p.add_argument("--k_list", type=int, nargs="+", default=[20, 50, 100], help="K values for R@K and mR@K")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    amp = True
    if args.no_amp:
        amp = False
    elif args.amp:
        amp = True
    
    if args.enable_vis and args.vis_dir is None:
        args.vis_dir = os.path.join(os.path.dirname(args.checkpoint_path), "visualizations")
    
    cfg = TestConfig(
        cache_dir=args.cache_dir,
        checkpoint_path=args.checkpoint_path,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=amp,
        enable_vis=args.enable_vis,
        vis_dir=args.vis_dir,
        vis_num_samples=args.vis_num_samples,
        k_list=args.k_list,
        seed=args.seed,
    )
    test(cfg)

