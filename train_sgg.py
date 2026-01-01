"""
Training script for Scene Graph Generation with SAM3
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import threading
import time
import torch.nn.functional as F

from datasets.vg_dataset import VGDataset
from models.frozen_sam3 import FrozenSAM3
from models.relation_head import RelationHead
from utils.matching import mask_to_box, match_by_iou
from utils.geometry import box_geom_feat


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Paper: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=501, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
        
        # Create alpha tensor: [num_classes]
        if isinstance(alpha, (float, int)):
            self.alpha_tensor = torch.ones(num_classes) * alpha
            self.alpha_tensor[0] = 1.0 - alpha  # Background class
        else:
            self.alpha_tensor = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        """
        inputs: [N, num_classes] logits
        targets: [N] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [N]
        pt = torch.exp(-ce_loss)  # p_t: probability of true class
        
        # Get alpha for each sample
        alpha_t = self.alpha_tensor[targets].to(inputs.device)
        
        # Focal loss: alpha_t * (1 - pt)^gamma * ce_loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def build_pair_labels(num_obj: int, gt_rels: torch.Tensor, bg_class: int = 0):
    """
    Build pair label matrix
    
    Args:
        num_obj: Number of objects
        gt_rels: [R, 3] (s_idx, o_idx, pred_id)
        bg_class: Background class index (usually 0)
        
    Returns:
        label_mat: [num_obj, num_obj] with labels
        pos_pairs: List of (i, j) positive pairs
    """
    label_mat = torch.zeros((num_obj, num_obj), dtype=torch.long) + bg_class
    pos_pairs = []
    
    for s, o, p in gt_rels.tolist():
        if s == o or s >= num_obj or o >= num_obj:
            continue
        label_mat[s, o] = p
        pos_pairs.append((s, o))
    
    return label_mat, pos_pairs


def sample_pairs(label_mat: torch.Tensor, pos_pairs: list, neg_ratio: int = 3):
    """
    Sample positive and negative pairs
    
    Args:
        label_mat: [num_obj, num_obj] label matrix
        pos_pairs: List of positive pairs
        neg_ratio: Ratio of negatives to positives
        
    Returns:
        pairs: List of (i, j) pairs
        labels: [P] labels
    """
    num_obj = label_mat.size(0)
    all_pairs = [(i, j) for i in range(num_obj) for j in range(num_obj) if i != j]
    
    pos = pos_pairs
    pos_set = set(pos)
    
    # Negatives
    neg = [p for p in all_pairs if p not in pos_set]
    num_neg = min(len(neg), len(pos) * neg_ratio if len(pos) > 0 else 256)
    
    import random
    if len(neg) > num_neg:
        neg = random.sample(neg, num_neg)
    
    pairs = pos + neg
    labels = torch.tensor([label_mat[i, j].item() for (i, j) in pairs], dtype=torch.long)
    
    return pairs, labels


def forward_one_image(
    sam3: FrozenSAM3,
    rel_head: RelationHead,
    image: Image.Image,
    gt_boxes: torch.Tensor,
    gt_obj_labels: torch.Tensor,
    gt_rels: torch.Tensor,
    max_props: int = 200,
    iou_thr: float = 0.3,  # Lower threshold for better matching
    bg_class: int = 0,
    neg_ratio: int = 3,
    device: str = "cuda",
):
    """
    Forward pass for one image
    
    Returns:
        (logits, labels, skip_reason): logits [P, num_predicates], labels [P], skip_reason str or None
    """
    # 1) SAM3 proposals
    masks, embs, scores = sam3.forward(image)  # masks:[N,H,W], embs:[N,D]
    N = embs.size(0)
    
    if N == 0:
        return None, None, "no_sam3_detections"
    
    if N > max_props:
        topk = torch.topk(scores, k=max_props).indices
        masks = masks[topk]
        embs = embs[topk]
        scores = scores[topk]
        N = max_props
    
    # 2) mask->box
    pred_boxes = []
    valid = []
    for n in range(masks.size(0)):
        box = mask_to_box(masks[n])
        if box is None:
            continue
        pred_boxes.append(box)
        valid.append(n)
    
    if len(pred_boxes) == 0:
        return None, None, "no_valid_boxes"
    
    pred_boxes = torch.stack(pred_boxes, dim=0).to(device)  # [N', 4]
    embs = embs[torch.tensor(valid, device=embs.device)]  # [N', D]
    
    # 3) Match GT->pred by IoU only (no label matching needed)
    # We only care about spatial matching (bbox/mask), not semantic labels
    gt_boxes = gt_boxes.to(device)
    if gt_boxes.numel() == 0:
        return None, None, "no_gt_boxes"
    
    gt_to_pred = match_by_iou(pred_boxes, gt_boxes, iou_thr=iou_thr)  # [G]
    
    # 4) Gather matched GT objects
    matched_g = (gt_to_pred >= 0).nonzero(as_tuple=False)
    if matched_g.numel() == 0:
        return None, None, "no_matched_objects"
    
    matched_g = matched_g.squeeze(1)
    if matched_g.numel() < 2:
        return None, None, f"matched_objects_less_than_2({matched_g.numel()})"
    
    # Remap indices
    old_to_new = {int(g.item()): k for k, g in enumerate(matched_g)}
    M = matched_g.numel()
    
    obj_emb = []
    obj_box = []
    for g in matched_g.tolist():
        n = gt_to_pred[g].item()
        obj_emb.append(embs[n])
        obj_box.append(gt_boxes[g])
    
    obj_emb = torch.stack(obj_emb, dim=0)  # [M, D]
    obj_box = torch.stack(obj_box, dim=0)  # [M, 4]
    
    # 5) Filter/remap relations
    rels = []
    for s, o, p in gt_rels.tolist():
        if s in old_to_new and o in old_to_new and s != o:
            rels.append((old_to_new[s], old_to_new[o], p))
    
    # 如果没有有效关系，仍然需要处理（给一个大的loss）
    # 我们创建所有可能的pair，但都标记为background（无关系）
    has_valid_relations = len(rels) > 0
    
    if has_valid_relations:
        gt_rels_m = torch.tensor(rels, device=device, dtype=torch.long)  # [R', 3]
        # 6) Build labels & sample pairs
        label_mat, pos_pairs = build_pair_labels(M, gt_rels_m, bg_class=bg_class)
        pairs, labels = sample_pairs(label_mat, pos_pairs, neg_ratio=neg_ratio)
    else:
        # 没有有效关系：创建所有可能的pair，都标记为background
        # 这样模型会学习到"这些物体之间没有关系"
        all_pairs = [(i, j) for i in range(M) for j in range(M) if i != j]
        # 限制pair数量，避免太多
        max_pairs_no_rel = min(len(all_pairs), 50)  # 最多50个pair
        import random
        if len(all_pairs) > max_pairs_no_rel:
            pairs = random.sample(all_pairs, max_pairs_no_rel)
        else:
            pairs = all_pairs
        # 所有pair都标记为background（无关系）
        labels = torch.zeros(len(pairs), dtype=torch.long)
    
    P = len(pairs)
    
    if P == 0:
        return None, None, "no_pairs_after_sampling"
    
    # 7) Build pair features
    z_list = []
    for (i, j) in pairs:
        geom = box_geom_feat(obj_box[i], obj_box[j])  # [geom_dim]
        z = torch.cat([obj_emb[i], obj_emb[j], geom], dim=0)
        z_list.append(z)
    
    z = torch.stack(z_list, dim=0)  # [P, 2D+geom]
    
    # 8) Logits
    logits = rel_head(z)  # [P, C]
    labels = labels.to(device)
    
    # Return skip_reason only if we had no valid relations (for tracking)
    skip_reason = None if has_valid_relations else "no_valid_relations"
    
    return logits, labels, skip_reason


def compute_grad_norm(model: nn.Module) -> float:
    """Compute gradient norm"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def train_step(
    batch: dict,
    sam3: FrozenSAM3,
    rel_head: RelationHead,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cuda",
    neg_ratio: int = 3,
):
    """Single training step"""
    rel_head.train()
    optimizer.zero_grad(set_to_none=True)
    
    total_loss = 0.0
    valid = 0
    loss_details = []
    skip_reasons = defaultdict(int)
    
    for sample in batch:
        # Convert tensor image back to PIL
        image_tensor = sample["image"]
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = image_tensor * std + mean
        image_tensor = image_tensor.clamp(0, 1)
        
        # Convert to PIL
        image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        
        gt_boxes = sample["gt_boxes"]
        gt_obj_labels = sample["gt_obj_labels"]
        gt_rels = sample["gt_rels"]
        
        out = forward_one_image(
            sam3, rel_head,
            image, gt_boxes, gt_obj_labels, gt_rels,
            bg_class=0, neg_ratio=neg_ratio, device=device
        )
        
        if out[0] is None:
            skip_reason = out[2] if len(out) > 2 else "unknown"
            skip_reasons[skip_reason] += 1
            continue
        
        logits, labels, _ = out
        # logits: [P, num_predicates], labels: [P] (0 for background, >0 for predicate classes)
        
        # 1. Relation Existence Loss: 判断两个物体之间是否有关系（二分类）
        # 对所有pair，判断是否有关系：label > 0 表示有关系，label == 0 表示无关系
        existence_labels = (labels > 0).float()  # [P]: 1.0 if has relation, 0.0 if no relation
        
        if logits.size(1) > 1:
            # 方法：使用log-sum-exp来聚合所有非background类的logits
            # 关系存在的logit = log(sum(exp(logits[:, 1:]))) - log(num_classes-1)
            # 更简单：使用max logit（近似log-sum-exp）
            # 或者：直接使用所有正类logits的log-sum-exp
            bg_logit = logits[:, 0]  # [P]: background logit
            pos_logits = logits[:, 1:]  # [P, num_predicates-1]: all non-background logits
            
            # 使用log-sum-exp来聚合正类logits（更稳定）
            # existence_logit = torch.logsumexp(pos_logits, dim=1)  # [P]
            # 或者使用max（更简单，近似log-sum-exp）
            existence_logit = pos_logits.max(dim=1)[0]  # [P]: max logit of non-background classes
            
            # 构建二分类：比较background logit和existence logit
            # 使用binary cross entropy: existence_logit - bg_logit 作为关系存在的logit
            existence_logit_binary = existence_logit - bg_logit  # [P]: positive if has relation
        else:
            # Fallback: 如果只有background类
            existence_logit_binary = -logits[:, 0]  # [P]: negative of background logit
        
        # 计算关系存在性loss（二分类交叉熵）
        existence_loss = F.binary_cross_entropy_with_logits(
            existence_logit_binary, existence_labels, reduction='mean'
        )
        
        # 2. Predicate Classification Loss: 只对有关系的pair，预测具体的关系类型
        pos_mask = labels > 0
        num_positives = pos_mask.sum().item()
        classification_loss_val = 0.0
        if num_positives > 0:
            # 只对有关系的pair计算分类loss
            pos_logits = logits[pos_mask]  # [num_pos, num_predicates]
            pos_labels = labels[pos_mask]  # [num_pos]: predicate class indices (>0)
            # 注意：这里我们仍然使用所有predicate类（包括background），但只对正样本计算
            # 实际上，对于正样本，label不会是0，所以可以直接使用CrossEntropyLoss
            classification_loss = criterion(pos_logits, pos_labels)
            classification_loss_val = classification_loss.item()
        else:
            classification_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # 3. Total Loss = 关系存在性loss + 关系分类loss
        total_sample_loss = existence_loss + classification_loss
        total_sample_loss.backward()
        
        # Record loss details
        loss_details.append({
            "existence_loss": existence_loss.item(),
            "classification_loss": classification_loss_val,
            "total_loss": total_sample_loss.item(),
            "num_pairs": len(labels),
            "num_positives": num_positives,
            "num_negatives": (labels == 0).sum().item(),
        })
        
        total_loss += total_sample_loss.item()
        valid += 1
    
    # Compute gradient norm before step
    grad_norm = compute_grad_norm(rel_head) if valid > 0 else 0.0
    
    if valid > 0:
        optimizer.step()
    
    # Calculate average loss per valid sample
    avg_loss = total_loss / max(valid, 1)
    
    # Aggregate loss details
    loss_info = {
        "avg_loss": avg_loss,  # Average total loss per valid sample
        "grad_norm": grad_norm,
        "valid_samples": valid,
        "total_samples": len(batch),
        "skip_reasons": dict(skip_reasons),
    }
    if loss_details:
        loss_info.update({
            "avg_existence_loss": np.mean([d["existence_loss"] for d in loss_details]),
            "avg_classification_loss": np.mean([d["classification_loss"] for d in loss_details]),
            "avg_num_pairs": np.mean([d["num_pairs"] for d in loss_details]),
            "avg_positives": np.mean([d["num_positives"] for d in loss_details]),
            "avg_negatives": np.mean([d["num_negatives"] for d in loss_details]),
        })
    else:
        loss_info.update({
            "avg_existence_loss": 0.0,
            "avg_classification_loss": 0.0,
            "avg_num_pairs": 0.0,
            "avg_positives": 0.0,
            "avg_negatives": 0.0,
        })
    
    return avg_loss, loss_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to Visual Genome dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Checkpoint save directory")
    parser.add_argument("--use_class_weights", action="store_true", default=False, help="Use class weights for imbalanced classes")
    parser.add_argument("--use_focal_loss", action="store_true", default=False, help="Use Focal Loss instead of CrossEntropyLoss")
    parser.add_argument("--neg_ratio", type=int, default=3, help="Ratio of negative to positive samples")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Dataset
    print("Loading dataset...")
    dataset = VGDataset(data_root=args.data_root, split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,  # Return list of samples
    )
    
    # Models
    print("Initializing models...")
    sam3 = FrozenSAM3(confidence_threshold=0.02, device=device)
    print(f"SAM3 confidence threshold set to: 0.02")
    
    predicate_vocab = dataset.get_predicate_vocab()
    num_predicates = len(predicate_vocab)
    print(f"Number of predicates: {num_predicates}")
    
    # Save predicate vocabulary to config file
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    os.makedirs(config_dir, exist_ok=True)
    vocab_config_path = os.path.join(config_dir, "predicate_vocab.json")
    
    # Create reverse mapping (index -> predicate name) for easier reading
    idx_to_predicate = {idx: pred for pred, idx in predicate_vocab.items()}
    vocab_config = {
        "predicate_to_idx": predicate_vocab,
        "idx_to_predicate": idx_to_predicate,
        "num_predicates": num_predicates,
        "created_at": datetime.now().isoformat()
    }
    
    with open(vocab_config_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_config, f, indent=2, ensure_ascii=False)
    print(f"✅ Predicate vocabulary saved to: {vocab_config_path}")
    
    # Print predicate vocabulary mapping
    print("\n" + "="*80)
    print("Predicate Vocabulary Mapping:")
    print("="*80)
    print(f"{'Index':<8} {'Predicate Name':<30}")
    print("-"*80)
    for idx in sorted(idx_to_predicate.keys()):
        pred_name = idx_to_predicate[idx]
        print(f"{idx:<8} {pred_name:<30}")
    print("="*80 + "\n")
    
    rel_head = RelationHead(
        emb_dim=256,
        geom_dim=6,
        num_predicates=num_predicates,
        hidden=512,
        dropout=0.1,
    ).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(rel_head.parameters(), lr=args.lr)
    
    # Loss function selection
    # Default: use class weights (unless focal loss is explicitly requested)
    use_class_weights = args.use_class_weights if args.use_class_weights else (not args.use_focal_loss)
    
    if args.use_focal_loss:
        # Focal Loss for handling class imbalance
        criterion = FocalLoss(alpha=0.25, gamma=2.0, num_classes=num_predicates).to(device)
        print(f"Using Focal Loss: alpha=0.25, gamma=2.0")
    elif use_class_weights:
        # Class weights for imbalanced dataset
        # Background class (0) is usually much more frequent than predicate classes
        class_weights = torch.ones(num_predicates, device=device)
        class_weights[0] = 0.1  # Background class weight (much less weight)
        # Predicate classes get weight 1.0 (default)
        print(f"Using class weights: background={class_weights[0]:.2f}, predicates=1.0")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss (no class balancing)")
    
    # Create log directory
    log_dir = os.path.join(os.path.dirname(__file__), "logfiles")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create TensorBoard log directory
    tb_log_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_log_{timestamp}.json")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, f"run_{timestamp}"))
    print(f"TensorBoard logs will be saved to: {os.path.join(tb_log_dir, f'run_{timestamp}')}")
    print(f"To view TensorBoard, run: tensorboard --logdir {tb_log_dir}")
    
    # Initialize training log
    training_log = {
        "start_time": datetime.now().isoformat(),
        "config": {
            "data_root": args.data_root,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "device": args.device,
            "num_predicates": num_predicates,
        },
        "epochs": []
    }
    
    # Thread lock for JSON file updates
    json_lock = threading.Lock()
    
    def save_json_log():
        """Thread-safe JSON log saving"""
        with json_lock:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(training_log, f, indent=2, ensure_ascii=False)
    
    # Training loop
    print("Starting training...")
    print(f"Training log will be saved to: {log_file}")
    
    total_steps = 0
    for epoch in range(args.num_epochs):
        rel_head.train()
        epoch_log = {
            "epoch": epoch + 1,
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
        total_loss = 0.0
        num_batches = 0
        epoch_start_time = datetime.now()
        
        # Calculate total steps
        total_batches = len(dataloader)
        
        print(f"\n{'='*80}")
        print(f"Starting Epoch {epoch+1}/{args.num_epochs} - Total steps: {total_batches}")
        print(f"{'='*80}\n")
        
        step_times = []
        for step_idx, batch in enumerate(dataloader, 1):
            step_start_time = datetime.now()
            
            loss, loss_info = train_step(
                batch, sam3, rel_head, optimizer, criterion, device=device
            )
            
            total_loss += loss
            num_batches += 1
            total_steps += 1
            
            # Calculate speed and ETA
            step_time = (datetime.now() - step_start_time).total_seconds()
            step_times.append(step_time)
            # Use average of last 10 steps for more stable speed estimate
            recent_times = step_times[-10:] if len(step_times) >= 10 else step_times
            avg_step_time = np.mean(recent_times) if recent_times else step_time
            speed = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
            
            remaining_steps = total_batches - step_idx
            eta_seconds = remaining_steps * avg_step_time if avg_step_time > 0 else 0
            eta_hours = int(eta_seconds // 3600)
            eta_minutes = int((eta_seconds % 3600) // 60)
            eta_secs = int(eta_seconds % 60)
            eta_str = f"{eta_hours:02d}:{eta_minutes:02d}:{eta_secs:02d}"
            
            # Format detailed loss info
            grad_norm = loss_info.get("grad_norm", 0.0)
            
            # Loss values: loss is average per valid sample
            # Total loss = existence loss + classification loss
            avg_existence_loss = loss_info.get("avg_existence_loss", 0.0)
            avg_classification_loss = loss_info.get("avg_classification_loss", 0.0)
            
            # Debug info (statistics, moved to valid section)
            avg_num_pairs = loss_info.get("avg_num_pairs", 0.0)  # Average number of pairs per sample
            avg_pos = loss_info.get("avg_positives", 0.0)
            avg_neg = loss_info.get("avg_negatives", 0.0)
            
            # Log to TensorBoard (always log, even if 0)
            global_step = total_steps
            writer.add_scalar("Train/GradNorm", grad_norm, global_step)
            writer.add_scalar("Train/Loss", loss, global_step)  # Total loss (average per valid sample)
            writer.add_scalar("Train/ExistenceLoss", avg_existence_loss, global_step)
            writer.add_scalar("Train/ClassificationLoss", avg_classification_loss, global_step)
            writer.add_scalar("Train/Speed", speed, global_step)
            writer.add_scalar("Train/AvgNumPairs", avg_num_pairs, global_step)
            writer.add_scalar("Train/AvgPositives", avg_pos, global_step)
            writer.add_scalar("Train/AvgNegatives", avg_neg, global_step)
            writer.add_scalar("Train/ValidSamples", loss_info.get('valid_samples', 0), global_step)
            
            # Print detailed info every 10 steps or on first step
            if step_idx == 1 or step_idx % 10 == 0:
                # Format: grad=... | loss=... existence_loss=... classification_loss=... | valid=... (avg_num_pairs=... pos=... neg=...)
                log_line = (
                    f"[Epoch {epoch+1}] Step {step_idx}/{total_batches} | "
                    f"Speed: {speed:.2f}it/s | ETA: {eta_str} | "
                    f"grad={grad_norm:.4f} | loss={loss:.4f} existence_loss={avg_existence_loss:.4f} classification_loss={avg_classification_loss:.4f} | "
                    f"valid={loss_info.get('valid_samples', 0)}/{loss_info.get('total_samples', 0)} "
                    f"(avg_num_pairs={avg_num_pairs:.1f} pos={avg_pos:.1f} neg={avg_neg:.1f})"
                )
                
                # Add skip reasons if any
                if loss_info.get("skip_reasons"):
                    skip_str = ", ".join([f"{k}:{v}" for k, v in loss_info["skip_reasons"].items()])
                    log_line += f" | skipped: {skip_str}"
                
                print(log_line)
            
            # Log step details (format matching print output, no timestamp)
            step_log = {
                "epoch": epoch + 1,
                "step": step_idx,
                "total_steps": total_batches,
                "speed": speed,
                "eta": eta_str,
                "grad": grad_norm,
                "loss": loss,
                "existence_loss": avg_existence_loss,
                "classification_loss": avg_classification_loss,
                "valid": f"{loss_info.get('valid_samples', 0)}/{loss_info.get('total_samples', 0)}",
            }
            # Add statistics as debug info in valid section (always include, even if 0)
            step_log["avg_num_pairs"] = avg_num_pairs
            step_log["pos"] = avg_pos
            step_log["neg"] = avg_neg
            
            # Add skip reasons for debugging
            if loss_info.get("skip_reasons"):
                step_log["skip_reasons"] = loss_info["skip_reasons"]
            
            epoch_log["steps"].append(step_log)
            
            # Real-time JSON update every 10 steps or on first step
            if step_idx == 1 or step_idx % 10 == 0:
                # Update epoch log in training_log (ensure epoch exists)
                while len(training_log["epochs"]) <= epoch:
                    training_log["epochs"].append({
                        "epoch": len(training_log["epochs"]) + 1,
                        "start_time": datetime.now().isoformat(),
                        "steps": []
                    })
                training_log["epochs"][epoch] = epoch_log
                
                # Save JSON in background thread
                threading.Thread(target=save_json_log, daemon=True).start()
        
        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        
        epoch_log.update({
            "end_time": datetime.now().isoformat(),
            "duration_seconds": epoch_time,
            "avg_loss": avg_loss,
            "total_steps": num_batches,
        })
        
        # Update epoch log (ensure epoch exists)
        while len(training_log["epochs"]) <= epoch:
            training_log["epochs"].append({
                "epoch": len(training_log["epochs"]) + 1,
                "start_time": datetime.now().isoformat(),
                "steps": []
            })
        training_log["epochs"][epoch] = epoch_log
        
        # Log epoch summary to TensorBoard
        writer.add_scalar("Epoch/AvgLoss", avg_loss, epoch + 1)
        writer.add_scalar("Epoch/Duration", epoch_time, epoch + 1)
        
        # Save JSON after each epoch
        save_json_log()
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.num_epochs} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Duration: {epoch_time/60:.2f} minutes")
        print(f"  Total Steps: {num_batches}")
        print(f"{'='*80}\n")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': rel_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'num_predicates': num_predicates,
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    training_log["end_time"] = datetime.now().isoformat()
    training_log["total_duration_seconds"] = (datetime.now() - datetime.fromisoformat(training_log["start_time"])).total_seconds()
    
    # Final save
    save_json_log()
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\n{'='*80}")
    print(f"✅ Training completed!")
    print(f"✅ Training log saved to: {log_file}")
    print(f"✅ TensorBoard logs saved to: {os.path.join(tb_log_dir, f'run_{timestamp}')}")
    print(f"   To view TensorBoard, run: tensorboard --logdir {tb_log_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

