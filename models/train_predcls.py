"""
Training script for SGG PredCls task
按照新流程：GT objects → SAM3 embedding → relation classification
"""
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import json
from datetime import datetime
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import threading
import time
from time import perf_counter

from sgg.datasets.vg150_dataset import VG150Dataset
from sgg.models.frozen_sam3_gt import FrozenSAM3GT
from sgg.models.relation_head import RelationHead
from sgg.utils.geometry import box_geom_feat
from sgg.utils.edge_builder import build_edges, sample_neg_pairs, build_pair_features


def forward_one_image(
    sam3: FrozenSAM3GT,
    rel_head: RelationHead,
    image: Image.Image,
    gt_boxes: torch.Tensor,
    gt_rels: torch.Tensor,
    neg_ratio: int = 3,
    max_negs: int = 50,
    device: str = "cuda",
):
    """
    对单张图像进行前向传播（新流程：无 IoU matching）
    
    Args:
        sam3: Frozen SAM3 model
        rel_head: Relation head
        image: PIL Image
        gt_boxes: [G, 4] normalized xyxy boxes
        gt_rels: [R, 3] (s_idx, o_idx, pred_idx)
        neg_ratio: Negative sampling ratio
        max_negs: Maximum negatives when no positives
        device: Device
        
    Returns:
        (logits, labels, skip_reason) or (None, None, skip_reason)
    """
    G = gt_boxes.size(0)
    
    # Check basic requirements
    if G == 0:
        return None, None, "no_gt_objects"
    if G < 2:
        return None, None, "less_than_2_objects"
    
    # Extract embeddings for GT boxes using box prompts
    try:
        embs = sam3.forward_batch_boxes(image, gt_boxes.to(device))  # [G, 256]
    except Exception as e:
        return None, None, f"sam3_error: {str(e)}"
    
    if embs.size(0) != G:
        return None, None, "embedding_mismatch"
    
    # Build edges from GT relations
    pos_edges, pos_pairs = build_edges(gt_rels)
    
    # Sample negative pairs
    neg_edges = sample_neg_pairs(G, pos_pairs, neg_ratio=neg_ratio, max_negs=max_negs)
    
    # Combine positive and negative edges
    all_edges = pos_edges + neg_edges
    
    if len(all_edges) == 0:
        return None, None, "no_edges"
    
    # Build pair features
    def geom_fn(box_s, box_o):
        return box_geom_feat(box_s, box_o)
    
    feats, labels = build_pair_features(embs, gt_boxes.to(device), all_edges, geom_fn)
    # feats: [P, 2*256 + 6], labels: [P]
    
    # Forward through relation head
    logits = rel_head(feats)  # [P, num_predicates]
    
    skip_reason = None if len(pos_edges) > 0 else "no_valid_relations"
    
    return logits, labels, skip_reason


def train_step(
    batch,
    sam3: FrozenSAM3GT,
    rel_head: RelationHead,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    neg_ratio: int = 3,
    max_negs: int = 50,
):
    """
    单个训练步骤
    """
    rel_head.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    valid = 0
    loss_details = []
    skip_reasons = defaultdict(int)
    
    for sample in batch:
        image_pil = sample["image_pil"]
        gt_boxes = sample["gt_boxes"]  # [G, 4]
        gt_rels = sample["gt_rels"]  # [R, 3]
        
        # Forward pass
        out = forward_one_image(
            sam3, rel_head, image_pil, gt_boxes, gt_rels,
            neg_ratio=neg_ratio, max_negs=max_negs, device=device
        )
        
        if out[0] is None:
            skip_reason = out[2]
            skip_reasons[skip_reason] += 1
            continue
        
        logits, labels, _ = out
        # logits: [P, num_predicates], labels: [P]
        
        # Compute loss (single softmax CE)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        
        # Record
        num_pairs = logits.size(0)
        num_positives = (labels > 0).sum().item()
        num_negatives = (labels == 0).sum().item()
        
        loss_details.append({
            "loss": loss.item(),
            "num_pairs": num_pairs,
            "num_positives": num_positives,
            "num_negatives": num_negatives,
        })
        
        total_loss += loss.item()
        valid += 1
    
    # Update parameters
    if valid > 0:
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(rel_head.parameters(), max_norm=1.0)
        optimizer.step()
    else:
        grad_norm = torch.tensor(0.0)
    
    # Aggregate statistics
    avg_loss = total_loss / max(valid, 1)
    avg_pairs = sum(d["num_pairs"] for d in loss_details) / max(valid, 1) if loss_details else 0
    avg_pos = sum(d["num_positives"] for d in loss_details) / max(valid, 1) if loss_details else 0
    avg_neg = sum(d["num_negatives"] for d in loss_details) / max(valid, 1) if loss_details else 0
    
    loss_info = {
        "loss": avg_loss,
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
        "valid": valid,
        "avg_num_pairs": avg_pairs,
        "avg_positives": avg_pos,
        "avg_negatives": avg_neg,
        "skip_reasons": dict(skip_reasons),
    }
    
    return avg_loss, loss_info


def main():
    parser = argparse.ArgumentParser(description="Train SGG PredCls")
    parser.add_argument("--data_root", type=str, required=True, help="VG dataset root")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--neg_ratio", type=int, default=3, help="Negative sampling ratio")
    parser.add_argument("--max_negs", type=int, default=50, help="Max negatives when no positives")
    parser.add_argument("--bg_weight", type=float, default=0.1, help="Background class weight")
    parser.add_argument("--save_dir", type=str, default="sgg/checkpoints", help="Save directory")
    parser.add_argument("--log_dir", type=str, default="sgg/logfiles", help="Log directory")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = VG150Dataset(
        data_root=args.data_root,
        split="train",
        image_size=1008,
    )
    
    # Get predicate vocabulary
    vocab = dataset.get_predicate_vocab()
    num_predicates = vocab["num_predicates"]
    print(f"Number of predicates: {num_predicates}")
    
    # Save vocabulary
    vocab_path = os.path.join("sgg/configs", "predicate_vocab.json")
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'w') as f:
        json.dump({
            **vocab,
            "created_at": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"Saved predicate vocabulary to {vocab_path}")
    
    # Print predicate vocabulary
    print("\n" + "="*80)
    print("Predicate Vocabulary:")
    print("="*80)
    print(f"{'Index':<8} {'Predicate Name':<30}")
    print("-"*80)
    for idx in sorted(vocab["idx_to_predicate"].keys()):
        pred_name = vocab["idx_to_predicate"][idx]
        print(f"{idx:<8} {pred_name:<30}")
    print("="*80 + "\n")
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,  # Return list of samples
    )
    
    # Models
    print("Initializing models...")
    sam3 = FrozenSAM3GT(device=device)
    rel_head = RelationHead(
        emb_dim=256,
        geom_dim=6,
        num_predicates=num_predicates,
        hidden=512,
        dropout=0.1,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(rel_head.parameters(), lr=args.lr)
    
    # Loss function (single softmax CE with background down-weighting)
    class_weights = torch.ones(num_predicates, device=device)
    class_weights[0] = args.bg_weight  # Background weight
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Using CrossEntropyLoss with background weight={args.bg_weight}")
    
    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"train_log_{timestamp}.json")
    tb_log_dir = os.path.join(args.log_dir, "tensorboard", timestamp)
    writer = SummaryWriter(tb_log_dir)
    
    log_data = {
        "config": vars(args),
        "vocab": vocab,
        "logs": [],
    }
    log_lock = threading.Lock()
    
    def save_log():
        with log_lock:
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    
    for epoch in range(args.num_epochs):
        print(f"\n[Epoch {epoch+1}/{args.num_epochs}]")
        
        for step, batch in enumerate(dataloader):
            # Train step
            avg_loss, loss_info = train_step(
                batch, sam3, rel_head, criterion, optimizer,
                device=device, neg_ratio=args.neg_ratio, max_negs=args.max_negs
            )
            
            global_step += 1
            
            # Logging
            if step % args.log_interval == 0:
                valid = loss_info["valid"]
                batch_size = len(batch)
                skip_info = ", ".join([f"{k}:{v}" for k, v in loss_info["skip_reasons"].items()])
                
                print(
                    f"Step {step}/{len(dataloader)} | "
                    f"grad={loss_info['grad_norm']:.4f} | "
                    f"loss={avg_loss:.4f} | "
                    f"valid={valid}/{batch_size} "
                    f"(avg_num_pairs={loss_info['avg_num_pairs']:.1f} "
                    f"pos={loss_info['avg_positives']:.1f} "
                    f"neg={loss_info['avg_negatives']:.1f}) | "
                    f"skipped: {skip_info}"
                )
                
                # TensorBoard
                writer.add_scalar("Loss", avg_loss, global_step)
                writer.add_scalar("GradNorm", loss_info["grad_norm"], global_step)
                writer.add_scalar("ValidSamples", valid, global_step)
                writer.add_scalar("AvgNumPairsPerSample", loss_info["avg_num_pairs"], global_step)
                writer.add_scalar("AvgPositives", loss_info["avg_positives"], global_step)
                writer.add_scalar("AvgNegatives", loss_info["avg_negatives"], global_step)
                
                # JSON log
                log_entry = {
                    "epoch": epoch + 1,
                    "step": step,
                    "global_step": global_step,
                    **loss_info,
                }
                log_data["logs"].append(log_entry)
                save_log()
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": rel_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab": vocab,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    writer.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
