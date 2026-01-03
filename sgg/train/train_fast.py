"""
Fast Training Script for Cached Pairs
训练只做：读缓存 → MLP → CE。速度会大幅提升。
"""
import argparse
import os
import sys
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sgg.datasets.cached_pairs import CachedPairDataset, collate_cached
from sgg.datasets.vg150_reader import VG150Reader
from sgg.models.relation_head_geom import RelationHeadMLP
from sgg.train.loss import MaskedCrossEntropy
from sgg.utils.seed import seed_everything
from sgg.utils.io import ensure_dir


@dataclass
class TrainConfig:
    cache_dir: str
    out_dir: str
    num_predicates: int           # K+1 includes bg
    data_root: str = ""           # VG150 dataset root (for index validation)
    split: str = "train"          # Dataset split
    epochs: int = 4
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    amp: bool = True
    grad_clip: float = 5.0
    bg_weight: float = 0.2
    log_every: int = 50
    save_mode: str = "epoch"       # "epoch" or "iter"
    save_frequency: int = 1       # 保存频率：每N个epoch或每N个iter
    validate_indices: bool = True  # 是否验证索引对齐
    seed: int = 42


def train(cfg: TrainConfig) -> None:
    """Training loop"""
    seed_everything(cfg.seed)
    ensure_dir(cfg.out_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    ds = CachedPairDataset(cfg.cache_dir)
    
    # 验证索引对齐（使用与测试端相同的策略）
    if cfg.validate_indices and cfg.data_root:
        print("\nValidating index alignment with h5 file (using official demo approach)...")
        reader = VG150Reader(data_root=cfg.data_root, split=cfg.split)
        
        # 随机采样几个cache文件进行验证
        import random
        random.seed(cfg.seed)
        sample_indices = random.sample(range(min(10, len(ds))), min(5, len(ds)))
        
        mismatch_count = 0
        for idx in sample_indices:
            cache_file = ds.files[idx]
            pack = torch.load(cache_file, map_location="cpu", weights_only=False)
            cache_image_id = pack["image_id"]
            
            # 临时修复：image n 对应 relation n+1
            # TODO: 修复后需要移除这个偏移
            image_id = cache_image_id + 1
            
            # 使用与测试端相同的方式：通过image_id直接从h5文件获取
            sample = reader.get_sample_by_image_id(image_id)
            if sample is None:
                print(f"  ⚠️  Warning: Image {image_id} (cache_id={cache_image_id}) not found in h5 file (cache file: {os.path.basename(cache_file)})")
                mismatch_count += 1
                continue
            
            # 检查pair_idx是否与boxes数量匹配
            if "pair_idx" in pack:
                import numpy as np
                cache_pair_idx = pack["pair_idx"]
                if not isinstance(cache_pair_idx, np.ndarray):
                    cache_pair_idx = np.array(cache_pair_idx)
                num_boxes = len(sample.boxes_xyxy)
                invalid_pairs = (cache_pair_idx[:, 0] >= num_boxes) | (cache_pair_idx[:, 1] >= num_boxes) | \
                               (cache_pair_idx[:, 0] < 0) | (cache_pair_idx[:, 1] < 0)
                if invalid_pairs.any():
                    invalid_count = invalid_pairs.sum()
                    print(f"  ⚠️  Warning: Image {image_id} (cache_id={cache_image_id}) has {invalid_count} invalid pair_idx entries "
                          f"(out of {len(cache_pair_idx)} total, num_boxes={num_boxes})")
                    print(f"     Using temporary fix: image_id = cache_image_id + 1")
                    mismatch_count += 1
                else:
                    print(f"  ✓ Image {image_id} (cache_id={cache_image_id}): index alignment OK (num_boxes={num_boxes}, pairs={len(cache_pair_idx)})")
            else:
                print(f"  ⚠️  Warning: Cache file {os.path.basename(cache_file)} missing pair_idx")
        
        reader.close()
        
        if mismatch_count > 0:
            print(f"\n  ⚠️  Found {mismatch_count} mismatches in sampled cache files!")
            print(f"  This may indicate cache files were generated with incorrect index mapping.")
            print(f"  Please regenerate cache files using the updated build_cache.py")
            print(f"  (build_cache.py now uses get_sample_by_image_id() for consistent indexing)")
        else:
            print(f"\n  ✓ Index validation passed for sampled cache files")
        print()
    
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_cached,
        drop_last=False,  # Keep all data, even if last batch is smaller
    )
    
    print(f"  DataLoader batches: {len(dl)} (dataset size: {len(ds)}, batch_size: {cfg.batch_size})")
    
    # Infer input dim from first sample and validate
    sample0 = ds[0]
    in_dim = int(sample0["geom_feat"].shape[1])
    print(f"Input dimension (geom_dim): {in_dim}")
    
    # Validate data: check for NaN/Inf
    geom0 = sample0["geom_feat"]
    if torch.isnan(geom0).any() or torch.isinf(geom0).any():
        print(f"WARNING: Found NaN/Inf in sample 0 geom_feat!")
        print(f"  NaN count: {torch.isnan(geom0).sum().item()}")
        print(f"  Inf count: {torch.isinf(geom0).sum().item()}")
    else:
        print(f"Sample 0 validation OK: geom_feat range [{geom0.min():.4f}, {geom0.max():.4f}]")
    
    # Model
    model = RelationHeadMLP(in_dim=in_dim, num_classes=cfg.num_predicates).to(device)
    criterion = MaskedCrossEntropy(num_classes=cfg.num_predicates, bg_weight=cfg.bg_weight).to(device)
    
    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.amp)
    
    print(f"\nStarting training...")
    print(f"  Dataset size: {len(ds)}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Epochs: {cfg.epochs}")
    print(f"  Learning rate: {cfg.lr}")
    print(f"  AMP: {cfg.amp}")
    print(f"  Output directory: {cfg.out_dir}\n")
    
    global_step = 0
    for ep in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_valid_pairs = 0
        epoch_total_pairs = 0
        
        for it, batch in enumerate(dl):
            geom = batch["geom_feat"].to(device, non_blocking=True)   # [B,P,Gd]
            labels = batch["pair_label"].to(device, non_blocking=True) # [B,P]
            mask = batch["pair_mask"].to(device, non_blocking=True)    # [B,P] bool
            
            # Flatten
            B, P, Gd = geom.shape
            x = geom.view(B * P, Gd)
            y = labels.view(B * P)
            m = mask.view(B * P)
            
            # Check for NaN/Inf in input
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"WARNING: NaN/Inf in batch geom_feat at step {global_step}")
                continue
            
            # Valid labels are >=0, and mask should filter padding already
            m = m & (y >= 0)
            
            if m.sum() == 0:
                # Skip if no valid pairs
                continue
            
            optim.zero_grad(set_to_none=True)
            
            with autocast(enabled=cfg.amp):
                logits = model(x)  # [B*P, C]
                
                # Check for NaN in logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print(f"WARNING: NaN/Inf in logits at step {global_step}")
                    print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
                    continue
                
                loss = criterion(logits, y.clamp_min(0), m)
                
                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: NaN/Inf loss at step {global_step}")
                    print(f"  Valid pairs: {m.sum().item()}")
                    print(f"  Label range: [{y[m].min().item()}, {y[m].max().item()}]")
                    continue
            
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
            scaler.step(optim)
            scaler.update()
            
            # Statistics
            with torch.no_grad():
                valid_n = int(m.sum().item())
                epoch_loss += loss.item()
                epoch_valid_pairs += valid_n
                epoch_total_pairs += B * P
            
            if global_step % cfg.log_every == 0:
                print(
                    f"[E{ep+1}/{cfg.epochs}] step={global_step} "
                    f"loss={loss.item():.4f} "
                    f"valid_pairs={valid_n}/{B*P}"
                )
            
            # Save checkpoint (根据save_mode决定)
            should_save = False
            if cfg.save_mode == "iter":
                # 按iter保存
                if global_step > 0 and global_step % cfg.save_frequency == 0:
                    should_save = True
                    ckpt_name = f"rel_head_step_{global_step}.pt"
            elif cfg.save_mode == "epoch":
                # 按epoch保存（在epoch结束时保存，这里不处理）
                should_save = False
            
            if should_save:
                ckpt_path = os.path.join(cfg.out_dir, ckpt_name)
                torch.save({
                    "model": model.state_dict(),
                    "in_dim": in_dim,
                    "num_predicates": cfg.num_predicates,
                    "epoch": ep + 1,
                    "step": global_step,
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")
            
            global_step += 1
        
        # Epoch summary
        if len(dl) > 0:
            avg_loss = epoch_loss / len(dl)
        else:
            avg_loss = 0.0
        print(f"\n[Epoch {ep+1} completed]")
        print(f"  Average loss: {avg_loss:.4f}")
        if epoch_total_pairs > 0:
            print(f"  Valid pairs: {epoch_valid_pairs}/{epoch_total_pairs} ({100*epoch_valid_pairs/epoch_total_pairs:.1f}%)")
        else:
            print(f"  Valid pairs: {epoch_valid_pairs}/{epoch_total_pairs} (no valid batches processed)")
        
        # Save epoch checkpoint（根据save_mode决定）
        if cfg.save_mode == "epoch":
            # 按epoch保存
            if (ep + 1) % cfg.save_frequency == 0:
                ckpt_path = os.path.join(cfg.out_dir, f"rel_head_ep{ep+1}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "in_dim": in_dim,
                    "num_predicates": cfg.num_predicates,
                    "epoch": ep + 1,
                    "step": global_step,
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")
        elif cfg.save_mode == "iter":
            # 按iter保存时，epoch结束时也保存一个（可选）
            # 这里可以选择是否保存，或者只在最后一个epoch保存
            if ep + 1 == cfg.epochs:
                ckpt_path = os.path.join(cfg.out_dir, f"rel_head_ep{ep+1}_final.pt")
                torch.save({
                    "model": model.state_dict(),
                    "in_dim": in_dim,
                    "num_predicates": cfg.num_predicates,
                    "epoch": ep + 1,
                    "step": global_step,
                }, ckpt_path)
                print(f"  Saved final checkpoint: {ckpt_path}")
        print()  # 空行


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast training with cached pairs")
    p.add_argument("--cache_dir", type=str, required=True, help="Cache directory")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints")
    p.add_argument("--num_predicates", type=int, required=True, help="Number of predicates (K+1 with bg)")
    p.add_argument("--data_root", type=str, default="", help="VG150 dataset root (for index validation)")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    p.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    p.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    p.add_argument("--amp", action="store_true", help="Use AMP")
    p.add_argument("--no_amp", action="store_true", help="Disable AMP")
    p.add_argument("--grad_clip", type=float, default=5.0, help="Gradient clipping")
    p.add_argument("--bg_weight", type=float, default=0.2, help="Background class weight")
    p.add_argument("--log_every", type=int, default=50, help="Log every N steps")
    p.add_argument("--save_mode", type=str, default="epoch", choices=["epoch", "iter"], 
                   help="Checkpoint save mode: 'epoch' or 'iter'")
    p.add_argument("--save_frequency", type=int, default=1, 
                   help="Save frequency: save every N epochs (if mode=epoch) or every N iters (if mode=iter)")
    p.add_argument("--no_validate_indices", action="store_true", 
                   help="Disable index validation (skip checking cache file alignment with h5)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    amp = True
    if args.no_amp:
        amp = False
    elif args.amp:
        amp = True
    
    cfg = TrainConfig(
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        num_predicates=args.num_predicates,
        data_root=args.data_root,
        split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=amp,
        grad_clip=args.grad_clip,
        bg_weight=args.bg_weight,
        log_every=args.log_every,
        save_mode=args.save_mode,
        save_frequency=args.save_frequency,
        validate_indices=not args.no_validate_indices,
        seed=args.seed,
    )
    train(cfg)

