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
from sgg.models.relation_head_geom import RelationHeadMLP
from sgg.train.loss import MaskedCrossEntropy
from sgg.utils.seed import seed_everything
from sgg.utils.io import ensure_dir


@dataclass
class TrainConfig:
    cache_dir: str
    out_dir: str
    num_predicates: int           # K+1 includes bg
    epochs: int = 4
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    amp: bool = True
    grad_clip: float = 5.0
    bg_weight: float = 0.2
    log_every: int = 50
    save_every: int = 1000
    seed: int = 42


def train(cfg: TrainConfig) -> None:
    """Training loop"""
    seed_everything(cfg.seed)
    ensure_dir(cfg.out_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    ds = CachedPairDataset(cfg.cache_dir)
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
            
            # Save checkpoint
            if global_step > 0 and global_step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.out_dir, f"rel_head_step_{global_step}.pt")
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
        
        # Save epoch checkpoint
        ckpt_path = os.path.join(cfg.out_dir, f"rel_head_ep{ep+1}.pt")
        torch.save({
            "model": model.state_dict(),
            "in_dim": in_dim,
            "num_predicates": cfg.num_predicates,
            "epoch": ep + 1,
            "step": global_step,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast training with cached pairs")
    p.add_argument("--cache_dir", type=str, required=True, help="Cache directory")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for checkpoints")
    p.add_argument("--num_predicates", type=int, required=True, help="Number of predicates (K+1 with bg)")
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
    p.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        amp=amp,
        grad_clip=args.grad_clip,
        bg_weight=args.bg_weight,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
    )
    train(cfg)

