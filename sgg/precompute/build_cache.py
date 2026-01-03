"""
Build cache for offline preprocessing
一次性生成缓存：SAM3（或 Dummy）只在这里跑，训练不会跑
"""
import argparse
import os
import sys
import json
from typing import Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sgg.datasets.vg150_reader import VG150Reader
from sgg.precompute.sam3_adapter import BaseSAM3MaskGenerator, RealSAM3MaskGenerator, DummySAM3MaskGenerator
from sgg.precompute.pair_sampler import sample_pairs_fixed_P
from sgg.precompute.geom_extractor import build_pair_geom
from sgg.utils.seed import seed_everything
from sgg.utils.io import ensure_dir, torch_save


def build_cache(
    vg_root: str,
    out_dir: str,
    split: str,
    P: int,
    neg_ratio: int,
    mask_size: int,
    seed: int,
    limit: int = -1,
    sam3_impl: str = "real",
    device: str = "cuda",
    save_every: int = 10,  # Save every N images
) -> None:
    """
    Build cache for VG150 dataset
    
    Args:
        vg_root: VG150 dataset root
        out_dir: Output directory for cache files
        split: "train" or "val"
        P: Fixed number of pairs per image
        neg_ratio: Negative sampling ratio
        mask_size: Mask resolution (Hm=Wm)
        seed: Random seed
        limit: Limit number of images to process (-1 for all)
        sam3_impl: "real" or "dummy"
        device: Device for SAM3
    """
    seed_everything(seed)
    ensure_dir(out_dir)
    
    reader = VG150Reader(data_root=vg_root, split=split)
    
    # Initialize SAM3 mask generator
    if sam3_impl == "real":
        sam3: BaseSAM3MaskGenerator = RealSAM3MaskGenerator(mask_size=mask_size, device=device)
    elif sam3_impl == "dummy":
        sam3: BaseSAM3MaskGenerator = DummySAM3MaskGenerator(mask_size=mask_size, device=device)
    else:
        raise ValueError(f"Unknown sam3_impl: {sam3_impl}, use 'real' or 'dummy'")
    
    n = 0
    success_count = 0
    fail_count = 0
    skip_reasons = {}
    last_save_count = 0
    
    print(f"\nBuilding cache for {split} split...")
    print(f"  Output directory: {out_dir}")
    print(f"  Fixed pairs per image: {P}")
    print(f"  Negative ratio: {neg_ratio}")
    print(f"  Mask size: {mask_size}x{mask_size}")
    print(f"  SAM3 implementation: {sam3_impl}")
    print(f"  Save every {save_every} images")
    if limit > 0:
        print(f"  ⚠️  LIMIT MODE: Processing only first {limit} SUCCESSFUL images (for quick validation)")
    else:
        print(f"  Processing all images (no limit)")
    print()
    
    # 计算total（用于tqdm进度条）
    # 注意：由于可能有很多图像被跳过，total只是一个估计值
    if limit > 0:
        # limit模式下，total设置为limit的2倍（考虑到可能有很多失败）
        total = min(limit * 2, len(reader))
    else:
        total = len(reader)
    
    # 使用pbar变量来访问tqdm的方法
    pbar = tqdm(reader.iter_samples(), total=total, desc="Precomputing")
    for sample in pbar:
        boxes = sample.boxes_xyxy.astype(np.float32)
        G = boxes.shape[0]
        if G < 2:
            fail_count += 1
            skip_reasons["less_than_2_objects"] = skip_reasons.get("less_than_2_objects", 0) + 1
            if limit > 0:
                print(f"  ⚠ Skipped image_id={sample.image_id}: less than 2 objects (G={G})")
            continue
        
        # Load image
        try:
            img = Image.open(sample.image_path).convert("RGB")
        except Exception as e:
            fail_count += 1
            skip_reasons[f"image_load_error: {str(e)}"] = skip_reasons.get(f"image_load_error: {str(e)}", 0) + 1
            if limit > 0:
                print(f"  ⚠ Skipped image_id={sample.image_id}: image load error: {e}")
            continue
        
        # 1) GT-driven masks
        try:
            sam3_out = sam3.predict(img, boxes)
            masks = sam3_out.masks
            if masks.dtype != np.bool_:
                masks = masks.astype(bool)
        except Exception as e:
            fail_count += 1
            skip_reasons[f"sam3_error: {str(e)}"] = skip_reasons.get(f"sam3_error: {str(e)}", 0) + 1
            if limit > 0:
                print(f"  ⚠ Skipped image_id={sample.image_id}: SAM3 error: {e}")
            continue
        
        # 2) Sample pairs (fixed P)
        ps = sample_pairs_fixed_P(
            num_obj=G,
            rels=sample.rels,
            P=P,
            neg_ratio=neg_ratio,
            seed=seed + sample.image_id,
        )
        
        # 3) Compute geometry (box + mask)
        geom_out = build_pair_geom(
            boxes_xyxy=boxes,
            masks=masks,
            pair_idx=ps.pair_idx,
            pair_mask=ps.pair_mask,
        )
        
        # 4) Write cache
        pack = {
            "image_id": int(sample.image_id),
            "pair_idx": ps.pair_idx.astype(np.int64),
            "pair_label": ps.pair_label.astype(np.int64),  # -1 for pad
            "pair_mask": ps.pair_mask.astype(np.uint8),
            "geom_feat": geom_out.geom_feat.astype(np.float32),
            # Optional metadata
            "geom_dim": int(geom_out.geom_dim),
            "num_obj": int(G),
        }
        out_path = os.path.join(out_dir, f"{sample.image_id:08d}.pt")
        
        try:
            torch_save(pack, out_path)
            success_count += 1
            n += 1
            
            # 调试信息：每成功处理一个就输出（特别是在limit模式下）
            # 使用print而不是pbar.write，确保立即输出
            if limit > 0:
                # limit模式下，每个成功都输出
                print(f"\n  ✓ [{success_count}/{limit}] Successfully saved pt file: {os.path.basename(out_path)} (image_id={sample.image_id}, num_obj={G})")
            elif success_count % 10 == 0:
                # 非limit模式下，每10个输出一次
                print(f"\n  ✓ Successfully processed {success_count} pt files...")
            
            # 更新进度条描述
            pbar.set_description(f"Precomputing (success: {success_count}, fail: {fail_count})")
            
            # Check limit after successful processing
            # limit <= 0 means no limit, process all images
            # limit > 0 means process only first N successful images
            if limit > 0 and success_count >= limit:
                print(f"\n  ✓ Reached limit of {limit} successful images. Stopping.")
                pbar.close()
                break
        except Exception as e:
            fail_count += 1
            skip_reasons[f"save_error: {str(e)}"] = skip_reasons.get(f"save_error: {str(e)}", 0) + 1
            print(f"\n  ✗ Failed to save pt file for image_id={sample.image_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Periodic save: update metadata every save_every images
        if n - last_save_count >= save_every:
            metadata = {
                "total_images": n,
                "success_count": success_count,
                "fail_count": fail_count,
                "skip_reasons": skip_reasons,
                "config": {
                    "vg_root": vg_root,
                    "split": split,
                    "P": P,
                    "neg_ratio": neg_ratio,
                    "mask_size": mask_size,
                    "seed": seed,
                    "sam3_impl": sam3_impl,
                },
            }
            metadata_path = os.path.join(out_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            last_save_count = n
            print(f"\n  [Progress] Saved {success_count} images, {fail_count} failed (total processed: {n})")
    
    reader.close()
    
    # Save metadata
    metadata = {
        "total_images": n,
        "success_count": success_count,
        "fail_count": fail_count,
        "skip_reasons": skip_reasons,
        "config": {
            "vg_root": vg_root,
            "split": split,
            "P": P,
            "neg_ratio": neg_ratio,
            "mask_size": mask_size,
            "seed": seed,
            "sam3_impl": sam3_impl,
        },
    }
    
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Cache building completed!")
    print(f"  Success: {success_count}/{n}")
    print(f"  Failed: {fail_count}/{n}")
    print(f"  Output directory: {out_dir}")
    print(f"  Metadata saved to: {metadata_path}")
    print(f"{'='*60}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cache for VG150 offline preprocessing")
    p.add_argument("--vg_root", type=str, required=True, help="VG150 dataset root")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for cache")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--P", type=int, default=128, help="Fixed number of pairs per image")
    p.add_argument("--neg_ratio", type=int, default=3, help="Negative sampling ratio")
    p.add_argument("--mask_size", type=int, default=256, help="Mask resolution (Hm=Wm)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--limit", type=int, default=-1, help="Limit number of images (-1 or 0 = no limit, or set to N for quick validation on first N images)")
    p.add_argument("--sam3_impl", type=str, default="real", choices=["real", "dummy"], help="SAM3 implementation")
    p.add_argument("--device", type=str, default="cuda", help="Device for SAM3")
    p.add_argument("--save_every", type=int, default=10, help="Save metadata every N images")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_cache(
        vg_root=args.vg_root,
        out_dir=args.out_dir,
        split=args.split,
        P=args.P,
        neg_ratio=args.neg_ratio,
        mask_size=args.mask_size,
        seed=args.seed,
        limit=args.limit,
        sam3_impl=args.sam3_impl,
        device=args.device,
        save_every=args.save_every,
    )

