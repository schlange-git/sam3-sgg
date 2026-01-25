"""
Inspect cache .pt files and convert to JSON format for debugging
"""

'''
python sgg/utils/inspect_cache.py --cache_file sgg/cache/train/00000001.pt --output_json single.json
'''


import argparse
import json
import os
import numpy as np
import torch
from typing import Dict, Any


def pt_to_dict(pt_path: str) -> Dict[str, Any]:
    """
    Load .pt file and convert to JSON-serializable dict
    
    Args:
        pt_path: Path to .pt file
        
    Returns:
        Dictionary with converted data
    """
    pack = torch.load(pt_path, map_location="cpu", weights_only=False)
    
    result = {
        "image_id": int(pack.get("image_id", -1)),
        "num_obj": int(pack.get("num_obj", -1)),
        "geom_dim": int(pack.get("geom_dim", -1)),
    }
    
    # Convert numpy arrays to lists with statistics
    if "geom_feat" in pack:
        geom = pack["geom_feat"]
        result["geom_feat"] = {
            "shape": list(geom.shape),
            "dtype": str(geom.dtype),
            "min": float(geom.min()) if geom.size > 0 else None,
            "max": float(geom.max()) if geom.size > 0 else None,
            "mean": float(geom.mean()) if geom.size > 0 else None,
            "std": float(geom.std()) if geom.size > 0 else None,
            "has_nan": bool(np.isnan(geom).any()),
            "has_inf": bool(np.isinf(geom).any()),
            "sample_values": geom[:5].tolist() if geom.size > 0 else [],  # First 5 values
        }
    
    if "pair_idx" in pack:
        pair_idx = pack["pair_idx"]
        result["pair_idx"] = {
            "shape": list(pair_idx.shape),
            "dtype": str(pair_idx.dtype),
            "min": int(pair_idx.min()) if pair_idx.size > 0 else None,
            "max": int(pair_idx.max()) if pair_idx.size > 0 else None,
            "sample_values": pair_idx[:5].tolist() if pair_idx.size > 0 else [],
        }
    
    if "pair_label" in pack:
        pair_label = pack["pair_label"]
        label_counts = {}
        unique_labels, counts = np.unique(pair_label, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_counts[int(label)] = int(count)
        
        result["pair_label"] = {
            "shape": list(pair_label.shape),
            "dtype": str(pair_label.dtype),
            "min": int(pair_label.min()) if pair_label.size > 0 else None,
            "max": int(pair_label.max()) if pair_label.size > 0 else None,
            "label_counts": label_counts,
            "num_valid": int((pair_label >= 0).sum()),
            "num_padding": int((pair_label < 0).sum()),
            "sample_values": pair_label[:10].tolist() if pair_label.size > 0 else [],
        }
    
    if "pair_mask" in pack:
        pair_mask = pack["pair_mask"]
        result["pair_mask"] = {
            "shape": list(pair_mask.shape),
            "dtype": str(pair_mask.dtype),
            "num_valid": int(pair_mask.sum()),
            "num_invalid": int((pair_mask == 0).sum()),
            "sample_values": pair_mask[:10].tolist() if pair_mask.size > 0 else [],
        }
    
    # Compute valid pairs statistics
    if "pair_label" in pack and "pair_mask" in pack:
        valid_mask = (pack["pair_mask"] > 0) & (pack["pair_label"] >= 0)
        num_valid = int(valid_mask.sum())
        result["statistics"] = {
            "total_pairs": int(pack["pair_mask"].shape[0]),
            "valid_pairs": num_valid,
            "invalid_pairs": int(pack["pair_mask"].shape[0] - num_valid),
            "valid_ratio": float(num_valid / pack["pair_mask"].shape[0]) if pack["pair_mask"].shape[0] > 0 else 0.0,
        }
        
        # Label distribution for valid pairs only
        if num_valid > 0:
            valid_labels = pack["pair_label"][valid_mask]
            unique_valid, counts_valid = np.unique(valid_labels, return_counts=True)
            valid_label_dist = {}
            for label, count in zip(unique_valid, counts_valid):
                valid_label_dist[int(label)] = int(count)
            result["statistics"]["valid_label_distribution"] = valid_label_dist
    
    return result


def inspect_cache_file(pt_path: str, output_json: str = None) -> None:
    """
    Inspect a single cache file
    
    Args:
        pt_path: Path to .pt file
        output_json: Optional path to save JSON output
    """
    print(f"Inspecting: {pt_path}")
    print("=" * 60)
    
    data = pt_to_dict(pt_path)
    
    # Print summary
    print(f"Image ID: {data['image_id']}")
    print(f"Number of objects: {data['num_obj']}")
    print(f"Geometry dimension: {data['geom_dim']}")
    
    if "statistics" in data:
        stats = data["statistics"]
        print(f"\nPair Statistics:")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Valid pairs: {stats['valid_pairs']} ({stats['valid_ratio']*100:.1f}%)")
        print(f"  Invalid pairs: {stats['invalid_pairs']}")
        
        if "valid_label_distribution" in stats:
            print(f"\n  Valid label distribution:")
            for label, count in sorted(stats["valid_label_distribution"].items()):
                label_name = "background" if label == 0 else f"predicate_{label}"
                print(f"    {label_name} (label={label}): {count}")
    
    if "pair_label" in data:
        print(f"\nLabel Statistics:")
        print(f"  Valid labels (>=0): {data['pair_label']['num_valid']}")
        print(f"  Padding labels (<0): {data['pair_label']['num_padding']}")
        print(f"  Label distribution: {data['pair_label']['label_counts']}")
    
    if "geom_feat" in data:
        geom = data["geom_feat"]
        print(f"\nGeometry Features:")
        print(f"  Shape: {geom['shape']}")
        print(f"  Range: [{geom['min']:.4f}, {geom['max']:.4f}]")
        print(f"  Mean: {geom['mean']:.4f}, Std: {geom['std']:.4f}")
        print(f"  Has NaN: {geom['has_nan']}, Has Inf: {geom['has_inf']}")
    
    # Save to JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to: {output_json}")
    
    print("=" * 60)


def inspect_cache_dir(cache_dir: str, num_samples: int = 5, output_json: str = None) -> None:
    """
    Inspect multiple cache files from a directory
    
    Args:
        cache_dir: Directory containing .pt files
        num_samples: Number of files to inspect
        output_json: Optional path to save JSON output for all files
    """
    pt_files = sorted([f for f in os.listdir(cache_dir) if f.endswith(".pt")])
    
    if len(pt_files) == 0:
        print(f"No .pt files found in {cache_dir}")
        return
    
    print(f"Found {len(pt_files)} cache files")
    print(f"Inspecting first {min(num_samples, len(pt_files))} files:\n")
    
    all_data = {}
    
    for i, pt_file in enumerate(pt_files[:num_samples]):
        pt_path = os.path.join(cache_dir, pt_file)
        data = pt_to_dict(pt_path)
        all_data[pt_file] = data
        
        print(f"\n[{i+1}/{min(num_samples, len(pt_files))}] {pt_file}")
        if "statistics" in data:
            stats = data["statistics"]
            print(f"  Valid pairs: {stats['valid_pairs']}/{stats['total_pairs']} ({stats['valid_ratio']*100:.1f}%)")
    
    # Overall statistics
    if len(all_data) > 0:
        total_pairs = sum(d.get("statistics", {}).get("total_pairs", 0) for d in all_data.values())
        total_valid = sum(d.get("statistics", {}).get("valid_pairs", 0) for d in all_data.values())
        print(f"\n{'='*60}")
        print(f"Overall Statistics (from {len(all_data)} files):")
        print(f"  Total pairs: {total_pairs}")
        print(f"  Valid pairs: {total_valid} ({100*total_valid/total_pairs:.1f}%)")
    
    # Save to JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(all_data, f, indent=2)
        print(f"\nSaved all data to: {output_json}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect cache .pt files")
    p.add_argument("--cache_file", type=str, help="Path to a single .pt file")
    p.add_argument("--cache_dir", type=str, help="Directory containing .pt files")
    p.add_argument("--num_samples", type=int, default=5, help="Number of files to inspect from directory")
    p.add_argument("--output_json", type=str, help="Path to save JSON output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.cache_file:
        inspect_cache_file(args.cache_file, args.output_json)
    elif args.cache_dir:
        inspect_cache_dir(args.cache_dir, args.num_samples, args.output_json)
    else:
        print("Please specify either --cache_file or --cache_dir")
        print("Example:")
        print("  python sgg/utils/inspect_cache.py --cache_file sgg/cache/train/00000001.pt")
        print("  python sgg/utils/inspect_cache.py --cache_dir sgg/cache/train --num_samples 10")

