"""
IO utilities
"""
import os
import torch


def ensure_dir(path: str) -> None:
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def torch_save(obj, path: str) -> None:
    """Save torch object with directory creation"""
    ensure_dir(os.path.dirname(path))
    torch.save(obj, path)


def torch_load(path: str, map_location="cpu"):
    """Load torch object"""
    return torch.load(path, map_location=map_location)

