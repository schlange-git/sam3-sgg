#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单图像校验脚本：
递归遍历指定目录，尝试用 PIL 打开所有图片文件；
如果失败（截断 / 损坏 / 非法格式），把路径记到一个 txt 里。
"""

import argparse
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, UnidentifiedImageError


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _collect_image_paths(root: str) -> List[str]:
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if is_image_file(path):
                image_paths.append(path)
    return image_paths


def _check_one_image(path: str):
    try:
        with Image.open(path) as im:
            im.load()  # 触发底层解码
        return path, None
    except (OSError, UnidentifiedImageError) as e:
        return path, str(e)


def scan_images(root: str, workers: int) -> List[str]:
    broken = []
    image_paths = _collect_image_paths(root)
    total = len(image_paths)

    print(f"[info] found {total} images, workers={workers}", flush=True)
    if total == 0:
        print("[done] scanned=0, broken=0")
        return broken

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, (path, err) in enumerate(executor.map(_check_one_image, image_paths), start=1):
            if idx % 1000 == 0:
                print(f"[check] scanned {idx} images, broken={len(broken)}", flush=True)
            if err is not None:
                print(f"[BROKEN] {path}  ({err})", flush=True)
                broken.append(path)

    print(f"[done] scanned={total}, broken={len(broken)}")
    return broken


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="要检查的图像根目录，比如 dataset/frames 或 dataset/videos_frames_root",
    )
    parser.add_argument(
        "--output",
        default="broken_images.txt",
        help="坏图像列表输出路径 (txt)，默认 broken_images.txt",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, (os.cpu_count() or 8) * 4)),
        help="线程数，默认按 CPU 数自动设置",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    print(f"[info] scanning image root: {root}")
    broken = scan_images(root, workers=max(1, args.workers))

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in broken:
            f.write(os.path.abspath(p) + "\n")
    print(f"[info] written {len(broken)} broken images to: {out_path}")


if __name__ == "__main__":
    main()