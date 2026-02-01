import sys
import os
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import argparse
from typing import List
import multiprocessing as mp

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.structures import ImageList

from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.data.tools import register_datasets
from SpeaQ.modeling.backbone.sam3_backbone import Sam3MaskedBackbone


def build_cfg(config_file: str, opts: List[str]):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def process_record(record, backbone, cfg, output_dir, device):
    """处理单个图像记录（用于多进程）"""
    try:
        image_id = record.get("image_id")
        if image_id is None:
            return None
        
        out_path = os.path.join(output_dir, f"{image_id}.pt")
        if os.path.isfile(out_path):
            return None  # 已存在，跳过

        image = utils.read_image(record["file_name"], format=cfg.INPUT.FORMAT)
        image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        images = ImageList.from_tensors([image_tensor.to(device)])

        with torch.no_grad():
            features = backbone(images)["sam3"]
        feat = features.tensors[0].to("cpu")
        payload = {
            "image_id": image_id,
            "image_size": (record.get("height"), record.get("width")),
            "feature": feat,
            "feature_stride": backbone.feature_stride,
        }
        torch.save(payload, out_path)
        
        # 清理显存
        del features, feat, images, image_tensor
        torch.cuda.empty_cache() if device.startswith("cuda") else None
        
        return image_id
    except Exception as e:
        print(f"Error processing image_id {record.get('image_id')}: {e}", flush=True)
        return None


def process_chunk_wrapper(records_chunk, config_file, opts, output_dir, device, chunk_idx):
    """包装函数：在多进程环境中重新构建 cfg（因为 cfg 对象不能序列化）"""
    # 在每个进程中重新构建 cfg 并注册数据集
    cfg = build_cfg(config_file, opts)
    register_datasets(cfg)  # 需要在每个进程中注册数据集
    return process_chunk(records_chunk, cfg, output_dir, device, chunk_idx)


def process_chunk(records_chunk, cfg, output_dir, device, chunk_idx):
    """处理一批记录（用于多进程）"""
    print(f"[GPU {device}] Processing chunk {chunk_idx}: {len(records_chunk)} images", flush=True)
    
    # 在 chunk 开始时创建一次 backbone（避免每个记录都创建）
    cfg_clone = cfg.clone()
    cfg_clone.defrost()
    cfg_clone.MODEL.DEVICE = device
    cfg_clone.MODEL.SAM3.DEVICE = device
    cfg_clone.MODEL.SAM3.USE_PRECOMPUTED = False
    cfg_clone.freeze()
    
    backbone = Sam3MaskedBackbone(cfg_clone)
    backbone.eval()
    backbone = backbone.to(device)
    
    processed = 0
    try:
        for record in records_chunk:
            result = process_record(record, backbone, cfg, output_dir, device)
            if result is not None:
                processed += 1
                if processed % 10 == 0:
                    print(f"[GPU {device}] Processed {processed}/{len(records_chunk)} images", flush=True)
    finally:
        # 清理显存
        del backbone
        torch.cuda.empty_cache() if device.startswith("cuda") else None
    
    print(f"[GPU {device}] Chunk {chunk_idx} completed: {processed}/{len(records_chunk)} images processed", flush=True)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute SAM3 feature maps.")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-dir", default="data/featuremaps")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use for parallel processing")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = build_cfg(args.config_file, args.opts)
    os.makedirs(args.output_dir, exist_ok=True)

    register_datasets(cfg)
    dataset_name = cfg.DATASETS.TRAIN[0]
    dataset_dicts = DatasetCatalog.get(dataset_name)
    limit = args.limit
    if limit <= 0:
        limit = cfg.DATASETS.VISUAL_GENOME.OVERFIT_NUM_IMAGES
    if limit and limit > 0:
        dataset_dicts = dataset_dicts[: limit]

    # 过滤已处理的图像
    records_to_process = []
    for record in dataset_dicts:
        image_id = record.get("image_id")
        if image_id is None:
            continue
        out_path = os.path.join(args.output_dir, f"{image_id}.pt")
        if not os.path.isfile(out_path):
            records_to_process.append(record)
    
    print(f"Total images: {len(dataset_dicts)}, Already processed: {len(dataset_dicts) - len(records_to_process)}, To process: {len(records_to_process)}")
    
    if len(records_to_process) == 0:
        print("All images already processed!")
        return

    # 确定使用的 GPU 数量
    num_gpus = args.num_gpus
    if num_gpus <= 0:
        num_gpus = 1
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        num_gpus = min(num_gpus, available_gpus)
    else:
        num_gpus = 1
    
    if num_gpus == 1:
        # 单 GPU 模式（原始逻辑）
        cfg_clone = cfg.clone()
        cfg_clone.defrost()
        cfg_clone.MODEL.DEVICE = cfg.MODEL.SAM3.DEVICE
        cfg_clone.MODEL.SAM3.USE_PRECOMPUTED = False
        cfg_clone.freeze()
        backbone = Sam3MaskedBackbone(cfg_clone)
        backbone.eval()

        for record in records_to_process:
            image_id = record.get("image_id")
            if image_id is None:
                continue
            out_path = os.path.join(args.output_dir, f"{image_id}.pt")
            if os.path.isfile(out_path):
                continue

            image = utils.read_image(record["file_name"], format=cfg.INPUT.FORMAT)
            image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            images = ImageList.from_tensors([image_tensor.to(cfg.MODEL.SAM3.DEVICE)])

            with torch.no_grad():
                features = backbone(images)["sam3"]
            feat = features.tensors[0].to("cpu")
            payload = {
                "image_id": image_id,
                "image_size": (record.get("height"), record.get("width")),
                "feature": feat,
                "feature_stride": backbone.feature_stride,
            }
            torch.save(payload, out_path)
    else:
        # 多 GPU 模式
        print(f"Using {num_gpus} GPUs for parallel processing")
        
        # 将数据集分成多个块
        chunk_size = (len(records_to_process) + num_gpus - 1) // num_gpus
        chunks = [records_to_process[i:i+chunk_size] for i in range(0, len(records_to_process), chunk_size)]
        
        # 确保块数不超过 GPU 数
        chunks = chunks[:num_gpus]
        devices = [f"cuda:{i}" for i in range(len(chunks))]
        
        print(f"Split into {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  GPU {i} ({devices[i]}): {len(chunk)} images")
        
        # 使用多进程处理（spawn 模式，因为 CUDA context 不能 fork）
        try:
            mp_start_method = mp.get_start_method()
            if mp_start_method != 'spawn':
                mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # 准备进程参数（将所有参数打包，因为 multiprocessing 需要可序列化的参数）
        # 注意：cfg 对象不能直接序列化，需要传递配置文件的路径和 opts
        process_args = [
            (chunk, args.config_file, args.opts, args.output_dir, device, i) 
            for i, (chunk, device) in enumerate(zip(chunks, devices))
        ]
        
        # 使用进程池并行处理
        try:
            with mp.Pool(processes=len(chunks)) as pool:
                results = pool.starmap(process_chunk_wrapper, process_args)
            
            total_processed = sum(results)
            print(f"Completed! Processed {total_processed} images using {num_gpus} GPUs")
        finally:
            # 确保所有 GPU 资源都被释放
            print("Cleaning up GPU resources...")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        torch.cuda.set_device(i)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception as e:
                        print(f"Warning: Failed to cleanup GPU {i}: {e}")
            print("GPU cleanup completed")


if __name__ == "__main__":
    main()
