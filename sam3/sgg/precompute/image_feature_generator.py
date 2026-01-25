"""
SAM3 Image Feature Generator
使用 SAM3 作为 backbone 提取整个图像的 feature（不作用于具体的 box）

区别于现有的 mask 生成工具（针对具体 GT box），本工具提取的是整图级别的 feature，
可以作为下游任务的 backbone feature。

参考设计：
- Phase B: 离线预处理阶段
- 使用 SAM3 的 image encoder 提取整图 feature
- 批量处理并保存，供后续使用
"""
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model


def collate_fn_pil_images(batch):
    """
    自定义 collate 函数：处理包含 PIL 图像的批次
    对于 PIL 图像，直接返回列表（不尝试 stack）
    对于其他数据（如 image_id, idx），使用默认 collate
    """
    if len(batch) == 0:
        return {}
    
    # 检查是否是错误批次
    if "error" in batch[0]:
        # 如果有错误标记，返回错误信息列表
        return {
            "error": [item.get("error", "Unknown error") for item in batch],
            "idx": [item.get("idx", -1) for item in batch],
        }
    
    # 分离 PIL 图像和其他数据
    result = {}
    pil_keys = []  # 包含 PIL 图像的 key
    other_keys = []  # 可以正常 collate 的 key
    
    # 先扫描所有 key
    for key in batch[0].keys():
        sample_value = batch[0][key]
        if isinstance(sample_value, Image.Image):
            pil_keys.append(key)
        else:
            other_keys.append(key)
    
    # 处理 PIL 图像（直接返回列表）
    for key in pil_keys:
        result[key] = [item[key] for item in batch]
    
    # 处理其他数据（使用默认 collate）
    if other_keys:
        try:
            from torch.utils.data._utils.collate import default_collate
            other_batch = [{k: item[k] for k in other_keys} for item in batch]
            collated_other = default_collate(other_batch)
            result.update(collated_other)
        except Exception as e:
            # 如果默认 collate 失败，手动处理
            for key in other_keys:
                values = [item[key] for item in batch]
                # 尝试转换为 tensor（如果是数字）
                try:
                    if isinstance(values[0], (int, float)):
                        result[key] = torch.tensor(values)
                    else:
                        result[key] = values
                except:
                    result[key] = values
    
    return result


class ImageOnlyDataset(Dataset):
    """
    包装数据集，只返回图像和 image_id（用于 DataLoader）
    便于使用 num_workers 进行多线程数据加载
    """
    def __init__(self, base_dataset, max_samples: Optional[int] = None):
        self.base_dataset = base_dataset
        self.max_samples = max_samples
        self.total_samples = len(base_dataset)
        if max_samples is not None and max_samples > 0:
            self.total_samples = min(self.total_samples, max_samples)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        try:
            item = self.base_dataset[idx]
            # 提取图像
            if "image_pil" in item:
                image = item["image_pil"]
            elif "image" in item:
                img_tensor = item["image"]
                if isinstance(img_tensor, torch.Tensor):
                    # 反归一化并转换为 PIL
                    img_tensor = img_tensor.permute(1, 2, 0)
                    img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(img_array)
                else:
                    image = img_tensor
            else:
                raise ValueError(f"Dataset item {idx} does not contain 'image_pil' or 'image'")
            
            image_id = item.get("image_id", idx)
            return {"image": image, "image_id": image_id, "idx": idx}
        except Exception as e:
            # 返回错误标记，后续会跳过
            return {"error": str(e), "idx": idx}


def _extract_features_worker_multi_gpu(
    data_root: str,
    indices: List[int],
    device: str,
    feature_dim: int,
    image_size: int,
    checkpoint_path: Optional[str],
    split: str,
    batch_size: int = 8,
    num_workers: int = 2,
    output_dir: str = None,
    filename_prefix: str = "image_feat",
    save_fp16: bool = False,
) -> Tuple[List[int], int, str]:
    """
    Worker 函数：在指定 GPU 上处理图像子集（用于多进程，使用流水线方法）
    
    注意：这个函数会在独立的进程中运行，每个进程独立加载模型和数据集
    使用 DataLoader 进行流水线处理，边加载边推理边保存
    
    Args:
        data_root: 数据集根目录
        indices: 要处理的图像索引列表
        device: GPU 设备（如 "cuda:0"）
        feature_dim: 特征维度
        image_size: 图像尺寸
        checkpoint_path: Checkpoint 路径
        split: 数据集 split
        batch_size: 批处理大小
        num_workers: DataLoader 工作线程数
        output_dir: 输出目录（每个进程保存到自己的子目录）
        filename_prefix: 文件名前缀
        
    Returns:
        (image_ids, processed_count): 处理的图像 ID 列表和处理数量
    """
    import sys
    import numpy as np
    from pathlib import Path
    
    # 添加项目根目录到路径（spawn 模式需要）
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from sgg.datasets.vg150_dataset import VG150Dataset
    from torch.utils.data import Subset, DataLoader
    
    # 在 worker 进程中创建数据集和模型（避免共享 CUDA context 问题）
    # 注意：使用 lazy loading，只加载必要的索引
    print(f"[GPU {device}] Loading dataset (lazy mode)...", flush=True)
    dataset = VG150Dataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
    )
    
    # 创建子集（只处理指定的索引）
    subset = Subset(dataset, indices)
    
    # 创建包装数据集
    wrapped_dataset = ImageOnlyDataset(subset, max_samples=None)
    
    print(f"[GPU {device}] Dataset loaded: {len(wrapped_dataset)} images", flush=True)
    
    # 在 worker 进程中创建 feature generator（每个进程独立加载模型）
    # 使用 torch.cuda.empty_cache() 清理缓存
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    print(f"[GPU {device}] Loading SAM3 model...", flush=True)
    generator = SAM3ImageFeatureGenerator(
        device=device,
        feature_dim=feature_dim,
        image_size=image_size,
        checkpoint_path=checkpoint_path,
        save_fp16=save_fp16,
    )
    print(f"[GPU {device}] Model loaded successfully", flush=True)
    
    # 再次清理缓存
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # 创建输出目录（直接保存到最终目录，因为文件名包含 image_id，不会冲突）
    if output_dir is None:
        # 如果未指定输出目录，使用临时目录
        output_dir = f"/tmp/sam3_features_{device.replace(':', '_')}"
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    # 创建 DataLoader（流水线处理）
    # 注意：在 multiprocessing spawn 模式的 worker 进程中，不要使用 num_workers
    # 因为这会导致嵌套的多进程，可能导致死锁。在主进程中才使用 num_workers
    # 需要使用自定义 collate_fn 来处理 PIL 图像
    dataloader = DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 在 worker 进程中禁用多线程，避免嵌套多进程问题
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        collate_fn=collate_fn_pil_images,  # 自定义 collate 函数，处理 PIL 图像
    )
    
    # 流水线处理：边加载边推理边保存
    image_ids_list = []
    processed_count = 0
    error_count = 0
    
    # 计算总批次数用于进度显示
    total_batches = (len(wrapped_dataset) + batch_size - 1) // batch_size
    
    print(f"[GPU {device}] Starting to process {len(indices)} images in {total_batches} batches (batch_size={batch_size})...", flush=True)
    print(f"[GPU {device}] Dataset indices range: [{indices[0] if len(indices) > 0 else 'N/A'}, {indices[-1] if len(indices) > 0 else 'N/A'}]", flush=True)
    print(f"[GPU {device}] Note: Each index maps to dataset[idx], which returns the actual image_id from the dataset", flush=True)
    start_time = time.time()
    last_progress_time = start_time
    
    for batch_idx, batch in enumerate(dataloader):
        # 检查是否是错误批次
        if "error" in batch:
            error_messages = batch["error"]
            error_indices = batch.get("idx", [])
            for error_msg, error_idx in zip(error_messages, error_indices):
                error_count += 1
                if error_count % 100 == 0 or error_count == 1:
                    print(f"[GPU {device}] Warning: {error_count} errors so far. Last error: {error_msg}", flush=True)
            continue
        
        # 处理正常批次
        batch_images = batch["image"]  # List[PIL.Image]
        batch_image_ids = batch["image_id"]  # List[int] 或 tensor
        
        # 确保 batch_image_ids 是列表
        if isinstance(batch_image_ids, torch.Tensor):
            batch_image_ids = batch_image_ids.tolist()
        elif not isinstance(batch_image_ids, list):
            batch_image_ids = list(batch_image_ids)
        
        for image, image_id in zip(batch_images, batch_image_ids):
            # 检查图像是否有效（应该是 PIL.Image）
            if not isinstance(image, Image.Image):
                error_count += 1
                if error_count % 100 == 0 or error_count == 1:
                    print(f"[GPU {device}] Warning: Invalid image type at batch {batch_idx}: {type(image)}", flush=True)
                continue
            
            try:
                # 提取 dense 特征
                feat, stride, spatial_hw = generator.extract_image_feature(image)
                
                # 立即保存并释放GPU内存
                feat_file = os.path.join(split_dir, f"{filename_prefix}_{int(image_id):08d}.pt")
                feat_cpu = feat.detach().cpu()
                if save_fp16:
                    feat_cpu = feat_cpu.half()
                del feat  # 立即释放GPU内存
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                
                torch.save({
                    "image_id": int(image_id),
                    "feature": feat_cpu,  # [C, Hf, Wf]
                    "feature_dim": int(feat_cpu.shape[0]),
                    "stride": int(stride),
                    "img_size": (generator.image_size, generator.image_size),
                    "spatial_shape": spatial_hw,
                    "dtype": "fp16" if save_fp16 else "fp32",
                }, feat_file)
                
                # 释放CPU tensor
                del feat_cpu
                
                image_ids_list.append(int(image_id))
                processed_count += 1
                
                # 每处理 50 张图像输出一次进度（多GPU模式下更频繁的输出，因为没有tqdm）
                current_time = time.time()
                if processed_count % 50 == 0 or (current_time - last_progress_time) >= 10.0:  # 每50张或每10秒
                    elapsed_time = current_time - start_time
                    progress_pct = (processed_count / len(indices)) * 100 if len(indices) > 0 else 0
                    speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (len(indices) - processed_count) / speed if speed > 0 and len(indices) > 0 else 0
                    eta_minutes = eta_seconds / 60
                    # 显示当前处理的image_id（用于验证一致性）
                    print(f"[GPU {device}] Progress: {processed_count}/{len(indices)} ({progress_pct:.1f}%) | "
                          f"Speed: {speed:.2f} img/s | ETA: {eta_minutes:.1f} min | Errors: {error_count} | "
                          f"Current image_id: {int(image_id)}", flush=True)
                    last_progress_time = current_time
                    
                    # 定期清理GPU缓存
                    if device.startswith('cuda') and processed_count % 500 == 0:
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                
            except Exception as e:
                error_count += 1
                if error_count % 100 == 0 or error_count == 1:
                    print(f"[GPU {device}] Warning: {error_count} errors so far. Last error: {e}", flush=True)
                # 使用零特征图作为 fallback
                default_h = max(1, generator.image_size // generator.feature_stride)
                spatial_hw = (default_h, default_h)
                zero_feat = torch.zeros((feature_dim, spatial_hw[0], spatial_hw[1]), dtype=torch.float32)
                feat_file = os.path.join(split_dir, f"{filename_prefix}_{int(image_id):08d}.pt")
                torch.save({
                    "image_id": int(image_id),
                    "feature": zero_feat,
                    "feature_dim": feature_dim,
                    "stride": generator.feature_stride,
                    "img_size": (generator.image_size, generator.image_size),
                    "spatial_shape": spatial_hw,
                }, feat_file)
                image_ids_list.append(int(image_id))
                processed_count += 1
        
        # 清理batch相关的tensor
        del batch_images, batch_image_ids, batch
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    print(f"[GPU {device}] Completed: Processed {processed_count} images (errors: {error_count})", flush=True)
    
    return image_ids_list, processed_count, split_dir


class SAM3ImageFeatureGenerator:
    """
    使用 SAM3 作为 backbone 提取整图 feature
    
    核心功能：
    1. 加载 SAM3 模型（冻结参数）
    2. 对输入图像提取整图 feature（不依赖于具体的 box）
    3. 批量处理数据集并保存 feature
    """
    
    def __init__(
        self,
        device: str = "cuda",
        feature_dim: int = 256,
        image_size: int = 1008,  # SAM3 默认分辨率为 1008（不是 1024）
        checkpoint_path: Optional[str] = None,
        save_fp16: bool = False,
    ):
        """
        Args:
            device: 计算设备
            feature_dim: 输出 feature 维度（默认 256）
            image_size: 输入图像尺寸（SAM3 的标准输入尺寸，默认 1008）
            checkpoint_path: 本地 checkpoint 文件路径（None 表示尝试自动查找或从 HF 下载）
        """
        self.device = device
        self.feature_dim = feature_dim
        self.image_size = image_size  # SAM3 默认 1008
        self.save_fp16 = save_fp16
        # SAM3 主干的下采样步长，视觉分支一般是 16
        self.feature_stride = 16
        # 针对 dense feature 的 1x1 投影层（懒加载，保持冻结）
        self._proj_layer: Optional[nn.Module] = None
        
        print("Loading SAM3 model as backbone for image feature extraction...")
        
        # 查找本地 checkpoint 文件（如果未指定）
        if checkpoint_path is None:
            checkpoint_path = self._find_local_checkpoint()
        
        # 加载 SAM3 模型
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            self.sam3_model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=False,  # 禁用从 HF 下载
            )
        else:
            # 如果找不到本地 checkpoint，尝试从 HF 下载（可能会失败）
            if checkpoint_path:
                print(f"Warning: Checkpoint path specified but file not found: {checkpoint_path}")
                print("Attempting to find checkpoint automatically or download from HuggingFace...")
            else:
                print("No checkpoint path specified. Attempting to find checkpoint automatically or download from HuggingFace...")
            
            # 尝试从 HF 下载（可能会因为权限问题失败）
            try:
                self.sam3_model = build_sam3_image_model(
                    checkpoint_path=None,
                    load_from_HF=True,
                )
            except Exception as e:
                error_msg = (
                    f"Failed to load SAM3 model:\n"
                    f"  Error: {str(e)}\n"
                    f"  Please ensure you have a local checkpoint file (sam3.pt) in one of these locations:\n"
                    f"    - ./sam3.pt (project root)\n"
                    f"    - ./checkpoints/sam3.pt\n"
                    f"    - ./sgg/sam3.pt\n"
                    f"  Or specify the checkpoint path using --checkpoint_path argument."
                )
                raise RuntimeError(error_msg) from e
        self.sam3_model = self.sam3_model.to(device)
        self.sam3_model.eval()
        
        # 冻结所有参数
        for p in self.sam3_model.parameters():
            p.requires_grad_(False)
        
        # 创建图像变换（使用与 Sam3Processor 相同的变换方式）
        from torchvision.transforms import v2
        self.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),  # 确保是 uint8
            v2.Resize(size=(image_size, image_size)),  # SAM3 默认 1008x1008
            v2.ToDtype(torch.float32, scale=True),  # 转换为 float32 [0, 255]
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 归一化到 [-1, 1]
        ])
        
        # 保存 checkpoint_path 以供多 GPU 使用
        self._checkpoint_path = checkpoint_path
        
        print(f"SAM3 model loaded on {device}")
        if checkpoint_path:
            print(f"  Checkpoint: {checkpoint_path}")
    
    def _find_local_checkpoint(self) -> Optional[str]:
        """
        查找本地 checkpoint 文件（使用相对路径）
        
        尝试的位置（按优先级）：
        1. 项目根目录下的 sam3.pt
        2. 项目根目录下的 checkpoints/sam3.pt
        3. sam3/checkpoints/sam3.pt
        4. ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/main/sam3.pt
        """
        from pathlib import Path
        
        # 获取项目根目录（当前文件所在目录的上级目录的上级目录）
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # sgg/precompute -> sgg -> project_root
        
        # 候选路径列表（相对路径优先）
        candidate_paths = [
            # 项目根目录下（包括 weights 目录）
            project_root / "sam3.pt",
            project_root / "checkpoints" / "sam3.pt",
            project_root / "weights" / "sam3.pt",  # 添加 weights 目录支持
            project_root / "sam3" / "checkpoints" / "sam3.pt",
            # 当前目录（sgg目录）
            project_root / "sgg" / "sam3.pt",
            project_root / "sgg" / "checkpoints" / "sam3.pt",
            project_root / "sgg" / "weights" / "sam3.pt",
            # 相对于运行目录
            Path("sam3.pt"),
            Path("checkpoints") / "sam3.pt",
            Path("weights") / "sam3.pt",
            Path("../sam3.pt"),
            Path("../checkpoints/sam3.pt"),
            Path("../weights/sam3.pt"),
            # HuggingFace 缓存目录
            Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--sam3" / "snapshots" / "main" / "sam3.pt",
        ]
        
        # 尝试查找存在的文件
        for checkpoint_path in candidate_paths:
            if checkpoint_path.exists() and checkpoint_path.is_file():
                print(f"Found local checkpoint: {checkpoint_path.resolve()}")
                return str(checkpoint_path.resolve())
        
        # 如果都找不到，返回 None（将导致 build_sam3_image_model 尝试从 HF 下载）
        print("Warning: No local checkpoint found, will try to download from HuggingFace (may fail if no access)")
        return None
    
    @torch.no_grad()
    def extract_image_feature(self, image: Image.Image) -> Tuple[torch.Tensor, int, Tuple[int, int]]:
        """
        提取整图的 dense feature map（不做全局池化）
        
        Args:
            image: PIL Image (RGB)
            
        Returns:
            dense_feat: FloatTensor [C, Hf, Wf]（默认 C=feature_dim）
            stride: 下采样步长（推断或默认 16）
            spatial_hw: (Hf, Wf)
        """
        dense_feat = None
        spatial_hw: Tuple[int, int] = (0, 0)
        
        # 将 PIL Image 转换为 tensor
        # 使用与 Sam3Processor 完全相同的变换方式（确保兼容性）
        from torchvision.transforms import v2
        import PIL
        
        # 步骤1: 转换为 tensor（与 Sam3Processor.set_image 相同）
        if isinstance(image, PIL.Image.Image):
            image_tensor = v2.functional.to_image(image)  # PIL -> tensor [H, W, 3], uint8, [0, 255]
        else:
            image_tensor = image
        
        # 步骤2: 移动到设备并应用变换
        image_tensor = image_tensor.to(self.device)  # [H, W, 3]
        image_tensor = self.transform(image_tensor)  # [3, H, W], float32, normalized to [-1, 1]
        image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]
        
        # 使用 SAM3 模型的 backbone 提取特征
        backbone_out = self.sam3_model.backbone.forward_image(image_tensor)
        
        # 立即释放输入tensor（节省GPU内存）
        del image_tensor
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        # 取 FPN 的最高分辨率层；若无则退回 vision_features
        candidate = None
        if 'backbone_fpn' in backbone_out:
            fpn_feats = backbone_out['backbone_fpn']
            if isinstance(fpn_feats, (list, tuple)) and len(fpn_feats) > 0:
                candidate = fpn_feats[-1]  # [1, C, H, W] 或 [1, C, HW]
        if candidate is None and 'vision_features' in backbone_out:
            candidate = backbone_out['vision_features']
        
        # 释放不再使用的输出
        if 'backbone_fpn' in backbone_out:
            del backbone_out['backbone_fpn']
        if 'vision_features' in backbone_out:
            del backbone_out['vision_features']
        del backbone_out
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        # 处理候选特征，得到 dense map
        if candidate is not None:
            if candidate.dim() == 3:
                candidate = candidate.unsqueeze(0)
            if candidate.dim() == 4:
                C_raw = candidate.shape[1]
                Hf = int(candidate.shape[2])
                Wf = int(candidate.shape[3])
                spatial_hw = (Hf, Wf)
                
                # 懒加载 1x1 投影层，将 C_raw -> feature_dim
                if self._proj_layer is None:
                    if C_raw == self.feature_dim:
                        self._proj_layer = nn.Identity()
                    else:
                        # 用确定性拷贝通道的 1x1 卷积：前 min(C_raw, feature_dim) 个通道直连，其余补零
                        self._proj_layer = nn.Conv2d(C_raw, self.feature_dim, kernel_size=1, bias=False)
                        with torch.no_grad():
                            self._proj_layer.weight.zero_()
                            copy_channels = min(C_raw, self.feature_dim)
                            for ch in range(copy_channels):
                                self._proj_layer.weight[ch, ch, 0, 0] = 1.0
                    self._proj_layer.to(self.device)
                    self._proj_layer.eval()
                    for p in self._proj_layer.parameters():
                        p.requires_grad_(False)
                
                # 投影到目标维度
                if isinstance(self._proj_layer, nn.Identity):
                    proj_feat = candidate
                else:
                    proj_feat = self._proj_layer(candidate)
                
                # 去掉 batch 维，得到 [C, Hf, Wf]
                dense_feat = proj_feat.squeeze(0).contiguous() if proj_feat.dim() == 4 else proj_feat
                
                # 释放中间 tensor
                del proj_feat, candidate
                if self.device.startswith('cuda'):
                    torch.cuda.empty_cache()
            else:
                # 非预期形状，直接释放
                del candidate
        
        # Fallback：生成零特征图
        if dense_feat is None:
            if spatial_hw == (0, 0):
                default_h = max(1, self.image_size // self.feature_stride)
                spatial_hw = (default_h, default_h)
            print(f"Warning: Failed to extract dense feature for image {getattr(image, 'size', None)}, returning zero map")
            dense_feat = torch.zeros(
                (self.feature_dim, spatial_hw[0], spatial_hw[1]),
                device=self.device,
                dtype=torch.float32,
            )
        
        # 计算 stride（基于输入尺寸和特征图尺寸的估计）
        stride = max(1, int(round(self.image_size / float(spatial_hw[0])))) if spatial_hw[0] > 0 else self.feature_stride
        
        # 最终清理GPU缓存（如果使用GPU）
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        return dense_feat.float(), stride, spatial_hw
    
    @torch.no_grad()
    def extract_batch_features(
        self,
        images: List[Image.Image],
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        批量提取 dense 图像 features
        
        Args:
            images: List of PIL Images
            batch_size: 批处理大小（SAM3 通常单图处理，但这里可以优化）
            
        Returns:
            features: [N, C, Hf, Wf] float32 tensor
        """
        if len(images) == 0:
            raise ValueError("Cannot extract features from empty image list")
        
        features_list = []
        
        for image in tqdm(images, desc="Extracting image features"):
            try:
                feat, _, spatial_hw = self.extract_image_feature(image)
                features_list.append(feat.cpu())
            except Exception as e:
                print(f"Warning: Failed to extract feature from image: {e}, using zero map")
                # 使用零特征图作为 fallback
                default_h = max(1, self.image_size // self.feature_stride)
                zero_feat = torch.zeros((self.feature_dim, default_h, default_h), dtype=torch.float32)
                features_list.append(zero_feat)
        
        if len(features_list) == 0:
            raise ValueError("No features were successfully extracted")
        
        features = torch.stack(features_list, dim=0)  # [N, C, Hf, Wf]
        return features
    
    def save_features(
        self,
        features: torch.Tensor,
        image_ids: List[int],
        output_dir: str,
        split: str = "train",
        filename_prefix: str = "image_feat"
    ):
        """
        保存 dense features 到文件
        
        Args:
            features: [N, C, Hf, Wf] float32 tensor
            image_ids: List of image IDs (length N)
            output_dir: 输出目录
            split: 数据集 split ("train" / "val")
            filename_prefix: 文件名前缀
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存方式 1: 每个图像一个文件（便于随机访问）
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for idx, img_id in enumerate(image_ids):
            feat = features[idx]  # [C, Hf, Wf]
            feat_to_save = feat.detach().cpu()
            if self.save_fp16:
                feat_to_save = feat_to_save.half()
            
            # 保存为 .pt 文件
            feat_file = os.path.join(split_dir, f"{filename_prefix}_{img_id:08d}.pt")
            torch.save({
                "image_id": img_id,
                "feature": feat_to_save,
                "feature_dim": self.feature_dim,
                "stride": self.feature_stride,
                "img_size": (self.image_size, self.image_size),
                "spatial_shape": (feat.shape[-2], feat.shape[-1]),
                "dtype": "fp16" if self.save_fp16 else "fp32",
            }, feat_file)
        
        # 保存方式 2: 元数据文件（索引映射）
        metadata = {
            "split": split,
            "num_images": len(image_ids),
            "feature_dim": self.feature_dim,
            "stride": self.feature_stride,
            "image_size": self.image_size,
            "image_ids": image_ids,
            "filename_prefix": filename_prefix,
            "dtype": "fp16" if self.save_fp16 else "fp32",
        }
        metadata_file = os.path.join(split_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(image_ids)} image features to {split_dir}")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Metadata: {metadata_file}")
    
    def process_dataset(
        self,
        dataset,
        output_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_idx: int = 0,
        filename_prefix: str = "image_feat",
        batch_size: int = 10,
        num_workers: int = 2,
    ):
        """
        处理整个数据集，提取并保存所有图像的 features（使用 DataLoader 流水线处理）
        
        Args:
            dataset: VG150Dataset 或类似的数据集对象
               需要支持 __len__ 和 __getitem__ 方法
               返回的 item 应该包含 "image_pil" (PIL Image) 和 "image_id" (int)
            output_dir: 输出目录
            split: 数据集 split
            max_samples: 最大处理样本数（None 表示处理全部）
            filename_prefix: 文件名前缀
            batch_size: 批处理大小（默认 8，根据内存调整）
            num_workers: DataLoader 的工作线程数（默认 2，不要设置太高，内存有限）
        """
        dataset_len = len(dataset)
        if start_idx < 0:
            start_idx = 0
        if start_idx >= dataset_len:
            print(f"Warning: start_idx {start_idx} >= dataset size {dataset_len}, nothing to do.")
            return
        total_available = dataset_len - start_idx
        if max_samples is not None and max_samples > 0:
            total_samples = min(total_available, max_samples)
        else:
            total_samples = total_available
        
        if total_samples <= 0:
            print(f"Warning: No samples to process (dataset size: {dataset_len}, start_idx: {start_idx}, max_samples: {max_samples})")
            return
        
        print(f"Processing {total_samples} images from {split} split (start_idx={start_idx})...")
        print(f"  Batch size: {batch_size}, Num workers: {num_workers}")
        
        # 创建包装数据集（只返回图像和 image_id），再用 Subset 做范围裁剪
        wrapped_dataset = ImageOnlyDataset(dataset)
        target_indices = list(range(start_idx, start_idx + total_samples))
        wrapped_dataset = Subset(wrapped_dataset, target_indices)
        
        # 创建 DataLoader（使用多线程预读取，流水线处理）
        # 注意：需要使用自定义 collate_fn 来处理 PIL 图像
        dataloader = DataLoader(
            wrapped_dataset,
            batch_size=batch_size,
            shuffle=False,  # 保持顺序
            num_workers=num_workers,
            pin_memory=False,  # 不使用 pin_memory，因为图像已经是 PIL，不需要 GPU 内存
            persistent_workers=True if num_workers > 0 else False,  # 保持 worker 进程存活
            prefetch_factor=2 if num_workers > 0 else None,  # 每个 worker 预取 2 个 batch
            collate_fn=collate_fn_pil_images,  # 自定义 collate 函数，处理 PIL 图像
        )
        
        # 准备输出目录
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # 流水线处理：按批次加载 -> 推理 -> 保存（实时保存，不积累内存）
        all_image_ids = []
        processed_count = 0
        error_count = 0
        
        print(f"Extracting and saving features (streaming mode with {num_workers} workers)...")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # batch 是一个 dict，包含：
            # - "image": List[PIL.Image]（或 "error" key 表示错误）
            # - "image_id": List[int] 或 tensor
            # - "idx": List[int] 或 tensor
            
            # 检查是否是错误批次
            if "error" in batch:
                error_messages = batch["error"]
                error_indices = batch.get("idx", [])
                for error_msg, error_idx in zip(error_messages, error_indices):
                    error_count += 1
                    if error_count % 100 == 0:
                        print(f"Warning: {error_count} errors encountered so far. Last error: {error_msg}")
                continue
            
            # 处理正常批次
            batch_images = batch["image"]  # List[PIL.Image]
            batch_image_ids = batch["image_id"]  # List[int] 或 tensor
            
            # 确保 batch_image_ids 是列表
            if isinstance(batch_image_ids, torch.Tensor):
                batch_image_ids = batch_image_ids.tolist()
            elif not isinstance(batch_image_ids, list):
                batch_image_ids = list(batch_image_ids)
            
            # 提取每个图像的特征并实时保存
            for i, (image, image_id) in enumerate(zip(batch_images, batch_image_ids)):
                # 检查图像是否有效（应该是 PIL.Image）
                if not isinstance(image, Image.Image):
                    error_count += 1
                    if error_count % 100 == 0:
                        print(f"Warning: Invalid image type at batch {batch_idx}, item {i}: {type(image)}")
                    # 保存零特征图作为 fallback
                    default_h = max(1, self.image_size // self.feature_stride)
                    spatial_hw = (default_h, default_h)
                    zero_feat = torch.zeros((self.feature_dim, spatial_hw[0], spatial_hw[1]), dtype=torch.float32)
                    feat_file = os.path.join(split_dir, f"{filename_prefix}_{int(image_id) if isinstance(image_id, (int, torch.Tensor)) else 0:08d}.pt")
                    torch.save({
                        "image_id": int(image_id) if isinstance(image_id, (int, torch.Tensor)) else 0,
                        "feature": zero_feat,
                        "feature_dim": self.feature_dim,
                        "stride": self.feature_stride,
                        "img_size": (self.image_size, self.image_size),
                        "spatial_shape": spatial_hw,
                        "dtype": "fp16" if self.save_fp16 else "fp32",
                    }, feat_file)
                    continue
                
                try:
                    # 提取 dense 特征
                    feat, stride, spatial_hw = self.extract_image_feature(image)
                    
                    # 立即保存（避免内存积累）
                    feat_file = os.path.join(split_dir, f"{filename_prefix}_{int(image_id):08d}.pt")
                    feat_cpu = feat.detach().cpu()
                    if self.save_fp16:
                        feat_cpu = feat_cpu.half()
                    torch.save({
                        "image_id": int(image_id),
                        "feature": feat_cpu,  # [C, Hf, Wf]
                        "feature_dim": int(feat_cpu.shape[0]),
                        "stride": int(stride),
                        "img_size": (self.image_size, self.image_size),
                        "spatial_shape": spatial_hw,
                        "dtype": "fp16" if self.save_fp16 else "fp32",
                    }, feat_file)
                    
                    all_image_ids.append(int(image_id))
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    if error_count % 100 == 0 or error_count == 1:
                        print(f"Warning: Failed to extract feature for image_id {image_id}: {e}")
                    # 使用零特征图作为 fallback
                    default_h = max(1, self.image_size // self.feature_stride)
                    spatial_hw = (default_h, default_h)
                    zero_feat = torch.zeros((self.feature_dim, spatial_hw[0], spatial_hw[1]), dtype=torch.float32)
                    feat_file = os.path.join(split_dir, f"{filename_prefix}_{int(image_id):08d}.pt")
                    if self.save_fp16:
                        zero_feat = zero_feat.half()
                    torch.save({
                        "image_id": int(image_id),
                        "feature": zero_feat,
                        "feature_dim": self.feature_dim,
                        "stride": self.feature_stride,
                        "img_size": (self.image_size, self.image_size),
                        "spatial_shape": spatial_hw,
                        "dtype": "fp16" if self.save_fp16 else "fp32",
                    }, feat_file)
                    all_image_ids.append(int(image_id))
                    processed_count += 1
        
        # 保存元数据
        metadata = {
            "split": split,
            "num_images": processed_count,
            "feature_dim": self.feature_dim,
            "stride": self.feature_stride,
            "image_size": self.image_size,
            "spatial_shape": (max(1, self.image_size // self.feature_stride), max(1, self.image_size // self.feature_stride)),
            "image_ids": sorted(all_image_ids),  # 排序以便查找
            "filename_prefix": filename_prefix,
        }
        metadata_file = os.path.join(split_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Successfully processed {processed_count} images")
        if error_count > 0:
            print(f"  (Encountered {error_count} errors during processing)")
        print(f"  - Output directory: {split_dir}")
        print(f"  - Metadata: {metadata_file}")
    
    def process_dataset_multi_gpu(
        self,
        dataset,
        output_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_idx: int = 0,
        filename_prefix: str = "image_feat",
        num_gpus: int = 2,
        devices: Optional[List[str]] = None,
        batch_size: int = 10,
        num_workers: int = 2,
        save_fp16: bool = False,
    ):
        """
        多 GPU 并行处理数据集（使用多进程，每个 GPU 一个进程）
        
        Args:
            dataset: VG150Dataset 或类似的数据集对象
            output_dir: 输出目录
            split: 数据集 split
            max_samples: 最大处理样本数（None/0/-1 表示处理全部）
            filename_prefix: 文件名前缀
            num_gpus: GPU 数量（默认 2）
            devices: GPU 设备列表（如 ["cuda:0", "cuda:1"]），None 表示自动检测
        """
        import multiprocessing as mp
        
        dataset_len = len(dataset)
        if start_idx < 0:
            start_idx = 0
        if start_idx >= dataset_len:
            print(f"Warning: start_idx {start_idx} >= dataset size {dataset_len}, nothing to do.")
            return
        total_available = dataset_len - start_idx
        if max_samples is not None and max_samples > 0:
            total_samples = min(total_available, max_samples)
        else:
            total_samples = total_available
        
        if total_samples <= 0:
            print(f"Warning: No samples to process (dataset size: {dataset_len}, start_idx: {start_idx}, max_samples: {max_samples})")
            return
        
        print(f"Processing {total_samples} images from {split} split using {num_gpus} GPUs (start_idx={start_idx})...")
        print(f"  Batch size: {batch_size}, Num workers per GPU: {num_workers}")
        
        # 检测可用 GPU
        if devices is None:
            if torch.cuda.is_available():
                num_available = torch.cuda.device_count()
                num_gpus = min(num_gpus, num_available)
                devices = [f"cuda:{i}" for i in range(num_gpus)]
                print(f"Detected {num_available} GPU(s), using {num_gpus}: {devices}")
            else:
                print("Warning: No CUDA available, falling back to CPU")
                devices = ["cpu"]
                num_gpus = 1
        
        # 直接使用连续索引，不需要验证所有数据集项（太慢）
        # 数据集本身会处理错误的索引，在 worker 中会跳过错误的图像
        print(f"Preparing {total_samples} indices for {num_gpus} GPUs (skipping index validation for speed)...")
        
        # 构建目标索引并分块
        all_indices = list(range(start_idx, start_idx + total_samples))
        chunk_size = total_samples // num_gpus
        chunks = []
        for i in range(num_gpus):
            s_idx = i * chunk_size
            e_idx = total_samples if i == num_gpus - 1 else (i + 1) * chunk_size
            chunk = all_indices[s_idx:e_idx]
            chunks.append(chunk)
        
        print(f"Split {total_samples} images into {num_gpus} chunks:")
        print(f"  Important: Dataset indices (0-based, after valid_indices filtering) are split.")
        print(f"  Each index 'idx' maps to actual image_id via: dataset[idx] -> image_id")
        print(f"  Saved .pt files use format: '{filename_prefix}_{{image_id:08d}}.pt' (using actual image_id from dataset)")
        print(f"  This ensures consistency: the same dataset index always maps to the same image_id and filename.")
        for i, chunk in enumerate(chunks):
            print(f"  GPU {i} ({devices[i]}): dataset indices [{chunk[0] if len(chunk) > 0 else 'N/A'}, {chunk[-1] if len(chunk) > 0 else 'N/A'}] (size: {len(chunk)})")
        
        # 使用多进程处理（每个 GPU 一个进程）
        # 注意：需要使用 spawn 模式，因为 CUDA context 不能在 fork 后共享
        try:
            mp_start_method = mp.get_start_method()
            if mp_start_method != 'spawn':
                mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # 如果已经设置过，不能再次设置
            pass
        
        # 获取数据集路径（VG150Dataset 有 data_root 属性）
        data_root = getattr(dataset, 'data_root', None)
        if data_root is None:
            raise ValueError("Dataset must have a 'data_root' attribute for multi-GPU processing")
        
        # 准备 worker 参数（使用流水线方法）
        # 每个 worker 直接保存到最终输出目录（文件名包含 image_id，不会冲突）
        worker_args = [
            (
                data_root,  # 传递数据集路径，在 worker 中重新加载
                chunk,
                device,
                self.feature_dim,
                self.image_size,
                self._checkpoint_path,
                split,
                batch_size,  # 批处理大小
                num_workers,  # DataLoader 工作线程数
                output_dir,  # 直接使用最终输出目录
                filename_prefix,
                self.save_fp16,
            )
            for chunk, device in zip(chunks, devices)
        ]
        
        # 使用进程池并行处理
        print(f"Starting {num_gpus} GPU workers with batch_size={batch_size}...")
        print(f"Note: num_workers is disabled in worker processes to avoid nested multiprocessing issues")
        print(f"Workers will use num_workers=0 for DataLoader (single-threaded loading in each GPU process)")
        
        # 设置超时（防止无限等待）
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Worker processes timeout after 2 hours")
        
        # 使用进程池并行处理（不设置超时，但添加更多调试信息）
        with mp.Pool(processes=num_gpus) as pool:
            print(f"Worker pool created, starting parallel processing...", flush=True)
            results = pool.starmap(_extract_features_worker_multi_gpu, worker_args)
            print(f"All worker processes completed", flush=True)
        
        # 合并结果
        # 每个 worker 已经直接保存到最终输出目录（文件名包含 image_id，不会冲突）
        all_image_ids = []
        total_processed = 0
        
        for worker_idx, (worker_image_ids, processed_count, worker_split_dir) in enumerate(results):
            all_image_ids.extend(worker_image_ids)
            total_processed += processed_count
            print(f"[GPU {devices[worker_idx]}] Processed {processed_count} images")
        
        # 保存最终元数据（所有 worker 都保存到同一个目录）
        final_split_dir = os.path.join(output_dir, split)
        os.makedirs(final_split_dir, exist_ok=True)
        
        metadata = {
            "split": split,
            "num_images": total_processed,
            "feature_dim": self.feature_dim,
            "stride": self.feature_stride,
            "image_size": self.image_size,
            "spatial_shape": (max(1, self.image_size // self.feature_stride), max(1, self.image_size // self.feature_stride)),
            "image_ids": sorted(all_image_ids),
            "filename_prefix": filename_prefix,
            "start_idx": start_idx,
            "dtype": "fp16" if self.save_fp16 else "fp32",
        }
        metadata_file = os.path.join(final_split_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Successfully processed {total_processed} images using {num_gpus} GPUs")
        print(f"  - Output directory: {final_split_dir}")
        print(f"  - Metadata: {metadata_file}")


def main():
    """
    示例使用：从命令行运行 feature 生成
    """
    import argparse
    import os
    import sys
    from pathlib import Path
    
    # 添加项目根目录到 PYTHONPATH（如果还没有）
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # sgg/precompute -> sgg -> project_root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from sgg.datasets.vg150_dataset import VG150Dataset
    
    parser = argparse.ArgumentParser(description="Generate SAM3 image features for VG150 dataset")
    parser.add_argument("--data_root", type=str, required=True, help="VG150 dataset root")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--feature_dim", type=int, default=256, help="Feature dimension")
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="Maximum number of samples to process. Use 0 or -1 for all images, or None (default) for all.")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Start index of dataset to process (default: 0). Useful to split runs manually.")
    parser.add_argument("--image_size", type=int, default=1008, help="Input image size (SAM3 default: 1008)")
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                       help="Path to local SAM3 checkpoint (sam3.pt). If not specified, will try to find automatically.")
    parser.add_argument("--num_gpus", type=int, default=1, 
                       help="Number of GPUs to use for parallel processing (default: 1, use 2 for dual GPU)")
    parser.add_argument("--use_multi_gpu", action="store_true",
                       help="Enable multi-GPU parallel processing (automatically enabled if num_gpus > 1)")
    parser.add_argument("--devices", type=str, nargs="+", default=None,
                       help="Specific GPU devices to use (e.g., cuda:0 cuda:1). If not specified, auto-detect.")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size for DataLoader (default: 10, adjust based on memory)")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of DataLoader worker threads (default: 2, don't set too high due to limited memory)")
    parser.add_argument("--save_fp16", action="store_true",
                       help="Save features as FP16 to reduce memory/disk footprint")
    
    args = parser.parse_args()
    
    # 处理相对路径：如果 checkpoint_path 是相对路径，相对于项目根目录解析
    checkpoint_path = args.checkpoint_path
    if checkpoint_path and not os.path.isabs(checkpoint_path):
        # 获取项目根目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # sgg/precompute -> sgg -> project_root
        checkpoint_path = (project_root / checkpoint_path).resolve()
        if checkpoint_path.exists():
            checkpoint_path = str(checkpoint_path)
        else:
            # 如果相对项目根目录找不到，尝试相对于当前工作目录
            checkpoint_path = str(Path(checkpoint_path).resolve())
    
    # 创建数据集
    print(f"Loading VG150 dataset from {args.data_root}...")
    dataset = VG150Dataset(
        data_root=args.data_root,
        split=args.split,
        image_size=args.image_size,
    )
    
    # 决定使用单 GPU 还是多 GPU
    use_multi_gpu = args.use_multi_gpu or (args.num_gpus > 1)
    
    # 决定使用单GPU还是多GPU模式
    # 只有当 num_gpus > 1 且确实有多个GPU可用时才使用多GPU模式
    effective_num_gpus = min(args.num_gpus, torch.cuda.device_count() if torch.cuda.is_available() else 0)
    
    if use_multi_gpu and torch.cuda.is_available() and effective_num_gpus > 1:
        # 多 GPU 模式（需要至少2个GPU）
        devices = args.devices
        if devices is None:
            num_gpus = effective_num_gpus
            devices = [f"cuda:{i}" for i in range(num_gpus)]
        else:
            num_gpus = len(devices)
        
        print(f"Using multi-GPU mode with {num_gpus} GPUs: {devices}")
        print(f"Warning: Multi-GPU mode will load {num_gpus}x memory (model + dataset).")
        print(f"  If you encounter memory issues, try using --num_gpus 1 for single-GPU mode.")
        
        # 创建 generator（主进程中的实例仅用于保存，实际处理在 worker 中）
        generator = SAM3ImageFeatureGenerator(
            device=devices[0],  # 主进程使用第一个 GPU（用于保存）
            feature_dim=args.feature_dim,
            image_size=args.image_size,
            checkpoint_path=checkpoint_path,
            save_fp16=args.save_fp16,
        )
        
        # 使用多 GPU 处理
        generator.process_dataset_multi_gpu(
            dataset=dataset,
            output_dir=args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
            start_idx=args.start_idx,
            num_gpus=len(devices),
            devices=devices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_fp16=args.save_fp16,
        )
    else:
        # 单 GPU 模式
        generator = SAM3ImageFeatureGenerator(
            device=args.device,
            feature_dim=args.feature_dim,
            image_size=args.image_size,
            checkpoint_path=checkpoint_path,
            save_fp16=args.save_fp16,
        )
        
        # 处理数据集
        generator.process_dataset(
            dataset=dataset,
            output_dir=args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
            start_idx=args.start_idx,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    
    print("Done!")


if __name__ == "__main__":
    main()

