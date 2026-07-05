import copy
import os
import sys
import torch
import numpy as np
import logging 
import detectron2.utils.comm as comm
import time 
import datetime
import pickle
import itertools
import pycocotools.mask as mask_util
from collections import OrderedDict, defaultdict
from torch.utils.data import Sampler
from detectron2.utils.logger import setup_logger, log_every_n_seconds
from detectron2.engine import DefaultTrainer
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    build_batch_data_loader
)
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator, inference_on_dataset, print_csv_format, inference_context
from imantics import Polygons, Mask

from detectron2.engine import hooks, HookBase
from detectron2.engine.hooks import get_bn_modules
from ..data import DetrDatasetMapper
from detectron2.evaluation import (
    COCOEvaluator,
    SemSegEvaluator
)
from ..checkpoint import PeriodicCheckpointerWithEval
from .patch_merge_heatmap_hook import PatchMergeHeatmapHook
from ..evaluation import scenegraph_inference_on_dataset, SceneGraphEvaluator
from detectron2.engine import hooks
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler
from detectron2.data.common import MapDataset, DatasetFromList
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import trivial_batch_collator
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter, EventWriter, get_event_storage
from typing import Any, Dict, List, Set
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.checkpoint import DetectionCheckpointer
import wandb
from PIL import Image
import weakref
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer

class WandbWriter(EventWriter):
    """
    Write all scalars to a wandb file.
    """

    def __init__(self, cfg, model, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): the directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size

        os.environ['WANDB_API_KEY'] = 'YOUR API KEY HERE'
        self._writer = wandb.init(
                entity=cfg.WANDB.ENTITY,
                project=cfg.WANDB.PROJECT,
                group=cfg.WANDB.GROUP,
                name=cfg.OUTPUT_DIR.split('/')[-1],
                config=cfg,
            )
        self._writer.watch(model,log_freq=20)
        # self._last_write = -1

    def write(self):
        storage = get_event_storage()
        # new_last_write = self._last_write
        log_dict = {}
        # import pdb;pdb.set_trace()
        for k, v in storage.latest_with_smoothing_hint().items():

            # if iter > self._last_write:
            log_dict.update({k: float(v[0])}, step=v[1])
                # self._writer.log({k: v}, step=iter)
                # new_last_write = max(new_last_write, iter)
        self._writer.log(log_dict)
        # self._last_write = new_last_write

        # visualize training samples

        # if len(storage.vis_data) >= 1:
        #
        #     for img_name, img, step_num in storage.vis_data:
        #         log_img = Image.fromarray(img.transpose(1, 2, 0))  # convert to (h, w, 3) PIL.Image
        #         log_img = wandb.Image(log_img, caption=img_name)
        #         self._writer.log({img_name: log_img}, step=step_num)
        #     # Storage stores all image data and rely on this writer to clear them.
        #     # As a result it assumes only one writer will use its image data.
        #     # An alternative design is to let storage store limited recent
        #     # data (e.g. only the most recent image) that all writers can access.
        #     # In that case a writer may not see all image data if its period is long.
        #     storage.clear_images()

    def close(self):
        if hasattr(self, "_writer"):
            self._writer.finish()

class EpochProgressWriter(EventWriter):
    """
    额外打印 epoch / iter 进度以及「总训练时间」估计的日志 Writer。
    不替换 detectron2 自带的 CommonMetricPrinter，而是额外打一行类似：
      epoch: 1/10  iter: 1235/50000  total_eta: 5:12:34
    """

    def __init__(self, cfg, max_iter: int, window_size: int = 20):
        """
        Args:
            cfg: detectron2 配置，用于读取自定义的 SOLVER.MAX_EPOCH（可选）
            max_iter: 训练的最大迭代次数
            window_size: 用于平滑 time/iter 的窗口大小
        """
        self._cfg = cfg
        self._max_iter = int(max_iter)
        self._window_size = int(window_size)
        # 允许在 config 里显式指定最大 epoch 数；否则退化为 “只有 1 个 epoch”
        self._max_epoch = int(getattr(getattr(cfg, "SOLVER", cfg), "MAX_EPOCH", 1))
        if self._max_epoch <= 0:
            self._max_epoch = 1
        self._logger = logging.getLogger("d2.utils.events")

    def write(self):
        storage = get_event_storage()
        cur_iter = int(storage.iter)

        # 基于最近若干次迭代的平均耗时，估计总训练时间（total_eta）
        iter_time = None
        try:
            # detectron2 的 EventStorage 会在 "time" 里记录每次迭代耗时
            iter_time = storage.history("time").median(self._window_size)
        except KeyError:
            pass

        total_eta_str = "N/A"
        if iter_time is not None and iter_time > 0:
            total_seconds = float(iter_time) * float(max(self._max_iter, 1))
            total_eta_str = str(datetime.timedelta(seconds=int(total_seconds)))

        # 通过 max_iter 和 max_epoch 之间的均分关系来近似 epoch 进度
        iters_per_epoch = max(self._max_iter // self._max_epoch, 1)
        cur_epoch = min(self._max_epoch, cur_iter // iters_per_epoch + 1)

        self._logger.info(
            f"epoch: {cur_epoch}/{self._max_epoch}  "
            f"iter: {cur_iter}/{self._max_iter}  "
            f"total_eta: {total_eta_str}"
        )

    def close(self):
        # 按接口需要实现，但这里没有资源需要手动释放
        pass

class MemoryMonitorHook(HookBase):
    """内存监控 Hook：当系统内存占用超过阈值时自动终止训练"""
    
    def __init__(self, threshold=90.0, check_interval=10):
        """
        Args:
            threshold: 内存使用率阈值（%），超过此值将终止训练
            check_interval: 每隔多少次迭代检查一次内存
        """
        self.threshold = threshold
        self.check_interval = check_interval
        self.check_count = 0
        self.logger = logging.getLogger(__name__)
    
    def after_step(self):
        super().after_step()
        """每次迭代后检查内存"""
        self.check_count += 1
        if self.check_count % self.check_interval == 0:
            try:
                import psutil
                mem = psutil.virtual_memory()
                mem_usage_percent = mem.percent
                
                if mem_usage_percent >= self.threshold:
                    self.logger.error(
                        f"⚠️  内存使用率 {mem_usage_percent:.1f}% 超过阈值 {self.threshold}%！"
                        f"正在终止训练以防止系统崩溃..."
                    )
                    import os
                    import signal
                    # 发送 SIGTERM 信号给当前进程
                    os.kill(os.getpid(), signal.SIGTERM)
                    # 如果 SIGTERM 无效，等待1秒后发送 SIGKILL
                    import time
                    time.sleep(1)
                    os.kill(os.getpid(), signal.SIGKILL)
            except ImportError:
                # psutil 未安装，跳过内存监控
                pass
            except Exception as e:
                self.logger.warning(f"内存监控出错: {e}")


class PerformanceMonitorHook(HookBase):
    """性能监控 Hook：记录训练过程中的时间分布、GPU利用率、精度等信息"""
    
    def __init__(self, period=50):
        """
        Args:
            period: 每隔多少次迭代输出一次性能报告
        """
        self.period = period
        self.stats = defaultdict(list)
        self.iter_times = []
        self.data_times = []
        self.forward_times = []
        self.backward_times = []
        self.optimizer_times = []
        self.gpu_utilizations = []
        self.precision_info = None
        self.logger = logging.getLogger(__name__)
        
    def before_train(self):
        """训练开始前初始化"""
        if comm.is_main_process():
            # 检测精度设置
            trainer = self.trainer
            if hasattr(trainer, 'model'):
                # 检查模型参数精度
                sample_param = next(iter(trainer.model.parameters()))
                if sample_param.dtype == torch.float16:
                    self.precision_info = "FP16"
                elif sample_param.dtype == torch.bfloat16:
                    self.precision_info = "BF16"
                else:
                    self.precision_info = "FP32"
                
                # 检查是否使用 AMP (通过检查是否有 grad_scaler)
                if hasattr(trainer, 'grad_scaler') and trainer.grad_scaler is not None:
                    self.precision_info = "AMP (自动混合精度) - " + self.precision_info
            else:
                self.precision_info = "未知"
            
            self.logger.info("=" * 80)
            self.logger.info("训练性能监控初始化")
            self.logger.info(f"  精度模式: {self.precision_info}")
            self.logger.info(f"  GPU 数量: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    self.logger.info(f"  GPU {i}: {props.name}, 显存: {props.total_memory / 1024**3:.1f} GB")
            self.logger.info("=" * 80)
    
    def before_step(self):
        """每次迭代前记录开始时间"""
        if comm.is_main_process():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            self.step_start_time = time.perf_counter()
            self.data_start_time = time.perf_counter()
    
    def after_step(self):
        super().after_step()
        """每次迭代后记录时间"""
        if comm.is_main_process():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            step_end_time = time.perf_counter()
            step_time = step_end_time - self.step_start_time
            
            self.iter_times.append(step_time)
            
            # 记录到 event storage
            storage = get_event_storage()
            if hasattr(storage, 'latest'):
                storage.put_scalar("time/iter_time", step_time)
                if hasattr(self, 'data_time'):
                    storage.put_scalar("time/data_time", self.data_time)
                if hasattr(self, 'forward_time'):
                    storage.put_scalar("time/forward_time", self.forward_time)
                if hasattr(self, 'backward_time'):
                    storage.put_scalar("time/backward_time", self.backward_time)
                if hasattr(self, 'optimizer_time'):
                    storage.put_scalar("time/optimizer_time", self.optimizer_time)
            
            # 获取 GPU 利用率
            if NVML_AVAILABLE and torch.cuda.is_available():
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_utilizations.append(util.gpu)
                except:
                    pass
            
            # 定期输出性能报告
            if len(self.iter_times) >= self.period:
                self._print_performance_report()
                # 清空统计（保留最近的数据用于平滑）
                keep_last = min(20, self.period // 2)
                self.iter_times = self.iter_times[-keep_last:]
                self.data_times = self.data_times[-keep_last:] if hasattr(self, 'data_times') else []
                self.forward_times = self.forward_times[-keep_last:] if hasattr(self, 'forward_times') else []
                self.backward_times = self.backward_times[-keep_last:] if hasattr(self, 'backward_times') else []
                self.optimizer_times = self.optimizer_times[-keep_last:] if hasattr(self, 'optimizer_times') else []
                self.gpu_utilizations = self.gpu_utilizations[-keep_last:]
    
    def _print_performance_report(self):
        """打印性能报告"""
        if not self.iter_times:
            return
        
        self.logger.info("=" * 80)
        self.logger.info("训练性能报告")
        self.logger.info("-" * 80)
        
        # 时间统计
        avg_iter_time = np.mean(self.iter_times)
        avg_data_time = np.mean(self.data_times) if self.data_times else 0.0
        avg_forward_time = np.mean(self.forward_times) if self.forward_times else 0.0
        avg_backward_time = np.mean(self.backward_times) if self.backward_times else 0.0
        avg_optimizer_time = np.mean(self.optimizer_times) if self.optimizer_times else 0.0
        
        total_compute_time = avg_forward_time + avg_backward_time + avg_optimizer_time
        other_time = avg_iter_time - avg_data_time - total_compute_time
        
        self.logger.info(f"精度模式: {self.precision_info}")
        self.logger.info(f"平均迭代时间: {avg_iter_time*1000:.2f} ms")
        self.logger.info("")
        self.logger.info("时间分布:")
        self.logger.info(f"  数据加载: {avg_data_time*1000:.2f} ms ({avg_data_time/avg_iter_time*100:.1f}%)")
        self.logger.info(f"  前向传播: {avg_forward_time*1000:.2f} ms ({avg_forward_time/avg_iter_time*100:.1f}%)")
        self.logger.info(f"  反向传播: {avg_backward_time*1000:.2f} ms ({avg_backward_time/avg_iter_time*100:.1f}%)")
        self.logger.info(f"  优化器更新: {avg_optimizer_time*1000:.2f} ms ({avg_optimizer_time/avg_iter_time*100:.1f}%)")
        self.logger.info(f"  其他开销: {other_time*1000:.2f} ms ({other_time/avg_iter_time*100:.1f}%)")
        self.logger.info("")
        
        # GPU 利用率
        if self.gpu_utilizations:
            avg_gpu_util = np.mean(self.gpu_utilizations)
            self.logger.info(f"平均 GPU 利用率: {avg_gpu_util:.1f}%")
        
        # GPU 内存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.logger.info(f"GPU 显存: 已分配={allocated:.2f} GB, 已保留={reserved:.2f} GB, 峰值={max_allocated:.2f} GB")
        
        # 吞吐量
        if avg_iter_time > 0:
            throughput = 1.0 / avg_iter_time
            self.logger.info(f"训练吞吐量: {throughput:.2f} iter/s")
        
        self.logger.info("=" * 80)


class Sam3UnfreezeHook(HookBase):
    """训练到指定 iter 后自动解冻 SAM3 backbone（仅执行一次）。"""

    def __init__(self, unfreeze_at_iter: int, cfg=None):
        self.unfreeze_at_iter = int(unfreeze_at_iter)
        self.cfg = cfg
        self._done = False
        self.logger = logging.getLogger(__name__)

    def after_step(self):
        super().after_step()
        if self._done:
            return
        cur_iter = self.trainer.iter + 1
        if cur_iter < self.unfreeze_at_iter:
            return
        self._done = True
        module = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
        backbone = self._find_sam3_backbone(module)
        if backbone is None:
            self.logger.warning("[Sam3UnfreezeHook] SAM3 backbone not found, skip unfreeze")
            return
        if hasattr(backbone, "set_freeze"):
            backbone.set_freeze(False)
        backbone.train()
        for p in backbone.parameters():
            p.requires_grad_(True)
        # 将新解冻参数加入优化器（初始化时因 requires_grad=False 被跳过）
        if self.cfg is not None and hasattr(self.trainer, "optimizer"):
            lr = self.cfg.SOLVER.BASE_LR * getattr(self.cfg.SOLVER, "BACKBONE_MULTIPLIER", 0.1)
            new_params = [p for p in backbone.parameters() if p.requires_grad]
            if new_params:
                self.trainer.optimizer.add_param_group({"params": new_params, "lr": lr})
        self.logger.info("[Sam3UnfreezeHook] Unfroze SAM3 backbone at iter=%d", cur_iter)

    def _find_sam3_backbone(self, module):
        if hasattr(module, "backbone") and hasattr(module.backbone, "bottom_up"):
            bb = module.backbone.bottom_up
            if hasattr(bb, "sam3_model") or hasattr(bb, "set_freeze"):
                return bb
        if hasattr(module, "backbone"):
            bb = module.backbone
            if isinstance(bb, (torch.nn.Sequential, torch.nn.ModuleList)):
                bb = bb[0] if len(bb) > 0 else None
            if bb is not None and (hasattr(bb, "sam3_model") or hasattr(bb, "set_freeze")):
                return bb
        if hasattr(module, "detr") and hasattr(module.detr, "backbone"):
            bb = module.detr.backbone
            if isinstance(bb, (torch.nn.Sequential, torch.nn.ModuleList)):
                bb = bb[0] if len(bb) > 0 else None
            if bb is not None and (hasattr(bb, "sam3_model") or hasattr(bb, "set_freeze")):
                return bb
        return None


class SameVideoBatchSampler(Sampler):
    """
    将同一视频的帧分组到同一个batch中的采样器。
    这个类必须在模块级别定义（而不是在方法内部），以便可以被pickle序列化，
    从而支持多GPU训练时的多进程数据加载。
    """
    def __init__(self, records, batch_size, base_seed):
        self.batch_size = max(1, int(batch_size))
        self.base_seed = int(base_seed)
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        grouped = defaultdict(list)
        for idx, rec in enumerate(records):
            grouped[str(rec.get("video_id", "__novid__"))].append(idx)
        videos = sorted(grouped.keys())
        self.video_to_indices = {k: grouped[k] for k in videos}
        self.videos = videos[self.rank :: self.world_size] if self.world_size > 1 else videos
        if len(self.videos) == 0:
            self.videos = videos

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.rank)
        while True:
            # Shuffle only at video granularity; keep frame order inside each video.
            order = torch.randperm(len(self.videos), generator=g).tolist()
            for vid_idx in order:
                vid = self.videos[vid_idx]
                indices = list(self.video_to_indices[vid])
                if len(indices) == 0:
                    continue
                # Keep temporal order for frames within this video.
                # `indices` follows dataset order, which is expected to be frame order.
                ordered = indices
                start = 0
                while start < len(ordered):
                    chunk = ordered[start : start + self.batch_size]
                    start += self.batch_size
                    if len(chunk) < self.batch_size:
                        # Pad using the last frame index to avoid introducing temporal jumps.
                        chunk.extend([ordered[-1]] * (self.batch_size - len(chunk)))
                    for x in chunk:
                        yield x

    def __len__(self):
        return sum(len(v) for v in self.video_to_indices.values())



class ClipSampleBatchSampler(Sampler):
    """
    multi-slot stateful clip-sample 采样器。

    一个 batch 由 num_slots 个 video slot 拼成，每个 slot 在当前 step 输出某视频连续
    clip_len 个关键帧；同一 slot 在后续 step 继续推进该视频的后续 chunk，直到该视频
    的完整 chunks 用完才切换到新视频。因此 batch 内是 clip 形态，跨 step 是按 slot
    持续推进的 streaming 形态；若 temporal memory 按 video_id 持久保存，memory 会跨
    batch/step 传递，而不是限制在单个 clip_len 窗口内。

    不 padding：每趟用随机相位选满 floor(n/clip_len) 个整块，丢弃边角 n%clip_len 帧；
    随机相位只能提高多趟采样下边角帧被覆盖的概率，不保证单趟或有限 iter 严格覆盖。
    产出扁平单帧索引流，由 build_batch_data_loader 每 batch_size 个聚成一个 batch。
    必须在模块级定义以支持 pickle / 多进程加载。per-rank 视频分片 videos[rank::world_size]。
    """
    def __init__(self, records, batch_size, base_seed, clip_len):
        self.batch_size = max(1, int(batch_size))
        self.clip_len = max(1, int(clip_len))
        assert self.batch_size % self.clip_len == 0, (
            f"ClipSampleBatchSampler: batch_size={self.batch_size} "
            f"必须能被 clip_len={self.clip_len} 整除。"
        )
        self.num_slots = self.batch_size // self.clip_len
        self.base_seed = int(base_seed)
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        grouped = defaultdict(list)
        for idx, rec in enumerate(records):
            grouped[str(rec.get("video_id", "__novid__"))].append(
                (int(rec.get("frame_idx", 0)), idx)
            )
        videos = sorted(grouped.keys())
        # 每个视频内严格按 frame_idx 升序，保证逐帧时序与 memory delta-t 正确
        self.video_to_indices = {
            k: [idx for _, idx in sorted(grouped[k])] for k in videos
        }
        shard = videos[self.rank :: self.world_size] if self.world_size > 1 else videos
        if len(shard) == 0:
            shard = videos
        self.videos = shard
        assert self.num_slots <= len(self.videos), (
            f"ClipSampleBatchSampler: slot 数(batch_size/clip_len)={self.num_slots} "
            f"必须 <= 本 rank 视频数={len(self.videos)}，否则无法保证 batch 内视频互不相同。"
        )
        for v in self.videos:
            assert len(self.video_to_indices[v]) >= self.clip_len, (
                f"ClipSampleBatchSampler: 视频 {v} 帧数={len(self.video_to_indices[v])} "
                f"< clip_len={self.clip_len}，无法构成一个 clip。"
            )

    def _buildChunks(self, vid, g):
        # 随机相位选满 floor(n/clip_len) 个整块；提高多趟覆盖概率，但不保证有限 iter 严格覆盖。
        frames = self.video_to_indices[vid]
        n = len(frames)
        K = self.clip_len
        num_chunks = n // K
        rem = n - num_chunks * K
        phase = int(torch.randint(0, rem + 1, (1,), generator=g).item()) if rem > 0 else 0
        return [frames[phase + c * K : phase + (c + 1) * K] for c in range(num_chunks)]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.rank)
        G = self.num_slots

        def makeQueue():
            order = torch.randperm(len(self.videos), generator=g).tolist()
            return [self.videos[i] for i in order]

        queue = makeQueue()

        def drawVideo(held):
            # 取一个当前未被任何 slot 占用的视频；耗尽则重洗，形成无限流。
            nonlocal queue
            scanned = 0
            while True:
                if not queue:
                    queue = makeQueue()
                vid = queue.pop(0)
                if vid not in held:
                    return vid
                scanned += 1
                assert scanned <= 2 * len(self.videos) + 2, "drawVideo 找不到空闲视频(不应发生)"

        held = set()
        slotVid = []
        slotChunks = []
        slotCursor = []
        for _ in range(G):
            vid = drawVideo(held)
            held.add(vid)
            slotVid.append(vid)
            slotChunks.append(self._buildChunks(vid, g))
            slotCursor.append(0)

        while True:
            for si in range(G):
                if slotCursor[si] >= len(slotChunks[si]):
                    # 该视频所有 clip 走完，换一个未占用视频从头开始（memory 靠 frame_idx 回跳 reset）
                    held.discard(slotVid[si])
                    newVid = drawVideo(held)
                    held.add(newVid)
                    slotVid[si] = newVid
                    slotChunks[si] = self._buildChunks(newVid, g)
                    slotCursor[si] = 0
                chunk = slotChunks[si][slotCursor[si]]
                slotCursor[si] += 1
                for x in chunk:
                    yield x

    def __len__(self):
        return sum(len(v) for v in self.video_to_indices.values())


class EvalFirstHook(HookBase):
    """训练前自动 eval，验证初始权重质量。由 SOLVER.EVAL_FIRST 控制。
    注意：eval 需要所有进程参与（COCO evaluator 内部用 comm.gather），
    因此不在 rank 0 处 return，而是只将日志输出限制在 rank 0。"""

    def before_train(self):
        cfg = self.trainer.cfg
        if not getattr(cfg.SOLVER, "EVAL_FIRST", False):
            return

        is_main = comm.is_main_process()
        logger = logging.getLogger('detectron2')

        if is_main:
            logger.info("=" * 60)
            logger.info("[EvalFirst] 训练前评估初始权重...")
            print("[EvalFirst] 训练前评估初始权重...", flush=True)

        # Determine eval dataset
        if cfg.DATASETS.TYPE == "ACTION GENOME":
            if cfg.DATASETS.ACTION_GENOME.NUM_VIDEOS_VAL > 0:
                test_dataset = "AG_val"
            else:
                test_dataset = "AG_train"
        elif cfg.DATASETS.TEST:
            test_dataset = cfg.DATASETS.TEST[0]
        else:
            test_dataset = "AG_train"

        eval_cfg = cfg.clone()
        eval_cfg.defrost()
        eval_cfg.DATASETS.TEST = (test_dataset,)
        eval_cfg.MODEL.TEMPORAL.EVAL_ENABLED = cfg.MODEL.TEMPORAL.ENABLED
        eval_cfg.freeze()

        if is_main:
            logger.info("[EvalFirst] 评估数据集: %s", test_dataset)
            print(f"[EvalFirst] 评估数据集: {test_dataset}", flush=True)

        # 所有进程参与 eval （COCO evaluator 使用 comm.gather 收集结果）
        model = self.trainer.model
        results = self.trainer.__class__.test(eval_cfg, model)

        if is_main and isinstance(results, dict):
            bbox = results.get('bbox', {})
            bbox_recall = results.get('bbox_recall', {})
            sg = results.get('SG', {})
            recall50 = bbox_recall.get('Recall@50', 0) if isinstance(bbox_recall, dict) else 0
            if isinstance(bbox, dict):
                print("[EvalFirst] bbox | AP: {:.4f}  AP50: {:.4f}  Rec@50: {:.4f}".format(
                           bbox.get('AP', 0), bbox.get('AP50', 0),
                           recall50), flush=True)
                logger.info("[EvalFirst] bbox | AP: %.4f  AP50: %.4f  Rec@50: %.4f",
                           bbox.get('AP', 0), bbox.get('AP50', 0),
                           recall50)
            if isinstance(sg, dict):
                print("[EvalFirst] SG   | R@20: {:.5f}  R@50: {:.5f}  mR@50: {:.5f}".format(
                           sg.get('SGRecall@20', 0), sg.get('SGRecall@50', 0),
                           sg.get('SGMeanRecall@50', 0)), flush=True)
                logger.info("[EvalFirst] SG   | R@20: %.5f  R@50: %.5f  mR@50: %.5f",
                           sg.get('SGRecall@20', 0), sg.get('SGRecall@50', 0),
                           sg.get('SGMeanRecall@50', 0))
        elif is_main:
            logger.info("[EvalFirst] 结果: %s", str(results)[:200])

        # Restore training mode (all ranks)
        model.train()
        if is_main:
            logger.info("=" * 60)

class JointTransformerTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        与 DefaultTrainer 基本一致，但在多卡训练时显式开启
        DDP find_unused_parameters，以兼容存在条件分支/未参与反传参数的模型。
        """
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(
            model,
            broadcast_buffers=False,
            find_unused_parameters=(comm.get_world_size() > 1),
        )
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @staticmethod
    def _build_same_video_sampler(dataset_dicts, per_worker_batch_size, seed=42):
        return SameVideoBatchSampler(dataset_dicts, per_worker_batch_size, seed)

    @staticmethod
    def _build_clip_sample_sampler(dataset_dicts, per_worker_batch_size, seed=42, clip_len=4):
        return ClipSampleBatchSampler(dataset_dicts, per_worker_batch_size, seed, clip_len)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.DATASETS.TYPE == "ACTION GENOME" and cfg.DATASETS.ACTION_GENOME.FORMAT_VID_WISE:
            dataset_dicts = get_detection_dataset_dicts(
                cfg.DATASETS.TRAIN,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if cfg.MODEL.KEYPOINT_ON
                else 0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            )
            dataset = DatasetFromList(dataset_dicts, copy=False)
            dataset = MapDataset(dataset, DetrDatasetMapper(cfg, True))
            world_size = max(1, get_world_size())
            assert (
                cfg.SOLVER.IMS_PER_BATCH % world_size == 0
            ), "SOLVER.IMS_PER_BATCH must be divisible by world size."
            per_worker_batch = cfg.SOLVER.IMS_PER_BATCH // world_size
            sampler_mode = cfg.DATASETS.ACTION_GENOME.SAMPLER_MODE
            assert sampler_mode in ("clip", "clip_sample"), (
                f"未知 SAMPLER_MODE={sampler_mode}，仅支持 'clip' / 'clip_sample'。"
            )
            if sampler_mode == "clip_sample":
                sampler = cls._build_clip_sample_sampler(
                    dataset_dicts,
                    per_worker_batch_size=per_worker_batch,
                    seed=cfg.DATASETS.VISUAL_GENOME.OVERFIT_SEED,
                    clip_len=cfg.DATASETS.ACTION_GENOME.CLIP_SAMPLE_LEN,
                )
            else:
                sampler = cls._build_same_video_sampler(
                    dataset_dicts,
                    per_worker_batch_size=per_worker_batch,
                    seed=cfg.DATASETS.VISUAL_GENOME.OVERFIT_SEED,
                )
            return build_batch_data_loader(
                dataset,
                sampler,
                total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
                aspect_ratio_grouping=False,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
            )
        return build_detection_train_loader(cfg, mapper=DetrDatasetMapper(cfg, True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DetrDatasetMapper(cfg, False))

    def resume_or_load(self, resume=True):
        """当启用 VG_PRETRAINED_FOR_AG 时，用「除分类头/关系头外」的 VG 权重加载并重建 AG 头，替代默认的 resume_or_load 初始权重加载。"""
        if not resume and getattr(self.cfg.MODEL.DETR, "VG_PRETRAINED_FOR_AG", False) and self.cfg.MODEL.WEIGHTS:
            if hasattr(self.model, "load_vg_pretrained_for_ag"):
                self.model.load_vg_pretrained_for_ag(self.cfg.MODEL.WEIGHTS)
                return
        super().resume_or_load(resume=resume)

    def build_writers(self):
        output_dir = self.cfg.OUTPUT_DIR
        max_iter = self.max_iter
        PathManager.mkdirs(output_dir)

        if self.cfg.WANDB.USE_WANDB:
            return [
                # It may not always print what you want to see, since it prints "common" metrics only.
                CommonMetricPrinter(max_iter),
                # 额外打印 epoch / iter / total_eta 信息
                EpochProgressWriter(self.cfg, max_iter),
                JSONWriter(os.path.join(output_dir, "metrics.json")),
                TensorboardXWriter(output_dir),
                WandbWriter(self.cfg, self.model)
            ]
        else:
            return [
                # It may not always print what you want to see, since it prints "common" metrics only.
                CommonMetricPrinter(max_iter),
                # 额外打印 epoch / iter / total_eta 信息
                EpochProgressWriter(self.cfg, max_iter),
                JSONWriter(os.path.join(output_dir, "metrics.json")),
                TensorboardXWriter(output_dir),
            ]

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            PerformanceMonitorHook(period=50),  # 每50次迭代输出一次性能报告
            MemoryMonitorHook(threshold=90.0, check_interval=10),  # 内存监控：每10次迭代检查一次，超过90%自动终止
            hooks.LRScheduler(self.optimizer, self.scheduler),
        ]
        if cfg.MODEL.SAM3.ENABLED and cfg.MODEL.SAM3.FREEZE and cfg.MODEL.SAM3.UNFREEZE_AT_ITER >= 0:
            ret.append(Sam3UnfreezeHook(cfg.MODEL.SAM3.UNFREEZE_AT_ITER))
        precise_bn = hooks.PreciseBN(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            self.build_train_loader(cfg),
            cfg.TEST.PRECISE_BN.NUM_ITER,
        ) if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model) else None
        ret.append(precise_bn)

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        # if comm.is_main_process():
        #     ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=1))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        def test_at_end_of_training():
            copy_cfg = copy.deepcopy(self.cfg)
            copy_cfg.defrost()
            if copy_cfg.DATASETS.TYPE == "ACTION GENOME":
                # use config TEST setting, don't force AG_val
                pass
            else:
                copy_cfg.DATASETS.TEST = ("VG_test",)
            copy_cfg.freeze()
            self._final_eval_results_on_test = self.test(copy_cfg, self.model)
            return self._final_eval_results_on_test

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        if cfg.MODEL.SAM3.ENABLED and cfg.MODEL.SAM3.USE_PATCH_MERGE:
            ret.append(PatchMergeHeatmapHook(cfg.TEST.EVAL_PERIOD, cfg.OUTPUT_DIR))
        ret.append(EvalFirstHook())
        ret.append(PeriodicCheckpointerWithEval(cfg.TEST.EVAL_PERIOD, test_and_save_results, test_at_end_of_training, self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=cfg.SOLVER.MAX_TO_KEEP))
        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def run_step(self):
        """
        重写训练步骤，添加详细的性能监控
        """
        # Inject current iter for gate schedule (unwrap DDP if needed)
        _model = self.model.module if hasattr(self.model, 'module') else self.model
        qi = getattr(_model.detr, "query_injector", None)
        if qi is not None:
            qi._current_iter = self.iter
        rqi = getattr(_model.detr, "relation_query_injector", None)
        if rqi is not None:
            rqi._current_iter = self.iter
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        start = time.perf_counter()

        def _flatten_params_by_name(model, name_substring):
            params = [
                p
                for name, p in model.named_parameters()
                if name_substring in name and p.requires_grad
            ]
            if len(params) == 0:
                return None, []
            flat = torch.cat([p.detach().flatten().cpu() for p in params])
            return flat, params

        def _grad_norm(params):
            parts = [
                p.grad.detach().flatten().float().cpu()
                for p in params
                if p.grad is not None
            ]
            if len(parts) == 0:
                return 0.0
            return float(torch.cat(parts).norm().item())

        def _scalar(stats, key, default=0.0):
            if stats is None:
                return default
            return float(stats.get(key, default))

        def _count(stats, key="count"):
            if stats is None:
                return 0
            return int(stats.get(key, 0))
        
        # 数据加载时间
        # 在 detectron2 中，_data_loader_iter 是一个 property（定义在 TrainerBase 中）
        # 它会在第一次访问时自动创建迭代器（如果 _data_loader_iter_obj 是 None）
        # 直接访问即可，property 会自动处理初始化
        # 如果属性不存在，说明可能是在旧版本的 detectron2 中，尝试其他方式
        try:
            # 尝试访问 property（新版本 detectron2）
            data = next(self._data_loader_iter)
        except AttributeError:
            # 如果 property 不存在，尝试直接访问 _data_loader_iter_obj
            # 或者通过 data_loader 手动创建迭代器
            if hasattr(self, '_data_loader_iter_obj') and self._data_loader_iter_obj is not None:
                data = next(self._data_loader_iter_obj)
            elif hasattr(self, 'data_loader'):
                # 如果 data_loader 存在，手动创建迭代器
                if not hasattr(self, '_data_loader_iter_obj'):
                    self._data_loader_iter_obj = iter(self.data_loader)
                data = next(self._data_loader_iter_obj)
            else:
                # 如果都找不到，回退到父类方法（但这样无法添加性能监控）
                # 这通常不应该发生，因为 run_step 应该在 train 循环中被调用
                logger = logging.getLogger(__name__)
                logger.warning("Cannot find data loader iterator, falling back to parent run_step")
                super().run_step()
                return
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        data_time = time.perf_counter() - start
        
        # 记录数据加载时间到 hook
        for hook in self._hooks:
            if isinstance(hook, PerformanceMonitorHook):
                hook.data_times.append(data_time)
                hook.data_time = data_time
                hook.data_start_time = time.perf_counter()
        
        # 前向传播时间（支持 AMP）
        forward_start = time.perf_counter()
        # 检查是否使用 AMP
        use_amp = hasattr(self, 'grad_scaler') and self.grad_scaler is not None
        if use_amp:
            from torch.cuda.amp import autocast
            with autocast():
                loss_dict = self.model(data)
        else:
            loss_dict = self.model(data)

        # 记录时序 query 使用数量（仅在启用 temporal memory 时生效）
        try:
            base_model = self.model
            # DDP 包装时取出真实模型
            if hasattr(base_model, "module"):
                base_model = base_model.module
            detr_module = getattr(base_model, "detr", None)
            temporal_used_queries = None
            if (
                detr_module is not None
                and getattr(detr_module, "object_memory_bank", None) is not None
                and hasattr(detr_module, "_memory_states")
                and isinstance(detr_module._memory_states, dict)
            ):
                used = 0
                for state in detr_module._memory_states.values():
                    if hasattr(state, "valid_mask") and state.valid_mask is not None:
                        used += int(state.valid_mask.sum().item())
                temporal_used_queries = float(used)

            if temporal_used_queries is not None:
                # 写入 EventStorage，方便 tensorboard / metrics.json 统计
                storage = get_event_storage()
                storage.put_scalar("temporal/used_queries", temporal_used_queries)
                # 直接打印到终端（仅主进程），保证在 log.txt 中可见
                if comm.is_main_process():
                    print(f"[temporal] used_queries={int(temporal_used_queries)}", flush=True)
        except Exception:
            # 任何异常都不应影响正常训练，仅跳过该统计
            pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_time = time.perf_counter() - forward_start
        
        # 记录前向传播时间
        for hook in self._hooks:
            if isinstance(hook, PerformanceMonitorHook):
                hook.forward_times.append(forward_time)
                hook.forward_time = forward_time

        self._ensure_late_patch_merge_optimizer_params(_model)
        
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        diagnostics_log_path = os.environ.get("LEARNABLE_DIAGNOSTICS_LOG_PATH", "")
        diagnostics_log_period = int(os.environ.get("LEARNABLE_DIAGNOSTICS_LOG_PERIOD", "50"))
        diagnostics_enabled = (
            bool(diagnostics_log_path)
            and comm.is_main_process()
            and diagnostics_log_period > 0
            and self.iter % diagnostics_log_period == 0
        )
        diagnostics_state = None
        if diagnostics_enabled:
            roi_before, roi_params = _flatten_params_by_name(_model, "roi_refine_head.gate")
            xsam_before, xsam_params = _flatten_params_by_name(_model, "_patch_merge_proj")
            diagnostics_state = {
                "roi_before": roi_before,
                "roi_params": roi_params,
                "xsam_before": xsam_before,
                "xsam_params": xsam_params,
            }
        
        # 反向传播时间（支持 AMP）
        backward_start = time.perf_counter()
        self.optimizer.zero_grad()
        if use_amp:
            self.grad_scaler.scale(losses).backward()
        else:
            losses.backward()
        if diagnostics_state is not None:
            diagnostics_state["roi_grad_norm"] = _grad_norm(diagnostics_state["roi_params"])
            diagnostics_state["xsam_grad_norm"] = _grad_norm(diagnostics_state["xsam_params"])
        gate_log_path = os.environ.get("ROI_GATE_LOG_PATH", "")
        gate_log_period = int(os.environ.get("ROI_GATE_LOG_PERIOD", "20"))
        roi_gate_before = None
        roi_gate_grad_norm = None
        roi_gate_output_stats = None
        if gate_log_path and comm.is_main_process() and gate_log_period > 0 and self.iter % gate_log_period == 0:
            try:
                detr_module = getattr(_model, "detr", None)
                roi_gate_output_stats = getattr(detr_module, "_last_roi_gate_stats", None)
                gate_params = [
                    p
                    for name, p in _model.named_parameters()
                    if "roi_refine_head.gate" in name and p.requires_grad
                ]
                if len(gate_params) > 0:
                    roi_gate_before = torch.cat([p.detach().flatten().cpu() for p in gate_params])
                    grad_parts = [
                        p.grad.detach().flatten().float().cpu()
                        for p in gate_params
                        if p.grad is not None
                    ]
                    if len(grad_parts) > 0:
                        roi_gate_grad_norm = float(torch.cat(grad_parts).norm().item())
                    else:
                        roi_gate_grad_norm = 0.0
            except Exception:
                roi_gate_before = None
                roi_gate_grad_norm = None
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        backward_time = time.perf_counter() - backward_start
        
        # 记录反向传播时间
        for hook in self._hooks:
            if isinstance(hook, PerformanceMonitorHook):
                hook.backward_times.append(backward_time)
                hook.backward_time = backward_time
        
        # 优化器更新时间（支持 AMP）
        optimizer_start = time.perf_counter()
        if use_amp:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        if roi_gate_before is not None:
            try:
                gate_params = [
                    p
                    for name, p in _model.named_parameters()
                    if "roi_refine_head.gate" in name and p.requires_grad
                ]
                roi_gate_after = torch.cat([p.detach().flatten().cpu() for p in gate_params])
                update_norm = float((roi_gate_after - roi_gate_before).norm().item())
                write_header = not os.path.exists(gate_log_path)
                os.makedirs(os.path.dirname(gate_log_path) or ".", exist_ok=True)
                obj_stats = (roi_gate_output_stats or {}).get("object") or {}

                def _stat(stats, key):
                    if key == "count":
                        return int(stats.get(key, 0))
                    return float(stats.get(key, 0.0))

                with open(gate_log_path, "a") as f:
                    if write_header:
                        f.write("iter,object_count,object_gate_mean,object_gate_std,object_gate_min,object_gate_max,grad_norm,update_norm")
                    cls_keys = sorted([k for k in obj_stats if k.endswith("_mean") or k.startswith("gate_") or k.startswith("area_")])
                    if cls_keys:
                        f.write("," + ",".join(cls_keys))
                    f.write("\n")
                    row = f"{self.iter},{_stat(obj_stats, 'count')},{_stat(obj_stats, 'mean'):.8f},{_stat(obj_stats, 'std'):.8f},{_stat(obj_stats, 'min'):.8f},{_stat(obj_stats, 'max'):.8f},{float(roi_gate_grad_norm):.8f},{update_norm:.8f}"
                    per_cls_keys = sorted([k for k in obj_stats if k.endswith("_mean") or k.startswith("gate_") or k.startswith("area_")])
                    for k in per_cls_keys:
                        row += f",{float(obj_stats.get(k, 0.0)):.8f}"
                    f.write(row + "\n")
            except Exception:
                pass
        if diagnostics_state is not None:
            try:
                detr_module = getattr(_model, "detr", None)
                backbone_module = None
                if detr_module is not None and hasattr(detr_module, "backbone"):
                    backbone_container = detr_module.backbone
                    if isinstance(backbone_container, torch.nn.Sequential) and len(backbone_container) > 0:
                        backbone_module = backbone_container[0]

                object_gate_stats = None
                relation_gate_stats = None
                if detr_module is not None:
                    obj_injector = getattr(detr_module, "query_injector", None)
                    rel_injector = getattr(detr_module, "relation_query_injector", None)
                    if obj_injector is not None and hasattr(obj_injector, "get_last_gate_stats"):
                        object_gate_stats = obj_injector.get_last_gate_stats()
                    if rel_injector is not None and hasattr(rel_injector, "get_last_gate_stats"):
                        relation_gate_stats = rel_injector.get_last_gate_stats()

                roi_stats = getattr(detr_module, "_last_roi_gate_stats", None) if detr_module is not None else None
                roi_obj_stats = (roi_stats or {}).get("object") or {}

                patch_stats = None
                if backbone_module is not None and hasattr(backbone_module, "get_patch_merge_stats"):
                    patch_stats = backbone_module.get_patch_merge_stats()

                roi_update_norm = 0.0
                if diagnostics_state["roi_before"] is not None:
                    roi_after, _ = _flatten_params_by_name(_model, "roi_refine_head.gate")
                    if roi_after is not None:
                        roi_update_norm = float((roi_after - diagnostics_state["roi_before"]).norm().item())

                xsam_update_norm = 0.0
                if diagnostics_state["xsam_before"] is not None:
                    xsam_after, _ = _flatten_params_by_name(_model, "_patch_merge_proj")
                    if xsam_after is not None:
                        xsam_update_norm = float((xsam_after - diagnostics_state["xsam_before"]).norm().item())

                write_header = not os.path.exists(diagnostics_log_path)
                os.makedirs(os.path.dirname(diagnostics_log_path) or ".", exist_ok=True)
                with open(diagnostics_log_path, "a") as f:
                    if write_header:
                        f.write(
                            "iter,"
                            "temporal_object_raw_gate,temporal_object_effective_gate,temporal_object_warmup,"
                            "temporal_object_gate_min,temporal_object_gate_max,"
                            "temporal_relation_raw_gate,temporal_relation_effective_gate,temporal_relation_warmup,"
                            "temporal_relation_gate_min,temporal_relation_gate_max,"
                            "roi_object_count,roi_object_gate_mean,roi_object_gate_std,roi_object_gate_min,roi_object_gate_max,"
                            "roi_gate_grad_norm,roi_gate_update_norm,"
                            "xsam_patch_factor,xsam_weight_mean,xsam_weight_std,xsam_delta_mean_abs,xsam_delta_max_abs,"
                            "xsam_delta_norm,xsam_init_norm,xsam_grad_norm,xsam_update_norm\n"
                        )
                    f.write(
                        f"{self.iter},"
                        f"{_scalar(object_gate_stats, 'raw_gate'):.8f},"
                        f"{_scalar(object_gate_stats, 'effective_gate'):.8f},"
                        f"{_scalar(object_gate_stats, 'warmup_factor'):.8f},"
                        f"{_scalar(object_gate_stats, 'gate_min'):.8f},"
                        f"{_scalar(object_gate_stats, 'gate_max'):.8f},"
                        f"{_scalar(relation_gate_stats, 'raw_gate'):.8f},"
                        f"{_scalar(relation_gate_stats, 'effective_gate'):.8f},"
                        f"{_scalar(relation_gate_stats, 'warmup_factor'):.8f},"
                        f"{_scalar(relation_gate_stats, 'gate_min'):.8f},"
                        f"{_scalar(relation_gate_stats, 'gate_max'):.8f},"
                        f"{_count(roi_obj_stats)},"
                        f"{_scalar(roi_obj_stats, 'mean'):.8f},"
                        f"{_scalar(roi_obj_stats, 'std'):.8f},"
                        f"{_scalar(roi_obj_stats, 'min'):.8f},"
                        f"{_scalar(roi_obj_stats, 'max'):.8f},"
                        f"{float(diagnostics_state.get('roi_grad_norm', 0.0)):.8f},"
                        f"{roi_update_norm:.8f},"
                        f"{int(_scalar(patch_stats, 'factor', 0.0))},"
                        f"{_scalar(patch_stats, 'weight_mean'):.8f},"
                        f"{_scalar(patch_stats, 'weight_std'):.8f},"
                        f"{_scalar(patch_stats, 'delta_mean_abs'):.8f},"
                        f"{_scalar(patch_stats, 'delta_max_abs'):.8f},"
                        f"{_scalar(patch_stats, 'delta_norm'):.8f},"
                        f"{_scalar(patch_stats, 'init_norm'):.8f},"
                        f"{float(diagnostics_state.get('xsam_grad_norm', 0.0)):.8f},"
                        f"{xsam_update_norm:.8f}\n"
                    )
            except Exception:
                pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        optimizer_time = time.perf_counter() - optimizer_start
        
        # 记录优化器更新时间
        for hook in self._hooks:
            if isinstance(hook, PerformanceMonitorHook):
                hook.optimizer_times.append(optimizer_time)
                hook.optimizer_time = optimizer_time
        
        # 写入指标（使用原有的方法）
        # 注意：_write_metrics 在 _trainer (SimpleTrainer) 中，不在 DefaultTrainer 中
        if hasattr(self, '_trainer') and hasattr(self._trainer, '_write_metrics'):
            self._trainer._write_metrics(loss_dict, data_time)
        else:
            # 如果 _trainer 不存在，尝试直接调用（不应该发生）
            logger = logging.getLogger(__name__)
            logger.warning("Cannot find _trainer._write_metrics, metrics may not be written")

    def _ensure_late_patch_merge_optimizer_params(self, model):
        """Add lazily-created X-SAM patch-merge params to the optimizer once."""
        if getattr(self, "_late_patch_merge_params_added", False):
            return
        patch_params = [
            p
            for name, p in model.named_parameters()
            if "_patch_merge_proj" in name and p.requires_grad
        ]
        if len(patch_params) == 0:
            return
        existing = {
            id(p)
            for group in self.optimizer.param_groups
            for p in group.get("params", [])
        }
        missing = [p for p in patch_params if id(p) not in existing]
        if len(missing) == 0:
            self._late_patch_merge_params_added = True
            return
        lr = self.cfg.SOLVER.BASE_LR * self.cfg.SOLVER.BACKBONE_MULTIPLIER * self.cfg.SOLVER.ENTITY_MULTIPLIER
        self.optimizer.add_param_group(
            {
                "params": missing,
                "lr": lr,
                "weight_decay": 0.0,
            }
        )
        self._late_patch_merge_params_added = True
        logging.getLogger("detectron2").info(
            "Added %d lazily-created X-SAM patch-merge parameters to optimizer with lr=%s weight_decay=0.0",
            len(missing),
            lr,
        )


    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        logger = logging.getLogger("detectron2")
        sam3_delayed_unfreeze = (
            cfg.MODEL.SAM3.ENABLED
            and cfg.MODEL.SAM3.FREEZE
            and int(getattr(cfg.MODEL.SAM3, "UNFREEZE_AFTER_ITER", -1)) >= 0
        )
        patch_merge_params = []
        for key, value in model.named_parameters(recurse=True):
            if "_patch_merge_proj" in key:
                if value.requires_grad:
                    patch_merge_params.append(value)
                continue
            include_frozen_sam3 = sam3_delayed_unfreeze and ("sam3_model" in key)
            if (not value.requires_grad) and (not include_frozen_sam3):
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "triplet_" in key or "temporal_" in key:
                lr = lr * cfg.SOLVER.TEMPORAL_LR_MULTIPLIER
                logger.info("Setting LR for temporal {} to {}".format(key, lr))
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            if "gate" in key:
                lr = lr * cfg.SOLVER.GATE_LR_MULTIPLIER
                logger.info("Setting LR for gate {} to {}".format(key, lr))
            if "relation" in key:
                lr = lr * cfg.SOLVER.RELATION_MULTIPLIER
                logger.info("Setting LR for {} to {}".format(key, lr)) 
            if "detr.transformer.encoder" in key or "detr.transformer.decoder.layers" in key or "detr.query_embed" in key or 'backbone' in key or 'detr.transformer.decoder.norm' in key:
                lr = lr * cfg.SOLVER.ENTITY_MULTIPLIER
                logger.info("Setting LR for {} to {}".format(key, lr))
            if include_frozen_sam3 and (not value.requires_grad):
                logger.info(
                    "Include frozen SAM3 param for delayed unfreeze: %s (iter=%d)",
                    key,
                    int(getattr(cfg.MODEL.SAM3, "UNFREEZE_AFTER_ITER", -1)),
                )
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if patch_merge_params:
            patch_lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BACKBONE_MULTIPLIER * cfg.SOLVER.ENTITY_MULTIPLIER
            params += [{"params": patch_merge_params, "lr": patch_lr, "weight_decay": 0.0}]
            logger.info(
                "Adding %d X-SAM patch-merge parameters to optimizer with lr=%s weight_decay=0.0",
                len(patch_merge_params),
                patch_lr,
            )

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)

        
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # import ipdb; ipdb.set_trace()
            
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            if cfg.MODEL.DETR.RELATION_HEAD and cfg.MODEL.META_ARCHITECTURE != 'DetrWithSGGBBox' and cfg.MODEL.META_ARCHITECTURE != 'Detr' and cfg.MODEL.META_ARCHITECTURE != 'QuerySplitObjectDetr' and cfg.MODEL.META_ARCHITECTURE != 'QuerySplitUnionBoxDetr' and cfg.MODEL.META_ARCHITECTURE != 'QuerySplitObjectDetrTest' and cfg.MODEL.META_ARCHITECTURE != 'RelationDetr' and cfg.MODEL.META_ARCHITECTURE != 'ConditionalDETR' and cfg.MODEL.META_ARCHITECTURE != 'QueryConditionalDETR' and cfg.MODEL.META_ARCHITECTURE != 'QueryConditionalDeformableDETR' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxConditionalDETR' and cfg.MODEL.META_ARCHITECTURE != 'LatentRelationDETRTest' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxDETR' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxDeformableDETR' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxDETRTest' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxConditionalDETRTest' and cfg.MODEL.META_ARCHITECTURE != 'LatentRelationCoordsDETRTest' and cfg.MODEL.META_ARCHITECTURE != 'LatentBoxCoordsDetr':
            #  and cfg.MODEL.META_ARCHITECTURE != 'LatentRelationCoordsNoAttentionDETR':
                if cfg.MODEL.ROI_REFINE.ENABLED and cfg.MODEL.ROI_REFINE.EVAL_DUAL:
                    # 单次前向同出 override(roi) 与 raw(origin) 两套指标，分别落盘到子目录
                    evaluator = {
                        "override": SceneGraphEvaluator(
                            dataset_name, cfg, True, os.path.join(output_folder, "override")
                        ),
                        "raw": SceneGraphEvaluator(
                            dataset_name, cfg, True, os.path.join(output_folder, "raw")
                        ),
                    }
                else:
                    evaluator = SceneGraphEvaluator(dataset_name, cfg, True, output_folder)
            else:
                evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
            results_i = scenegraph_inference_on_dataset(cfg, model, data_loader, evaluator)
            
            # print("Out of sg inference")
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
        comm.synchronize()
        if len(results) == 1:
            results = list(results.values())[0]
        return results
