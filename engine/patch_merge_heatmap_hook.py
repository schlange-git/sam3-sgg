import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import detectron2.utils.comm as comm
from detectron2.engine import HookBase


def resolveSam3Backbone(model):
    base = model.module if hasattr(model, "module") else model
    detr = getattr(base, "detr", None)
    assert detr is not None and hasattr(detr, "backbone"), "SAM3 DETR backbone not found."
    backbone = detr.backbone
    if isinstance(backbone, torch.nn.Sequential):
        backbone = backbone[0]
    return backbone


class PatchMergeHeatmapHook(HookBase):
    def __init__(self, period: int, output_dir: str):
        self.period = int(period)
        self.out_dir = os.path.join(output_dir, "patch_merge_heatmaps")
        self.logger = logging.getLogger("detectron2")
        self.done_iters = set()

    def before_train(self):
        if comm.is_main_process():
            os.makedirs(self.out_dir, exist_ok=True)
            self.dumpHeatmap(0)

    def after_step(self):
        if not comm.is_main_process():
            return
        next_iter = int(self.trainer.iter) + 1
        is_eval_iter = self.period > 0 and next_iter % self.period == 0
        is_final_iter = next_iter >= int(self.trainer.max_iter)
        if is_eval_iter or is_final_iter:
            self.dumpHeatmap(next_iter)

    def dumpHeatmap(self, iteration: int):
        if iteration in self.done_iters:
            return
        backbone = resolveSam3Backbone(self.trainer.model)
        proj = getattr(backbone, "_patch_merge_proj", None)
        assert proj is not None, "PatchMergeHeatmapHook requires _patch_merge_proj."

        weight = proj.weight.detach().float().cpu().squeeze().numpy()
        channels, input_channels = weight.shape
        subpixel_count = input_channels // channels
        factor = int(round(subpixel_count ** 0.5))
        assert factor * factor == subpixel_count, f"bad patch_merge weight shape {weight.shape}"

        allocation = np.zeros((channels, subpixel_count), dtype=np.float32)
        for out_channel in range(channels):
            for subpixel_idx in range(subpixel_count):
                allocation[out_channel, subpixel_idx] = weight[out_channel, out_channel + subpixel_idx * channels]

        uniform_value = 1.0 / subpixel_count
        max_delta = float(np.abs(allocation - uniform_value).max())
        mean_delta = float(np.abs(allocation - uniform_value).mean())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes[0].imshow(allocation, aspect="auto", cmap="viridis")
        axes[0].set_title(
            f"sub-pixel allocation, iter={iteration}\n"
            f"uniform={uniform_value:.4f}, mean|delta|={mean_delta:.4g}, max|delta|={max_delta:.4g}"
        )
        axes[0].set_xlabel("sub-pixel index")
        axes[0].set_ylabel("output channel")
        fig.colorbar(im0, ax=axes[0], shrink=0.85)

        im1 = axes[1].imshow(
            allocation - uniform_value,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-uniform_value,
            vmax=uniform_value,
        )
        axes[1].set_title("deviation from uniform avg-pool allocation")
        axes[1].set_xlabel("sub-pixel index")
        axes[1].set_ylabel("output channel")
        fig.colorbar(im1, ax=axes[1], shrink=0.85)

        fig.suptitle(f"patch_merge 1x1 conv heatmap (factor={factor}, C={channels})")
        fig.tight_layout()
        path = os.path.join(self.out_dir, f"iter_{iteration:06d}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        self.done_iters.add(iteration)
        self.logger.info("[patch_merge_heatmap] saved %s", path)
