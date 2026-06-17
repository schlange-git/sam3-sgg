import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import detectron2.utils.comm as comm
from detectron2.engine import HookBase


GLOBAL_LOCAL_ROW_START = 120
GLOBAL_LOCAL_ROW_COUNT = 16
GLOBAL_LOCAL_COL_START = 480
GLOBAL_LOCAL_COL_COUNT = 64


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
        assert channels == 256 and input_channels == 1024, f"expected [256,1024], got {weight.shape}"

        uniform_value = 1.0 / float(input_channels)
        max_delta = float(np.abs(weight - uniform_value).max())
        mean_delta = float(np.abs(weight - uniform_value).mean())

        global_path = os.path.join(self.out_dir, f"iter_{iteration:06d}_global_256x1024.png")
        local_path = os.path.join(self.out_dir, f"iter_{iteration:06d}_local_r120_136_c480_544.png")
        self.saveGlobalHeatmap(weight, iteration, factor, uniform_value, mean_delta, max_delta, global_path)
        self.saveLocalHeatmap(weight, iteration, uniform_value, local_path)
        self.done_iters.add(iteration)
        self.logger.info("[patch_merge_heatmap] saved %s and %s", global_path, local_path)

    def saveGlobalHeatmap(self, weight, iteration, factor, uniform_value, mean_delta, max_delta, path):
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        im0 = axes[0].imshow(weight, aspect="auto", cmap="viridis")
        axes[0].set_title(
            f"full patch_merge weight [256,1024], iter={iteration}\n"
            f"global uniform={uniform_value:.6f}, mean|delta|={mean_delta:.4g}, max|delta|={max_delta:.4g}"
        )
        axes[0].set_xlabel("input channel after pixel_unshuffle")
        axes[0].set_ylabel("output channel")
        fig.colorbar(im0, ax=axes[0], shrink=0.85)

        im1 = axes[1].imshow(weight - uniform_value, aspect="auto", cmap="RdBu_r")
        axes[1].set_title("full deviation from global-uniform init")
        axes[1].set_xlabel("input channel after pixel_unshuffle")
        axes[1].set_ylabel("output channel")
        fig.colorbar(im1, ax=axes[1], shrink=0.85)
        fig.suptitle(f"patch_merge dense 1x1 conv (factor={factor})")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def saveLocalHeatmap(self, weight, iteration, uniform_value, path):
        row_end = GLOBAL_LOCAL_ROW_START + GLOBAL_LOCAL_ROW_COUNT
        col_end = GLOBAL_LOCAL_COL_START + GLOBAL_LOCAL_COL_COUNT
        local = weight[GLOBAL_LOCAL_ROW_START:row_end, GLOBAL_LOCAL_COL_START:col_end]
        assert local.shape == (GLOBAL_LOCAL_ROW_COUNT, GLOBAL_LOCAL_COL_COUNT), f"bad local shape {local.shape}"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes[0].imshow(local, aspect="auto", cmap="viridis")
        axes[0].set_title(
            f"fixed local weight block, iter={iteration}\n"
            f"rows {GLOBAL_LOCAL_ROW_START}:{row_end}, cols {GLOBAL_LOCAL_COL_START}:{col_end}"
        )
        axes[0].set_xlabel("input channel")
        axes[0].set_ylabel("output channel")
        fig.colorbar(im0, ax=axes[0], shrink=0.85)

        im1 = axes[1].imshow(local - uniform_value, aspect="auto", cmap="RdBu_r")
        axes[1].set_title("local deviation from global-uniform init")
        axes[1].set_xlabel("input channel")
        axes[1].set_ylabel("output channel")
        fig.colorbar(im1, ax=axes[1], shrink=0.85)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
