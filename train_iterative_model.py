import sys
import os
import gc
import numpy as np
import torch
import warnings
import math
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

# Suppress FutureWarning from detectron2 autocast usage
warnings.filterwarnings("ignore", category=FutureWarning, module="detectron2.engine.train_loop")

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import get_detection_dataset_dicts

from SpeaQ.engine import JointTransformerTrainer
from SpeaQ.data import VisualGenomeTrainData, register_datasets, DatasetCatalog, MetadataCatalog
from SpeaQ.configs.defaults import add_dataset_config, add_scenegraph_config
from SpeaQ.modeling import Detr
from detectron2.data.datasets import register_coco_instances
from glob import glob
import pathlib
from shutil import copyfile

parser = default_argument_parser()

def log_gpu_memory(prefix: str) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        print(f"{prefix} GPU memory: total={total:.1f} MiB, allocated={allocated:.1f} MiB, reserved={reserved:.1f} MiB")
    else:
        print(f"{prefix} GPU memory: CUDA not available")

def cleanup_cuda(prefix: str) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    log_gpu_memory(prefix)

def backup_source_codes(cfg):
    if comm.is_main_process():
        output_dir = cfg.OUTPUT_DIR
        # Only backup Python source files and configs, skip large directories
        skip_dirs = {'wandb', '__pycache__', '.git', 'z_outputs', 'tmp', 'sam3', 'node_modules', '.pytest_cache', 'build', 'dist', '.eggs'}
        skip_extensions = {'.pth', '.pkl', '.h5', '.pt', '.pth.tar', '.ckpt', '.log', '.pyc', '.pyo', '.so', '.dylib'}
        source_files = glob('**/*.py', recursive=True)
        source_files.extend(glob('**/*.yaml', recursive=True))
        source_files.extend(glob('**/*.yml', recursive=True))
        source_files.extend(glob('**/*.sh', recursive=True))

        for file in source_files:
            # Skip files in excluded directories
            if any(skip_dir in file for skip_dir in skip_dirs):
                continue
            # Skip if extension is excluded
            if any(file.endswith(ext) for ext in skip_extensions):
                continue
            if os.path.isdir(file):
                continue
                target_dir = os.path.join(output_dir, 'code_backup', file)
                os.makedirs(os.path.dirname(target_dir), exist_ok=True)
            try:
                copyfile(file, target_dir)
            except Exception as e:
                # Skip files that can't be copied (e.g., broken symlinks)
                pass

def setup(args):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    assert(cfg.MODEL.ROI_SCENEGRAPH_HEAD.MODE in ['predcls', 'sgls', 'sgdet']), "Mode {} not supported".format(cfg.MODEL.ROI_SCENEGRaGraph.MODE)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # 仅在训练阶段清空 MODEL.WEIGHTS：
    # - 训练时若启用 SAM3 / LOAD_HEAD_ONLY，避免错误地整模型加载 backbone 权重
    # - eval_only 必须保留用户传入的 MODEL.WEIGHTS（如 model_final.pth），否则会退化为“未加载checkpoint评测”
    if (not args.eval_only) and (
        cfg.MODEL.SAM3.ENABLED or (cfg.MODEL.DETR.LOAD_HEAD_ONLY and cfg.MODEL.DETR.HEAD_WEIGHTS)
    ):
        cfg.MODEL.WEIGHTS = ""
    register_datasets(cfg)
    if cfg.DATASETS.TYPE == "ACTION GENOME" and len(cfg.DATASETS.TRAIN) > 0:
        train_name = cfg.DATASETS.TRAIN[0]
        metadata = MetadataCatalog.get(train_name)
        num_obj = len(getattr(metadata, "thing_classes", []))
        num_rel = len(getattr(metadata, "predicate_classes", []))
        if num_obj > 0 and num_rel > 0:
            cfg.MODEL.DETR.NUM_CLASSES = num_obj
            cfg.MODEL.DETR.NUM_RELATION_CLASSES = num_rel
            cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES = num_rel  # 评估用关系类别数，与 relationship_classes.txt 一致
            print(f"[ActionGenome] Auto-set classes: NUM_CLASSES={num_obj}, NUM_RELATION_CLASSES={num_rel} (from AG_ANNOTATIONS class files)")

        # 若显式设置了 MAX_EPOCH，则基于“每个 epoch = 全部帧”推导 MAX_ITER
        if getattr(cfg.SOLVER, "MAX_EPOCH", 0) > 0:
            train_sets = cfg.DATASETS.TRAIN
            dataset_dicts = get_detection_dataset_dicts(
                train_sets,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if getattr(cfg.MODEL, "KEYPOINT_ON", False)
                else 0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
                if getattr(cfg.DATASETS, "PROPOSAL_FILES_TRAIN", None)
                else None,
            )
            num_samples = max(1, len(dataset_dicts))
            ims_per_batch = max(1, int(cfg.SOLVER.IMS_PER_BATCH))
            iters_per_epoch = math.ceil(num_samples / float(ims_per_batch))
            max_epoch = int(cfg.SOLVER.MAX_EPOCH)
            new_max_iter = iters_per_epoch * max_epoch
            print(
                f"[EpochMode] ACTION GENOME: num_samples={num_samples}, "
                f"ims_per_batch={ims_per_batch}, iters_per_epoch={iters_per_epoch}, "
                f"MAX_EPOCH={max_epoch} -> MAX_ITER={new_max_iter}"
            )
            cfg.SOLVER.MAX_ITER = new_max_iter
    cfg.freeze()
    # register_coco_data(cfg)
    default_setup(cfg, args)
    
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="LSDA")
    return cfg

def main(args):
    cfg = setup(args)
    cleanup_cuda("Before building model")
    if args.eval_only:
        model = JointTransformerTrainer.build_model(cfg)
        # from thop import profile
        # input = torch.randn(1, 3, 800, 1333)
        # macs, params = profile(model, inputs=(input, ))

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        res = JointTransformerTrainer.test(cfg, model)
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        return res
    backup_source_codes(cfg)
    trainer = JointTransformerTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == '__main__':
    args = parser.parse_args()
    # 如果 dist_url 已经是完整的URL（以 tcp:// 开头），直接使用
    # 否则，从环境变量或默认值获取端口号并拼接URL
    if args.dist_url and args.dist_url.startswith('tcp://'):
        # 已经是完整的URL，直接使用
        pass
    else:
        try:
            # use the last 4 numbers in the job id as the id
            # default_port = os.environ['SLURM_JOB_ID']
            # default_port = default_port[-4:]
            #
            # # all ports should be in the 10k+ range
            # default_port = int(default_port) + 15000
            if args.dist_url:
                # 如果 dist_url 是端口号字符串，提取端口号
                default_port = int(args.dist_url) if args.dist_url.isdigit() else 30050
            else:
                default_port = 30050
        except Exception:
            default_port = 30050
        
        args.dist_url = 'tcp://127.0.0.1:'+str(default_port)
    print(args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
