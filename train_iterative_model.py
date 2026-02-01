import sys
import os
import gc
import numpy as np
import torch
import warnings
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

# Suppress FutureWarning from detectron2 autocast usage
warnings.filterwarnings("ignore", category=FutureWarning, module="detectron2.engine.train_loop")

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

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
    # If using SAM3 backbone, we should not load ResNet backbone weights
    # If LOAD_HEAD_ONLY is True, clear MODEL.WEIGHTS to avoid loading full model
    if cfg.MODEL.SAM3.ENABLED or (cfg.MODEL.DETR.LOAD_HEAD_ONLY and cfg.MODEL.DETR.HEAD_WEIGHTS):
        # Avoid full-model loading when using SAM3 or when only DETR head weights are intended
        cfg.MODEL.WEIGHTS = ""
    cfg.freeze()
    register_datasets(cfg)
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
    try:
        # use the last 4 numbers in the job id as the id
        # default_port = os.environ['SLURM_JOB_ID']
        # default_port = default_port[-4:]
        #
        # # all ports should be in the 10k+ range
        # default_port = int(default_port) + 15000
        default_port = args.dist_url
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
