import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from collections import Counter
import torch

from detectron2.utils.comm import get_world_size, is_main_process


from detectron2.evaluation import COCOEvaluator


def scenegraph_inference_on_dataset(cfg, model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger('detectron2')
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    # evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)

    # ROI EVAL_DUAL: evaluator 可能是 {"override":..., "raw":...} 字典，单次前向同出两套指标
    dual_eval = isinstance(evaluator, dict)
    if dual_eval:
        assert "override" in evaluator and "raw" in evaluator, (
            "Dual evaluator dict must contain 'override' and 'raw' keys."
        )
        evaluator["override"].reset(total * num_devices)
        evaluator["raw"].reset(total * num_devices)
    else:
        evaluator.reset(total*num_devices)
    num_warmup = min(5, total - 1)
    
    # 在评估开始前清理显存，释放训练阶段占用的内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache before evaluation")
    
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

    #        if len(inputs[0]['instances']) > 40:
    #           continue
            start_compute_time = time.perf_counter()
            
            outputs = model(inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            if dual_eval:
                assert isinstance(outputs, dict) and outputs.get("__roi_dual__"), (
                    "Dual evaluator expects model outputs dict with '__roi_dual__'; "
                    "got non-dual outputs. Check MODEL.ROI_REFINE.EVAL_DUAL wiring."
                )
                evaluator["override"].process(inputs, outputs["override"])
                evaluator["raw"].process(inputs, outputs["raw"])
            else:
                evaluator.process(inputs, outputs)
            
            # 定期清理显存，避免内存碎片化（每100个样本清理一次）
            if torch.cuda.is_available() and (idx + 1) % 100 == 0:
                torch.cuda.empty_cache()

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start

            # 不再每 N 秒打印 "Inference done ..."，避免 log 被刷屏；仅保留结束时汇总

            if cfg.DEV_RUN and idx==2:
                break
    
    
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    if dual_eval:
        results_override = evaluator["override"].evaluate()
        results_raw = evaluator["raw"].evaluate()
        if results_override is None:
            results_override = {}
        if results_raw is None:
            results_raw = {}
        # override(roi) 指标保持原 key（向后兼容 EVAL_FIRST / best 追踪）；
        # raw(origin) 指标加 raw_ 前缀附加，单次前向同时产出两套结果。
        results = dict(results_override)
        for k, v in results_raw.items():
            results["raw_" + str(k)] = v
        return results

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

# _LOG_COUNTER = Counter()
# _LOG_TIMER = {}

# def log_every_n_seconds(lvl, msg, n=1, *, name=None):
#     """
#     Log no more than once per n seconds.

#     Args:
#         lvl (int): the logging level
#         msg (str):
#         n (int):
#         name (str): name of the logger to use. Will use the caller's module by default.
#     """
#     caller_module, key = _find_caller()
#     last_logged = _LOG_TIMER.get(key, None)
#     current_time = time.time()
#     if last_logged is None or current_time - last_logged >= n:
#         logging.getLogger('detectron2').log(lvl, msg)
#         _LOG_TIMER[key] = current_time

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
