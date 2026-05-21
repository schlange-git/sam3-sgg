import logging
import copy 
import os
from typing import OrderedDict
from collections import defaultdict
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
import itertools
from tabulate import tabulate

from abc import ABC, abstractmethod

from ..data import DetrDatasetMapper

from fvcore.common.file_io import PathManager
import detectron2.utils.comm as comm
from detectron2.structures import Instances
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation import COCOEvaluator
from detectron2.structures.boxes import pairwise_iou, Boxes
from detectron2.utils.registry import Registry


from .utils import intersect_2d, argsort_desc

idx2predicate = {0: 'above', 1: 'across', 2: 'against', 3: 'along', 4: 'and', 5: 'at', 6: 'attached to', 7: 'behind', 8: 'belonging to', 9: 'between', 10: 'carrying',
                11: 'covered in', 12: 'covering', 13: 'eating', 14: 'flying in', 15: 'for', 16: 'from', 17: 'growing on', 18: 'hanging from', 19: 'has', 20: 'holding',
                21: 'in', 22: 'in front of', 23: 'laying on', 24: 'looking at', 25: 'lying on', 26: 'made of', 27: 'mounted on', 28: 'near', 29: 'of', 30: 'on',
                31: 'on back of', 32: 'over', 33: 'painted on', 34: 'parked on', 35: 'part of', 36: 'playing', 37: 'riding', 38: 'says', 39: 'sitting on', 40: 'standing on',
                41: 'to', 42: 'under', 43: 'using', 44: 'walking in', 45: 'walking on', 46: 'watching', 47: 'wearing', 48: 'wears', 49: 'with'}

SCENEGRAPH_METRIC_REGISTRY = Registry("SCENEGRAPH_METRIC_REGISTRY")

class SceneGraphEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, cfg, distributed, output_dir=None, metrics=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "instances_results.json" a json file in COCO's result
                   format. #TODO: fix the commnent after implementation
            metrics (tuple): The metrics using which the scene graphs performance should be evaluated
                Options: ('SGRecall', 'SGNoGraphConstraintRecall', 'SGZeroShotRecall', 'SGPairAccuracy', 'SGMeanRecall')
        """

        SGMETRICS = ('SGRecall', 'SGNoGraphConstraintRecall', 'SGZeroShotRecall', 'SGPairAccuracy', 'SGMeanRecall','query_count_per_rel_class','rel_class_info_per_query','recall_per_class','SGErrorAnalysis')

        self._mode = self._mode_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir
        self.cfg = cfg

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger('detectron2')

        if metrics is None:
            self._metrics = SGMETRICS
        else:
            for metric in metrics:
                assert metric in SGMETRICS, "Specified scene graph evaluation metric {} not suppoted. Currently supported metrics : {}".format(metric, SGMETRICS)
            self._metrics = metrics
        
        self._logger.info("Following metrics will be use for evaluation")
        self._logger.info("{}".format(self._metrics))

        self.detection_evaluator = COCOEvaluator(dataset_name, cfg, distributed, output_dir)
        self.detection_evaluator._tasks =  ("bbox",)
        #Register a filed for each of the metric
        self._evaluators = build_scenegraph_evaluators(self._metrics, cfg, {}, dataset_name)
        # self._register_evaluator_containers()

        self._metadata = MetadataCatalog.get(dataset_name)

        self._ground_truths = []
        self._predictions = []
        self._zero_shot_triplets = self._get_zero_shot_triplets() - 1
        self._debug_eval = os.environ.get("SPEAQ_DEBUG_EVAL", "0") == "1"
        self._debug_limit = int(os.environ.get("SPEAQ_DEBUG_EVAL_LIMIT", "20"))
        self._debug_count = 0

    def reset(self,total):
        self.detection_evaluator.reset()
        self._register_evaluator_containers(total)

    def _get_zero_shot_triplets(self):
        self._logger.info('Loading zero shot triplets')
        return torch.load(self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.ZERO_SHOT_TRIPLETS , map_location=torch.device("cpu")).long().numpy()

    def _mode_from_config(self, cfg):
        '''
        Estimate mode from configuration
        '''
        if cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL:
                mode = 'predcls'
            else:
                mode = 'sgcls'
        else:
            mode = 'sgdet'

        return mode
    
    def _register_evaluator_containers(self,total):
        for evaluator in self._evaluators.keys():
            self._evaluators[evaluator].register_container(self._mode,self.cfg,total)

    def process(self, inputs, outputs):
        processed_outputs = []
        # import pdb; pdb.set_trace()
        for idx, input in enumerate(inputs):
            height, width = outputs[idx]['instances'].image_size
            input['instances'] = resize_instance(input['instances'], height, width)

        for output in outputs:
            instances = output.get("instances", None)
            if instances is None:
                processed_outputs.append(output)
                continue
            rel_pair_idxs = output.get("rel_pair_idxs", getattr(instances, "_rel_pair_idxs", None))
            pred_rel_scores = output.get("pred_rel_scores", getattr(instances, "_pred_rel_scores", None))
            pred_rel_labels = output.get("pred_rel_labels", getattr(instances, "_pred_rel_labels", None))
            query_index = output.get("query_index", getattr(instances, "_query_index", None))
            new_instances, rel_pair_idxs, pred_rel_scores, pred_rel_labels, query_index = self._apply_classwise_nms(
                instances.to(self._cpu_device),
                rel_pair_idxs.to(self._cpu_device) if rel_pair_idxs is not None else None,
                pred_rel_scores.to(self._cpu_device) if pred_rel_scores is not None else None,
                pred_rel_labels.to(self._cpu_device) if pred_rel_labels is not None else None,
                query_index.to(self._cpu_device) if query_index is not None else None,
                iou_thresh=getattr(
                    self.cfg.TEST.RELATION,
                    "CLASSWISE_MINIOU_THRESH",
                    getattr(self.cfg.TEST.RELATION, "CLASSWISE_NMS_THRESH", 0.9),
                ),
            )
            processed_outputs.append(
                {
                    **output,
                    "instances": new_instances,
                    "rel_pair_idxs": rel_pair_idxs,
                    "pred_rel_scores": pred_rel_scores,
                    "pred_rel_labels": pred_rel_labels,
                    "query_index": query_index,
                }
            )

        self.detection_evaluator.process(inputs, processed_outputs)

        for input, output in zip(inputs, processed_outputs):
            ground_truth = {}
            prediction = {}

            ground_truth['relation_tuple'] = input['relations'].to(self._cpu_device) #Relation tupe (obj_id, sub_id, relation label)
            ground_truth['gt_boxes'] = input['instances'].gt_boxes.to(self._cpu_device) #Ground truth object boxes
            ground_truth['labels'] = input['instances'].gt_classes.to(self._cpu_device) #Ground truth object classes
            ground_truth['rel_pair_idxs'] = input['relations'][:,:2].to(self._cpu_device) #Realtion pair index (shape: (num of relations, 2))

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["image_id"] = input["image_id"]
                prediction["instances"] = instances
                num_rel_category = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
                rel_pair_idxs = output.get("rel_pair_idxs", getattr(output["instances"], "_rel_pair_idxs", None))
                pred_rel_scores = output.get("pred_rel_scores", getattr(output["instances"], "_pred_rel_scores", None))
                pred_rel_labels = output.get("pred_rel_labels", getattr(output["instances"], "_pred_rel_labels", None))
                query_index = output.get("query_index", getattr(output["instances"], "_query_index", None))
                obj_split_sub_head_source = output.get("obj_split_sub_head_source", getattr(output["instances"], "_obj_split_sub_head_source", None))
                obj_split_obj_head_source = output.get("obj_split_obj_head_source", getattr(output["instances"], "_obj_split_obj_head_source", None))
                if rel_pair_idxs is None:
                    rel_pair_idxs = torch.zeros((0, 2), dtype=torch.int64)
                if pred_rel_scores is None:
                    pred_rel_scores = torch.zeros((0, num_rel_category), dtype=torch.float32)
                if pred_rel_labels is None:
                    pred_rel_labels = torch.zeros((0,), dtype=torch.int64)
                if query_index is None:
                    query_index = torch.zeros((0,), dtype=torch.int64)
                if obj_split_sub_head_source is None:
                    obj_split_sub_head_source = torch.full((query_index.shape[0],), -1, dtype=torch.int64)
                if obj_split_obj_head_source is None:
                    obj_split_obj_head_source = torch.full((query_index.shape[0],), -1, dtype=torch.int64)
                prediction['rel_pair_idxs'] = rel_pair_idxs.to(self._cpu_device)
                prediction['pred_rel_scores'] = pred_rel_scores.to(self._cpu_device)
                prediction['pred_rel_labels'] = pred_rel_labels.to(self._cpu_device)
                prediction['query_index'] = query_index.to(self._cpu_device)
                prediction['obj_split_sub_head_source'] = obj_split_sub_head_source.to(self._cpu_device)
                prediction['obj_split_obj_head_source'] = obj_split_obj_head_source.to(self._cpu_device)
                if self._debug_eval and self._debug_count < self._debug_limit:
                    self._logger.info(
                        "[SPEAQ_DEBUG_EVAL] image_id=%s gt_rel=%d gt_boxes=%d "
                        "pred_boxes=%d pred_rel=%d pred_rel_scores_shape=%s",
                        input.get("image_id", "unknown"),
                        int(ground_truth['relation_tuple'].shape[0]),
                        int(ground_truth['gt_boxes'].tensor.shape[0]),
                        int(instances.pred_boxes.tensor.shape[0]),
                        int(prediction['rel_pair_idxs'].shape[0]),
                        tuple(prediction['pred_rel_scores'].shape),
                    )
                    self._debug_count += 1
            else:
                if self._debug_eval and self._debug_count < self._debug_limit:
                    self._logger.warning(
                        "[SPEAQ_DEBUG_EVAL] image_id=%s output has no instances",
                        input.get("image_id", "unknown"),
                    )
                    self._debug_count += 1
#                 prediction['subject_scores'] = output['subject_scores'].to(self._cpu_device)
#                 prediction['object_scores'] = output['object_scores'].to(self._cpu_device)
#                 prediction['predicate_scores'] = output['predicate_scores'].to(self._cpu_device)
#                 prediction['triplet_scores'] = output['triplet_scores'].to(self._cpu_device)
                
                if 'pred_rel_scores_1' in output:
                    instances_1 = output['instances_1'].to(self._cpu_device)
                    prediction['instances_1'] = instances_1
                    prediction['rel_pair_idxs_1'] = output["rel_pair_idxs_1"].to(self._cpu_device)
                    prediction['pred_rel_scores_1'] = output["pred_rel_scores_1"].to(self._cpu_device)

                if 'pred_rel_scores_3' in output:
                    instances_3 = output['instances_3'].to(self._cpu_device)
                    prediction['instances_3'] = instances_3
                    prediction['rel_pair_idxs_3'] = output["rel_pair_idxs_3"].to(self._cpu_device)
                    prediction['pred_rel_scores_3'] = output["pred_rel_scores_3"].to(self._cpu_device)

            
            ground_truth_cp = copy.deepcopy(ground_truth)
            prediction_cp = copy.deepcopy(prediction)
        
            del ground_truth 
            del prediction
            self._ground_truths.append(ground_truth_cp)
            self._predictions.append(prediction_cp)
        del outputs
        del processed_outputs
        del inputs 

    def _apply_classwise_nms(
        self,
        instances,
        rel_pair_idxs=None,
        pred_rel_scores=None,
        pred_rel_labels=None,
        query_index=None,
        iou_thresh=0.4,
    ):
        if instances is None or len(instances) == 0:
            return instances, rel_pair_idxs, pred_rel_scores, pred_rel_labels, query_index
        if (not hasattr(instances, "pred_boxes")) or (not hasattr(instances, "scores")) or (not hasattr(instances, "pred_classes")):
            return instances, rel_pair_idxs, pred_rel_scores, pred_rel_labels, query_index

        boxes = instances.pred_boxes.tensor
        scores = instances.scores
        classes = instances.pred_classes
        keep = self._classwise_min_iou_keep_indices(boxes, scores, classes, float(iou_thresh))
        keep = keep.long()
        if keep.numel() == len(instances):
            return instances, rel_pair_idxs, pred_rel_scores, pred_rel_labels, query_index

        new_instances = Instances(instances.image_size)
        new_instances.pred_boxes = Boxes(boxes[keep])
        new_instances.scores = scores[keep]
        new_instances.pred_classes = classes[keep]

        # remap relation pairs to filtered object indices
        if rel_pair_idxs is None:
            return new_instances, rel_pair_idxs, pred_rel_scores, pred_rel_labels, query_index

        keep_list = keep.detach().cpu().tolist()
        remap = {int(old): int(new) for new, old in enumerate(keep_list)}
        valid_rel_idx = []
        new_pairs = []
        for rel_i, pair in enumerate(rel_pair_idxs.detach().cpu().tolist()):
            s, o = int(pair[0]), int(pair[1])
            if s in remap and o in remap:
                valid_rel_idx.append(rel_i)
                new_pairs.append([remap[s], remap[o]])

        if len(valid_rel_idx) == 0:
            device = rel_pair_idxs.device
            empty_long = torch.zeros((0,), dtype=torch.long, device=device)
            empty_pair = torch.zeros((0, 2), dtype=torch.long, device=device)
            empty_scores = (
                torch.zeros((0, pred_rel_scores.shape[1]), dtype=pred_rel_scores.dtype, device=pred_rel_scores.device)
                if pred_rel_scores is not None and pred_rel_scores.ndim == 2
                else torch.zeros((0,), dtype=torch.float32, device=device)
            )
            return new_instances, empty_pair, empty_scores, empty_long, empty_long

        valid_rel_idx = torch.as_tensor(valid_rel_idx, dtype=torch.long, device=rel_pair_idxs.device)
        new_rel_pair_idxs = torch.as_tensor(new_pairs, dtype=torch.long, device=rel_pair_idxs.device)
        new_pred_rel_scores = pred_rel_scores.index_select(0, valid_rel_idx) if pred_rel_scores is not None else pred_rel_scores
        new_pred_rel_labels = pred_rel_labels.index_select(0, valid_rel_idx) if pred_rel_labels is not None else pred_rel_labels
        new_query_index = query_index.index_select(0, valid_rel_idx) if query_index is not None else query_index

        return new_instances, new_rel_pair_idxs, new_pred_rel_scores, new_pred_rel_labels, new_query_index

    def _classwise_min_iou_keep_indices(self, boxes, scores, classes, thresh):
        """
        Class-wise NMS using minIoU = inter / min(area_a, area_b).
        Better at suppressing nested duplicates (large box almost covering small box).
        """
        device = boxes.device
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=device)

        keep_all = []
        unique_classes = torch.unique(classes)
        for cls in unique_classes:
            cls_idx = torch.where(classes == cls)[0]
            if cls_idx.numel() == 0:
                continue
            cls_scores = scores[cls_idx]
            order = cls_idx[torch.argsort(cls_scores, descending=True)]
            cls_keep = []
            while order.numel() > 0:
                i = order[0]
                cls_keep.append(i)
                if order.numel() == 1:
                    break
                rest = order[1:]

                b_i = boxes[i]
                b_r = boxes[rest]

                xx1 = torch.maximum(b_i[0], b_r[:, 0])
                yy1 = torch.maximum(b_i[1], b_r[:, 1])
                xx2 = torch.minimum(b_i[2], b_r[:, 2])
                yy2 = torch.minimum(b_i[3], b_r[:, 3])
                inter_w = torch.clamp(xx2 - xx1, min=0.0)
                inter_h = torch.clamp(yy2 - yy1, min=0.0)
                inter = inter_w * inter_h

                area_i = torch.clamp((b_i[2] - b_i[0]) * (b_i[3] - b_i[1]), min=1e-6)
                area_r = torch.clamp((b_r[:, 2] - b_r[:, 0]) * (b_r[:, 3] - b_r[:, 1]), min=1e-6)
                min_area = torch.minimum(area_i, area_r)
                miniou = inter / min_area

                keep_mask = miniou <= thresh
                order = rest[keep_mask]
            keep_all.extend(cls_keep)

        if len(keep_all) == 0:
            return torch.zeros((0,), dtype=torch.long, device=device)
        keep = torch.stack(keep_all).long()
        keep = keep[torch.argsort(scores[keep], descending=True)]
        return keep

    def evaluate(self):
        #First evaluate the detection precisions
        result_detector = self.detection_evaluator.evaluate()

        if self._distributed:
            comm.synchronize()
            self._logger.info("Gathering data")
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            ground_truths = comm.gather(self._ground_truths, dst=0)
            ground_truths = list(itertools.chain(*ground_truths))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            ground_truths = self._ground_truths

        self._logger.info("Predictions Gathered")

        if len(predictions) == 0:
            self._logger.warning("[SceneGraphEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "scenegraph_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save({'groundtruths':ground_truths, 'predictions':predictions}, f)
        self._logger.info("Saving output prediction")

        # 额外输出 bbox recall 表格（IoU=0.50），便于和 AP 对照分析 precision/recall 失衡
        bbox_recall = self._compute_bbox_recall(ground_truths, predictions, iou_thresh=0.5)
        result_detector["bbox_recall"] = bbox_recall
        # 检测错误分析（逐类别）
        det_error_analysis = self._compute_detection_error_analysis(ground_truths, predictions, iou_thresh=0.5)
        result_detector["det_error_analysis"] = det_error_analysis

        result_detector['SG'] = self._evaluate_scenegraphs(ground_truths, predictions)
        
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "result_dict.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._evaluators['SGRecall'].result_dict, f)
        
        return result_detector 

    def _summarize_obj_split_head_source(self, predictions):
        """
        Summarize triplets by subject/object head source.
        head source code:
          0=regular, 1=small, 2+=fine group index
         -1=unknown/disabled
        """
        summary = {
            "subject_head_source_count": {},
            "object_head_source_count": {},
            "pair_head_source_count": {},
        }
        sub_count = defaultdict(int)
        obj_count = defaultdict(int)
        pair_count = defaultdict(int)
        for pred in predictions:
            s = pred.get("obj_split_sub_head_source", None)
            o = pred.get("obj_split_obj_head_source", None)
            if s is None or o is None:
                continue
            s = s.detach().cpu().numpy().tolist()
            o = o.detach().cpu().numpy().tolist()
            for ss in s:
                sub_count[int(ss)] += 1
            for oo in o:
                obj_count[int(oo)] += 1
            for ss, oo in zip(s, o):
                pair_count[f"{int(ss)}->{int(oo)}"] += 1
        summary["subject_head_source_count"] = dict(sorted(sub_count.items(), key=lambda kv: kv[0]))
        summary["object_head_source_count"] = dict(sorted(obj_count.items(), key=lambda kv: kv[0]))
        summary["pair_head_source_count"] = dict(sorted(pair_count.items(), key=lambda kv: kv[0]))
        return summary

    def _compute_bbox_recall(self, ground_truths, predictions, iou_thresh=0.5):
        thing_classes = list(getattr(self._metadata, "thing_classes", []))
        num_classes = len(thing_classes)
        gt_count = np.zeros(num_classes, dtype=np.int64)
        matched_count = np.zeros(num_classes, dtype=np.int64)

        for gt, pred in zip(ground_truths, predictions):
            gt_boxes = gt["gt_boxes"].tensor.detach().cpu()
            gt_labels = gt["labels"].long().detach().cpu()
            pred_instances = pred.get("instances", None)
            if pred_instances is None:
                for cls in gt_labels.tolist():
                    if 0 <= cls < num_classes:
                        gt_count[cls] += 1
                continue

            pred_boxes = pred_instances.pred_boxes.tensor.detach().cpu()
            pred_labels = pred_instances.pred_classes.long().detach().cpu()

            for cls in range(num_classes):
                gt_mask = gt_labels == cls
                pred_mask = pred_labels == cls
                n_gt = int(gt_mask.sum().item())
                if n_gt == 0:
                    continue
                gt_count[cls] += n_gt
                if int(pred_mask.sum().item()) == 0:
                    continue

                cls_gt_boxes = gt_boxes[gt_mask]
                cls_pred_boxes = pred_boxes[pred_mask]
                ious = pairwise_iou(Boxes(cls_gt_boxes), Boxes(cls_pred_boxes))
                # 逐个 GT 做贪心 1-to-1 匹配，统计被命中的 GT 数（Recall）
                used_pred = set()
                local_match = 0
                for gt_i in range(ious.shape[0]):
                    vals, idxs = torch.sort(ious[gt_i], descending=True)
                    for v, p_idx in zip(vals.tolist(), idxs.tolist()):
                        if v < iou_thresh:
                            break
                        if p_idx not in used_pred:
                            used_pred.add(p_idx)
                            local_match += 1
                            break
                matched_count[cls] += local_match

        total_gt = int(gt_count.sum())
        total_matched = int(matched_count.sum())
        overall_recall = (float(total_matched) / float(total_gt)) if total_gt > 0 else 0.0

        self._logger.info("Evaluation results for bbox recall (IoU=0.50):")
        overall_table = [
            ["Recall", "Matched/GT"],
            [f"{overall_recall * 100:.3f}", f"{total_matched}/{total_gt}"],
        ]
        self._logger.info("\n" + tabulate(overall_table, headers="firstrow", tablefmt="pipe"))

        per_cls_rows = []
        for cls_idx, cls_name in enumerate(thing_classes):
            cls_gt = int(gt_count[cls_idx])
            if cls_gt == 0:
                cls_recall = float("nan")
            else:
                cls_recall = float(matched_count[cls_idx]) / float(cls_gt) * 100.0
            per_cls_rows.append([cls_name, cls_recall, f"{int(matched_count[cls_idx])}/{cls_gt}"])

        if len(per_cls_rows) > 0:
            self._logger.info("Per-category bbox Recall@50:")
            self._logger.info(
                "\n"
                + tabulate(
                    per_cls_rows,
                    headers=["category", "R50", "Matched/GT"],
                    tablefmt="pipe",
                    floatfmt=".3f",
                )
            )

        recall_dict = {"Recall@50": overall_recall * 100.0}
        for cls_idx, cls_name in enumerate(thing_classes):
            cls_gt = int(gt_count[cls_idx])
            recall_dict[f"{cls_name}_R50"] = (
                float(matched_count[cls_idx]) / float(cls_gt) * 100.0 if cls_gt > 0 else float("nan")
            )
        return recall_dict

    def _compute_detection_error_analysis(self, ground_truths, predictions, iou_thresh=0.5):
        """
        For each GT object box, classify the detection failure reason.
        Error types:
          1. loc_miss:      no predicted box with IoU >= iou_thresh
          2. cls_wrong:     has box IoU >= iou_thresh but all wrong class
          3. low_score:     correct class & IoU but not matched in greedy 1-to-1 (top-N effect / suppressed by higher-score preds)
          4. success:       correctly matched in greedy 1-to-1
        Output: per-category table logged to logger.
        """
        thing_classes = list(getattr(self._metadata, "thing_classes", []))
        num_classes = len(thing_classes)
        total_count = np.zeros(num_classes, dtype=np.int64)
        loc_miss_count = np.zeros(num_classes, dtype=np.int64)
        cls_wrong_count = np.zeros(num_classes, dtype=np.int64)
        low_score_count = np.zeros(num_classes, dtype=np.int64)
        success_count = np.zeros(num_classes, dtype=np.int64)

        for gt, pred in zip(ground_truths, predictions):
            gt_boxes = gt["gt_boxes"].tensor.detach().cpu()
            gt_labels = gt["labels"].long().detach().cpu()

            pred_instances = pred.get("instances", None)
            if pred_instances is None or len(pred_instances) == 0:
                for cls in gt_labels.tolist():
                    if 0 <= cls < num_classes:
                        total_count[cls] += 1
                        loc_miss_count[cls] += 1
                continue

            pred_boxes = pred_instances.pred_boxes.tensor.detach().cpu()
            pred_labels = pred_instances.pred_classes.long().detach().cpu()
            pred_scores = pred_instances.scores.detach().cpu()

            # For greedy 1-to-1 matching: sort preds by score descending
            score_order = torch.argsort(pred_scores, descending=True)
            sorted_pred_boxes = pred_boxes[score_order]
            sorted_pred_labels = pred_labels[score_order]
            used_pred = set()

            for gt_i in range(len(gt_boxes)):
                gt_cls = int(gt_labels[gt_i])
                if gt_cls < 0 or gt_cls >= num_classes:
                    continue

                total_count[gt_cls] += 1
                gt_box = gt_boxes[gt_i:gt_i+1]  # (1, 4)

                # Step 1: find all pred boxes with IoU >= threshold
                ious = pairwise_iou(Boxes(gt_box), Boxes(pred_boxes))[0]  # (N_pred,)
                box_match = ious >= iou_thresh

                if not box_match.any():
                    loc_miss_count[gt_cls] += 1
                    continue

                # Step 2: among box-matched preds, check if any has correct class
                box_match_indices = torch.where(box_match)[0]
                cls_of_box_matched = pred_labels[box_match_indices]
                cls_correct = cls_of_box_matched == gt_cls

                if not cls_correct.any():
                    cls_wrong_count[gt_cls] += 1
                    continue

                # Step 3: correct class & box exists. Check if it is matched in greedy 1-to-1.
                # We do the same greedy matching as _compute_bbox_recall.
                matched = False
                vals, idxs = torch.sort(ious, descending=True)
                for v, p_idx in zip(vals.tolist(), idxs.tolist()):
                    if v < iou_thresh:
                        break
                    if p_idx not in used_pred:
                        used_pred.add(p_idx)
                        matched = True
                        break

                if matched:
                    success_count[gt_cls] += 1
                else:
                    low_score_count[gt_cls] += 1

        # Log the results
        self._logger.info("")
        self._logger.info("=" * 120)
        self._logger.info("Detection Error Analysis (IoU={})".format(iou_thresh))
        self._logger.info("=" * 120)
        self._logger.info("Error types:")
        self._logger.info("  1. loc_miss:     no predicted box with IoU >= {}".format(iou_thresh))
        self._logger.info("  2. cls_wrong:    box matched but class predicted wrong")
        self._logger.info("  3. low_score:    correct class & IoU but not in greedy 1-to-1 match (suppressed)")
        self._logger.info("  4. success:      correctly matched in greedy 1-to-1")

        headers = ["category", "total_gt", "loc_miss", "cls_wrong", "low_score", "success", "recall"]
        rows = []
        for cls_idx, cls_name in enumerate(thing_classes):
            t = int(total_count[cls_idx])
            if t == 0:
                continue
            lm = int(loc_miss_count[cls_idx])
            cw = int(cls_wrong_count[cls_idx])
            ls = int(low_score_count[cls_idx])
            sc = int(success_count[cls_idx])
            recall = float(sc) / float(t) * 100.0
            pct_lm = float(lm) / float(t) * 100.0
            pct_cw = float(cw) / float(t) * 100.0
            pct_ls = float(ls) / float(t) * 100.0
            rows.append((
                cls_name, t, lm,
                f"{lm} ({pct_lm:.1f}%)",
                f"{cw} ({pct_cw:.1f}%)",
                f"{ls} ({pct_ls:.1f}%)",
                f"{sc} ({recall:.1f}%)",
                f"{recall:.2f}%",
            ))

        # 按 loc_miss 数量降序排序
        rows.sort(key=lambda r: r[2], reverse=True)

        # 构建显示行（只保留显示用的字段，去掉排序用的原始 lm）
        display_rows = [r[:2] + r[3:] for r in rows]

        t_total = int(total_count.sum())
        lm_total = int(loc_miss_count.sum())
        cw_total = int(cls_wrong_count.sum())
        ls_total = int(low_score_count.sum())
        sc_total = int(success_count.sum())
        total_recall = float(sc_total) / float(t_total) * 100.0 if t_total > 0 else 0.0
        pct_lm_t = float(lm_total) / float(t_total) * 100.0 if t_total > 0 else 0.0
        pct_cw_t = float(cw_total) / float(t_total) * 100.0 if t_total > 0 else 0.0
        pct_ls_t = float(ls_total) / float(t_total) * 100.0 if t_total > 0 else 0.0
        display_rows.append([
            'TOTAL', t_total,
            f"{lm_total} ({pct_lm_t:.1f}%)",
            f"{cw_total} ({pct_cw_t:.1f}%)",
            f"{ls_total} ({pct_ls_t:.1f}%)",
            f"{sc_total} ({total_recall:.1f}%)",
            f"{total_recall:.2f}%",
        ])

        self._logger.info("\n" + tabulate(display_rows, headers=headers, tablefmt="pipe"))
        self._logger.info("=" * 120)

        result_dict = {"det_Recall@50": total_recall}
        for cls_idx, cls_name in enumerate(thing_classes):
            t = int(total_count[cls_idx])
            if t == 0:
                continue
            lm = int(loc_miss_count[cls_idx])
            cw = int(cls_wrong_count[cls_idx])
            ls = int(low_score_count[cls_idx])
            sc = int(success_count[cls_idx])
            result_dict[f"{cls_name}_loc_miss"] = int(loc_miss_count[cls_idx])
            result_dict[f"{cls_name}_cls_wrong"] = int(cls_wrong_count[cls_idx])
            result_dict[f"{cls_name}_low_score"] = int(low_score_count[cls_idx])
            result_dict[f"{cls_name}_success"] = int(success_count[cls_idx])
            result_dict[f"{cls_name}_total"] = int(total_count[cls_idx])
        return result_dict

    def _evaluate_scenegraphs(self, ground_truths, predictions):
        
        # result_detector = None
        
        self._logger.info("Computing Scene Graph Metrics")
        num_rel_category = self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        multiple_preds = self.cfg.TEST.RELATION.MULTIPLE_PREDS
        iou_thres = self.cfg.TEST.RELATION.IOU_THRESHOLD

        self._logger.info("Preparing Global Container")
        #Prepare Global container
        global_container = {}
        global_container['zeroshot_triplet'] = self._zero_shot_triplets
        global_container['result_dict'] = {}
        global_container['mode'] = self._mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        
        for i , (groundtruth, prediction) in tqdm(enumerate(zip(ground_truths, predictions)),desc='Computing recalls'):
            self.evaluate_relation_of_one_image(groundtruth, prediction, global_container,i)

        self._logger.info("Scene Graph Metric Evaluation Complete. Computing recall statistics...")
        # ('SGRecall', 'SGNoGraphConstraintRecall', 'SGZeroShotRecall', 'SGPairAccuracy', 'SGMeanRecall')
        if 'SGMeanRecall' in self._evaluators:
            # calculate mean recall
            self._evaluators['SGMeanRecall'].calculate_mean_recall(self._mode)

        result_str =''
        # print result
        if 'SGRecall' in self._evaluators:
            result_str += self._evaluators['SGRecall'].generate_print_string(self._mode)
        if 'SGNoGraphConstraintRecall' in self._evaluators:
            result_str += self._evaluators['SGNoGraphConstraintRecall'].generate_print_string(self._mode)
        if 'SGZeroShotRecall' in self._evaluators:
            result_str += self._evaluators['SGZeroShotRecall'].generate_print_string(self._mode)
        if 'SGMeanRecall' in self._evaluators:
            result_str += self._evaluators['SGMeanRecall'].generate_print_string(self._mode)
        
        if self.cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX and 'SGPairAccuracy' in self._evaluators:
            result_str += self._evaluators['SGPairAccuracy'].generate_print_string(self._mode)
        result_str += self._evaluators['recall_per_class'].generate_print_string(self._mode,self.cfg,len(ground_truths))
        if 'SGErrorAnalysis' in self._evaluators:
            result_str += self._evaluators['SGErrorAnalysis'].generate_print_string(self._mode)
        result_str += '=' * 100 + '\n'
        
        torch.save(self._evaluators['SGRecall'].result_dict, 'temp.pth')
        self._logger.info('Scene Graph Results for mode: {}'.format(self._mode))
        for line in result_str.rstrip("\n").split("\n"):
            if line.strip():
                self._logger.info(line)
        ret = OrderedDict()
        for k, v in self._evaluators['SGMeanRecall'].result_dict[self._mode + '_mean_recall'].items():
            ret['SGMeanRecall@{}'.format(k)] = float(v)
        for k, v in self._evaluators['SGRecall'].result_dict[self._mode + '_recall'].items():
            ret['SGRecall@{}'.format(k)] = np.mean(v)
        # Add No Graph Constraint Recall
        if 'SGNoGraphConstraintRecall' in self._evaluators:
            for k, v in self._evaluators['SGNoGraphConstraintRecall'].result_dict[self._mode + '_recall_nogc'].items():
                ret['ng-R@{}'.format(k)] = np.mean(v)
        # Add Zero-shot Recall
        if 'SGZeroShotRecall' in self._evaluators:
            for k, v in self._evaluators['SGZeroShotRecall'].result_dict[self._mode + '_zeroshot_recall'].items():
                ret['zR@{}'.format(k)] = np.mean(v)
        return ret

    def evaluate_relation_of_one_image(self, groundtruth, prediction, global_container, i):
        """
        Returns:
            pred_to_gt: Matching from predicate to GT
            pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
            pred_triplet_scores: [cls_0score, relscore, cls1_score]
        """
        #unpack all inputs
        mode = global_container['mode']

        local_container = {}
        local_container['gt_rels'] = groundtruth['relation_tuple'].long().detach().cpu().numpy()

        # if there is no gt relations for current image, then skip it
        if len(local_container['gt_rels']) == 0:
            return

        local_container['gt_boxes'] = groundtruth['gt_boxes'].tensor.detach().cpu().numpy()                   # (#gt_objs, 4)
        local_container['gt_classes'] = groundtruth['labels'].long().detach().cpu().numpy()           # (#gt_objs, )
        # import ipdb; ipdb.set_trace()
        # about relations
        local_container['pred_rel_inds'] = prediction['rel_pair_idxs'].long().detach().cpu().numpy()  # (#pred_rels, 2)
        local_container['rel_scores'] = prediction['pred_rel_scores'].detach().cpu().numpy()          # (#pred_rels, num_pred_class)
        local_container['rel_labels'] = prediction['pred_rel_labels'].detach().cpu().numpy()
        local_container['query_index']=prediction['query_index'].detach().cpu().numpy()
#         local_container['subject_scores']=prediction['subject_scores'].detach().cpu().numpy()
#         local_container['object_scores']=prediction['object_scores'].detach().cpu().numpy()
#         local_container['predicate_scores']=prediction['predicate_scores'].detach().cpu().numpy()
#         local_container['triplet_scores']=prediction['triplet_scores'].detach().cpu().numpy()

        # about objects
        local_container['pred_boxes'] = prediction['instances'].pred_boxes.tensor.detach().cpu().numpy()                  # (#pred_objs, 4)
        local_container['pred_classes'] = prediction['instances'].pred_classes.long().detach().cpu().numpy()     # (#pred_objs, )
        local_container['obj_scores'] = prediction['instances'].scores.detach().cpu().numpy()              # (#pred_objs, )
        # import pdb; pdb.set_trace()
        # to calculate accuracy, only consider those gt pairs
        # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
        # for sgcls and predcls
        if mode != 'sgdet' and 'SGPairAccuracy' in self._metrics:
            self._evaluators['SGPairAccuracy'].prepare_gtpair(local_container)

        # to calculate the prior label based on statistics
        if 'SGZeroShotRecall' in self._metrics:
            self._evaluators['SGZeroShotRecall'].prepare_zeroshot(global_container, local_container)

        if mode == 'predcls':
            local_container['pred_boxes'] = local_container['gt_boxes']
            local_container['pred_classes'] = local_container['gt_classes']
            local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

        elif mode == 'sgcls':
            if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
                print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
        elif mode == 'sgdet' or mode == 'phrdet':
            pass
        else:
            raise ValueError('invalid mode')

        if local_container['pred_rel_inds'].shape[0] == 0:
            return

        # Traditional Metric with Graph Constraint
        # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
        # ('SGRecall', 'SGNoGraphConstraintRecall', 'SGZeroShotRecall', 'SGPairAccuracy', 'SGMeanRecall')

        local_container = self._evaluators['SGRecall'].calculate_recall(global_container, local_container, mode)

        self._evaluators['query_count_per_rel_class'].calculate(global_container, local_container, mode,i)
        self._evaluators['rel_class_info_per_query'].calculate(global_container, local_container, mode)
        self._evaluators['recall_per_class'].calculate(self.cfg,global_container, local_container, mode,i)        
        if 'SGNoGraphConstraintRecall' in self._metrics:
            # No Graph Constraint
            self._evaluators['SGNoGraphConstraintRecall'].calculate_recall(global_container, local_container, mode)
        if 'SGPairAccuracy' in self._metrics:
            # GT Pair Accuracy
            self._evaluators['SGPairAccuracy'].calculate_recall(global_container, local_container, mode)
        if 'SGMeanRecall' in self._metrics:
            # Mean Recall
            self._evaluators['SGMeanRecall'].collect_mean_recall_items(global_container, local_container, mode)
        if 'SGZeroShotRecall' in self._metrics:
            # Zero shot Recall
            self._evaluators['SGZeroShotRecall'].calculate_recall(global_container, local_container, mode, i)
        if 'SGErrorAnalysis' in self._metrics:
            self._evaluators['SGErrorAnalysis'].calculate(self.cfg, global_container, local_container, mode, i)
        return 
        

class SceneGraphEvaluation(ABC):
    def __init__(self, result_dict):
        super().__init__()
        self.result_dict = result_dict
 
    @abstractmethod
    def register_container(self, mode, cfg):
        print("Register Result Container")
        pass
    
    @abstractmethod
    def generate_print_string(self, mode):
        print("Generate Print String")
        pass

@SCENEGRAPH_METRIC_REGISTRY.register()
class query_count_per_rel_class(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(query_count_per_rel_class, self).__init__(result_dict)
        

    def register_container(self, mode, cfg,total):
        query_num=cfg.MODEL.DETR.NUM_RELATION_QUERIES*cfg.MODEL.DETR.MULTIPLY_QUERY*cfg.TEST.NUM_REL
        num_rel_category=cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        self.result_dict['query_count_per_rel_class'] = {20: torch.zeros(num_rel_category,query_num), 50: torch.zeros(num_rel_category,query_num), 100: torch.zeros(num_rel_category,query_num)}
        self.result_dict['gt_match_query_per_image']={}
    def generate_print_string(self, mode):
        pass

    def calculate(self, global_container, local_container, mode,i):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        rel_labels = local_container['rel_labels']
        query_index = local_container['query_index']

        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, rel_labels)) #Backround index at the end
        pred_scores = rel_scores[:,:-1].max(1) #Backround index at the end

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt,index_list,recall_per_class_index = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )
        self.result_dict['gt_match_query_per_image'][i]=index_list
        for k in self.result_dict['query_count_per_rel_class']:
            for prediction_index in np.where(index_list<k)[0]:
                index=index_list[prediction_index]
                self.result_dict['query_count_per_rel_class'][k][rel_labels[index]][query_index[index]]+=1

@SCENEGRAPH_METRIC_REGISTRY.register()
class rel_class_info_per_query(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(rel_class_info_per_query, self).__init__(result_dict)
        
    def register_container(self, mode, cfg,total):
        query_num=cfg.MODEL.DETR.NUM_RELATION_QUERIES*cfg.MODEL.DETR.MULTIPLY_QUERY*cfg.TEST.NUM_REL
        num_rel_category=cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        self.result_dict['rel_class_count_per_query'] = {20: torch.zeros(query_num,num_rel_category), 50: torch.zeros(query_num,num_rel_category), 100: torch.zeros(query_num,num_rel_category)}
        self.result_dict['rel_class_logit_per_query'] = {20: torch.zeros(query_num,num_rel_category+1), 50: torch.zeros(query_num,num_rel_category+1), 100: torch.zeros(query_num,num_rel_category+1)}

    def generate_print_string(self, mode):
        pass

    def calculate(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        rel_labels = local_container['rel_labels']
        query_index = local_container['query_index']

        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']
    
        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, rel_labels)) #Backround index at the end
        pred_scores = rel_scores[:,:-1].max(1) #Backround index at the end

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt,index_list,recall_per_class_index = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )

        for k in self.result_dict['rel_class_count_per_query']:
            for prediction_index in np.where(index_list<k)[0]:
                index=index_list[prediction_index]
                self.result_dict['rel_class_count_per_query'][k][query_index[index]][rel_labels[index]]+=1

        for k in self.result_dict['rel_class_logit_per_query']:
            for prediction_index in np.where(index_list<k)[0]:
                index=index_list[prediction_index]
                self.result_dict['rel_class_logit_per_query'][k][query_index[index]].add_(torch.tensor(rel_scores[index]))

@SCENEGRAPH_METRIC_REGISTRY.register()
class recall_per_class(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(recall_per_class, self).__init__(result_dict)
        

    def register_container(self, mode,cfg,total):
        num_rel_category=cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        self.result_dict['count_per_class'] = torch.zeros(num_rel_category,total)
        self.result_dict['hit_per_class'] = {20: torch.zeros(num_rel_category,total), 50: torch.zeros(num_rel_category,total), 100: torch.zeros(num_rel_category,total)}
        self.result_dict['gt_pair']={i:{} for i in range(total)}
        self.result_dict['gt_pair_count']={i:0 for i in range(total)}
        self.result_dict['pred_pair']={k:{i:{} for i in range(total)} for k in [20,50,100]}
        self.result_dict['pred_pair_count']={k:{i:0 for i in range(total)} for k in [20,50,100]}
    
    def generate_print_string(self, mode,cfg,total):
        num_rel_category=cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        result_str = 'recall_per_class\n'
        for k in self.result_dict['hit_per_class']:
            result_str+= 'R @ %d\n' % (k)
            for predicate_num in range(num_rel_category):
                per_class_recall=[]
                for j in range(total):
                    count=self.result_dict['count_per_class'][predicate_num][j]
                    hit=self.result_dict['hit_per_class'][k][predicate_num][j]
                    if count>0:
                        per_class_recall.append(hit/count)
                result_str += '%15s: %6.3f; ' % (idx2predicate[predicate_num], np.mean(per_class_recall))
            result_str += '\n'
        return result_str

    def calculate(self, cfg, global_container, local_container, mode,i):
        num_rel_category=cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES

        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        rel_labels = local_container['rel_labels']
        query_index = local_container['query_index']

        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, rel_labels)) #Backround index at the end
        pred_scores = rel_scores[:,:-1].max(1) #Backround index at the end

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt,index_list,recall_per_class_index = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )

        for predicate in gt_rels[:,2]:
            self.result_dict['count_per_class'][predicate][i]+=1
        
        for gt_rel in gt_rels:
            subj,obj,predicate=gt_rel[0],gt_rel[1],gt_rel[2]
            if f'{subj}_{obj}' not in self.result_dict['gt_pair'][i]:
                self.result_dict['gt_pair'][i][f'{subj}_{obj}']=torch.zeros(num_rel_category)
            self.result_dict['gt_pair'][i][f'{subj}_{obj}'][predicate]+=1
        self.result_dict['gt_pair_count'][i]=len(gt_rels)

        for k in self.result_dict['hit_per_class']:
            # for prediction_index in np.where(recall_per_class_index<k)[0]:
            #     index=recall_per_class_index[prediction_index]
            #     self.result_dict['hit_per_class'][k][rel_labels[index]][i]+=1
            match=reduce(np.union1d,pred_to_gt[:k])
            match_count=reduce(lambda a,b:a+b,pred_to_gt[:k])
            for index in range(len(match)):
                match_subj,match_obj,match_predicate=gt_rels[int(match[index]),0],gt_rels[int(match[index]),1],gt_rels[int(match[index]),2]
                self.result_dict['hit_per_class'][k][match_predicate][i]+=1
                if f'{match_subj}_{match_obj}' not in self.result_dict['pred_pair'][k][i]:
                    self.result_dict['pred_pair'][k][i][f'{match_subj}_{match_obj}']=torch.zeros(num_rel_category)
                self.result_dict['pred_pair'][k][i][f'{match_subj}_{match_obj}'][match_predicate]+=match_count.count(match[index])
            self.result_dict['pred_pair_count'][k][i]=len(match_count)


"""
Traditional Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGRecall(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(SGRecall, self).__init__(result_dict)
        

    def register_container(self, mode, cfg,total):
        self.result_dict[mode + '_recall'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall'].items():
            result_str += '    R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        rel_labels = local_container['rel_labels']

        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        gt_boxes = local_container['gt_boxes']
        pred_classes = local_container['pred_classes']
        pred_boxes = local_container['pred_boxes']
        obj_scores = local_container['obj_scores']

        iou_thres = global_container['iou_thres']

        pred_rels = np.column_stack((pred_rel_inds, rel_labels)) #Backround index at the end
        pred_scores = rel_scores[:,:-1].max(1) #Backround index at the end

        gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_classes, gt_boxes)
        local_container['gt_triplets'] = gt_triplets
        local_container['gt_triplet_boxes'] = gt_triplet_boxes
        
        pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
                pred_rels, pred_classes, pred_boxes, pred_scores, obj_scores)

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt,index_list,recall_per_class_index = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )
        local_container['pred_to_gt'] = pred_to_gt

        for k in self.result_dict[mode + '_recall']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall'][k].append(rec_i) 
        return local_container

"""
No Graph Constraint Recall, implement based on:
https://github.com/rowanz/neural-motifs
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGNoGraphConstraintRecall(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(SGNoGraphConstraintRecall, self).__init__(result_dict)

    def register_container(self, mode, cfg,total):
        self.result_dict[mode + '_recall_nogc'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_recall_nogc'].items():
            result_str += ' ng-R @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=No Graph Constraint Recall(Main).' % mode
        result_str += '\n'
        return result_str

    def calculate_recall(self, global_container, local_container, mode):
        obj_scores = local_container['obj_scores']
        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        gt_rels = local_container['gt_rels']

        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:,None] * rel_scores[:,:-1] #Backround index at the end
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        nogc_pred_rels = np.column_stack((pred_rel_inds[nogc_score_inds[:,0]], nogc_score_inds[:,1]))#Backround index at the end(removed +1)
        nogc_pred_scores = rel_scores[nogc_score_inds[:,0], nogc_score_inds[:,1]]#Backround index at the end(removed +1)

        nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(
                nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores
        )

        # No Graph Constraint
        gt_triplets = local_container['gt_triplets']
        gt_triplet_boxes = local_container['gt_triplet_boxes']
        iou_thres = global_container['iou_thres']

        nogc_pred_to_gt,index_list,recall_per_class_index = _compute_pred_matches(
            gt_triplets,
            nogc_pred_triplets,
            gt_triplet_boxes,
            nogc_pred_triplet_boxes,
            iou_thres,
            phrdet=mode=='phrdet',
        )

        local_container['nogc_pred_to_gt'] = nogc_pred_to_gt

        for k in self.result_dict[mode + '_recall_nogc']:
            match = reduce(np.union1d, nogc_pred_to_gt[:k])
            rec_i = float(len(match)) / float(gt_rels.shape[0])
            self.result_dict[mode + '_recall_nogc'][k].append(rec_i)

        return local_container

"""
Zero Shot Scene Graph
Only calculate the triplet that not occurred in the training set
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGZeroShotRecall(SceneGraphEvaluation):
    def __init__(self, cfg,  result_dict, dataset_name):
        super(SGZeroShotRecall, self).__init__(result_dict)

    def register_container(self, mode, cfg,total):
        self.result_dict[mode + '_zeroshot_recall'] = {20: [], 50: [], 100: []} 
        self.result_dict[mode + '_zs_id'] = {20: [], 50: [], 100: []} 

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_zeroshot_recall'].items():
            result_str += '   zR @ %d: %.4f; ' % (k, np.mean(v))
        result_str += ' for mode=%s, type=Zero Shot Recall.' % mode
        result_str += '\n'
        return result_str

    def prepare_zeroshot(self, global_container, local_container):
        gt_rels = local_container['gt_rels']
        gt_classes = local_container['gt_classes']
        zeroshot_triplets = global_container['zeroshot_triplet']

        sub_id, ob_id, pred_label = gt_rels[:, 0], gt_rels[:, 1], gt_rels[:, 2]
        gt_triplets = np.column_stack((gt_classes[sub_id], gt_classes[ob_id], pred_label))  # num_rel, 3

        self.zeroshot_idx = np.where( intersect_2d(gt_triplets, zeroshot_triplets).sum(-1) > 0 )[0].tolist()

    def calculate_recall(self, global_container, local_container, mode, i):
        pred_to_gt = local_container['pred_to_gt']

        for k in self.result_dict[mode + '_zeroshot_recall']:
            # Zero Shot Recall
            match = reduce(np.union1d, pred_to_gt[:k])
            if len(self.zeroshot_idx) > 0:
                if not isinstance(match, (list, tuple)):
                    match_list = match.tolist()
                else:
                    match_list = match
                zeroshot_match = len(self.zeroshot_idx) + len(match_list) - len(set(self.zeroshot_idx + match_list))
                zero_rec_i = float(zeroshot_match) / float(len(self.zeroshot_idx))
                self.result_dict[mode + '_zeroshot_recall'][k].append(zero_rec_i)
                self.result_dict[mode + '_zs_id'][k].append(i)


"""
Give Ground Truth Object-Subject Pairs
Calculate Recall for SG-Cls and Pred-Cls
Only used in https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGPairAccuracy(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(SGPairAccuracy, self).__init__(result_dict)

    def register_container(self, mode, cfg,total):
        self.result_dict[mode + '_accuracy_hit'] = {20: [], 50: [], 100: []}
        self.result_dict[mode + '_accuracy_count'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accuracy_hit'].items():
            a_hit = np.mean(v)
            a_count = np.mean(self.result_dict[mode + '_accuracy_count'][k])
            result_str += '    A @ %d: %.4f; ' % (k, a_hit/a_count)
        result_str += ' for mode=%s, type=TopK Accuracy.' % mode
        result_str += '\n'
        return result_str

    def prepare_gtpair(self, local_container):
        pred_pair_idx = local_container['pred_rel_inds'][:, 0] * 1024 + local_container['pred_rel_inds'][:, 1]
        gt_pair_idx = local_container['gt_rels'][:, 0] * 1024 + local_container['gt_rels'][:, 1]
        self.pred_pair_in_gt = (pred_pair_idx[:, None] == gt_pair_idx[None, :]).sum(-1) > 0

    def calculate_recall(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_accuracy_hit']:
            # to calculate accuracy, only consider those gt pairs
            # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
            # for sgcls and predcls
            if mode != 'sgdet':
                gt_pair_pred_to_gt = []
                for p, flag in zip(pred_to_gt, self.pred_pair_in_gt):
                    if flag:
                        gt_pair_pred_to_gt.append(p)
                if len(gt_pair_pred_to_gt) > 0:
                    gt_pair_match = reduce(np.union1d, gt_pair_pred_to_gt[:k])
                else:
                    gt_pair_match = []
                self.result_dict[mode + '_accuracy_hit'][k].append(float(len(gt_pair_match)))
                self.result_dict[mode + '_accuracy_count'][k].append(float(gt_rels.shape[0]))


"""
Mean Recall: Proposed in:
https://arxiv.org/pdf/1812.01880.pdf CVPR, 2019
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGMeanRecall(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name, print_detail=True):
        super(SGMeanRecall, self).__init__(result_dict)
        self.num_rel = cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        self.print_detail = print_detail
        self.rel_name_list = MetadataCatalog.get(dataset_name).predicate_classes # remove __background__

    def register_container(self, mode, cfg,total):
        #self.result_dict[mode + '_recall_hit'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        #self.result_dict[mode + '_recall_count'] = {20: [0]*self.num_rel, 50: [0]*self.num_rel, 100: [0]*self.num_rel}
        self.result_dict[mode + '_mean_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}
        self.result_dict[mode + '_mean_recall_collect'] = {20: [[] for i in range(self.num_rel)], 50: [[] for i in range(self.num_rel)], 100: [[] for i in range(self.num_rel)]}
        self.result_dict[mode + '_mean_recall_list'] = {20: [], 50: [], 100: []}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            result_str += '   mR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Mean Recall.' % mode
        result_str += '\n'
        if self.print_detail:
            # import ipdb; ipdb.set_trace()
            result_str += '----------------------- Details ------------------------\n'
            for n, r in zip(self.rel_name_list, self.result_dict[mode + '_mean_recall_list'][100]):
                result_str += '({}:{:.4f}) '.format(str(n), r)
            result_str += '\n'
            result_str += '--------------------------------------------------------\n'

        return result_str

    def collect_mean_recall_items(self, global_container, local_container, mode):
        pred_to_gt = local_container['pred_to_gt']
        gt_rels = local_container['gt_rels']

        for k in self.result_dict[mode + '_mean_recall_collect']:
            # the following code are copied from Neural-MOTIFS
            match = reduce(np.union1d, pred_to_gt[:k])
            # NOTE: by kaihua, calculate Mean Recall for each category independently
            # this metric is proposed by: CVPR 2019 oral paper "Learning to Compose Dynamic Tree Structures for Visual Contexts"
            recall_hit = [0] * self.num_rel
            recall_count = [0] * self.num_rel
            for idx in range(gt_rels.shape[0]):
                local_label = gt_rels[idx,2]
                recall_count[int(local_label)] += 1
                # if self.num_rel != 50:
                # recall_count[0] += 1

            for idx in range(len(match)):
                local_label = gt_rels[int(match[idx]),2]
                recall_hit[int(local_label)] += 1
                # if self.num_rel != 50:
                # recall_hit[0] += 1
            
            for n in range(self.num_rel):
                if recall_count[n] > 0:
                    self.result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
 

    def calculate_mean_recall(self, mode):
        for k, v in self.result_dict[mode + '_mean_recall'].items():
            sum_recall = 0
            num_rel_no_bg = self.num_rel
            for idx in range(num_rel_no_bg):
                if len(self.result_dict[mode + '_mean_recall_collect'][k][idx]) == 0:
                    tmp_recall = 0.0
                else:
                    tmp_recall = np.mean(self.result_dict[mode + '_mean_recall_collect'][k][idx])
                self.result_dict[mode + '_mean_recall_list'][k].append(tmp_recall)
                sum_recall += tmp_recall

            self.result_dict[mode + '_mean_recall'][k] = sum_recall / float(num_rel_no_bg)
        return



"""
Accumulate Recall:
calculate recall on the whole dataset instead of each image
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGAccumulateRecall(SceneGraphEvaluation):
    def __init__(self, result_dict):
        super(SGAccumulateRecall, self).__init__(result_dict)

    def register_container(self, mode, cfg,total):
        self.result_dict[mode + '_accumulate_recall'] = {20: 0.0, 50: 0.0, 100: 0.0}

    def generate_print_string(self, mode):
        result_str = 'SGG eval: '
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            result_str += '   aR @ %d: %.4f; ' % (k, float(v))
        result_str += ' for mode=%s, type=Accumulate Recall.' % mode
        result_str += '\n'
        return result_str

    def calculate_accumulate(self, mode):
        for k, v in self.result_dict[mode + '_accumulate_recall'].items():
            self.result_dict[mode + '_accumulate_recall'][k] = float(self.result_dict[mode + '_recall_hit'][k][0]) / float(self.result_dict[mode + '_recall_count'][k][0] + 1e-10)

        return 


"""
Error Analysis: categorize each GT triplet into one of four error types.
"""
@SCENEGRAPH_METRIC_REGISTRY.register()
class SGErrorAnalysis(SceneGraphEvaluation):
    def __init__(self, cfg, result_dict, dataset_name):
        super(SGErrorAnalysis, self).__init__(result_dict)
        self.cfg = cfg
        self.num_rel = cfg.MODEL.ROI_SCENEGRAPH_HEAD.NUM_CLASSES
        self.iou_thresh = cfg.TEST.RELATION.IOU_THRESHOLD
        self.rel_name_list = MetadataCatalog.get(dataset_name).predicate_classes
        self.obj_name_list = MetadataCatalog.get(dataset_name).thing_classes

    def register_container(self, mode, cfg, total):
        self.result_dict['error_analysis_total'] = np.zeros(self.num_rel, dtype=np.int64)
        self.result_dict['error_analysis_loc_miss'] = np.zeros(self.num_rel, dtype=np.int64)
        self.result_dict['error_analysis_cls_wrong'] = np.zeros(self.num_rel, dtype=np.int64)
        self.result_dict['error_analysis_triplet_fail'] = np.zeros(self.num_rel, dtype=np.int64)
        self.result_dict['error_analysis_low_score'] = np.zeros(self.num_rel, dtype=np.int64)
        self.result_dict['error_analysis_success'] = np.zeros(self.num_rel, dtype=np.int64)

    def generate_print_string(self, mode):
        total = self.result_dict['error_analysis_total']
        loc_miss = self.result_dict['error_analysis_loc_miss']
        cls_wrong = self.result_dict['error_analysis_cls_wrong']
        triplet_fail = self.result_dict['error_analysis_triplet_fail']
        low_score = self.result_dict['error_analysis_low_score']
        success = self.result_dict['error_analysis_success']

        result_str = '\n' + '=' * 120 + '\n'
        result_str += 'Error Analysis of GT triplets (by predicate category)\n'
        result_str += '=' * 120 + '\n'
        result_str += 'Error types:\n'
        result_str += '  1. loc_miss:      no predicted sub/obj box pair with IoU >= {}\n'.format(self.iou_thresh)
        result_str += '  2. cls_wrong:     box matched but sub/obj class predicted wrong\n'
        result_str += '  3. triplet_fail:  sub/obj correct but predicate wrong\n'
        result_str += '  4. low_score:     full triplet correct but score rank below top-100\n'
        result_str += '  5. success:       correctly recalled in top-100\n'
        result_str += '-' * 120 + '\n'

        headers = ['predicate', 'total_gt', 'loc_miss', 'cls_wrong', 'triplet_fail', 'low_score', 'success', 'recall@100']
        rows = []
        for rel_idx in range(self.num_rel):
            t = int(total[rel_idx])
            if t == 0:
                continue
            lm = int(loc_miss[rel_idx])
            cw = int(cls_wrong[rel_idx])
            tf = int(triplet_fail[rel_idx])
            ls = int(low_score[rel_idx])
            sc = int(success[rel_idx])
            recall = float(sc) / float(t) * 100.0
            pct_lm = float(lm) / float(t) * 100.0
            pct_cw = float(cw) / float(t) * 100.0
            pct_tf = float(tf) / float(t) * 100.0
            pct_ls = float(ls) / float(t) * 100.0
            name = self.rel_name_list[rel_idx] if rel_idx < len(self.rel_name_list) else f"rel_{rel_idx}"
            rows.append((
                name, t, lm,
                f"{lm} ({pct_lm:.1f}%)",
                f"{cw} ({pct_cw:.1f}%)",
                f"{tf} ({pct_tf:.1f}%)",
                f"{ls} ({pct_ls:.1f}%)",
                f"{sc} ({recall:.1f}%)",
                f"{recall:.2f}%",
            ))

        # 按 loc_miss 数量降序排序
        rows.sort(key=lambda r: r[2], reverse=True)
        display_rows = [r[:2] + r[3:] for r in rows]

        t_total = int(total.sum())
        lm_total = int(loc_miss.sum())
        cw_total = int(cls_wrong.sum())
        tf_total = int(triplet_fail.sum())
        ls_total = int(low_score.sum())
        sc_total = int(success.sum())
        total_recall = float(sc_total) / float(t_total) * 100.0 if t_total > 0 else 0.0
        pct_lm_t = float(lm_total) / float(t_total) * 100.0 if t_total > 0 else 0.0
        pct_cw_t = float(cw_total) / float(t_total) * 100.0 if t_total > 0 else 0.0
        pct_tf_t = float(tf_total) / float(t_total) * 100.0 if t_total > 0 else 0.0
        pct_ls_t = float(ls_total) / float(t_total) * 100.0 if t_total > 0 else 0.0
        display_rows.append([
            'TOTAL', t_total,
            f"{lm_total} ({pct_lm_t:.1f}%)",
            f"{cw_total} ({pct_cw_t:.1f}%)",
            f"{tf_total} ({pct_tf_t:.1f}%)",
            f"{ls_total} ({pct_ls_t:.1f}%)",
            f"{sc_total} ({total_recall:.1f}%)",
            f"{total_recall:.2f}%",
        ])

        result_str += tabulate(display_rows, headers=headers, tablefmt="pipe")
        result_str += '\n' + '=' * 120 + '\n'
        return result_str

    def calculate(self, cfg, global_container, local_container, mode, i):
        """Analyze each GT triplet and classify the error type."""
        gt_rels = local_container['gt_rels']
        gt_boxes = local_container['gt_boxes']
        gt_classes = local_container['gt_classes']

        pred_rel_inds = local_container['pred_rel_inds']
        rel_scores = local_container['rel_scores']
        rel_labels = local_container['rel_labels']
        pred_boxes = local_container['pred_boxes']
        pred_classes = local_container['pred_classes']
        obj_scores = local_container['obj_scores']

        if len(gt_rels) == 0:
            return

        pred_sub_ids = pred_rel_inds[:, 0].astype(np.int64)
        pred_obj_ids = pred_rel_inds[:, 1].astype(np.int64)

        # Sort predictions by combined score descending
        pred_score_all = rel_scores[:, :-1].max(1)
        pred_combined_scores = obj_scores[pred_sub_ids] * pred_score_all * obj_scores[pred_obj_ids]
        sort_order = np.argsort(-pred_combined_scores)

        iou_thresh = self.iou_thresh

        for gt_idx in range(len(gt_rels)):
            gt_sub_id, gt_obj_id, gt_pred_label = gt_rels[gt_idx]
            gt_sub_id, gt_obj_id = int(gt_sub_id), int(gt_obj_id)
            gt_pred_label = int(gt_pred_label)
            gt_sub_box = gt_boxes[gt_sub_id:gt_sub_id+1]
            gt_obj_box = gt_boxes[gt_obj_id:gt_obj_id+1]
            gt_sub_class = int(gt_classes[gt_sub_id])
            gt_obj_class = int(gt_classes[gt_obj_id])

            # Box IoU matching for all predictions
            pred_sub_boxes_all = pred_boxes[pred_sub_ids]
            pred_obj_boxes_all = pred_boxes[pred_obj_ids]

            sub_iou_all = pairwise_iou(
                Boxes(torch.from_numpy(gt_sub_box).float()),
                Boxes(torch.from_numpy(pred_sub_boxes_all).float()),
            )[0].numpy()
            obj_iou_all = pairwise_iou(
                Boxes(torch.from_numpy(gt_obj_box).float()),
                Boxes(torch.from_numpy(pred_obj_boxes_all).float()),
            )[0].numpy()
            box_match_all = (sub_iou_all >= iou_thresh) & (obj_iou_all >= iou_thresh)

            if not box_match_all.any():
                self.result_dict['error_analysis_loc_miss'][gt_pred_label] += 1
                self.result_dict['error_analysis_total'][gt_pred_label] += 1
                continue

            box_match_indices = np.where(box_match_all)[0]
            matched_pred_sub_classes = pred_classes[pred_sub_ids[box_match_indices]]
            matched_pred_obj_classes = pred_classes[pred_obj_ids[box_match_indices]]
            matched_pred_rel_labels = rel_labels[box_match_indices]

            sub_obj_class_ok = (matched_pred_sub_classes == gt_sub_class) & \
                               (matched_pred_obj_classes == gt_obj_class)

            if not sub_obj_class_ok.any():
                self.result_dict['error_analysis_cls_wrong'][gt_pred_label] += 1
                self.result_dict['error_analysis_total'][gt_pred_label] += 1
                continue

            # Among those with correct sub/obj class, check predicate match
            triplet_ok_mask = sub_obj_class_ok
            full_match_global_idx = box_match_indices[triplet_ok_mask]
            full_match_pred_labels = matched_pred_rel_labels[triplet_ok_mask]
            pred_match = full_match_pred_labels == gt_pred_label

            if not pred_match.any():
                self.result_dict['error_analysis_triplet_fail'][gt_pred_label] += 1
                self.result_dict['error_analysis_total'][gt_pred_label] += 1
                continue

            # Full match exists: check rank
            full_match_global_set = set(full_match_global_idx[pred_match].tolist())
            rank = None
            for rk, orig_idx in enumerate(sort_order):
                if orig_idx in full_match_global_set:
                    rank = rk
                    break

            if rank is None:
                self.result_dict['error_analysis_triplet_fail'][gt_pred_label] += 1
            elif rank >= 100:
                self.result_dict['error_analysis_low_score'][gt_pred_label] += 1
            else:
                self.result_dict['error_analysis_success'][gt_pred_label] += 1
            self.result_dict['error_analysis_total'][gt_pred_label] += 1


def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns: 
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id], predicate_scores, class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores

def resize_instance(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """

    # Converts integer tensors to float temporaries
    #   to ensure true division is performed when
    #   computing scale_x and scale_y.
    if isinstance(output_width, torch.Tensor):
        output_width_tmp = output_width.float()
    else:
        output_width_tmp = output_width

    if isinstance(output_height, torch.Tensor):
        output_height_tmp = output_height.float()
    else:
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("gt_boxes"):
        output_boxes = results.gt_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        results.pred_masks = retry_if_cuda_oom(paste_masks_in_image)(
            results.pred_masks[:, 0, :, :],  # N, 1, M, M
            results.pred_boxes,
            results.image_size,
            threshold=mask_threshold,
        )

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thres, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    index_list=[]
    recall_per_class_index=[]
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = pairwise_iou(gt_box_union[None], box_union)[0] >= iou_thres

        else:
            #FIXME Check for indexing
            sub_iou = pairwise_iou(Boxes(gt_box[None,:4]), Boxes(boxes[:, :4]))[0]
            obj_iou = pairwise_iou(Boxes(gt_box[None,4:]), Boxes(boxes[:, 4:]))[0]

            inds = ((sub_iou >= iou_thres) & (obj_iou >= iou_thres)).numpy()
        for order,i in enumerate(np.where(keep_inds)[0][inds]):
            if order==0:
                recall_per_class_index.append(i)
            pred_to_gt[i].append(int(gt_ind))
            index_list.append(i)
    index_list=np.array(index_list)
    recall_per_class_index=np.array(recall_per_class_index)
    return pred_to_gt,index_list,recall_per_class_index


def build_scenegraph_evaluators(metrics, cfg, result_dict, dataset_name):
    
    evaluators = {}
    for name in metrics:
        evaluators[name] = SCENEGRAPH_METRIC_REGISTRY.get(name)(cfg, result_dict, dataset_name)

    return evaluators
