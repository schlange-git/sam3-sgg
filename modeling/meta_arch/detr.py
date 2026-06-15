import gc
import logging
import math
import os
from multiprocessing import Condition
from turtle import back
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks, pairwise_iou
from detectron2.utils.logger import log_first_n
from detectron2.data import MetadataCatalog
from fvcore.nn import giou_loss, smooth_l1_loss
from ..transformer import build_detr, build_criterion, build_transformer, build_matcher, build_position_encoding
from ..transformer.segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm
from ..transformer.util.utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, NestedTensor, convert_coco_poly_to_mask
from ..transformer.util import box_ops
from ..backbone import MaskedBackbone, Joiner, Backbone, DeformableDETRMaskedBackbone, DeformableDETRJoiner, Sam3MaskedBackbone
from detectron2.layers import batched_nms

__all__ = ["Detr"]


def validate_obj_split_config(cfg, num_classes: int):
    obj_cfg = cfg.MODEL.OBJ_SPLIT
    if not bool(getattr(obj_cfg, "ENABLED", False)):
        return

    regular = [int(x) for x in list(getattr(obj_cfg, "REGULAR_CLASSES", []))]
    small = [int(x) for x in list(getattr(obj_cfg, "SMALL_SHARED_CLASSES", []))]
    fine_groups = [[int(x) for x in list(g)] for g in list(getattr(obj_cfg, "FINE_GROUPS", []))]
    fine_names = [str(x) for x in list(getattr(obj_cfg, "FINE_GROUP_NAMES", []))]
    fine_use_upsample = [bool(x) for x in list(getattr(obj_cfg, "FINE_GROUPS_USE_UPSAMPLE", []))]

    if len(fine_names) not in (0, len(fine_groups)):
        raise ValueError(
            f"MODEL.OBJ_SPLIT.FINE_GROUP_NAMES length {len(fine_names)} must be 0 or equal to FINE_GROUPS length {len(fine_groups)}"
        )
    if len(fine_use_upsample) not in (0, len(fine_groups)):
        raise ValueError(
            f"MODEL.OBJ_SPLIT.FINE_GROUPS_USE_UPSAMPLE length {len(fine_use_upsample)} must be 0 or equal to FINE_GROUPS length {len(fine_groups)}"
        )

    all_sets = [("REGULAR_CLASSES", regular), ("SMALL_SHARED_CLASSES", small)]
    all_sets.extend([(f"FINE_GROUPS[{i}]", g) for i, g in enumerate(fine_groups)])

    owner = {}
    for set_name, ids in all_sets:
        for cls_id in ids:
            if cls_id < 0 or cls_id >= int(num_classes):
                raise ValueError(f"{set_name} contains invalid class id {cls_id}, valid range is [0, {num_classes - 1}]")
            if cls_id in owner:
                raise ValueError(f"class id {cls_id} appears in both {owner[cls_id]} and {set_name}; split sets must be disjoint")
            owner[cls_id] = set_name

    missing = [c for c in range(int(num_classes)) if c not in owner]
    if missing:
        raise ValueError(
            f"OBJ_SPLIT requires full foreground coverage. Missing class ids: {missing[:20]}{' ...' if len(missing) > 20 else ''}"
        )


@META_ARCH_REGISTRY.register()
class Detr(nn.Module):
    """
    Implement Detr
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.num_relation_classes = cfg.MODEL.DETR.NUM_RELATION_CLASSES
        validate_obj_split_config(cfg, self.num_classes)
        self.mask_on = cfg.MODEL.MASK_ON
        self.use_gt_box = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_BOX
        self.use_gt_label = cfg.MODEL.ROI_SCENEGRAPH_HEAD.USE_GT_OBJECT_LABEL
        self.later_nms_thres = cfg.MODEL.DETR.LATER_NMS_THRESHOLD
        self.use_freq_bias = cfg.MODEL.DETR.USE_FREQ_BIAS
        self.test_index = cfg.MODEL.DETR.TEST_INDEX
        self.M = cfg.TEST.NUM_REL
        self.pnms = cfg.TEST.PNMS
        
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES if cfg.MODEL.DETR.ONLY_PREDICATE_MULTIPLY else cfg.MODEL.DETR.NUM_OBJECT_QUERIES *cfg.MODEL.DETR.MULTIPLY_QUERY
        num_relation_queries = cfg.MODEL.DETR.NUM_RELATION_QUERIES * cfg.MODEL.DETR.MULTIPLY_QUERY
        create_bg_pairs = cfg.MODEL.DETR.CREATE_BG_PAIRS
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        obj_dec_layers = cfg.MODEL.DETR.OBJECT_DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT
        no_rel_weight = cfg.MODEL.DETR.NO_REL_WEIGHT
        cost_class = cfg.MODEL.DETR.COST_CLASS
        nms_weight = cfg.MODEL.DETR.NMS_WEIGHT
        cost_selection = cfg.MODEL.DETR.COST_SELECTION
        beta = cfg.MODEL.DETR.BETA
        matcher_topk = cfg.MODEL.DETR.MATCHER_TOPK
        self.nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        d2_backbone = Sam3MaskedBackbone(cfg) if cfg.MODEL.SAM3.ENABLED else MaskedBackbone(cfg)
        position_embedding = build_position_encoding(cfg.MODEL.DETR.POSITION_EMBEDDING, hidden_dim)
        backbone = Joiner(d2_backbone, position_embedding)
        backbone.num_channels = d2_backbone.num_channels

        self.one_to_many = cfg.MODEL.DETR.ONE_TO_MANY

        transformer = build_transformer(cfg.MODEL.DETR.TRANSFORMER, d_model=hidden_dim, dropout=dropout, nhead=nheads, dim_feedforward=dim_feedforward, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, normalize_before=pre_norm, return_intermediate_dec=deep_supervision, cfg=cfg, num_object_decoder_layers=obj_dec_layers, num_classes=self.num_classes, num_relation_classes=self.num_relation_classes, beta=beta)

        self.detr = build_detr(cfg.MODEL.DETR.NAME, backbone, transformer, num_classes=self.num_classes, num_queries=num_queries,num_relation_queries=num_relation_queries, aux_loss=deep_supervision, use_gt_box=self.use_gt_box, use_gt_label = self.use_gt_label,cfg=cfg)
        if self.mask_on:
            frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS
            if frozen_weights != '':
                print("LOAD pre-trained weights")
                weight = torch.load(frozen_weights, map_location=lambda storage, loc: storage)['model']
                new_weight = {}
                for k, v in weight.items():
                    if 'detr.' in k:
                        new_weight[k.replace('detr.', '')] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                gc.collect()
                self.detr.load_state_dict(new_weight)
                del new_weight
                gc.collect()
            self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ''))
            self.seg_postprocess = PostProcessSegm
        if cfg.MODEL.DETR.FROZEN_WEIGHTS != '':
            self.load_detr_from_pretrained(cfg.MODEL.DETR.FROZEN_WEIGHTS)
        self.detr.to(self.device)

        statistics = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).statistics
        # building criterion
        matcher = build_matcher(cfg.MODEL.DETR.MATCHER, cost_class=cost_class, cost_bbox=l1_weight, cost_giou=giou_weight, topk=matcher_topk, cfg=cfg, statistics=statistics)
        weight_dict = {"loss_ce": cost_class, "loss_bbox": l1_weight, 'loss_ce_subject': cost_class, 'loss_ce_object': cost_class, 'loss_bbox_subject': l1_weight, 'loss_bbox_object': l1_weight, 'loss_giou_subject': giou_weight, 'loss_giou_object': giou_weight, 'loss_relation': 1, 'loss_bbox_relation': l1_weight, 'loss_giou_relation':giou_weight, 'loss_nms':nms_weight, 'loss_selection_subject': cost_selection, 'loss_selection_object': cost_selection}
        weight_dict["loss_giou"] = giou_weight
        if cfg.MODEL.OBJ_SPLIT.ENABLED:
            weight_dict["loss_reg_aux"] = float(cfg.MODEL.OBJ_SPLIT.AUX_LOSS_WEIGHT_REG)
            weight_dict["loss_small_aux"] = float(cfg.MODEL.OBJ_SPLIT.AUX_LOSS_WEIGHT_SMALL)
            weight_dict["loss_fine_aux"] = float(cfg.MODEL.OBJ_SPLIT.AUX_LOSS_WEIGHT_FINE)
        if cfg.MODEL.ROI_REFINE.ENABLED and cfg.MODEL.ROI_REFINE.LOSS_ENABLED:
            # REPLACE_BEFORE_MATCHER 开启时 roi 分类作为主分类参与全程, roi loss 权重 x5
            _roi_loss_scale = 5.0 if cfg.MODEL.ROI_REFINE.REPLACE_BEFORE_MATCHER else 1.0
            weight_dict["loss_roi_subject_cls"] = float(cfg.MODEL.ROI_REFINE.LOSS_WEIGHT) * _roi_loss_scale
            weight_dict["loss_roi_object_cls"] = float(cfg.MODEL.ROI_REFINE.LOSS_WEIGHT) * _roi_loss_scale
        corner_weight = cfg.MODEL.DETR.CORNER_LOSS_WEIGHT if cfg.MODEL.DETR.USE_CORNER_LOSS else 0.0
        if corner_weight > 0:
            weight_dict["loss_corner"] = corner_weight
            weight_dict["loss_corner_subject"] = corner_weight
            weight_dict["loss_corner_object"] = corner_weight
            weight_dict["loss_corner_relation"] = corner_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(max(dec_layers, obj_dec_layers) - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        print (weight_dict)
        losses = ["labels", "boxes"]

        if self.mask_on:
            losses += ["masks"]
        self.criterion = build_criterion(cfg.MODEL.DETR.CRITERION, self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses, use_gt_box=self.use_gt_box, use_gt_label=self.use_gt_label, num_relation_classes=self.num_relation_classes, intersection_iou_threshold=cfg.MODEL.DETR.INTERSECTION_IOU_THRESHOLD, intersection_iou_lambda=cfg.MODEL.DETR.INTERSECTION_IOU_LAMBDA, intersection_loss=cfg.MODEL.DETR.INTERSECTION_LOSS, rel_eos_coef=no_rel_weight, statistics=statistics, reweight_relations=cfg.MODEL.DETR.REWEIGHT_RELATIONS, reweight_rel_eos_coef=cfg.MODEL.DETR.REWEIGHT_REL_EOS_COEF, neg_rel_fraction=cfg.MODEL.DETR.NEGATIVE_RELATION_FRACTION, max_rel_pairs=cfg.MODEL.DETR.MAX_RELATION_PAIRS, use_reweight_log=cfg.MODEL.DETR.REWEIGHT_USE_LOG, focal_alpha=cfg.MODEL.DETR.FOCAL_ALPHA, create_bg_pairs=create_bg_pairs, oversample_param=cfg.MODEL.DETR.OVERSAMPLE_PARAM, undersample_param=cfg.MODEL.DETR.UNDERSAMPLE_PARAM, person_class_weight=cfg.MODEL.DETR.PERSON_CLASS_WEIGHT, obj_split_enabled=cfg.MODEL.OBJ_SPLIT.ENABLED, obj_split_regular_classes=list(cfg.MODEL.OBJ_SPLIT.REGULAR_CLASSES), obj_split_small_classes=list(cfg.MODEL.OBJ_SPLIT.SMALL_SHARED_CLASSES), obj_split_fine_groups=[list(g) for g in list(cfg.MODEL.OBJ_SPLIT.FINE_GROUPS)], obj_split_fine_names=list(cfg.MODEL.OBJ_SPLIT.FINE_GROUP_NAMES), roi_refine_enabled=cfg.MODEL.ROI_REFINE.ENABLED, roi_refine_loss_enabled=cfg.MODEL.ROI_REFINE.LOSS_ENABLED, roi_refine_small_area_thresh=cfg.MODEL.ROI_REFINE.SMALL_AREA_THRESH, \
            obj_missed_aux_cfg=cfg.MODEL.DETR.OBJ_MISSED_AUX, \
            one2many_scheme =cfg.MODEL.DETR.ONE2MANY_SCHEME, match_independent = cfg.MODEL.DETR.MATCH_INDEPENDENT, \
            box_loss_type=cfg.MODEL.DETR.BOX_LOSS_TYPE, use_corner_loss=cfg.MODEL.DETR.USE_CORNER_LOSS, corner_loss_weight=cfg.MODEL.DETR.CORNER_LOSS_WEIGHT,
            detection_only=cfg.MODEL.DETR.DETECTION_ONLY)
        self.criterion.to(self.device)

        self.use_sam3_backbone = cfg.MODEL.SAM3.ENABLED
        # QualityAux flags (inference-only, default OFF)
        self._aux_discount = float(getattr(cfg.MODEL.DETR.OBJ_MISSED_AUX, "DISCOUNT_FACTOR", 0.0))
        self._aux_discount_iou = float(getattr(cfg.MODEL.DETR.OBJ_MISSED_AUX, "DISCOUNT_IOU_THRESH", 0.5))
        self._triplet_conf_alpha = float(getattr(cfg.MODEL.DETR.OBJ_MISSED_AUX, "TRIPLET_CONF_ALPHA", 0.0))
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)
        self._freeze_layers(layers=cfg.MODEL.DETR.FREEZE_LAYERS)
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print ("Number of Parameters:", pytorch_total_params)

        # Load DETR pretrained weights from MODEL.DETR.HEAD_WEIGHTS:
        # - LOAD_HEAD_ONLY=True  -> exclude class/relation heads, then re-init AG heads
        # - LOAD_HEAD_ONLY=False -> load all compatible DETR params (including class/relation heads)
        if cfg.MODEL.DETR.HEAD_WEIGHTS:
            if cfg.MODEL.DETR.LOAD_HEAD_ONLY:
                self._load_detr_head_only(cfg.MODEL.DETR.HEAD_WEIGHTS, cfg)
            else:
                self._load_detr_full(cfg.MODEL.DETR.HEAD_WEIGHTS)

    def _freeze_layers(self, layers):
        # Freeze layers
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers):
                logging.getLogger('detectron2').log(logging.WARN, "Freezed Layer: {}".format(name))
                param.requires_grad = False

    def load_detr_from_pretrained(self, path):
        print("Loading DETR checkpoint from pretrained: ", path)
        ckpt = torch.load(path, map_location="cpu")
        pretrained_detr = ckpt.get("model", ckpt)
        del ckpt
        pretrained_detr_without_class_head = {k: v for k, v in pretrained_detr.items() if 'class_embed' not in k}
        del pretrained_detr
        gc.collect()
        self.detr.load_state_dict(pretrained_detr_without_class_head, strict=False)
        del pretrained_detr_without_class_head
        gc.collect()

    def _load_detr_head_only(self, path, cfg):
        """
        按「除分类头以外所有权重 + 重建 AG 分类头」两步加载，与当前数据集 anno 的 class 配置一致，
        避免与 vg_objectdetector_pretrained.pth 中 VG 类别数冲突。
        LOAD_CLASS_HEAD=False（默认）：跳过 class_embed，加载后重建；relation 始终跳过并重建。
        LOAD_CLASS_HEAD=True：加载 class_embed（分类头已匹配时）；relation 仍跳过并重建。
        """
        load_class_head = getattr(cfg.MODEL.DETR, "LOAD_CLASS_HEAD", False)
        logger = logging.getLogger("detectron2")
        logger.info(
            "Loading DETR head: path=%s, LOAD_CLASS_HEAD=%s (False=skip class_embed & re-init, True=load class_embed)",
            path, load_class_head
        )
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        del ckpt
        model_dict = self.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if not load_class_head and "class_embed" in k:
                continue
            if "relation" in k:
                continue
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered[k] = v
        del state_dict
        model_dict.update(filtered)
        self.load_state_dict(model_dict, strict=False)
        logger.info(
            "DETR head: loaded %d keys (excluded relation heads%s)",
            len(filtered), ", excluded class_embed" if not load_class_head else ""
        )
        del model_dict
        del filtered
        gc.collect()

        # 第二步：仅当未加载 class_embed 时重建 object 分类头；relation 始终重建
        if not load_class_head and hasattr(self.detr, "class_embed"):
            torch.nn.init.normal_(self.detr.class_embed.weight, std=0.01)
            torch.nn.init.constant_(self.detr.class_embed.bias, 0)
        if hasattr(self.detr, "relation_class_embed"):
            torch.nn.init.normal_(self.detr.relation_class_embed.weight, std=0.01)
            torch.nn.init.constant_(self.detr.relation_class_embed.bias, 0)
        if hasattr(self.detr, "transformer"):
            t = self.detr.transformer
            if hasattr(t, "object_embed") and (t.object_embed is not None) and not load_class_head:
                torch.nn.init.normal_(t.object_embed.weight, std=0.01)
                torch.nn.init.constant_(t.object_embed.bias, 0)
            if hasattr(t, "relation_embed"):
                torch.nn.init.normal_(t.relation_embed.weight, std=0.01)
                torch.nn.init.constant_(t.relation_embed.bias, 0)
        logger.info(
            "DETR head: re-inited relation heads; class_embed%s",
            " (skipped, was loaded)" if load_class_head else " re-inited"
        )

    def _load_detr_full(self, path):
        """
        加载所有 shape 匹配的 DETR 相关权重（包含 class/relation heads）。
        适用于类别头已对齐当前任务的专用预训练权重。
        """
        logger = logging.getLogger("detectron2")
        logger.info("Loading full DETR weights (including class/relation heads): %s", path)
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        del ckpt
        model_dict = self.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered[k] = v
        del state_dict
        model_dict.update(filtered)
        self.load_state_dict(model_dict, strict=False)
        logger.info("Full DETR load: loaded %d compatible keys", len(filtered))
        del model_dict
        del filtered
        gc.collect()

    def set_sam3_freeze(self, freeze: bool) -> bool:
        """
        Toggle SAM3 backbone freeze state at runtime.
        Returns True when SAM3 backbone is found and updated.
        """
        if not self.use_sam3_backbone:
            return False
        if not hasattr(self.detr, "backbone"):
            return False
        backbone = self.detr.backbone[0] if isinstance(self.detr.backbone, nn.Sequential) else self.detr.backbone
        if hasattr(backbone, "set_freeze"):
            backbone.set_freeze(freeze)
            return True
        return False

    def load_vg_pretrained_for_ag(self, path):
        """
        从 VG 预训练权重加载，排除 object 分类头与 relation 头，避免与 AG 的 class 文件冲突；
        然后按当前模型结构重建并初始化 AG 分类头与关系头。
        - 跳过: class_embed（VG 旧名）, object_embed（当前物体头）, 所有含 relation 的键
        - 之后对 detr.transformer.object_embed 与 detr.transformer.relation_embed 做 normal(0, 0.01) / constant(0)
        """
        logger = logging.getLogger("detectron2")
        logger.info("Loading VG pretrained for AG (excluding class/relation heads): %s", path)
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        del ckpt
        model_dict = self.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            if "class_embed" in k:
                continue
            if "object_embed" in k:
                continue
            if "relation" in k:
                continue
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered[k] = v
        del state_dict
        model_dict.update(filtered)
        self.load_state_dict(model_dict, strict=False)
        del model_dict
        del filtered
        gc.collect()
        # 按 AG 类别数重建并初始化两个头
        if hasattr(self.detr, "transformer"):
            t = self.detr.transformer
            if hasattr(t, "object_embed") and (t.object_embed is not None):
                torch.nn.init.normal_(t.object_embed.weight, std=0.01)
                torch.nn.init.constant_(t.object_embed.bias, 0)
            if hasattr(t, "relation_embed"):
                torch.nn.init.normal_(t.relation_embed.weight, std=0.01)
                torch.nn.init.constant_(t.relation_embed.bias, 0)
        logger.info("VG-for-AG: loaded %d keys, re-inited object_embed and relation_embed", len(filtered))

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        output = self.detr(images)
        
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            # import pdb;pdb.set_trace()
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            # 提取image_sizes
            if hasattr(self.detr, 'use_pre_extracted_features') and self.detr.use_pre_extracted_features:
                # 从batched_inputs中提取image_sizes
                image_sizes = [(x['height'], x['width']) for x in batched_inputs]
            else:
                image_sizes = images.image_sizes
            results = self.inference(box_cls, box_pred, mask_pred, image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({
                    "instances": r,
                    "rel_pair_idxs": results_per_image._rel_pair_idxs,
                    "pred_rel_scores": results_per_image._pred_rel_scores,
                    "pred_rel_labels": results_per_image._pred_rel_labels,
                    "query_index": results_per_image._query_index,
                    "obj_split_sub_head_source": getattr(results_per_image, "_obj_split_sub_head_source", None),
                    "obj_split_obj_head_source": getattr(results_per_image, "_obj_split_obj_head_source", None),
                })
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({'masks': gt_masks})
        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, image_sizes
        )):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        if self.use_sam3_backbone:
            images = [x["image"].to(self.device) for x in batched_inputs]
        else:
            images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        images.image_ids = [x.get("image_id") for x in batched_inputs]
        images.video_ids = [x.get("video_id", x.get("image_id")) for x in batched_inputs]
        return images

@META_ARCH_REGISTRY.register()
class IterativeRelationDetr(Detr):
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        targets = None
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_relations = [x["relations"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets((gt_instances, gt_relations))

        # 检查是否使用预提取特征
        if hasattr(self.detr, 'use_pre_extracted_features') and self.detr.use_pre_extracted_features:
            # 预提取特征模式：直接传递特征字典给DETR模型
            output = self.detr(batched_inputs, targets=targets)
        else:
            # 标准图像处理模式
            images = self.preprocess_image(batched_inputs)
            # Attach video metadata for triplet memory (temporal_v3)
            video_ids = [x.get("video_id", "__novid__") for x in batched_inputs]
            frame_idxs = [x.get("frame_idx", 0) for x in batched_inputs]
            images.video_ids = video_ids
            images.frame_idxs = frame_idxs
            output = self.detr(images, targets=targets)

        if self.training:
            loss_dict = self.criterion(output, targets)
            
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            # import pdb;pdb.set_trace()
            return loss_dict
        else:
            # 提取image_sizes
            if hasattr(self.detr, 'use_pre_extracted_features') and self.detr.use_pre_extracted_features:
                # 从batched_inputs中提取image_sizes
                image_sizes = [(x['height'], x['width']) for x in batched_inputs]
            else:
                image_sizes = images.image_sizes
            roi_dual = getattr(self.detr, "roi_refine_enabled", False) and getattr(self.detr, "roi_eval_dual", False)
            if roi_dual:
                assert "relation_subject_logits_roi" in output and "relation_object_logits_roi" in output, (
                    "ROI_REFINE.EVAL_DUAL=True but roi logits missing from detr output."
                )
                proc_override = self._postprocess_results(
                    self.inference(output, image_sizes, "relation_subject_logits_roi", "relation_object_logits_roi"),
                    batched_inputs, image_sizes,
                )
                proc_raw = self._postprocess_results(
                    self.inference(output, image_sizes),
                    batched_inputs, image_sizes,
                )
                return {"__roi_dual__": True, "override": proc_override, "raw": proc_raw}
            return self._postprocess_results(
                self.inference(output, image_sizes), batched_inputs, image_sizes
            )

    def _postprocess_results(self, results, batched_inputs, image_sizes):
        processed_results = []
        debug_rel = os.environ.get("SPEAQ_DEBUG_REL_OUT", "0") == "1"
        debug_limit = int(os.environ.get("SPEAQ_DEBUG_REL_OUT_LIMIT", "20"))
        debug_count = getattr(self, "_debug_rel_out_count", 0)
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            if debug_rel and debug_count < debug_limit:
                rel_pair = getattr(results_per_image, "_rel_pair_idxs", None)
                rel_scores = getattr(results_per_image, "_pred_rel_scores", None)
                rel_labels = getattr(results_per_image, "_pred_rel_labels", None)
                logging.getLogger("detectron2").info(
                    "[SPEAQ_DEBUG_REL_OUT] image_id=%s pred_boxes=%d "
                    "rel_pair=%s rel_scores_shape=%s rel_labels_shape=%s",
                    input_per_image.get("image_id", "unknown"),
                    int(r.pred_boxes.tensor.shape[0]) if hasattr(r, "pred_boxes") else -1,
                    "None" if rel_pair is None else str(tuple(rel_pair.shape)),
                    "None" if rel_scores is None else str(tuple(rel_scores.shape)),
                    "None" if rel_labels is None else str(tuple(rel_labels.shape)),
                )
                debug_count += 1
            processed_results.append({
                "instances": r,
                "rel_pair_idxs": results_per_image._rel_pair_idxs,
                "pred_rel_scores": results_per_image._pred_rel_scores,
                "pred_rel_labels": results_per_image._pred_rel_labels,
                "query_index": results_per_image._query_index,
                "obj_split_sub_head_source": getattr(results_per_image, "_obj_split_sub_head_source", None),
                "obj_split_obj_head_source": getattr(results_per_image, "_obj_split_obj_head_source", None),
            })
        self._debug_rel_out_count = debug_count
        return processed_results


    def inference(self, output, image_sizes, sub_key="relation_subject_logits", obj_key="relation_object_logits"):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes
        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []
        if self.test_index == -1:
            M = self.M
            logits_r = F.softmax(output['relation_logits'], -1)

            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores_s, labels_s = F.softmax(output[sub_key], -1)[:, :, :-1].max(-1)
            scores_o, labels_o = F.softmax(output[obj_key], -1)[:, :, :-1].max(-1)
            B, _ = scores_s.size()
            
            scores_s, labels_s, scores_o, labels_o = map(lambda u: u.unsqueeze(-1).repeat(1,1,M).reshape(B, -1), (scores_s, labels_s, scores_o, labels_o))
            src_head_s = output.get('obj_split_subject_head_source_idx', None)
            src_head_o = output.get('obj_split_object_head_source_idx', None)
            if src_head_s is not None and src_head_o is not None:
                src_head_s = src_head_s.unsqueeze(-1).repeat(1, 1, M).reshape(B, -1)
                src_head_o = src_head_o.unsqueeze(-1).repeat(1, 1, M).reshape(B, -1)
            else:
                src_head_s = torch.full_like(labels_s, -1)
                src_head_o = torch.full_like(labels_o, -1)

            scores_r, labels_r = map(lambda u: u.reshape(B, -1), logits_r[:,:,:-1].topk(M,-1))
            # Use dynamic number of relation classes (+bg) instead of hard-coded 51
            num_rel_classes_plus_bg = logits_r.shape[-1]
            logits_r = logits_r.repeat(1,1,M).reshape(1,-1, num_rel_classes_plus_bg)
            box_s = output['relation_subject_boxes']
            box_o = output['relation_object_boxes']
            box_s, box_o = map(lambda u: u.repeat(1,1,M).reshape(B,-1,4), (box_s, box_o))
        else:
            logits_r = F.softmax(output['aux_outputs_r'][self.test_index]['pred_logits'], -1)
             # For each box we assign the best class or the second best if the best on is `no_object`.
            scores_s, labels_s = F.softmax(output['aux_outputs_r_sub'][self.test_index]['pred_logits'], -1)[:, :, :-1].max(-1)
            scores_o, labels_o = F.softmax(output['aux_outputs_r_obj'][self.test_index]['pred_logits'], -1)[:, :, :-1].max(-1)
            scores_r, labels_r = logits_r[:, :, :-1].max(-1)
            src_head_s = torch.full_like(labels_s, -1)
            src_head_o = torch.full_like(labels_o, -1)
            
            box_s = output['aux_outputs_r_sub'][self.test_index]['pred_boxes']
            box_o = output['aux_outputs_r_obj'][self.test_index]['pred_boxes']

        # QualityAux discount: penalize shadow detections
        discount_factor = float(getattr(self, "_aux_discount", 0.0))
        discount_iou_thresh = float(getattr(self, "_aux_discount_iou", 0.5))
        if discount_factor > 0 and not self.training:
            for b in range(B):
                for br_s, br_l, br_b in [(scores_s, labels_s, box_s), (scores_o, labels_o, box_o)]:
                    for cls_id in br_l[b].unique():
                        if cls_id < 0:
                            continue
                        mask = br_l[b] == cls_id
                        idxs = mask.nonzero(as_tuple=True)[0]
                        if len(idxs) <= 1:
                            continue
                        boxes_xyxy = Boxes(box_cxcywh_to_xyxy(br_b[b][idxs]))
                        ious = pairwise_iou(boxes_xyxy, boxes_xyxy)
                        for k in range(len(idxs)):
                            penalty = ious[k, :k].max().item() if k > 0 else 0.0
                            if penalty > discount_iou_thresh:
                                br_s[b, idxs[k]] *= discount_factor

        for i, (scores_per_image_s, labels_per_image_s, box_per_image_s, scores_per_image_o, labels_per_image_o, box_per_image_o, scores_per_image_r, labels_per_image_r, logits_per_image_r, src_per_image_s, src_per_image_o, image_size) in enumerate(zip(
            scores_s, labels_s, box_s, scores_o, labels_o, box_o, scores_r, labels_r, logits_r, src_head_s, src_head_o, image_sizes
        )):
            if self.pnms:
                subject_boxes = Boxes(box_cxcywh_to_xyxy(box_per_image_s))
                object_boxes = Boxes(box_cxcywh_to_xyxy(box_per_image_o))

                sop_labels = torch.stack([labels_per_image_s, labels_per_image_o, labels_per_image_r], dim=1) # shape : (600, 3)
                unique_sop_labels = torch.unique(sop_labels, dim=0)

                indices = {}

                for sop in unique_sop_labels:
                    mask = torch.eq(sop_labels, sop).sum(dim=1) == 3
                    index = torch.where(mask)[0].tolist()
                    indices[sop.data] = index

                global_scores = scores_per_image_s * scores_per_image_o * scores_per_image_r
                for triplet_label, index in indices.items():
                    keep_inds_in_class = self.pairwise_nms(subject_boxes[index], object_boxes[index], global_scores[index])

            else:
                image_boxes = Boxes(box_cxcywh_to_xyxy(torch.cat([box_per_image_s, box_per_image_o])))
                # Dual scoring: blend triplet confidence into detection scores
                triplet_alpha = float(getattr(self, "_triplet_conf_alpha", 0.0))
                if triplet_alpha > 0 and not self.training:
                    r_logits = logits_per_image_r
                    r_prob = F.softmax(r_logits, dim=-1)
                    r_conf = r_prob[:, :-1].max(dim=-1).values
                    r_conf = r_conf.unsqueeze(-1).repeat(1, M).reshape(-1)
                    scores_per_image_s = scores_per_image_s * (1 - triplet_alpha + triplet_alpha * r_conf[:scores_per_image_s.shape[0]])
                    scores_per_image_o = scores_per_image_o * (1 - triplet_alpha + triplet_alpha * r_conf[:scores_per_image_o.shape[0]])
                    image_scores = torch.cat([scores_per_image_s, scores_per_image_o])
                else:
                    image_scores = torch.cat([scores_per_image_s, scores_per_image_o])

                image_pred_classes = torch.cat([labels_per_image_s, labels_per_image_o])
                keep = batched_nms(image_boxes.tensor, image_scores, image_pred_classes, self.nms_thresh) # shape : (169,)
                keep_classes = image_pred_classes[keep] # shape : (169,) (may vary)
                ious = pairwise_iou(image_boxes, image_boxes[keep]) # shape : (1200, 169)
                iou_assignments = torch.zeros_like(image_pred_classes) # shape : (1200,)
                for class_id in torch.unique(keep_classes):
                    curr_indices = torch.where(image_pred_classes == class_id)[0]
                    curr_keep_indices = torch.where(keep_classes == class_id)[0]
                    curr_ious = ious[curr_indices][:, curr_keep_indices]
                    curr_iou_assignment = curr_keep_indices[curr_ious.argmax(-1)]
                    iou_assignments[curr_indices] = curr_iou_assignment

                result = Instances(image_size)
                result.pred_boxes = image_boxes[keep]
                result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
                result.scores = image_scores[keep]
                result.pred_classes = image_pred_classes[keep]

                # rel_pair_indexer = torch.arange(labels_per_image_s.size(0)).to(labels_per_image_s.device)
                # rel_pair_idx = torch.stack([rel_pair_indexer, rel_pair_indexer + labels_per_image_s.size(0)], 1)
                rel_pair_idx = torch.stack(torch.split(iou_assignments, labels_per_image_s.size(0)), 1)

                triple_scores = scores_per_image_r * scores_per_image_s * scores_per_image_o
                # triple_scores = scores_per_image_r

                _, sorting_idx = torch.sort(triple_scores, descending=True)
                rel_pair_idx = rel_pair_idx[sorting_idx]
                rel_class_prob = logits_per_image_r[sorting_idx]
                rel_labels = labels_per_image_r[sorting_idx]
                rel_source_sub = src_per_image_s[sorting_idx]
                rel_source_obj = src_per_image_o[sorting_idx]

                triplets = torch.cat((rel_pair_idx, rel_labels.unsqueeze(-1)), -1)
                unique_triplets = {}
                keep_triplet = torch.zeros_like(rel_labels)
                # for idx, triplet in enumerate(triplets):
                #     if "{}-{}-{}".format(triplet[0], triplet[1], triplet[2]) not in unique_triplets:
                #         unique_triplets[ "{}-{}-{}".format(triplet[0], triplet[1], triplet[2])] = 1
                #         keep_triplet[idx] = 1

                unique, idx, counts = torch.unique(triplets, dim=0, sorted=True, return_inverse=True, return_counts=True)
                _, ind_sorted = torch.sort(idx, stable=True)
                cum_sum = counts.cumsum(0)
                cum_sum = torch.cat((torch.tensor([0]).to('cuda'), cum_sum[:-1]))
                first_indices = ind_sorted[cum_sum]
                keep_triplet[first_indices] = 1

                result._rel_pair_idxs = rel_pair_idx[keep_triplet == 1] # (#rel, 2)
                result._pred_rel_scores = rel_class_prob[keep_triplet == 1] # (#rel, #rel_class)
                result._pred_rel_labels = rel_labels[keep_triplet == 1] # (#rel, )
                result._query_index = sorting_idx[keep_triplet == 1]
                result._obj_split_sub_head_source = rel_source_sub[keep_triplet == 1]
                result._obj_split_obj_head_source = rel_source_obj[keep_triplet == 1]
                results.append(result)
        return results

    def pairwise_nms(self, subject_boxes, object_boxes, scores):

        sx1, sy1, sx2, sy2 = subject_boxes.tensor.cpu().split(1, dim=-1)
        ox1, oy1, ox2, oy2 = object_boxes.tensor.cpu().split(1, dim=-1)

        sub_areas = (sx2 - sx1) * (sy2 - sy1)
        obj_areas = (ox2 - ox1) * (oy2 - oy1)
        # max_scores = np.max(scores)
        # order = scores.argsort()[::-1]
        order = torch.flip(scores.argsort(), [-1])
        # scores[torch.flip(scores.argsort(), [-1])]
        keep = []
        while order.size()[0] > 0:
            i = order[0]
            keep.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            # ovr = np.power(sub_inter / sub_union, self.nms_alpha) * np.power(obj_inter / obj_union, self.nms_beta)
            # inds = np.where(ovr <= self.thres_nms)[0]

            ovr = np.power(sub_inter / sub_union, 1) * np.power(obj_inter / obj_union, 1)
            inds = np.where(ovr <= 0.7)[0]

            order = order[inds + 1]
        keep = torch.as_tensor(keep, dtype=torch.int64).to('cuda')
        return keep

    def boxes_union(self, boxes1, boxes2):
        """
        Compute the union region of two set of boxes
        Arguments:
        box1: (Boxes) bounding boxes, sized [N,4].
        box2: (Boxes) bounding boxes, sized [N,4].
        Returns:
        (Boxes) union, sized [N,4].
        """
        assert len(boxes1) == len(boxes2)

        union_box = torch.cat((
            torch.min(boxes1.tensor[:,:2], boxes2.tensor[:,:2]),
            torch.max(boxes1.tensor[:,2:], boxes2.tensor[:,2:])
            ),dim=1)
        return Boxes(union_box)

    def get_center_coords(self, boxes):
        x0, y0, x1, y1 = boxes.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2]
        return torch.stack(b, dim=-1)

    def center_xyxy_to_cxcywh(self, x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            torch.abs(x1 - x0), torch.abs(y1 - y0)]
        return torch.stack(b, dim=-1)

    def prepare_targets(self, targets, box_threshold=1e-5):
        new_targets = []
        for image_idx, (targets_per_image, relations_per_image) in enumerate(zip(targets[0], targets[1])):
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            subject_boxes = targets_per_image.gt_boxes[relations_per_image[:, 0]]
            object_boxes = targets_per_image.gt_boxes[relations_per_image[:, 1]]

            subject_object_giou = torch.diag(box_ops.generalized_box_iou(subject_boxes.tensor, object_boxes.tensor))

            gt_boxes = self.boxes_union(subject_boxes, object_boxes)
            gt_boxes = gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            gt_subject_classes = targets_per_image.gt_classes[relations_per_image[:, 0]]
            gt_subject_boxes = subject_boxes.tensor / image_size_xyxy
            gt_subject_boxes = box_xyxy_to_cxcywh(gt_subject_boxes)

            gt_object_classes = targets_per_image.gt_classes[relations_per_image[:, 1]]
            gt_object_boxes = object_boxes.tensor / image_size_xyxy
            gt_object_boxes = box_xyxy_to_cxcywh(gt_object_boxes)

            gt_combined_classes = targets_per_image.gt_classes
            gt_combined_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_combined_boxes = box_xyxy_to_cxcywh(gt_combined_boxes)

            subject_boxes_center = self.get_center_coords(subject_boxes.tensor)
            object_boxes_center = self.get_center_coords(object_boxes.tensor)
            center_boxes = torch.cat([subject_boxes_center, object_boxes_center], -1)
            center_boxes = center_boxes / image_size_xyxy
            center_boxes = self.center_xyxy_to_cxcywh(center_boxes)

            # Remove degenerate boxes
            center_boxes_xyxy = Boxes(box_cxcywh_to_xyxy(center_boxes))
            center_boxes_xyxy.scale(scale_x=targets_per_image.image_size[1], scale_y=targets_per_image.image_size[0])
            center_masks = center_boxes_xyxy.nonempty(threshold=box_threshold)
            center_boxes[~center_masks] = gt_subject_boxes[~center_masks]

            gt_classes = relations_per_image[:, 2]
            # new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'subject_boxes': gt_subject_boxes, 'object_boxes': gt_object_boxes, 'combined_boxes': gt_combined_boxes, 'subject_labels': gt_subject_classes, 'object_labels': gt_object_classes, 'combined_labels': gt_combined_classes, 'image_relations': relations_per_image, 'relation_boxes':center_boxes, 'relation_labels':relations_per_image[:, 2], 'subject_object_giou':subject_object_giou})
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, 'subject_boxes': gt_subject_boxes,
                                'object_boxes': gt_object_boxes, 'combined_boxes': gt_combined_boxes,
                                'subject_labels': gt_subject_classes, 'object_labels': gt_object_classes,
                                'combined_labels': gt_combined_classes, 'image_relations': relations_per_image,
                                'relation_boxes':center_boxes, 'relation_labels':relations_per_image[:, 2],
                                'relation_pairwise_giou':subject_object_giou})
        return new_targets
