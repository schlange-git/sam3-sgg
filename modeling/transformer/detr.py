"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

import copy
import numpy as np
from detectron2.utils.registry import Registry
import math 
from ..temporal.object_memory import ObjectMemoryBank
from .roi_refine import ROIRefineHead

DETR_REGISTRY = Registry("DETR_REGISTRY")


class TemporalAggregator(nn.Module):
    """
    Two-state temporal memory:
      - current frame feature F_t
      - historical aggregated feature H_{t-1}
    Update:
      H_t = sigma(alpha) * H_{t-1} + (1 - sigma(alpha)) * g(F_t)
    """

    def __init__(self, hidden_dim: int = 256, alpha_init: float = 0.7):
        super().__init__()
        self.update = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.memory = {}

    def reset(self):
        self.memory.clear()

    def forward(self, x: torch.Tensor, video_ids):
        # x: [B, C, H, W]
        if video_ids is None or len(video_ids) == 0:
            return x
        uniq = {str(v) for v in video_ids if v is not None}
        if len(uniq) != 1:
            # Not a single-video batch, skip temporal aggregation for safety.
            return x
        vid = next(iter(uniq))

        x_proj = self.update(x)
        x_state = x_proj.mean(dim=0, keepdim=True)

        h_prev = self.memory.get(vid, None)
        if h_prev is None or h_prev.shape != x_state.shape or h_prev.device != x_state.device:
            self.memory[vid] = x_state.detach()
            return x

        gate = torch.sigmoid(self.alpha)
        h_new = gate * h_prev + (1.0 - gate) * x_state
        self.memory[vid] = h_new.detach()

        h_expand = h_new.expand(x_proj.shape[0], -1, -1, -1)
        # Return temporally enhanced feature with unchanged shape.
        return gate * h_expand + (1.0 - gate) * x_proj


class TemporalQueryInjector(nn.Module):
    """Inject memory queries into the first K object query slots."""

    def __init__(self, d_model: int):
        super().__init__()
        self.memory_proj = nn.Linear(d_model, d_model)
        self.memory_type_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        self.memory_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, base_queries_qbd: torch.Tensor, memory_queries_bmd: torch.Tensor) -> torch.Tensor:
        if memory_queries_bmd is None:
            return base_queries_qbd
        q = base_queries_qbd.clone()
        q_len = q.shape[0]
        m_len = memory_queries_bmd.shape[1]
        k = min(q_len, m_len)
        if k <= 0:
            return q
        memory_q = memory_queries_bmd[:, :k, :].transpose(0, 1)  # [k, B, D]
        q[:k] = q[:k] + self.memory_scale * self.memory_proj(memory_q) + self.memory_type_embed
        return q


@DETR_REGISTRY.register()
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, use_gt_box=False, use_gt_label=False, **kwargs):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

@DETR_REGISTRY.register()
class IterativeRelationDETR(DETR):
    def __init__(self, backbone, transformer, num_classes, num_queries,num_relation_queries, aux_loss=False, use_gt_box=False, use_gt_label=False,cfg=None, **kwargs):
        super().__init__(backbone=backbone, transformer=transformer, num_classes=num_classes, num_queries=num_queries, aux_loss=aux_loss, use_gt_box=use_gt_box, use_gt_label=use_gt_label, **kwargs)

        self.relation_query_embed = nn.Embedding(num_relation_queries, transformer.d_model)
        self.object_query_embed = nn.Embedding(num_queries, transformer.d_model)

        del self.class_embed
        del self.bbox_embed

        self.only_predicate_multiply = cfg.MODEL.DETR.ONLY_PREDICATE_MULTIPLY
        self.multiply_query = cfg.MODEL.DETR.MULTIPLY_QUERY
        self.temporal_enabled = bool(getattr(cfg.MODEL.TEMPORAL, "ENABLED", False))
        self.temporal_eval = bool(getattr(cfg.MODEL.TEMPORAL, "EVAL_ENABLED", False))
        self.temporal_mode = str(getattr(cfg.MODEL.TEMPORAL, "MODE", "feature_ema")).lower()
        self.temporal_agg = None
        self.query_injector = None
        self.object_memory_bank = None
        self._memory_states = {}
        self.person_score_scale = float(getattr(cfg.MODEL.DETR, "PERSON_SCORE_SCALE", 1.0))
        self.person_class_index = int(getattr(cfg.MODEL.DETR, "PERSON_CLASS_INDEX", 0))
        roi_refine_cfg = cfg.MODEL.ROI_REFINE
        self.roi_refine_enabled = bool(roi_refine_cfg.ENABLED)
        self.roi_refine_stride = int(roi_refine_cfg.STRIDE)
        self.roi_refine_loss_enabled = bool(roi_refine_cfg.LOSS_ENABLED)
        self.roi_refine_head = None
        self.sam3_image_size = int(getattr(cfg.MODEL.SAM3, "IMAGE_SIZE", 1008))
        if self.roi_refine_enabled:
            assert self.roi_refine_stride > 0, "MODEL.ROI_REFINE.STRIDE must be positive."
            self.roi_refine_head = ROIRefineHead(
                hidden_dim=transformer.d_model,
                pool_size=int(roi_refine_cfg.POOL_SIZE),
                stride=self.roi_refine_stride,
                small_area_thresh=float(roi_refine_cfg.SMALL_AREA_THRESH),
                detach_boxes=bool(roi_refine_cfg.DETACH_BOXES),
                use_gate=bool(roi_refine_cfg.USE_GATE),
                apply_to=str(roi_refine_cfg.APPLY_TO),
            )
            self.roi_refine_head.only_roi_cls = bool(roi_refine_cfg.ONLY_ROI_CLS)
        if self.temporal_enabled and self.temporal_mode == "feature_ema":
            alpha_init = float(getattr(cfg.MODEL.TEMPORAL, "ALPHA_INIT", 0.7))
            self.temporal_agg = TemporalAggregator(
                hidden_dim=transformer.d_model, alpha_init=alpha_init
            )
        if self.temporal_enabled and self.temporal_mode == "object_query_memory_v1":
            self.query_injector = TemporalQueryInjector(transformer.d_model)
            self.object_memory_bank = ObjectMemoryBank(transformer.d_model, cfg)

    def _classify_object_embeddings(self, embeddings):
        assert embeddings.dim() == 3, f"Expected [B,Q,D] embeddings, got {tuple(embeddings.shape)}."
        if getattr(self.transformer, "obj_split_enabled", False):
            classifier = getattr(self.transformer, "split_object_classifier", None)
            assert classifier is not None, "OBJ_SPLIT enabled but split_object_classifier is missing."
            logits_qbd = classifier(embeddings.transpose(0, 1).unsqueeze(0))["full_logits"][0]
            return logits_qbd.transpose(0, 1)
        object_embed = getattr(self.transformer, "object_embed", None)
        assert object_embed is not None, "Transformer object_embed is missing for ROI refinement."
        return object_embed(embeddings)

    def reset_temporal_memory(self):
        self._memory_states = {}
        if self.temporal_agg is not None:
            self.temporal_agg.reset()

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        
        src, mask = features[-1].decompose()
        assert mask is not None
        if self.temporal_agg is not None and (self.training or self.temporal_eval):
            video_ids = getattr(samples, "video_ids", None)
            src = self.temporal_agg(src, video_ids)

        bs = src.shape[0]
        subject_embed = self.query_embed.weight
        object_embed = self.object_query_embed.weight
        relation_embed = self.relation_query_embed.weight

        # Object-query memory mode: inject memory into subject/object query embeddings.
        if (
            self.query_injector is not None
            and self.object_memory_bank is not None
            and (self.training or self.temporal_eval)
        ):
            video_ids = getattr(samples, "video_ids", None)
            if video_ids is None:
                video_ids = [None] * bs
            memory_batch = []
            for i in range(bs):
                vid = str(video_ids[i]) if video_ids[i] is not None else "__novid__"
                state = self._memory_states.get(vid)
                if state is None:
                    state = self.object_memory_bank.init_empty(src.device)
                memory_batch.append(self.object_memory_bank.get_memory_queries(state))
            memory_batch = torch.stack(memory_batch, dim=0)  # [B, M, D]

            subject_bqd = subject_embed.unsqueeze(1).repeat(1, bs, 1)
            object_bqd = object_embed.unsqueeze(1).repeat(1, bs, 1)
            subject_embed = self.query_injector(subject_bqd, memory_batch)
            object_embed = self.query_injector(object_bqd, memory_batch)

        output = self.transformer(
            self.input_proj(src),
            mask,
            subject_embed,
            object_embed,
            relation_embed,
            pos[-1],
        )

        # Score-time prior for person class on relation entity logits:
        # logit_person <- logit_person + log(PERSON_SCORE_SCALE)
        if self.person_score_scale > 0 and abs(self.person_score_scale - 1.0) > 1e-8:
            log_scale = math.log(self.person_score_scale)
            for key in ("relation_subject_logits", "relation_object_logits"):
                logits = output.get(key)
                if logits is None:
                    continue
                if logits.shape[-1] <= self.person_class_index:
                    continue
                logits[..., self.person_class_index] = logits[..., self.person_class_index] + log_scale

        if self.roi_refine_enabled:
            assert self.roi_refine_head is not None, "ROI_REFINE enabled but roi_refine_head was not built."
            assert isinstance(self.backbone, nn.Sequential) and len(self.backbone) > 0, (
                "ROI_REFINE requires Joiner-style backbone with SAM3 as self.backbone[0]."
            )
            backbone_module = self.backbone[0]
            assert hasattr(backbone_module, "get_last_aux_features"), (
                "ROI_REFINE requires SAM3 backbone exposing get_last_aux_features()."
            )
            aux_features = backbone_module.get_last_aux_features()
            assert isinstance(aux_features, dict), "ROI_REFINE aux_features must be a dict keyed by stride."
            assert self.roi_refine_stride in aux_features, (
                f"ROI_REFINE requires native stride{self.roi_refine_stride} feature. "
                f"Available strides: {sorted(aux_features.keys())}."
            )
            roi_nested = aux_features[self.roi_refine_stride]
            roi_feature, _ = roi_nested.decompose()
            assert roi_feature.shape[0] == bs, (
                f"ROI_REFINE feature batch {roi_feature.shape[0]} does not match DETR batch {bs}."
            )
            # SAM3 always resizes input to IMAGE_SIZE x IMAGE_SIZE internally,
            # so the feature map spatial size is always IMAGE_SIZE / stride.
            image_h = self.sam3_image_size
            image_w = self.sam3_image_size
            assert image_h % self.roi_refine_stride == 0 and image_w % self.roi_refine_stride == 0, (
                f"ROI_REFINE SAM3 IMAGE_SIZE={image_h} must be divisible by stride {self.roi_refine_stride}."
            )
            expected_feat_size = image_h // self.roi_refine_stride
            assert roi_feature.shape[-2] == expected_feat_size and roi_feature.shape[-1] == expected_feat_size, (
                f"ROI_REFINE feature shape {tuple(roi_feature.shape[-2:])} does not match "
                f"SAM3 image {image_h} / stride {self.roi_refine_stride} = {expected_feat_size}."
            )
            sub_emb = output["hs_subject_last"]
            obj_emb = output["hs_object_last"]
            sub_boxes = output["relation_subject_coords"][-1]
            obj_boxes = output["relation_object_coords"][-1]
            sub_refined, sub_mask = self.roi_refine_head(
                sub_emb, sub_boxes, roi_feature, image_h, image_w
            )
            obj_refined, obj_mask = self.roi_refine_head(
                obj_emb, obj_boxes, roi_feature, image_h, image_w
            )
            sub_roi_logits = self._classify_object_embeddings(sub_refined)
            obj_roi_logits = self._classify_object_embeddings(obj_refined)
            if self.person_score_scale > 0 and abs(self.person_score_scale - 1.0) > 1e-8:
                log_scale = math.log(self.person_score_scale)
                if sub_roi_logits.shape[-1] > self.person_class_index:
                    sub_roi_logits[..., self.person_class_index] = sub_roi_logits[..., self.person_class_index] + log_scale
                if obj_roi_logits.shape[-1] > self.person_class_index:
                    obj_roi_logits[..., self.person_class_index] = obj_roi_logits[..., self.person_class_index] + log_scale
            output["relation_subject_logits_roi"] = sub_roi_logits
            output["relation_object_logits_roi"] = obj_roi_logits
            output["roi_subject_mask"] = sub_mask
            output["roi_object_mask"] = obj_mask

        if self.only_predicate_multiply:
            output['relation_subject_logits'] = output['relation_subject_logits'].repeat_interleave(self.multiply_query,2)
            output['relation_object_logits'] = output['relation_object_logits'].repeat_interleave(self.multiply_query,2)
            output['relation_subject_coords'] = output['relation_subject_coords'].repeat_interleave(self.multiply_query,2)
            output['relation_object_coords'] = output['relation_object_coords'].repeat_interleave(self.multiply_query,2)
            if "relation_subject_logits_roi" in output:
                output['relation_subject_logits_roi'] = output['relation_subject_logits_roi'].repeat_interleave(self.multiply_query,1)
                output['relation_object_logits_roi'] = output['relation_object_logits_roi'].repeat_interleave(self.multiply_query,1)
                output['roi_subject_mask'] = output['roi_subject_mask'].repeat_interleave(self.multiply_query,1)
                output['roi_object_mask'] = output['roi_object_mask'].repeat_interleave(self.multiply_query,1)

        out = dict()

        out['relation_boxes'] = output['relation_coords'][-1]
        out['relation_logits'] = output['relation_logits'][-1]
        out['relation_subject_logits'] = output['relation_subject_logits'][-1]
        out['relation_object_logits'] = output['relation_object_logits'][-1]
        out['relation_subject_boxes'] = output['relation_subject_coords'][-1]
        out['relation_object_boxes'] = output['relation_object_coords'][-1]
        if self.roi_refine_enabled:
            assert "relation_subject_logits_roi" in output and "relation_object_logits_roi" in output, (
                "ROI_REFINE enabled but refined logits were not produced."
            )
            out['relation_subject_logits_roi'] = output['relation_subject_logits_roi']
            out['relation_object_logits_roi'] = output['relation_object_logits_roi']
            out['roi_subject_mask'] = output['roi_subject_mask']
            out['roi_object_mask'] = output['roi_object_mask']
            if not self.training:
                out['relation_subject_logits'] = out['relation_subject_logits_roi']
                out['relation_object_logits'] = out['relation_object_logits_roi']
        if 'obj_split_subject_head_source_idx' in output:
            out['obj_split_subject_head_source_idx'] = output['obj_split_subject_head_source_idx'][-1]
        if 'obj_split_object_head_source_idx' in output:
            out['obj_split_object_head_source_idx'] = output['obj_split_object_head_source_idx'][-1]
        if 'raw_split_logits_subject' in output:
            out['raw_split_logits_subject'] = {
                k: v[-1] for k, v in output['raw_split_logits_subject'].items()
            }
        if 'raw_split_logits_object' in output:
            out['raw_split_logits_object'] = {
                k: v[-1] for k, v in output['raw_split_logits_object'].items()
            }
        out['hs_subject_last'] = output.get('hs_subject_last')
        out['hs_object_last'] = output.get('hs_object_last')
        out['hs_relation_last'] = output.get('hs_relation_last')

        if self.aux_loss:
            out['aux_outputs_r'] = self._set_aux_loss(output['relation_logits'], output['relation_coords'])
            out['aux_outputs_r_sub'] = self._set_aux_loss(output['relation_subject_logits'], output['relation_subject_coords'])
            out['aux_outputs_r_obj'] = self._set_aux_loss(output['relation_object_logits'], output['relation_object_coords'])
            if 'raw_split_logits_subject' in output and 'raw_split_logits_object' in output:
                aux_outputs_obj_split = []
                num_aux = len(out['aux_outputs_r_sub'])
                for i in range(num_aux):
                    aux_outputs_obj_split.append({
                        'raw_split_logits_subject': {k: v[i] for k, v in output['raw_split_logits_subject'].items()},
                        'raw_split_logits_object': {k: v[i] for k, v in output['raw_split_logits_object'].items()},
                    })
                out['aux_outputs_obj_split'] = aux_outputs_obj_split

        # Update memory after each step using current frame predictions.
        if (
            self.object_memory_bank is not None
            and (self.training or self.temporal_eval)
            and out.get('hs_subject_last') is not None
        ):
            video_ids = getattr(samples, "video_ids", None)
            if video_ids is None:
                video_ids = [None] * bs
            for i in range(bs):
                vid = str(video_ids[i]) if video_ids[i] is not None else "__novid__"
                state = self._memory_states.get(vid)
                if state is None:
                    state = self.object_memory_bank.init_empty(src.device)
                hs_obj = torch.cat([out['hs_subject_last'][i], out['hs_object_last'][i]], dim=0)
                pred_logits = torch.cat([out['relation_subject_logits'][i], out['relation_object_logits'][i]], dim=0)
                pred_boxes = torch.cat([out['relation_subject_boxes'][i], out['relation_object_boxes'][i]], dim=0)
                state = self.object_memory_bank.update(state, hs_obj, pred_logits, pred_boxes)
                self._memory_states[vid] = state
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_coord is not None:
            return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        else:
            return [{'pred_logits': a}
                for a in outputs_class[:-1]]



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos

def build_detr(name, backbone, transformer, num_classes, num_queries,num_relation_queries, aux_loss=False, use_gt_box=False, use_gt_label=False,cfg=None, **kwargs):
    return DETR_REGISTRY.get(name)(backbone, transformer, num_classes, num_queries,num_relation_queries, aux_loss=aux_loss, use_gt_box=use_gt_box, use_gt_label=use_gt_label,cfg=cfg, **kwargs)