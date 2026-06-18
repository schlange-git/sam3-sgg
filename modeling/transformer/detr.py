import os
"""
DETR model and criterion classes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from detectron2.data import MetadataCatalog

from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .util.box_ops import box_cxcywh_to_xyxy

import copy
import numpy as np
from detectron2.utils.registry import Registry
import math 
from ..temporal.object_memory import ObjectMemoryBank
from ..temporal.triplet_memory import (
    TripletMemoryManager, TripletMemoryEncoder, TemporalTripletInjector,
    get_temporal_gate, make_union_box,
)
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
    """Lightweight cross-attention: all queries attend to memory bank."""

    def __init__(
        self,
        d_model: int,
        max_iter: int = 160000,
        output_dir: str = "/tmp",
        name: str = "object",
        gate_min: float = 0.0,
        gate_max: float = 1.0,
        gate_warmup_iters: int = 0,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.name = str(name)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.gate_warmup_iters = int(gate_warmup_iters)
        assert self.gate_max >= self.gate_min, "gate_max must be >= gate_min."
        self._gate_file = os.path.join(output_dir, f"gate_log_{self.name}.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(self._gate_file, "w") as gf:
            gf.write("iter,raw_gate,effective_gate,warmup_factor,gate_min,gate_max\n")
        self.memory_proj_k = nn.Linear(d_model, d_model)
        self.memory_proj_v = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Parameter(torch.tensor(-4.0))
        self._call_count = 0
        self._current_iter = 0
        self._last_gate_stats = None
        self.scale = d_model ** -0.5

    def forward(self, base_queries_qbd: torch.Tensor, memory_queries_bmd: torch.Tensor) -> torch.Tensor:
        if memory_queries_bmd is None:
            return base_queries_qbd
        # base_queries_qbd: [Q, B, D], memory_queries_bmd: [B, M, D]
        Q = self.query_proj(base_queries_qbd)  # [Q, B, D]
        K = self.memory_proj_k(memory_queries_bmd.transpose(0, 1))  # [M, B, D]
        V = self.memory_proj_v(memory_queries_bmd.transpose(0, 1))  # [M, B, D]

        # Cross-attention: Q attends to K, output attends to V
        attn = torch.einsum('qbd,mbd->qbm', Q, K) * self.scale
        attn = torch.softmax(attn, dim=2)  # normalize over memory slots: [Q, B, M]
        mem_out = torch.einsum('qbm,mbd->qbd', attn, V)
        mem_out = self.out_proj(mem_out)

        # Learned gating: blend memory with original
        raw_gate = torch.sigmoid(self.gate)
        bounded_gate = self.gate_min + (self.gate_max - self.gate_min) * raw_gate
        self._call_count += 1
        it = int(getattr(self, "_current_iter", self._call_count))
        if self.gate_warmup_iters > 0:
            warmup_factor = min(max(float(it) / float(self.gate_warmup_iters), 0.0), 1.0)
        else:
            warmup_factor = 1.0
        gate = bounded_gate * warmup_factor
        self._last_gate_stats = {
            "raw_gate": float(raw_gate.detach().mean().item()),
            "effective_gate": float(gate.detach().mean().item()),
            "warmup_factor": float(warmup_factor),
            "gate_min": float(self.gate_min),
            "gate_max": float(self.gate_max),
        }
        if self.training and it % 50 == 0:
            with open(self._gate_file, "a") as gf:
                gf.write(f"{it},{float(raw_gate.mean().item()):.6f},{float(gate.mean().item()):.6f},{float(warmup_factor):.6f},{self.gate_min:.6f},{self.gate_max:.6f}\n")
        return gate * mem_out + (1.0 - gate) * base_queries_qbd

    def get_last_gate_stats(self):
        return dict(self._last_gate_stats) if self._last_gate_stats is not None else None


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

    def forward(self, samples: NestedTensor, targets=None):
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
        self.relation_query_injector = None
        self.object_memory_bank = None
        self.relation_memory_bank = None
        self._memory_states = {}
        # ---- Triplet Memory (temporal_v3) ----
        self.triplet_memory_enabled = bool(
            self.temporal_enabled and getattr(cfg.MODEL.TEMPORAL, "TRIPLET_MEMORY_ENABLED", False)
        )
        self.triplet_memory_manager = None
        self.triplet_encoder = None
        self.triplet_injector_obj = None
        self.triplet_injector_rel = None
        self.triplet_iter_counter = 0
        self._triplet_max_iter = int(getattr(cfg.SOLVER, "MAX_ITER", 80000))
        self._triplet_gate_max_obj = float(getattr(cfg.MODEL.TEMPORAL, "GATE_MAX_OBJECT", 0.15))
        self._triplet_gate_max_rel = float(getattr(cfg.MODEL.TEMPORAL, "GATE_MAX_RELATION", 0.30))
        self._triplet_gate_zero_ratio = float(getattr(cfg.MODEL.TEMPORAL, "GATE_ZERO_END_RATIO", 0.10))
        self._triplet_gate_warmup_ratio = float(getattr(cfg.MODEL.TEMPORAL, "GATE_WARMUP_END_RATIO", 0.30))
        self._triplet_topk_update = int(getattr(cfg.MODEL.TEMPORAL, "TRIPLET_MEMORY_TOPK_UPDATE", 16))
        self._triplet_update_thresh = float(getattr(cfg.MODEL.TEMPORAL, "UPDATE_SCORE_THRESH", 0.10))
        self._triplet_debug_memory = bool(getattr(cfg.MODEL.TEMPORAL, "DEBUG_MEMORY", False))

        self._relation_memory_states = {}
        tcfg = cfg.MODEL.TEMPORAL
        self.memory_update_mode = str(getattr(tcfg, "MEMORY_UPDATE_MODE", "prediction")).lower()
        self.relation_memory_source = str(getattr(tcfg, "RELATION_MEMORY_SOURCE", "object")).lower()
        self.relation_memory_update_mode = str(getattr(tcfg, "RELATION_MEMORY_UPDATE_MODE", "prediction")).lower()
        self.person_score_scale = float(getattr(cfg.MODEL.DETR, "PERSON_SCORE_SCALE", 1.0))
        self.person_class_index = int(getattr(cfg.MODEL.DETR, "PERSON_CLASS_INDEX", 0))
        roi_refine_cfg = cfg.MODEL.ROI_REFINE
        self.roi_refine_enabled = bool(roi_refine_cfg.ENABLED)
        self.roi_eval_dual = bool(getattr(roi_refine_cfg, "EVAL_DUAL", False))
        self.roi_refine_stride = int(roi_refine_cfg.STRIDE)
        self.resnet_fpn_level = int(getattr(roi_refine_cfg, "RESNET_FPN_LEVEL", 0))
        self.roi_refine_loss_enabled = bool(roi_refine_cfg.LOSS_ENABLED)
        self.roi_replace_main = bool(getattr(roi_refine_cfg, "REPLACE_BEFORE_MATCHER", False))
        if self.roi_replace_main:
            assert self.roi_refine_enabled, "REPLACE_BEFORE_MATCHER 需要 ROI_REFINE.ENABLED=True。"
            assert self.roi_refine_loss_enabled, "REPLACE_BEFORE_MATCHER 需要 ROI_REFINE.LOSS_ENABLED=True (x5 roi loss 才能生效)。"
            assert not self.roi_eval_dual, "REPLACE_BEFORE_MATCHER 与 EVAL_DUAL 互斥: 替换后 main==roi, eval 仅一套结果。"
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
                fusion=str(roi_refine_cfg.FUSION),
                class_names=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes,
            )
            self.roi_refine_head.only_roi_cls = bool(roi_refine_cfg.ONLY_ROI_CLS)
        if self.temporal_enabled and self.temporal_mode == "feature_ema":
            alpha_init = float(getattr(cfg.MODEL.TEMPORAL, "ALPHA_INIT", 0.7))
            self.temporal_agg = TemporalAggregator(
                hidden_dim=transformer.d_model, alpha_init=alpha_init
            )
        if self.temporal_enabled and self.temporal_mode == "object_query_memory_v1":
            self.query_injector = TemporalQueryInjector(
                transformer.d_model,
                max_iter=cfg.SOLVER.MAX_ITER,
                output_dir=cfg.OUTPUT_DIR,
                name="object",
                gate_min=float(getattr(tcfg, "GATE_MIN", 0.0)),
                gate_max=float(getattr(tcfg, "GATE_MAX", 1.0)),
                gate_warmup_iters=int(getattr(tcfg, "GATE_WARMUP_ITERS", 0)),
            )
            if cfg.MODEL.TEMPORAL.RELATION_MEMORY_ENABLED:
                self.relation_query_injector = TemporalQueryInjector(
                    transformer.d_model,
                    max_iter=cfg.SOLVER.MAX_ITER,
                    output_dir=cfg.OUTPUT_DIR,
                    name="relation",
                    gate_min=float(getattr(tcfg, "RELATION_GATE_MIN", getattr(tcfg, "GATE_MIN", 0.0))),
                    gate_max=float(getattr(tcfg, "RELATION_GATE_MAX", getattr(tcfg, "GATE_MAX", 1.0))),
                    gate_warmup_iters=int(getattr(tcfg, "RELATION_GATE_WARMUP_ITERS", getattr(tcfg, "GATE_WARMUP_ITERS", 0))),
                )
            else:
                self.relation_query_injector = None
            self.object_memory_bank = ObjectMemoryBank(transformer.d_model, cfg)
            if self.relation_query_injector is not None and self.relation_memory_source == "relation":
                self.relation_memory_bank = ObjectMemoryBank(transformer.d_model, cfg)
        # ---- Build Triplet Memory modules (temporal_v3) ----
        if self.temporal_enabled and getattr(tcfg, "TRIPLET_MEMORY_ENABLED", False):
            mem_dim = int(getattr(tcfg, "TRIPLET_MEMORY_DIM", 128))
            num_rel_cls = int(cfg.MODEL.DETR.NUM_RELATION_CLASSES)
            self.triplet_memory_enabled = True
            self.triplet_memory_manager = TripletMemoryManager(cfg)
            self.triplet_encoder = TripletMemoryEncoder(
                d_model=transformer.d_model, num_rel_classes=num_rel_cls,
                mem_dim=mem_dim)
            assert getattr(tcfg, "INJECT_OBJECT", True) or getattr(tcfg, "INJECT_RELATION", True), (
                "[TripletMemory] ENABLED=True but INJECT_OBJECT=False and INJECT_RELATION=False!")
            if getattr(tcfg, "INJECT_OBJECT", True):
                self.triplet_injector_obj = TemporalTripletInjector(
                    d_model=transformer.d_model, mem_dim=mem_dim,
                    nhead=cfg.MODEL.DETR.NHEADS, dropout=cfg.MODEL.DETR.DROPOUT)
            if getattr(tcfg, "INJECT_RELATION", True):
                self.triplet_injector_rel = TemporalTripletInjector(
                    d_model=transformer.d_model, mem_dim=mem_dim,
                    nhead=cfg.MODEL.DETR.NHEADS, dropout=cfg.MODEL.DETR.DROPOUT)


    def _classify_object_embeddings(self, embeddings):
        assert embeddings.dim() == 3, f"Expected [B,Q,D] embeddings, got {tuple(embeddings.shape)}."
        return self._predict_object_from_embeddings(embeddings)[0]

    def _predict_object_from_embeddings(self, embeddings):
        assert embeddings.dim() == 3, f"Expected [B,Q,D] embeddings, got {tuple(embeddings.shape)}."
        if getattr(self.transformer, "obj_split_enabled", False):
            classifier = getattr(self.transformer, "split_object_classifier", None)
            assert classifier is not None, "OBJ_SPLIT enabled but split_object_classifier is missing."
            cls_outputs = classifier(embeddings.transpose(0, 1).unsqueeze(0))
            logits = cls_outputs["full_logits"][0].transpose(0, 1)
            raw_split = {
                k: v[0].transpose(0, 1)
                for k, v in cls_outputs["raw_split_logits"].items()
            }
            head_source = cls_outputs["head_source_idx"][0].transpose(0, 1)
            return logits, raw_split, head_source
        object_embed = getattr(self.transformer, "object_embed", None)
        assert object_embed is not None, "Transformer object_embed is missing for ROI refinement."
        return object_embed(embeddings), None, None

    def _apply_triplet_memory_to_output(self, output, samples, device):
        if not (
            self.triplet_memory_enabled
            and self.triplet_memory_manager is not None
            and (self.training or self.temporal_eval)
        ):
            return output
        video_ids = getattr(samples, "video_ids", None)
        frame_idxs = getattr(samples, "frame_idxs", None)
        if video_ids is None or frame_idxs is None:
            return output

        mem, mem_mask = self.triplet_memory_manager.get_batch_memory(
            video_ids=video_ids, frame_idxs=frame_idxs, device=device)
        self.triplet_iter_counter += 1
        _tdbg = os.environ.get("TRIPLET_DEBUG") == "1" and (self.triplet_iter_counter <= 30 or self.triplet_iter_counter % 200 == 0)
        if mem is None:
            if _tdbg:
                print(f"[TRIPLET_DBG] apply iter={self.triplet_iter_counter} mem=None -> NO injection", flush=True)
            return output

        gate_obj = get_temporal_gate(
            self.triplet_iter_counter, self._triplet_max_iter,
            self._triplet_gate_max_obj, self._triplet_gate_zero_ratio, self._triplet_gate_warmup_ratio)
        gate_rel = get_temporal_gate(
            self.triplet_iter_counter, self._triplet_max_iter,
            self._triplet_gate_max_rel, self._triplet_gate_zero_ratio, self._triplet_gate_warmup_ratio)
        if _tdbg:
            print(f"[TRIPLET_DBG] apply iter={self.triplet_iter_counter} mem_shape={tuple(mem.shape)} "
                  f"gate_obj={gate_obj:.4f} gate_rel={gate_rel:.4f}", flush=True)
        mask_exp = mem_mask if (mem_mask is not None and mem_mask.dim() == 2) else None

        if self.triplet_injector_obj is not None and output.get("hs_object_last") is not None:
            obj_q = self.triplet_injector_obj(
                output["hs_object_last"], mem, memory_mask=mask_exp, gate=gate_obj)
            output["hs_object_last"] = obj_q
            obj_logits, raw_obj, src_obj = self._predict_object_from_embeddings(obj_q)
            output["relation_object_logits"] = output["relation_object_logits"].clone()
            output["relation_object_coords"] = output["relation_object_coords"].clone()
            output["relation_object_logits"][-1] = obj_logits
            output["relation_object_coords"][-1] = self.transformer.object_bbox_coords(obj_q).sigmoid()
            if raw_obj is not None:
                for k, v in raw_obj.items():
                    output["raw_split_logits_object"][k] = output["raw_split_logits_object"][k].clone()
                    output["raw_split_logits_object"][k][-1] = v
                output["obj_split_object_head_source_idx"] = output["obj_split_object_head_source_idx"].clone()
                output["obj_split_object_head_source_idx"][-1] = src_obj

        if self.triplet_injector_rel is not None and output.get("hs_relation_last") is not None:
            rel_q = self.triplet_injector_rel(
                output["hs_relation_last"], mem, memory_mask=mask_exp, gate=gate_rel)
            output["hs_relation_last"] = rel_q
            output["relation_logits"] = output["relation_logits"].clone()
            output["relation_coords"] = output["relation_coords"].clone()
            output["relation_logits"][-1] = self.transformer.relation_embed(rel_q)
            output["relation_coords"][-1] = self.transformer.object_bbox_coords(rel_q).sigmoid()

        return output

    def reset_temporal_memory(self):
        self._memory_states = {}
        self._relation_memory_states = {}
        if self.temporal_agg is not None:
            self.temporal_agg.reset()

    def forward(self, samples: NestedTensor, targets=None):
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
            if self.relation_query_injector is not None:
                relation_memory_batch = memory_batch
                if self.relation_memory_bank is not None:
                    relation_memory_batch = []
                    for i in range(bs):
                        vid = str(video_ids[i]) if video_ids[i] is not None else "__novid__"
                        state = self._relation_memory_states.get(vid)
                        if state is None:
                            state = self.relation_memory_bank.init_empty(src.device)
                        relation_memory_batch.append(self.relation_memory_bank.get_memory_queries(state))
                    relation_memory_batch = torch.stack(relation_memory_batch, dim=0)  # [B, M, D]
                relation_bqd = relation_embed.unsqueeze(1).repeat(1, bs, 1)
                relation_embed = self.relation_query_injector(relation_bqd, relation_memory_batch)

        output = self.transformer(
            self.input_proj(src),
            mask,
            subject_embed,
            object_embed,
            relation_embed,
            pos[-1],
        )

        output = self._apply_triplet_memory_to_output(output, samples, src.device)

        # PERSON_SCORE_SCALE no longer scales output logits here: it must not enter the
        # criterion or eval predictions. The person prior is applied only to the triplet
        # quality gate's subject score (see triplet update below).

        if self.roi_refine_enabled:
            assert self.roi_refine_head is not None, "ROI_REFINE enabled but roi_refine_head was not built."
            assert isinstance(self.backbone, nn.Sequential) and len(self.backbone) > 0, (
                "ROI_REFINE requires Joiner-style backbone."
            )
            backbone_module = self.backbone[0]
            if hasattr(backbone_module, "get_last_aux_features"):
                # SAM3 backbone path
                aux_features = backbone_module.get_last_aux_features()
                assert isinstance(aux_features, dict), "ROI_REFINE aux_features must be a dict keyed by stride."
                assert self.roi_refine_stride in aux_features, (
                    f"ROI_REFINE requires native stride{self.roi_refine_stride} feature. "
                    f"Available strides: {sorted(aux_features.keys())}."
                )
                roi_nested = aux_features[self.roi_refine_stride]
                roi_feature, _ = roi_nested.decompose()
                # SAM3 always resizes input to IMAGE_SIZE x IMAGE_SIZE internally
                image_h = self.sam3_image_size
                image_w = self.sam3_image_size
            else:
                # ResNet backbone path: features list from fine to coarse
                # Without FPN: res2(256ch,s4), res3(512ch,s8), res4(1024ch,s16), res5(2048ch,s32)
                # With FPN: all levels 256ch. Use RESNET_FPN_LEVEL to pick the feature index.
                fpn_level = int(getattr(self, "resnet_fpn_level", 1))  # default: res4 (stride 16)
                feat_idx = -(1 + fpn_level)  # 0->-1(res5), 1->-2(res4), 2->-3(res3), 3->-4(res2)
                assert abs(feat_idx) <= len(features), (
                    f"ROI_REFINE RESNET_FPN_LEVEL={fpn_level} out of range. "f"{len(features)} levels available."
                )
                roi_feature, roi_mask = features[feat_idx].decompose()
                # Project to hidden_dim if channels don't match
                fc = roi_feature.shape[1]
                target_c = self.roi_refine_head.hidden_dim
                if fc != target_c:
                    if not hasattr(self, "_roi_resnet_proj") or self._roi_resnet_proj is None:
                        self._roi_resnet_proj = nn.Conv2d(fc, target_c, 1).to(roi_feature.device)
                        nn.init.kaiming_normal_(self._roi_resnet_proj.weight)
                    roi_feature = self._roi_resnet_proj(roi_feature)
                image_h = roi_feature.shape[2] * self.roi_refine_stride
                image_w = roi_feature.shape[3] * self.roi_refine_stride
            expected_feat_h = image_h // self.roi_refine_stride
            expected_feat_w = image_w // self.roi_refine_stride
            assert roi_feature.shape[-2] == expected_feat_h and roi_feature.shape[-1] == expected_feat_w, (
                f"ROI_REFINE feature shape {tuple(roi_feature.shape[-2:])} does not match "
                f"image ({image_h},{image_w}) / stride {self.roi_refine_stride} = ({expected_feat_h},{expected_feat_w})."
            )
            sub_emb = output["hs_subject_last"]
            obj_emb = output["hs_object_last"]
            sub_boxes = output["relation_subject_coords"][-1]
            obj_boxes = output["relation_object_coords"][-1]
            all_lbls_t = torch.cat([t["combined_labels"] for t in targets]) if targets is not None else None
            sub_refined, sub_mask = self.roi_refine_head(
                sub_emb, sub_boxes, roi_feature, image_h, image_w,
                labels=all_lbls_t,  # TODO: align with ROI subset
            )
            obj_refined, obj_mask = self.roi_refine_head(
                obj_emb, obj_boxes, roi_feature, image_h, image_w,
                labels=all_lbls_t,
            )
            obj_gate_stats = getattr(self.roi_refine_head, "_last_gate_stats", None)
            self._last_roi_gate_stats = {
                "object": dict(obj_gate_stats) if obj_gate_stats is not None else None,
            }
            sub_roi_logits = self._classify_object_embeddings(sub_refined)
            obj_roi_logits = self._classify_object_embeddings(obj_refined)
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
            # [REPLACE_BEFORE_MATCHER] 无条件 (train+eval) 用 roi cls 覆盖 main logits ->
            #   matcher / 主 relation cls loss / eval 全程使用 roi 分类结果。
            if self.roi_replace_main:
                out['relation_subject_logits'] = out['relation_subject_logits_roi']
                out['relation_object_logits'] = out['relation_object_logits_roi']
            roi_eval_area_thresh = os.environ.get("ROI_EVAL_AREA_THRESH", "")
            if not self.training and roi_eval_area_thresh:
                assert not self.roi_eval_dual, "ROI_EVAL_AREA_THRESH requires MODEL.ROI_REFINE.EVAL_DUAL=False."
                area_thresh = float(roi_eval_area_thresh)
                assert 0.0 < area_thresh <= 1.0, f"bad ROI_EVAL_AREA_THRESH={area_thresh}"
                sub_area = out['relation_subject_boxes'][..., 2].clamp_min(0.0) * out['relation_subject_boxes'][..., 3].clamp_min(0.0)
                obj_area = out['relation_object_boxes'][..., 2].clamp_min(0.0) * out['relation_object_boxes'][..., 3].clamp_min(0.0)
                sub_small = sub_area < area_thresh
                obj_small = obj_area < area_thresh
                out['relation_subject_logits'] = torch.where(
                    sub_small.unsqueeze(-1),
                    out['relation_subject_logits_roi'],
                    out['relation_subject_logits'],
                )
                out['relation_object_logits'] = torch.where(
                    obj_small.unsqueeze(-1),
                    out['relation_object_logits_roi'],
                    out['relation_object_logits'],
                )
            # [ROI_EVAL_RAW gate] 置 ROI_EVAL_RAW=1 时跳过替换 -> eval 用原始(pre-refine) logits, 仅用于对照评测
            # [EVAL_DUAL gate] EVAL_DUAL=True 时不在此处覆盖, 由 meta_arch 单次前向同出 override/raw 两套结果
            elif not self.training and not self.roi_eval_dual and os.environ.get("ROI_EVAL_RAW", "0") != "1":
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

        # Update memory after each step. In matched_gt mode, only predictions that
        # can be associated with GT boxes/classes are written into memory.
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
                if self.memory_update_mode == "matched_gt" and targets is not None:
                    state = self.object_memory_bank.update_matched_gt(
                        state,
                        hs_obj,
                        pred_logits,
                        pred_boxes,
                        targets[i]["combined_boxes"].to(pred_boxes.device),
                        targets[i]["combined_labels"].to(pred_logits.device),
                    )
                else:
                    state = self.object_memory_bank.update(state, hs_obj, pred_logits, pred_boxes)
                self._memory_states[vid] = state
        if (
            self.relation_memory_bank is not None
            and (self.training or self.temporal_eval)
            and out.get('hs_relation_last') is not None
        ):
            video_ids = getattr(samples, "video_ids", None)
            if video_ids is None:
                video_ids = [None] * bs
            for i in range(bs):
                vid = str(video_ids[i]) if video_ids[i] is not None else "__novid__"
                state = self._relation_memory_states.get(vid)
                if state is None:
                    state = self.relation_memory_bank.init_empty(src.device)
                if self.relation_memory_update_mode == "matched_gt" and targets is not None:
                    state = self.relation_memory_bank.update_matched_gt(
                        state,
                        out['hs_relation_last'][i],
                        out['relation_logits'][i],
                        out['relation_boxes'][i],
                        targets[i]["relation_boxes"].to(out['relation_boxes'].device),
                        targets[i]["relation_labels"].to(out['relation_logits'].device),
                    )
                else:
                    state = self.relation_memory_bank.update(
                        state,
                        out['hs_relation_last'][i],
                        out['relation_logits'][i],
                        out['relation_boxes'][i],
                    )
                self._relation_memory_states[vid] = state

        # ---- Triplet Memory: update with current predictions (temporal_v3) ----
        if self.triplet_memory_enabled and self.triplet_memory_manager is not None and (self.training or self.temporal_eval):
            video_ids = getattr(samples, "video_ids", None)
            frame_idxs = getattr(samples, "frame_idxs", None)
            if video_ids is not None and frame_idxs is not None and out.get("hs_relation_last") is not None:
                with torch.no_grad():
                    rel_logits = out["relation_logits"]
                    rel_sub_logits = out["relation_subject_logits"]
                    rel_obj_logits = out["relation_object_logits"]
                    rel_sub_boxes = out["relation_subject_boxes"]
                    rel_obj_boxes = out["relation_object_boxes"]
                    hs_rel = out["hs_relation_last"]
                    hs_obj = out["hs_object_last"]

                    batch_candidates = []
                    for b in range(bs):
                        sub_logits_b = rel_sub_logits[b]
                        # person prior for the quality gate ONLY (subject is always person).
                        # clone so out["relation_subject_logits"] (criterion path) is never mutated.
                        if self.person_score_scale > 0 and abs(self.person_score_scale - 1.0) > 1e-8 \
                                and sub_logits_b.shape[-1] > self.person_class_index:
                            sub_logits_b = sub_logits_b.clone()
                            sub_logits_b[..., self.person_class_index] = (
                                sub_logits_b[..., self.person_class_index] + math.log(self.person_score_scale)
                            )
                        sub_prob = F.softmax(sub_logits_b, dim=-1)
                        obj_prob = F.softmax(rel_obj_logits[b], dim=-1)
                        rel_prob = F.softmax(rel_logits[b], dim=-1)
                        sub_score, sub_label = sub_prob[..., :-1].max(-1)
                        obj_score, obj_label = obj_prob[..., :-1].max(-1)
                        pred_score, pred_label = rel_prob[..., :-1].max(-1)

                        cands = []
                        quality = sub_score * obj_score * pred_score
                        topk_upd = min(self._triplet_topk_update, len(pred_score))
                        thresh = self._triplet_update_thresh
                        _, topk_idx = quality.topk(min(topk_upd, quality.shape[0]))

                        for r in topk_idx:
                            q = quality[r]
                            if q < thresh:
                                continue
                            signature = (int(sub_label[r]), int(pred_label[r]), int(obj_label[r]))
                            sub_bx = box_cxcywh_to_xyxy(rel_sub_boxes[b, r].unsqueeze(0))[0].clamp(0.0, 1.0)
                            obj_bx = box_cxcywh_to_xyxy(rel_obj_boxes[b, r].unsqueeze(0))[0].clamp(0.0, 1.0)
                            union_bx = make_union_box(sub_bx, obj_bx)
                            mem_feat = self.triplet_encoder(
                                rel_query=hs_rel[b, r].unsqueeze(0),
                                sub_box=sub_bx.unsqueeze(0),
                                obj_box=obj_bx.unsqueeze(0),
                                pred_prob=rel_prob[r, :-1].unsqueeze(0),
                            )[0]
                            cands.append({
                                "signature": signature,
                                "feat": mem_feat,
                                "sub_box": sub_bx,
                                "obj_box": obj_bx,
                                "union_box": union_bx,
                                "score": float(q),
                                "sub_score": float(sub_score[r]),
                                "obj_score": float(obj_score[r]),
                                "pred_score": float(pred_score[r]),
                            })
                        if os.environ.get("TRIPLET_DEBUG") == "1" and (self.triplet_iter_counter <= 30 or self.triplet_iter_counter % 200 == 0):
                            _qpass = int((quality[topk_idx] >= thresh).sum().item()) if topk_idx.numel() > 0 else 0
                            print(f"[TRIPLET_DBG] upd iter={self.triplet_iter_counter} b={b} "
                                  f"nq={int(quality.numel())} "
                                  f"sub[max={float(sub_score.max()):.3f},mean={float(sub_score.mean()):.3f}] "
                                  f"obj[max={float(obj_score.max()):.3f},mean={float(obj_score.mean()):.3f}] "
                                  f"pred[max={float(pred_score.max()):.3f},mean={float(pred_score.mean()):.3f}] "
                                  f"qual_max={float(quality.max()):.4f} thresh={thresh} "
                                  f"npass={_qpass} ncand={len(cands)}", flush=True)
                        batch_candidates.append(cands)

                    self.triplet_memory_manager.update_batch(
                        video_ids=video_ids,
                        frame_idxs=frame_idxs,
                        batch_candidates=batch_candidates,
                    )
                    if os.environ.get("TRIPLET_DEBUG") == "1" and (self.triplet_iter_counter <= 30 or self.triplet_iter_counter % 200 == 0):
                        for _vid in dict.fromkeys(video_ids):
                            _bk = self.triplet_memory_manager.banks.get(_vid)
                            _occ = len(_bk.get_valid_slots()) if _bk is not None else 0
                            print(f"[TRIPLET_DBG] bank vid={_vid} occ={_occ}", flush=True)

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