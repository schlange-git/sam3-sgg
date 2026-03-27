import copy
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor
from detectron2.utils.registry import Registry
from .detr import MLP
from .detr import gen_sineembed_for_position
from .util.misc import inverse_sigmoid
import math
from detectron2.utils.events import get_event_storage
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

TRANSFORMER_REGISTRY = Registry("TRANSFORMER_REGISTRY")

@TRANSFORMER_REGISTRY.register()
class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, **kwargs):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class SplitObjectClassifier(nn.Module):
    """
    Split object classifier that assembles branch logits into unified full-class logits.
    """

    def __init__(self, d_model: int, num_classes: int, cfg):
        super().__init__()
        self.num_classes = int(num_classes)
        obj_cfg = cfg.MODEL.OBJ_SPLIT
        self.use_temp_scaling = bool(getattr(obj_cfg, "USE_TEMP_SCALING", False))
        self.small_use_upsample = bool(getattr(obj_cfg, "SMALL_SHARED_USE_UPSAMPLE", False))

        self.regular_classes = [int(x) for x in list(getattr(obj_cfg, "REGULAR_CLASSES", []))]
        self.small_classes = [int(x) for x in list(getattr(obj_cfg, "SMALL_SHARED_CLASSES", []))]
        self.fine_groups = [[int(x) for x in list(g)] for g in list(getattr(obj_cfg, "FINE_GROUPS", []))]
        fine_names = [str(x) for x in list(getattr(obj_cfg, "FINE_GROUP_NAMES", []))]
        self.fine_group_names = fine_names if len(fine_names) == len(self.fine_groups) else [
            f"group_{i}" for i in range(len(self.fine_groups))
        ]
        self.fine_group_use_upsample = [bool(x) for x in list(getattr(obj_cfg, "FINE_GROUPS_USE_UPSAMPLE", []))]
        if len(self.fine_group_use_upsample) != len(self.fine_groups):
            self.fine_group_use_upsample = [False] * len(self.fine_groups)

        self.regular_head = nn.Linear(d_model, len(self.regular_classes))
        self.small_shared_head = nn.Linear(d_model, len(self.small_classes))
        self.bg_head = nn.Linear(d_model, 1)
        self.fine_head_dict = nn.ModuleDict()
        for name, group in zip(self.fine_group_names, self.fine_groups):
            self.fine_head_dict[name] = nn.Linear(d_model, len(group))

        class_to_head = torch.full((self.num_classes,), -1, dtype=torch.long)
        for c in self.regular_classes:
            class_to_head[c] = 0
        for c in self.small_classes:
            class_to_head[c] = 1
        for gidx, group in enumerate(self.fine_groups):
            for c in group:
                class_to_head[c] = 2 + gidx
        self.register_buffer("class_to_head", class_to_head)

        self.tau_regular = nn.Parameter(torch.tensor(1.0))
        self.tau_small = nn.Parameter(torch.tensor(1.0))
        self.tau_fine = nn.Parameter(torch.ones(len(self.fine_groups)))

        self.reg_local_map = {c: i for i, c in enumerate(self.regular_classes)}
        self.small_local_map = {c: i for i, c in enumerate(self.small_classes)}
        self.fine_local_map = {}
        for name, group in zip(self.fine_group_names, self.fine_groups):
            self.fine_local_map[name] = {c: i for i, c in enumerate(group)}

    def _apply_temp(self, logits: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if (not self.use_temp_scaling) or logits.numel() == 0:
            return logits
        return logits / tau.clamp_min(1e-4)

    def _upsample_tokens_2x(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D] -> [N, D, 1, 1] -> 2x bilinear -> [N, D]
        x2 = x.view(x.shape[0], x.shape[1], 1, 1)
        x2 = F.interpolate(x2, scale_factor=2.0, mode="bilinear", align_corners=False)
        x2 = F.adaptive_avg_pool2d(x2, output_size=1).flatten(1)
        return x2

    def forward(self, hs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # hs: [L, Q, B, D]
        l, q, b, d = hs.shape
        x = hs.reshape(-1, d)  # [L*Q*B, D]
        out_full = x.new_full((x.shape[0], self.num_classes + 1), -1e4)
        raw_logits = {}

        if len(self.regular_classes) > 0:
            reg_logits = self._apply_temp(self.regular_head(x), self.tau_regular)
            out_full[:, self.regular_classes] = reg_logits
            raw_logits["regular"] = reg_logits
        else:
            raw_logits["regular"] = x.new_zeros((x.shape[0], 0))

        if len(self.small_classes) > 0:
            x_small = self._upsample_tokens_2x(x) if self.small_use_upsample else x
            small_logits = self._apply_temp(self.small_shared_head(x_small), self.tau_small)
            out_full[:, self.small_classes] = small_logits
            raw_logits["small"] = small_logits
        else:
            raw_logits["small"] = x.new_zeros((x.shape[0], 0))

        for gidx, (name, classes) in enumerate(zip(self.fine_group_names, self.fine_groups)):
            x_fine = self._upsample_tokens_2x(x) if self.fine_group_use_upsample[gidx] else x
            fine_logits = self._apply_temp(self.fine_head_dict[name](x_fine), self.tau_fine[gidx])
            out_full[:, classes] = fine_logits
            raw_logits[f"fine:{name}"] = fine_logits

        bg_logit = self.bg_head(x).squeeze(-1)
        out_full[:, self.num_classes] = bg_logit

        fg_pred_cls = out_full[:, :-1].argmax(-1)
        head_source = self.class_to_head[fg_pred_cls]

        out_full = out_full.view(l, q, b, self.num_classes + 1)
        head_source = head_source.view(l, q, b)
        for k in list(raw_logits.keys()):
            raw_logits[k] = raw_logits[k].view(l, q, b, -1)

        return {
            "full_logits": out_full,
            "head_source_idx": head_source,
            "raw_split_logits": raw_logits,
        }

@TRANSFORMER_REGISTRY.register()
class IterativeRelationTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, cfg=None, **kwargs):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        layer_norm = nn.LayerNorm(d_model)
        relation_layer_norm = nn.LayerNorm(d_model)

        self.decoder = IterativeRelationDecoder(decoder_layer, num_decoder_layers, layer_norm, relation_layer_norm,
                                                      return_intermediate=return_intermediate_dec, d_model=d_model,
                                                    nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)

        self.d_model = d_model
        self.nhead = nhead
        self.obj_split_enabled = bool(getattr(getattr(getattr(cfg, "MODEL", None), "OBJ_SPLIT", None), "ENABLED", False))
        if self.obj_split_enabled:
            self.object_embed = None
            self.split_object_classifier = SplitObjectClassifier(
                d_model=d_model,
                num_classes=kwargs['num_classes'],
                cfg=cfg,
            )
        else:
            self.object_embed = nn.Linear(d_model, kwargs['num_classes'] + 1)
            self.split_object_classifier = None
        self.object_bbox_coords = MLP(d_model, d_model, 4, 3)

        self.relation_embed = nn.Linear(d_model, kwargs['num_relation_classes'] + 1)


        self.num_relation_classes = kwargs['num_relation_classes']
        self.num_object_classes = kwargs['num_classes']


        self._reset_parameters()
        for layer in range(self.decoder.num_layers - 1):
            nn.init.constant_(self.decoder.subject_graph_query_residual[layer].weight, 0)
            nn.init.constant_(self.decoder.subject_graph_query_residual[layer].bias, 0)

            nn.init.constant_(self.decoder.object_graph_query_residual[layer].weight, 0)
            nn.init.constant_(self.decoder.object_graph_query_residual[layer].bias, 0)

            nn.init.constant_(self.decoder.relation_graph_query_residual[layer].weight, 0)
            nn.init.constant_(self.decoder.relation_graph_query_residual[layer].bias, 0)

        for layer in range(self.decoder.num_layers):
            nn.init.constant_(self.decoder.object_pos_linear[layer].weight, 0)
            nn.init.constant_(self.decoder.object_pos_linear[layer].bias, 0)
            nn.init.constant_(self.decoder.relation_pos_linear[layer].weight, 0)
            nn.init.constant_(self.decoder.relation_pos_linear[layer].bias, 0)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, subject_embed, object_embed, relation_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        if subject_embed.dim() == 2:
            subject_query_embed = subject_embed.unsqueeze(1).repeat(1, bs, 1)
        else:
            subject_query_embed = subject_embed
        if object_embed.dim() == 2:
            object_query_embed = object_embed.unsqueeze(1).repeat(1, bs, 1)
        else:
            object_query_embed = object_embed
        if relation_embed.dim() == 2:
            relation_query_embed = relation_embed.unsqueeze(1).repeat(1, bs, 1)
        else:
            relation_query_embed = relation_embed

        # Condition on subject
        tgt_sub = torch.zeros_like(subject_query_embed)
        tgt_obj = torch.zeros_like(object_query_embed)
        tgt_rel = torch.zeros_like(relation_query_embed)


        hs_subject, hs_object, hs_relation = self.decoder(tgt_sub, tgt_obj, tgt_rel, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, subject_pos=subject_query_embed, object_pos=object_query_embed, relation_pos=relation_query_embed)
        if self.obj_split_enabled:
            subject_cls_outputs = self.split_object_classifier(hs_subject)
            object_cls_outputs = self.split_object_classifier(hs_object)
            relation_subject_class = subject_cls_outputs["full_logits"]
            relation_object_class = object_cls_outputs["full_logits"]
        else:
            relation_subject_class = self.object_embed(hs_subject)
            relation_object_class = self.object_embed(hs_object)
        relation_subject_coords = self.object_bbox_coords(hs_subject).sigmoid()
        relation_object_coords = self.object_bbox_coords(hs_object).sigmoid()
        relation_class = self.relation_embed(hs_relation)
        relation_coords = self.object_bbox_coords(hs_relation).sigmoid()

        output = {
            'relation_coords': relation_coords.transpose(1, 2),
            'relation_logits': relation_class.transpose(1, 2),
            'relation_subject_logits': relation_subject_class.transpose(1, 2),
            'relation_object_logits': relation_object_class.transpose(1, 2),
            'relation_subject_coords': relation_subject_coords.transpose(1, 2),
            'relation_object_coords': relation_object_coords.transpose(1, 2),
            # [num_layers, bs, num_queries, hidden_dim]
            'hs_subject': hs_subject.transpose(1, 2),
            'hs_object': hs_object.transpose(1, 2),
            'hs_relation': hs_relation.transpose(1, 2),
            # [bs, num_queries, hidden_dim]
            'hs_subject_last': hs_subject[-1].transpose(0, 1),
            'hs_object_last': hs_object[-1].transpose(0, 1),
            'hs_relation_last': hs_relation[-1].transpose(0, 1),
        }
        if self.obj_split_enabled:
            output['obj_split_subject_head_source_idx'] = subject_cls_outputs['head_source_idx'].transpose(1, 2)
            output['obj_split_object_head_source_idx'] = object_cls_outputs['head_source_idx'].transpose(1, 2)
            output['raw_split_logits_subject'] = {
                k: v.transpose(1, 2) for k, v in subject_cls_outputs['raw_split_logits'].items()
            }
            output['raw_split_logits_object'] = {
                k: v.transpose(1, 2) for k, v in object_cls_outputs['raw_split_logits'].items()
            }

        return output

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class IterativeRelationDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, relation_norm=None, return_intermediate=False, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.subject_layers = _get_clones(decoder_layer, num_layers)
        self.object_layers = _get_clones(decoder_layer, num_layers)
        self.relation_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.subject_norm = norm
        self.relation_norm = relation_norm
        self.return_intermediate = return_intermediate

        self.object_pos_attn = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout) for _ in range(num_layers)])
        self.object_pos_linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.object_pos_dropout= nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        self.relation_pos_attn = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=2*d_model, vdim=2*d_model) for _ in range(num_layers)])
        self.relation_pos_linear = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.relation_pos_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        self.subject_graph_query_attn = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=3*d_model, vdim=3*d_model) for _ in range(num_layers-1)])
        self.subject_graph_query_residual = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers-1)])
        self.subject_graph_query_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers-1)])
        self.subject_graph_query_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers-1)])

        self.object_graph_query_attn = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=3*d_model, vdim=3*d_model) for _ in range(num_layers-1)])
        self.object_graph_query_residual = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers-1)])
        self.object_graph_query_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers-1)])
        self.object_graph_query_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers-1)])

        self.relation_graph_query_attn = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=3*d_model, vdim=3*d_model) for _ in range(num_layers-1)])
        self.relation_graph_query_residual = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers-1)])
        self.relation_graph_query_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers-1)])
        self.relation_graph_query_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers-1)])

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def object_query_generator(self, object_query_embed, subject_features, subject_pos, layer):
        sub = self.with_pos_embed(subject_features, subject_pos)
        tgt = self.object_pos_attn[layer](object_query_embed, sub, value=subject_features)[0]
        tgt_residual = self.object_pos_linear[layer](tgt)
        object_query_embed = object_query_embed + self.object_pos_dropout[layer](tgt_residual)

        return object_query_embed

    def relation_query_generator(self, relation_query_embed, subject_features, object_features, subject_pos, object_pos, layer):
        sub = self.with_pos_embed(subject_features, subject_pos)
        obj = self.with_pos_embed(object_features, object_pos)

        k = torch.cat([sub, obj], -1)
        v = torch.cat([subject_features, object_features], -1)
        tgt = self.relation_pos_attn[layer](relation_query_embed, k, value=v)[0]
        tgt_residual = self.relation_pos_linear[layer](tgt)
        relation_query_embed = relation_query_embed + self.relation_pos_dropout[layer](tgt_residual)
        return relation_query_embed

    def forward(self, tgt_sub, tgt_obj, tgt_rel, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                subject_pos: Optional[Tensor] = None,
                object_pos: Optional[Tensor] = None,
                relation_pos: Optional[Tensor] = None):
        output_subject = tgt_sub
        output_object = tgt_obj
        output_relation = tgt_rel

        intermediate_relation = []
        intermediate_subject = []
        intermediate_object = []
        for layer_id in range(self.num_layers):
            sub_features = self.subject_layers[layer_id](output_subject, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=subject_pos)

            conditional_object_pos = self.object_query_generator(object_pos, sub_features, subject_pos, layer_id)
            obj_features = self.object_layers[layer_id](output_object, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=conditional_object_pos)

            conditional_relation_pos = self.relation_query_generator(relation_pos, sub_features, obj_features, subject_pos, conditional_object_pos, layer_id)
            rel_features = self.relation_layers[layer_id](output_relation, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=conditional_relation_pos)

            if self.return_intermediate:
                intermediate_subject.append(self.subject_norm(sub_features))
                intermediate_object.append(self.subject_norm(obj_features))
                intermediate_relation.append(self.relation_norm(rel_features))
            # import pdb;pdb.set_trace()
            rel_q = rel_features.shape[0]
            obj_q = obj_features.shape[0]
            multiply_q = rel_q//obj_q
            if layer_id != self.num_layers - 1:
                # Get queries for each decoder
                triplet_features = torch.cat([sub_features.repeat_interleave(multiply_q,0), \
                    obj_features.repeat_interleave(multiply_q,0), rel_features], -1)
                triplet_pos = torch.cat([subject_pos.repeat_interleave(multiply_q,0),\
                     conditional_object_pos.repeat_interleave(multiply_q,0),\
                     conditional_relation_pos], -1)
                triplet_features_pos = self.with_pos_embed(triplet_features, triplet_pos)

                subject_with_pos = self.with_pos_embed(sub_features, subject_pos)
                object_with_pos = self.with_pos_embed(obj_features, conditional_object_pos)
                relation_with_pos = self.with_pos_embed(rel_features, conditional_relation_pos)

                # Relation queries
                subject_graph_residual = self.subject_graph_query_residual[layer_id](self.subject_graph_query_attn[layer_id](subject_with_pos, triplet_features_pos, value=triplet_features)[0])
                output_subject = sub_features + self.subject_graph_query_norm[layer_id](self.subject_graph_query_dropout[layer_id](subject_graph_residual))

                object_graph_residual = self.object_graph_query_residual[layer_id](self.object_graph_query_attn[layer_id](object_with_pos, triplet_features_pos, value=triplet_features)[0])
                output_object = obj_features + self.object_graph_query_norm[layer_id](self.object_graph_query_dropout[layer_id](object_graph_residual))

                relation_graph_residual = self.relation_graph_query_residual[layer_id](self.relation_graph_query_attn[layer_id](relation_with_pos, triplet_features_pos, value=triplet_features)[0])
                output_relation = rel_features + self.relation_graph_query_norm[layer_id](self.relation_graph_query_dropout[layer_id](relation_graph_residual))


        if self.subject_norm is not None:
            sub_features = self.subject_norm(sub_features)
            obj_features = self.subject_norm(obj_features)
            rel_features = self.relation_norm(rel_features)
            if self.return_intermediate:
                intermediate_subject.pop()
                intermediate_subject.append(sub_features)
                intermediate_object.pop()
                intermediate_object.append(obj_features)
                intermediate_relation.pop()
                intermediate_relation.append(rel_features)

        if self.return_intermediate:
            return torch.stack(intermediate_subject), torch.stack(intermediate_object), torch.stack(intermediate_relation)

        return sub_features.unsqueeze(0), obj_features.unsqueeze(0), rel_features.unsqueeze(0)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(name, d_model, dropout, nhead, dim_feedforward, num_encoder_layers, num_decoder_layers, normalize_before, return_intermediate_dec, **kwargs):
    return TRANSFORMER_REGISTRY.get(name)(
        d_model=d_model,
        dropout=dropout,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        normalize_before=normalize_before,
        return_intermediate_dec=return_intermediate_dec,
        **kwargs
    )

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")