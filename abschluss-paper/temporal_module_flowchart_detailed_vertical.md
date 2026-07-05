# 详细竖版流程图：Temporal Triplet Memory（不含 Tracking Mask）

本图基于当前代码库实现重新梳理，重点对应 `SpeaQ/modeling/transformer/detr.py` 与 `SpeaQ/modeling/temporal/triplet_memory.py` 的真实数据流。图中不包含 SAM3 tracking mask 分支，也不展开 ROI refinement、quality auxiliary loss 等非时序主线。

## Figure: Detailed Vertical Flowchart

```mermaid
flowchart TB
    %% =========================
    %% Main frame path
    %% =========================
    subgraph S0[Input and single-frame SGG backbone]
        A0[Video mini-batch / same-video clip sampling] --> A1[Current frame image I_t]
        A0 --> A2[Metadata: video_id, frame_idx]
        A1 --> B0[SAM3 / X-SAM visual backbone\npatch-merge feature extraction]
        B0 --> B1[Feature map + padding mask + position embedding]
        B1 --> C0[IterativeRelationTransformer\nencoder + decoder]
        C0 --> C1[Decoder hidden states\nhs_subject_last, hs_object_last, hs_relation_last]
        C1 --> C2[Initial prediction tensors\nsubject/object/relation logits and boxes]
    end

    %% =========================
    %% Memory read path
    %% =========================
    subgraph S1[Per-video triplet memory read]
        A2 --> M0[TripletMemoryManager.get_batch_memory]
        M0 --> M1[maybe_clear_on_video_jump\nclear bank if frame_idx regresses]
        M1 --> M2[Select or create TripletMemoryBank for video_id]
        M2 --> M3[Collect valid slots\nvalid=True and feat exists]
        M3 --> M4{Any valid memory slots?}
        M4 -- No --> M5[Skip temporal injection\nkeep original decoder output]
        M4 -- Yes --> M6[Stack slot features\nfeat: M x 128]
        M6 --> M7[Add delta-time bucket embedding\nΔt = current frame_idx - slot frame_idx]
        M7 --> M8[Score-weight memory features\nfeat = feat * slot_score]
        M8 --> M9[Pad batch memory\nmemory: B x M_max x 128\nmask: B x M_max]
    end

    %% =========================
    %% Gate and injection path
    %% =========================
    subgraph S2[Gated cross-attention injection]
        G0[triplet_iter_counter += 1] --> G1[get_temporal_gate]
        G1 --> G2[Gate schedule\n0 before 10% max_iter\nlinear warmup 10%-30%\nthen gate_max]
        G2 --> G3[Object gate max = 0.15\nRelation gate max = 0.30]
        M9 --> I0[TemporalTripletInjector]
        C1 --> I0
        G3 --> I0
        I0 --> I1[Object branch injection if enabled\nQ_obj* = Q_obj + gate_obj * CrossAttn(Q_obj, memory)]
        I0 --> I2[Relation branch injection if enabled\nQ_rel* = Q_rel + gate_rel * CrossAttn(Q_rel, memory)]
        I1 --> I3[Recompute final object logits and boxes\nreplace only last decoder layer]
        I2 --> I4[Recompute final relation logits and boxes\nreplace only last decoder layer]
        I3 --> I5[Temporally enhanced output]
        I4 --> I5
        M5 --> I5
    end

    %% =========================
    %% Output/loss path
    %% =========================
    subgraph S3[Current-frame prediction and training objective]
        I5 --> P0[Optional downstream modules\nROI refine / obj split / aux outputs]
        P0 --> P1[Assemble final out dict\nrelation boxes, logits, subject/object boxes]
        P1 --> P2[Scene graph losses and evaluation outputs\ncurrent-frame supervision]
    end

    %% =========================
    %% Candidate construction
    %% =========================
    subgraph S4[Memory candidate construction under no_grad]
        P1 --> U0[Triplet memory update block\ntraining or temporal_eval]
        U0 --> U1[Read final outputs\nrelation logits, subject/object logits, boxes, hs_relation_last]
        U1 --> U2[Compute probabilities\nsoftmax subject/object/relation]
        U2 --> U3[Choose labels and scores\nsub_score, obj_score, pred_score]
        U3 --> U4[Triplet quality\nq = sub_score * obj_score * pred_score]
        U4 --> U5[Top-k selection\nK = TRIPLET_MEMORY_TOPK_UPDATE, default 16]
        U5 --> U6[Threshold filtering\nq >= UPDATE_SCORE_THRESH, default 0.10]
        U6 --> U7[Build candidate signature\n(sub_label, pred_label, obj_label)]
        U7 --> U8[Convert subject/object boxes\ncxcywh -> xyxy, clamp to [0,1]]
        U8 --> U9[Make union box]
        U9 --> U10[TripletMemoryEncoder]
        U10 --> U11[Encode memory feature from\nrelation query + boxes + relative geometry + predicate distribution]
        U11 --> U12[Candidate dict\nsignature, feat, boxes, score components]
    end

    %% =========================
    %% Memory update
    %% =========================
    subgraph S5[Per-video memory write]
        U12 --> W0[TripletMemoryManager.update_batch]
        A2 --> W0
        W0 --> W1[For each video bank: update candidates]
        W1 --> W2[Find matching slot\nsame signature + mean IoU(sub,obj,union) > 0.3]
        W2 --> W3{Matched existing slot?}
        W3 -- Yes --> W4[EMA update\nfeat and boxes momentum = 0.9\nscore = max(history, current)\nmiss = 0, age += 1]
        W3 -- No --> W5{Bank has free slot?}
        W5 -- Yes --> W6[Insert new slot]
        W5 -- No --> W7[Replace weakest slot\nmin score - 0.05 * miss]
        W4 --> W8[Expire unmatched slots]
        W6 --> W8
        W7 --> W8
        W8 --> W9[miss += 1 for unmatched\nset invalid if miss > 2]
        W9 --> M2
    end

    %% =========================
    %% Styling
    %% =========================
    classDef input fill:#f7f7f7,stroke:#333,stroke-width:1px;
    classDef backbone fill:#e8f1ff,stroke:#1d4e89,stroke-width:1px;
    classDef memory fill:#fff4df,stroke:#b36b00,stroke-width:1px;
    classDef inject fill:#e9f8ee,stroke:#2a7f3f,stroke-width:1px;
    classDef pred fill:#f3e8ff,stroke:#6f42c1,stroke-width:1px;
    classDef update fill:#fce8e6,stroke:#b3261e,stroke-width:1px;
    classDef decision fill:#ffffff,stroke:#555,stroke-width:1px,stroke-dasharray: 4 2;

    class A0,A1,A2 input;
    class B0,B1,C0,C1,C2 backbone;
    class M0,M1,M2,M3,M6,M7,M8,M9,W0,W1,W2,W4,W6,W7,W8,W9 memory;
    class G0,G1,G2,G3,I0,I1,I2,I3,I4,I5 inject;
    class P0,P1,P2 pred;
    class U0,U1,U2,U3,U4,U5,U6,U7,U8,U9,U10,U11,U12 update;
    class M4,W3,W5 decision;
```

## 图注建议

**Figure X. Detailed vertical flowchart of the Temporal Triplet Memory module without the tracking-mask branch.** The model first produces single-frame subject, object, and relation query features through the SAM3/X-SAM backbone and the iterative relation transformer. For each sample, a video-specific triplet memory bank is indexed by `video_id` and `frame_idx`; valid memory slots are temporally encoded, score-weighted, padded across the batch, and injected into the final object and relation decoder features through gated cross-attention. Only the final decoder-layer predictions are recomputed after injection. After current-frame prediction and loss construction, high-quality triplet candidates are built under `torch.no_grad()` and written back to the corresponding video bank by signature-and-IoU matching, EMA update, insertion, replacement, and expiration.

## 中文图注

**图 X. 不含 tracking mask 分支的三元组时序记忆模块详细竖版流程。** 当前帧首先经过 SAM3/X-SAM 主干和迭代关系 Transformer，得到主体、客体与关系查询特征及初始预测。随后模型根据 `video_id` 和 `frame_idx` 读取对应视频的三元组记忆库，将有效记忆槽加入时间差编码和置信度加权后，以门控交叉注意力注入最终层的客体查询与关系查询，并仅重算最终层预测。当前帧输出完成后，模型在 `torch.no_grad()` 下从最终预测构造高质量三元组候选，并通过类别签名与空间 IoU 匹配执行 EMA 更新、插入、替换和过期清理，从而形成下一帧可读取的历史上下文。

## 代码对应关系

- `SpeaQ/modeling/transformer/detr.py`：初始化 `TripletMemoryManager`、`TripletMemoryEncoder`、`TemporalTripletInjector`；调用 `_apply_triplet_memory_to_output()`；在 forward 末尾执行候选构造和 `update_batch()`。
- `SpeaQ/modeling/temporal/triplet_memory.py`：定义 memory slot、per-video bank、delta-time embedding、triplet encoder、cross-attention injector、gate schedule 和 bank update 规则。
- `configs/defaults.py`：定义 `MODEL.TEMPORAL.TRIPLET_MEMORY_*`、`INJECT_OBJECT`、`INJECT_RELATION`、gate warmup、update threshold、EMA momentum、IoU threshold 等配置。
