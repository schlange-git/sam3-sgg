# SAM 3 时序建模详解：记忆库、对象指针与跨帧传播

本文档详细论述 SAM 3 如何进行视频时序建模，包括架构设计、记忆库结构、数据流和核心伪代码。

---

## 1. 架构总览：解耦检测-跟踪双路径设计

SAM 3 的视频系统由两个共享同一 Vision Backbone（Perception Encoder）的独立模块组成：

```
                        概念提示 P
                     (文本短语 + 图像示例)
                            │
                            ▼
┌──────────────────────────────────────────────────────┐
│                  Perception Encoder                   │
│        ViT (446M, 32层, 1024-dim, 16头, RoPE)        │
│         输入: 1008×1008, patch 14×14                  │
│         输出: 多尺度特征金字塔 (72×72 feature map)       │
└──────────────────────┬───────────────────────────────┘
                       │ 共享视觉特征
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐     ┌──────────────────────┐
│    检测器 (Detector) │     │   跟踪器 (Tracker)     │
│   identity-agnostic │     │   identity-aware      │
│   找出"所有实例"      │     │   保持"同一实例"         │
│   DETR-style       │     │   Memory-Conditioned  │
│   + Presence Head  │     │   Transformer         │
└──────────────────┘     └──────────────────────┘
          │                         │
          ▼                         ▼
    每帧独立检测结果          跨帧一致的 masklet
   (class + box + mask)     (mask + obj_ptr + memory)
```

**核心分离原则：**
- **检测器**：回答"这一帧里有哪些匹配该概念的实例？"——无身份概念，每帧独立运行
- **跟踪器**：回答"这个实例在下一帧的哪个位置？"——跨帧维护身份一致性

---

## 2. 记忆库结构 (Memory Bank)

跟踪器的核心是一个**滑动窗口记忆库**，为每个被跟踪对象独立维护历史信息。默认容量 `num_maskmem = 7` 帧。

### 2.1 每帧记忆条目

| 字段 | 维度 | 说明 |
|---|---|---|
| `maskmem_features` | `(B, 64, H_mem, W_mem)` | 压缩的时空视觉记忆，由 SimpleMaskEncoder 编码 |
| `maskmem_pos_enc` | `List[(B, 64, H_i, W_i)]` | 多尺度 RoPE 空间位置编码 |
| `obj_ptr` | `(B, 256)` | 对象指针：紧凑的对象外观描述符，用于跨帧"重识别" |
| `pred_masks` | `(B, 1, H_low, W_low)` | 该帧的低分辨率预测掩码 |
| `object_score_logits` | `(B, 1)` | 对象是否存在于该帧的置信度逻辑值 |

### 2.2 两类记忆

```
记忆库 (最多 7 帧)
├─ Conditioning Frame Memories (t_pos = 0)
│   ├─ 来源：用户显式提供点击/框/掩码的帧
│   ├─ 策略：始终保留，不受 mf_threshold 过滤
│   └─ 意义：提供可靠的"锚点"，防止跟踪漂移
│
└─ Non-Conditioning Frame Memories (t_pos = 1..6)
    ├─ 来源：跟踪器自动传播的帧
    ├─ 策略：按置信度过滤 (cal_mem_score > mf_threshold)
    │         评估时按 stride 采样
    └─ 意义：节省计算，只用高质量帧做 Cross-Attention
```

### 2.3 时序位置编码 (maskmem_tpos_enc)

```python
# 可学习的参数：让模型知道每段记忆距离当前帧有多远
maskmem_tpos_enc = nn.Parameter(
    torch.randn(num_maskmem, 1, 1, mem_dim)  # (7, 1, 1, 64)
)

# 使用时直接加到记忆特征上
for i, mem_feat in enumerate(all_memories):
    mem_feat = mem_feat + maskmem_tpos_enc[i]
```

**设计意图**：不像 NLP 用三角函数式的位置编码，这里使用可学习参数。因为"距离当前帧 3 帧"和"距离当前帧 5 帧"在视觉跟踪中的语义不同，且只有 7 个离散位置，可学习编码更灵活。

---

## 3. 记忆编码管道 (Memory Encoding Pipeline)

### 3.1 核心函数：`_encode_new_memory()`

```python
def _encode_new_memory(
    self,
    current_vision_feats: Tensor,    # (B, C, H, W) backbone 特征
    pred_masks_high_res: Tensor,     # (B, 1, H_high, W_high) 预测掩码
    object_score_logits: Tensor,     # (B, 1) 对象存在分数
) -> Tuple[Tensor, List[Tensor]]:
    """
    将当前帧的预测掩码 + 视觉特征压缩为 64 维记忆表示。
    返回: (maskmem_features, maskmem_pos_enc)
    """
    # ── 阶段 1: Sigmoid 温度缩放 ──
    # 参数: sigmoid_scale=20.0, sigmoid_bias=-10.0
    # 将软掩码 logit 推向几乎二值化的范围
    # 目的: 训练时用布尔 GT 掩码，推理时用软掩码，
    #       温度缩放减少两者间的 domain shift
    mask_for_mem = torch.sigmoid(
        pred_masks_high_res * 20.0 + (-10.0)
    )
    # 效果: logit≈0.5 → sigmoid(0)=0.5 (不太确定)
    #       logit≈1.0 → sigmoid(10)≈1.0 (几乎确定是前景)
    #       logit≈-1.0 → sigmoid(-30)≈0.0 (几乎确定是背景)

    # ── 阶段 2: SimpleMaskDownSampler ──
    # 将掩码与图像特征逐步融合并降采样
    # 结构: stem Conv2d → N 个降采样 block → 输出 64-dim
    maskmem_out = self.maskmem_backbone(
        pix_feat=current_vision_feats,
        mask=mask_for_mem,
        skip_mask_sigmoid=True,
    )

    # ── 阶段 3: 提取输出 ──
    return (
        maskmem_out["vision_features"],   # (B, 64, H/16, W/16)
        maskmem_out["vision_pos_enc"],    # 多尺度 RoPE 位置编码
    )
```

### 3.2 SimpleMaskEncoder 内部结构

```
输入: (mask_pred, pix_feat)

Stage 0: Stem
  mask_proj = Conv2d(in_ch=1+3, out_ch=dim) + LayerNorm2d
  → (B, dim, H, W)

Stage 1..S: SimpleMaskDownSampler Blocks
  每个 block:
    ┌─ mask_down: Conv2d(dim, dim, stride=2) + LayerNorm2d
    └─ fuse: CXBlock (ConvNeXt-style)
         ├─ image_feat_proj: Conv2d(backbone_dim, dim)
         ├─ fused = mask_feat + image_feat_proj(pix_feat)
         ├─ bottleneck: Conv2d(dim, hidden_dim) → GELU → Conv2d(hidden_dim, dim)
         ├─ LayerNorm2d
         └─ residual: output = fused + bottleneck(fused)

  stride 累积 = 2^S (默认 total stride=16)
  最终输出: vision_features = (B, 64, H/16, W/16)
          vision_pos_enc = [每层的 RoPE 编码]
```

**为什么是 64 维？**
- 完整 backbone 特征通常 256-1024 维，但记忆库需要存 7 帧 × N 个对象
- 64 维是压缩比和质量之间的平衡：足够区分不同对象的外观，又不至于显存爆炸
- 也是 SAM 2 遗留下来的经过验证的设计

---

## 4. 记忆条件特征融合 (Memory-Conditioned Features)

### 4.1 核心函数：`_prepare_memory_conditioned_features()`

这是 SAM 3 时序建模的**核心**——让当前帧"看见"历史记忆。

```python
def _prepare_memory_conditioned_features(
    self,
    current_vision_feats: Tensor,      # (B, C, H, W) 当前帧特征
    vision_pos_enc: List[Tensor],      # 当前帧的空间位置编码
    feat_sizes: List[List[int]],       # 特征图尺寸列表
) -> Tensor:
    """
    从记忆库检索历史记忆，通过 Cross-Attention 与当前帧特征融合。
    """

    # ── 步骤 1: 选择 Conditioning 帧记忆 ──
    # _select_valid_memories() 找出所有用户提示帧
    # 这些帧获得 t_pos = 0，在 Cross-Attention 中永远有最高优先级
    cond_features, cond_pos_encs = self._select_cond_memories()

    # ── 步骤 2: 选择 Non-Conditioning 帧记忆 ──
    # 按 stride 从最近帧中采样
    # 质量过滤:
    #   cal_mem_score = sigmoid(object_score_logits) * iou_score
    #   如果 cal_mem_score < mf_threshold (0.01)，丢弃该帧
    noncond_features, noncond_pos_encs = (
        self._select_non_cond_memories()
    )

    # ── 步骤 3: 拼接 + 时序位置编码 ──
    all_memories = cond_features + noncond_features  # 最多 7 帧
    all_pos_encs = cond_pos_encs + noncond_pos_encs

    for i in range(len(all_memories)):
        # maskmem_tpos_enc[i]: (1, 1, 64) 可学习参数
        all_memories[i] = all_memories[i] + self.maskmem_tpos_enc[i]

    # ── 步骤 4: Transformer 融合 ──
    # 4 层 encoder-only Transformer
    # 每层:
    #   Self-Attention:  当前帧特征内部 (空间注意力)
    #   Cross-Attention: Q = 当前帧特征, K/V = 拼接的记忆库
    #   均使用 RoPE 位置编码
    fused = self.transformer(
        q=current_vision_feats,
        kv=all_memories,            # 拼接后: (B, 7*C_mem, H_mem, W_mem)
        q_pe=vision_pos_enc,
        kv_pe=all_pos_encs,
    )

    return fused  # (B, C, H, W)  与输入同维度
```

### 4.2 Cross-Attention 中的数据流动

```
当前帧 (t)                    记忆库 (t-6, t-4, t-2, t-1, ...)
──────────                    ──────────────────────────

Q: "这一帧的 (x,y) 位置           K/V: "过去某帧相同空间区域
    有什么特征？"                      是什么样的特征？"

具体计算:
  Q = current_feats · W_Q      # 来自当前帧
  K = memory_feats · W_K       # 来自记忆库中所有帧
  V = memory_feats · W_V       # 来自记忆库中所有帧

  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

RoPE 增强:
  Q_rot = apply_rotary_pos_emb(Q, spatial_coords)
  K_rot = apply_rotary_pos_emb(K, spatial_coords)
  → 使注意力能显式编码空间位置关系

时序信息增强:
  K += maskmem_tpos_enc[frame_position]
  → 使注意力能区分"3 帧前的记忆"和"1 帧前的记忆"

输出:
  fused_feats = Attention(Q_rot, K_rot+时序, V)
  → 当前帧的每个位置都"看见了"历史上该对象的外观变化
```

### 4.3 RoPE Attention 实现

```python
class RoPEAttention(nn.Module):
    """
    使用 Rotary Position Embeddings 的注意力机制。
    SAM 3 中 embedding_dim=256, num_heads=1。
    """

    def __init__(self, embedding_dim, num_heads, rope_theta=10000.0):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.rope = RotaryPositionEmbedding(
            dim=embedding_dim // num_heads,
            theta=rope_theta,          # 10000.0
        )

    def forward(self, q, k, v, q_pe=None, k_pe=None):
        # ── 应用轴向 RoPE ──
        # compute_axial_cis() 生成沿 H 和 W 轴的复数旋转因子
        q_rope = apply_rotary_enc(q, compute_axial_cis(q.shape))
        k_rope = apply_rotary_enc(k, compute_axial_cis(k.shape))

        # ── 可选降采样 (Cross-Attention 优化) ──
        if self.attention_downsample_rate > 1:
            k_rope = downsample(k_rope, self.attention_downsample_rate)
            v = downsample(v, self.attention_downsample_rate)

        # ── 注意力计算 ──
        # 支持 Flash Attention (flashattn / fa3)
        output = scaled_dot_product_attention(q_rope, k_rope, v)

        return output
```

**为什么 num_heads=1？**
单头注意力在 SAM 3 的特定分辨率下已被验证足够，减少计算开销。跟踪任务的空间模式相对简单（"同一物体在邻近位置"），多头不一定带来增益。

---

## 5. 单帧追踪：完整 track_step 流程

### 5.1 伪代码

```python
def track_step(
    self,
    frame_idx: int,
    video_frame: Tensor,          # (3, H, W) 单帧 RGB
) -> TrackingOutput:
    """
    对视频第 t 帧执行单步跟踪。
    此函数被 propagate_in_video 的 for 循环逐帧调用。
    """

    # ═══════════════════════════════════════════════════
    # 阶段 A: 特征提取 (共享 Perception Encoder)
    # ═══════════════════════════════════════════════════
    current_feats, pos_enc, feat_sizes = self.pe(video_frame)
    # current_feats: 多尺度特征金字塔
    # pos_enc:      各层 RoPE 位置编码
    # feat_sizes:   [(H_i, W_i) for each level]

    # ═══════════════════════════════════════════════════
    # 阶段 B: 记忆条件融合
    # ═══════════════════════════════════════════════════
    conditioned_feats = self._prepare_memory_conditioned_features(
        current_vision_feats=current_feats,
        vision_pos_enc=pos_enc,
        feat_sizes=feat_sizes,
    )
    # conditioned_feats: 融合了历史记忆的当前帧特征
    # 维度与 current_feats 相同
    # 这是整个时序建模的核心输出

    # ═══════════════════════════════════════════════════
    # 阶段 C: SAM Heads 预测
    # ═══════════════════════════════════════════════════
    # 沿袭 SAM 2 的 PromptEncoder + MaskDecoder 设计
    masks, iou_pred, obj_score = self._forward_sam_heads(
        conditioned_feats,
        prompt_embeddings=self.prev_prompt_embeddings,
    )
    # masks:     (3, H, W) 三个候选掩码
    # iou_pred:  (3,)  每个候选掩码的 IoU 预测
    # obj_score: (1,)  该对象是否存在于当前帧

    # 选择 IoU 最高的掩码
    best_idx = iou_pred.argmax()
    best_mask = masks[best_idx]

    # ═══════════════════════════════════════════════════
    # 阶段 D: 对象指针提取
    # ═══════════════════════════════════════════════════
    # SAM MaskDecoder 的输出包含一个 output_token
    # 这个 token 编码了"这是什么对象"的紧凑信息
    sam_output_token = self.last_mask_decoder_output["output_token"]
    obj_ptr = self.obj_ptr_proj(sam_output_token)  # MLP 投影到 (256,)

    # 对象不存在时使用"空指针"
    if obj_score <= 0:
        obj_ptr = self.obj_ptr_no_obj  # nn.Parameter, (256,)

    # ═══════════════════════════════════════════════════
    # 阶段 E: 编码新记忆 (为下一帧准备)
    # ═══════════════════════════════════════════════════
    maskmem_features, maskmem_pos_enc = self._encode_new_memory(
        current_vision_feats=current_feats,
        pred_masks_high_res=best_mask,
        object_score_logits=obj_score,
    )

    # ═══════════════════════════════════════════════════
    # 阶段 F: 返回结果 (调用方负责更新记忆库)
    # ═══════════════════════════════════════════════════
    return TrackingOutput(
        masks=masks,
        best_mask=best_mask,
        iou_pred=iou_pred,
        object_score_logits=obj_score,
        obj_ptr=obj_ptr,
        maskmem_features=maskmem_features,
        maskmem_pos_enc=maskmem_pos_enc,
    )
```

### 5.2 数据流图

```
帧 t 输入
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Perception Encoder (ViT, RoPE, 窗口注意力)             │
│   输出: 多尺度特征金字塔                                │
└──────────────────────┬──────────────────────────────┘
                       │ current_feats
                       ▼
┌─────────────────────────────────────────────────────┐
│ _prepare_memory_conditioned_features()               │
│                                                      │
│   记忆库 (来自帧 t-1, t-2, ...)                       │
│   ┌────────────────────────────────────┐             │
│   │ maskmem_features (64-dim)          │             │
│   │ maskmem_pos_enc (多尺度 RoPE)       │──────┐      │
│   │ maskmem_tpos_enc (可学习时序编码)    │      │      │
│   └────────────────────────────────────┘      │      │
│                                                 │      │
│   ┌────────────────────────────────────┐      │      │
│   │ 4-Layer Transformer Encoder        │      │      │
│   │   Self-Attn(Q=current)             │◄─────┘      │
│   │   Cross-Attn(Q=current, K/V=mem)   │             │
│   │   + RoPE + Flash Attention         │             │
│   └────────────────────┬───────────────┘             │
└────────────────────────┼─────────────────────────────┘
                         │ conditioned_feats
                         ▼
┌─────────────────────────────────────────────────────┐
│ SAM Heads (PromptEncoder + MaskDecoder)              │
│   输出: masks(3候选), iou_pred, obj_score            │
│   额外输出: sam_output_token (用于生成 obj_ptr)        │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    ┌─────────────┐         ┌──────────────────┐
    │ obj_ptr_proj│         │ _encode_new_memory│
    │ (256-dim)   │         │ → maskmem_features│
    │             │         │   (64-dim)        │
    └──────┬──────┘         └────────┬─────────┘
           │                         │
           └────────┬────────────────┘
                    │
                    ▼
            写入记忆库 (FIFO, 最多 7 帧)
                    │
                    ▼
            帧 t+1 的 Cross-Attention 将使用这些记忆
```

---

## 6. 视频级传播：propagate_in_video 完整流程

### 6.1 主循环

```python
def propagate_in_video(
    video_frames: List[Image],        # T 帧视频
    concept_prompt: ConceptPrompt,    # 文本 + 图像示例
) -> List[Masklet]:
    """
    完整视频传播：检测器启动 + 跟踪器逐帧传播 + 状态管理。
    """

    # ═══════════════════════════════════════════
    # 初始化：第 0 帧
    # ═══════════════════════════════════════════
    I_0 = video_frames[0]

    # 检测器输出该概念的所有实例
    detections_0 = detector(I_0, concept_prompt)
    # detections_0 = [
    #   {"mask": M, "bbox": B, "score": S, "class": C}
    #   for each detected instance
    # ]

    # 为每个检测到的对象创建 masklet
    # (spatio-temporal mask sequence)
    masklets = [
        Masklet(
            id=i,
            masks=[det["mask"]],         # 第 0 帧的掩码
            obj_ptrs=[],                  # 对象指针 (后续填充)
            scores=[det["score"]],
            state=MaskletState.UNCONFIRMED,  # 初始为 "待验证"
        )
        for i, det in enumerate(detections_0)
    ]

    # 初始化记忆库 (每个对象独立)
    memory_banks = {m.id: MemoryBank() for m in masklets}

    # 编码第 0 帧记忆
    for m in masklets:
        mem_feat, mem_pos = tracker._encode_new_memory(
            I_0_feats, m.masks[0], m.scores[0]
        )
        memory_banks[m.id].append(mem_feat, mem_pos)

    # ═══════════════════════════════════════════
    # 逐帧传播：第 1 帧到第 T-1 帧
    # ═══════════════════════════════════════════
    for t in range(1, len(video_frames)):
        I_t = video_frames[t]

        # ── 阶段 A: 跟踪器传播 ──
        for m in active_masklets(masklets):
            # 用记忆库中该对象的历史记忆做 Cross-Attention
            tracking_out = tracker.track_step(
                frame_idx=t,
                video_frame=I_t,
                memory_bank=memory_banks[m.id],  # 该对象的记忆
            )
            # 记录跟踪结果
            m.tracked_masks[t] = tracking_out.best_mask
            m.tracked_scores[t] = tracking_out.object_score_logits

        # ── 阶段 B: 检测器重新检测 ──
        # 周期性运行以发现新出现/丢失后重现的对象
        if t % recondition_every_nth_frame == 0:
            detections_t = detector(I_t, concept_prompt)

        # ── 阶段 C: 匹配与状态更新 ──
        for m in masklets:
            # 匹配：跟踪结果 vs 检测结果 (IoU 匹配)
            match_result = match_track_with_detection(
                m.tracked_masks[t], detections_t
            )

            if match_result.matched:
                m.scores[t] = max(
                    m.tracked_scores[t],
                    match_result.detection.score
                )
                # 可选：用检测掩码重新初始化跟踪
                if match_result.detection.score > 0.8:
                    m.masks[t] = match_result.detection.mask  # 替代跟踪掩码
                    m.state = MaskletState.ACTIVE
            else:
                m.unmatched_count += 1

            # 时序消歧义
            m = apply_temporal_disambiguation(m, t)

        # ── 阶段 D: 更新记忆库 ──
        for m in active_masklets(masklets):
            # 置信度门控：只有 cal_mem_score > mf_threshold 才写入
            cal_score = compute_cal_mem_score(
                m.tracked_scores[t],
                m.tracked_ious[t],
            )
            if cal_score > mf_threshold:
                new_mem = tracker._encode_new_memory(
                    I_t_feats,
                    m.tracked_masks[t],
                    m.tracked_scores[t],
                )
                memory_banks[m.id].append(new_mem)
                # FIFO 淘汰：超过 num_maskmem 帧时移除最旧的

        # ── 阶段 E: 生成新 masklet (新检测到的对象) ──
        for det in unmatched_detections(detections_t):
            new_m = Masklet(
                id=next_id(),
                first_frame=t,
                state=MaskletState.UNCONFIRMED,
            )
            masklets.append(new_m)
            memory_banks[new_m.id] = MemoryBank()

    return masklets
```

### 6.2 masklet 状态机

```
                    ┌─────────────┐
    新检测 ──────→  │ UNCONFIRMED │  等待 hotstart_delay 帧验证
                    └──────┬──────┘
                           │ hotstart_delay 帧内匹配成功 ≥ K 次
                           ▼
                    ┌─────────────┐
                    │   ACTIVE    │  正常跟踪
                    └──────┬──────┘
                           │ 连续 unmatched 超过阈值
                           ▼
                    ┌─────────────┐
                    │  OCCLUDED   │  暂时遮挡，保留记忆
                    └──────┬──────┘
                           │ 超过 max_trk_keep_alive 帧仍未匹配
                           ▼
                    ┌─────────────┐
                    │    DEAD     │  删除 masklet，释放记忆库
                    └─────────────┘
```

---

## 7. 时序消歧义机制

### 7.1 关注确认延迟 (Hotstart)

```python
def apply_temporal_disambiguation(masklet, current_frame):
    """
    对新检测的对象，观察 hotstart_delay 帧再决定是否保留。
    防止 false positive 污染跟踪结果。
    """

    # ── 确认延迟 ──
    if masklet.state == MaskletState.UNCONFIRMED:
        frames_since_birth = current_frame - masklet.first_frame
        if frames_since_birth >= hotstart_delay:  # 默认 15 帧
            # 检查 hotstart 期间的匹配率
            match_rate = masklet.matched_frames / frames_since_birth
            if match_rate >= hotstart_unmatch_thresh:  # 默认 8/15
                masklet.state = MaskletState.ACTIVE
            else:
                masklet.state = MaskletState.DEAD

    # ── 死亡判断 ──
    if masklet.unmatched_count > max_trk_keep_alive:  # 默认 30 帧
        masklet.state = MaskletState.DEAD

    # ── 重复遮罩抑制 ──
    # 同一概念的两个 masklet 互相重叠超过阈值 → 保留高分者
    if masklet.max_iou_with_other > suppress_overlapping_threshold:  # 0.7
        if masklet.score < other_masklet.score:
            masklet.state = MaskletState.DEAD

    return masklet
```

### 7.2 非重叠约束

```python
def _apply_non_overlapping_constraints(masks: Tensor, scores: Tensor):
    """
    每个像素只能属于一个对象 (取最高分)。
    被抑制的对象 logit 被 clamp 到 -10.0。
    """
    # 找出每个像素的最高分对象索引
    best_per_pixel = scores.argmax(dim=0)  # (H, W)

    # 对非最优对象，将该像素的 logit 压低
    for obj_idx in range(len(masks)):
        suppressed_pixels = (best_per_pixel != obj_idx)
        masks[obj_idx][suppressed_pixels] = -10.0
        # -10.0 经过 sigmoid 后 ≈ 4.5e-5，实际为零

    return masks
```

### 7.3 周期性重条件化

```python
# recondition_every_nth_frame = 16 (默认)

if frame_idx % 16 == 0:
    # 在当前帧运行检测器
    dets = detector(current_frame, concept_prompt)

    for masklet in active_masklets:
        # 找出与该 masklet 当前位置重叠且高置信度的检测
        match = find_best_match(masklet.current_mask, dets)
        if match is not None and match.det_score > 0.8:
            # 用检测掩码替换跟踪掩码，重新初始化
            # 防止长时间跟踪的 drift 积累
            masklet.masks[frame_idx] = match.det_mask
            # 清除该 masklet 记忆库中的 non-cond 记忆
            memory_banks[masklet.id].clear_non_cond()
```

---

## 8. 完整数据流总结

```
时间轴 →

帧 0           帧 1            帧 2            ...     帧 T
──────────────────────────────────────────────────────────

检测器          跟踪器           跟踪器                   跟踪器
  │               │               │                       │
  ├─ PE(I₀)       ├─ PE(I₁)       ├─ PE(I₂)               ├─ PE(I_T)
  ├─ DETR        ├─ CrossAttn    ├─ CrossAttn            ├─ CrossAttn
  │               │  (Q=I₁_feat,  │  (Q=I₂_feat,          │  (Q=I_T_feat,
  │               │   K/V=mem₀)   │   K/V=mem₀+mem₁)      │   K/V=mem₀..mem_T-1)
  │               ├─ MaskDecoder  ├─ MaskDecoder           ├─ MaskDecoder
  │               ├─ obj_ptr₁     ├─ obj_ptr₂              ├─ obj_ptr_T
  │               ├─ maskmem₁     ├─ maskmem₂              ├─ maskmem_T
  │               │               │                       │
  ▼               ▼               ▼                       ▼
检测结果₀       跟踪结果₁        跟踪结果₂                跟踪结果_T
  │               │               │                       │
  └──→ 记忆库 →──┘               │                       │
         (mem₀)                   │                       │
         └──────────→ 记忆库 →───┘                       │
                      (mem₀, mem₁)                        │
                      └──────────→ ... → 记忆库 →────────┘
                                   (FIFO, max 7 entries)

记忆库数据流 (以单个对象为例):

  t=0:  B = [mem₀]                    容量: 1/7
  t=1:  B = [mem₀, mem₁]              容量: 2/7
       Cross-Attention 见到的 K/V: mem₀, mem₁
  ...
  t=7:  B = [mem₀, mem₁, ..., mem₇]   容量: 7/7 (满)
       Cross-Attention 见到的 K/V: 所有 7 帧
  t=8:  B = [mem₁, mem₂, ..., mem₈]   容量: 7/7
       FIFO 淘汰 mem₀，加入 mem₈
       Cross-Attention 见到的 K/V: mem₁..mem₈
```

---

## 9. 与本项目的关联分析

### 9.1 对象指针 ↔ Quality Aux 豁免

| SAM 3 中的概念 | SGG 中的对应 | 相似性 |
|---|---|---|
| Object Pointer (256-dim 紧凑描述符) | 检测头的 softmax score | 都编码了"这个预测有多可信" |
| 记忆库只保留 `cal_mem_score > mf_threshold` 的帧 | Quality Aux 只豁免 `IoU > threshold 且 score > threshold` 的 query | 都是置信度门控：只有高质量信号才被保留/豁免 |
| obj_ptr_no_obj 学得的"空指针" | CE ignore_index (-100) | 都是"这里没有有效信号"的标记 |

### 9.2 Presence Head ↔ 解耦检测与谓词

```
SAM 3: final_score = presence_score × localization_score
       解耦"概念是否存在"与"实例在哪里"
       → 避免错误的存在判断压低正确的定位

SGG:   triplet_cost = cost(sub) + cost(obj) + cost(pred)
       Quality Aux: 检测正确的 bg query 不参与 CE 惩罚
       → 解耦"检测是否对"与"谓词是否对"
       → 避免错误的谓词判断惩罚正确的检测
```

### 9.3 Confidence-Gated Memory Update ↔ START_ITER Gate

```
SAM 3:    只有 cal_mem_score > mf_threshold 的帧才写入记忆库
          原因：低质量记忆会污染 Cross-Attention，导致跟踪漂移

Quality Aux: 只在 iter > START_ITER 时激活 (训练过半之后)
             原因：训练初期检测不准确，此时豁免会误放错误检测
```

### 9.4 Reconditioning ↔ Dual Scoring

```
SAM 3 Periodic Reconditioning (每 16 帧用检测器矫正跟踪器)
  → 目的：防止长时间跟踪的 drift 积累

SGG Dual Scoring (TRIPLET_CONF_ALPHA)
  → 目的：用 relation head 的谓词置信度矫正检测分数的偏差
  → final_score = det_score × (1-α + α × triplet_conf)
```

---

## 参考文献

- **SAM 3 论文**: SAM 3: Segment Anything with Concepts. https://openreview.net/forum?id=r35clVtGzw
- **SAM 3 GitHub**: https://github.com/facebookresearch/sam3
- **DeepWiki - SAM 3 Tracker Architecture**: https://deepwiki.com/facebookresearch/sam3/4.2-tracker-architecture
- **DeepWiki - SAM 3 Tracker Component**: https://deepwiki.com/facebookresearch/sam3/5.6-tracker-component
- **DeepWiki - SAM 3 Model Architecture**: https://deepwiki.com/facebookresearch/sam3/5-model-architecture-deep-dive
- **DeepWiki - SAM 3 Video Segmentation**: https://deepwiki.com/facebookresearch/sam3/4-video-segmentation
- **EfficientSAM3**: https://arxiv.org/abs/2511.15833
