# SAM 3 with SGG head 

## 0) 你这一阶段建议采用的任务设置：PredCls（最稳）

因为 **SAM3 不输出 VG/AG 的 object 类别**，所以第一阶段最稳的是做 **Predicate Classification（PredCls）**：

* **输入**：GT boxes + GT object labels（来自 VG/AG 标注）
* **模型预测**：predicate（关系类别）
* **你用 SAM3 提供**：每个 GT object 的**mask embedding / mask**（通过匹配获得）

这样你可以**只训练 relation head**，避免额外做 object classifier，评估也最清楚。

---

## 1) Pipeline 总览（训练时）

1. DataLoader 读一张图像 `img` 与标注：

   * `gt_boxes: [G,4]`（xyxy）
   * `gt_obj_labels: [G]`
   * `gt_rels: [R,3]`，每行 `(s_idx, o_idx, pred_label)`（索引指向 GT objects）
2. 冻结 SAM3 推理得到 proposals：

   * `pred_masks: [N,H,W]`
   * `pred_embs:  [N,D]`（mask/query embedding）
   * 可选：`pred_scores: [N]`
3. 把每个 `pred_mask` 转成 `pred_box`（mask->bbox）
4. **匹配**：把 SAM3 proposals 与 `gt_boxes` 做 Hungarian 或 greedy IoU 匹配

   * 得到 `gt_to_pred[g] = n` 或 `-1`
5. 为每个 GT object 取到对应的 SAM3 embedding：

   * `e_g = pred_embs[gt_to_pred[g]]`
6. 构造 pair 样本（正负）：

   * 对所有 ordered pairs `(i,j), i!=j`：

     * 若在 `gt_rels` 中存在 predicate：label=pred
     * 否则：label=BG（背景/无关系）
   * 负样本下采样，保持正负比（例如 1:3 或 1:5）
7. 对每个 pair 计算特征 `z_ij = concat(e_i, e_j, geom(i,j), optional pooled_feat)`
8. Relation head 输出 `logits_ij -> predicate_class`
9. Loss：CrossEntropy（可加 class weight）
10. 反向传播：只更新 relation head（和可选 adapter），SAM3 全程 no_grad

---

## 2) 关键：pair 的几何特征（建议固定一套，可复用）

用 bbox（来自 GT 或 mask bbox 都行；PredCls 阶段推荐直接用 GT bbox）：

```text
$$
c_x = (x_1 + x_2)/2,\quad c_y = (y_1 + y_2)/2,\quad w = x_2-x_1,\quad h = y_2-y_1
$$

$$
dx = (c_x^s - c_x^o) / (w^o + \epsilon),\quad
dy = (c_y^s - c_y^o) / (h^o + \epsilon)
$$

$$
dw = \log\left(\frac{w^s+\epsilon}{w^o+\epsilon}\right),\quad
dh = \log\left(\frac{h^s+\epsilon}{h^o+\epsilon}\right)
$$

$$
iou = IoU(box_s, box_o),\quad
da = \log\left(\frac{w^s h^s+\epsilon}{w^o h^o+\epsilon}\right)
$$
```

再加一个 mask overlap（如果你愿意用 SAM3 mask）：

* `mask_iou = IoU(mask_s, mask_o)` 或 overlap ratio

最终 `geom(i,j)` 维度大概 6~8 维，很稳。

---

## 3) 伪代码：模块与核心函数（Cursor 可直接照抄扩写）

### 3.1 冻结 SAM3 的推理封装

```python
class FrozenSAM3:
    def __init__(self, sam3_model):
        self.model = sam3_model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        """
        image: [3,H,W] float, normalized as SAM3 expects
        returns:
          masks: [N,H,W] (bool or float)
          embs:  [N,D]   (float)
          scores:[N]     (float, optional)
        """
        masks, embs, scores = self.model.predict(image)
        return masks, embs, scores
```

---

### 3.2 mask -> bbox

```python
def mask_to_box(mask: torch.Tensor):
    """
    mask: [H,W] bool/0-1
    return: [4] xyxy (float)
    """
    ys, xs = torch.where(mask > 0.5)
    if ys.numel() == 0:
        return None
    x1, x2 = xs.min().float(), xs.max().float()
    y1, y2 = ys.min().float(), ys.max().float()
    return torch.stack([x1, y1, x2, y2], dim=0)
```

---

### 3.3 proposal 与 GT 匹配（Hungarian 或 greedy）

**最简 greedy**（先用它，够用；后面再换 Hungarian）：

```python
def match_by_iou(pred_boxes, gt_boxes, iou_thr=0.5):
    """
    pred_boxes: [N,4]
    gt_boxes:   [G,4]
    return:
      gt_to_pred: LongTensor [G], -1 if unmatched
    """
    iou = box_iou(gt_boxes, pred_boxes)  # [G,N]
    gt_to_pred = torch.full((gt_boxes.size(0),), -1, dtype=torch.long)

    # greedy: for each gt pick best pred not taken
    taken = set()
    for g in range(gt_boxes.size(0)):
        vals, idxs = torch.sort(iou[g], descending=True)
        for v, n in zip(vals.tolist(), idxs.tolist()):
            if v < iou_thr:
                break
            if n not in taken:
                gt_to_pred[g] = n
                taken.add(n)
                break
    return gt_to_pred
```

---

### 3.4 几何特征

```python
def box_geom_feat(box_s, box_o, eps=1e-6):
    # box: [4] xyxy
    x1s, y1s, x2s, y2s = box_s
    x1o, y1o, x2o, y2o = box_o

    cxs, cys = (x1s + x2s) * 0.5, (y1s + y2s) * 0.5
    cxo, cyo = (x1o + x2o) * 0.5, (y1o + y2o) * 0.5
    ws, hs = (x2s - x1s).clamp(min=eps), (y2s - y1s).clamp(min=eps)
    wo, ho = (x2o - x1o).clamp(min=eps), (y2o - y1o).clamp(min=eps)

    dx = (cxs - cxo) / wo
    dy = (cys - cyo) / ho
    dw = torch.log(ws / wo)
    dh = torch.log(hs / ho)
    da = torch.log((ws * hs) / (wo * ho))

    iou = single_box_iou(box_s, box_o)  # scalar
    return torch.stack([dx, dy, dw, dh, da, iou], dim=0)  # [6]
```

---

### 3.5 Pair 采样（正负）

```python
def build_pair_labels(num_obj, gt_rels, bg_class=0):
    """
    gt_rels: [R,3] (s,o,pred) where pred in [1..C-1], bg=0
    return:
      label_mat: [num_obj,num_obj] with bg default
      pos_pairs: list of (i,j)
    """
    label_mat = torch.zeros((num_obj, num_obj), dtype=torch.long) + bg_class
    pos_pairs = []
    for s, o, p in gt_rels.tolist():
        if s == o:
            continue
        label_mat[s, o] = p
        pos_pairs.append((s, o))
    return label_mat, pos_pairs


def sample_pairs(label_mat, pos_pairs, neg_ratio=3):
    """
    keep all positives, sample negatives
    """
    num_obj = label_mat.size(0)
    all_pairs = [(i, j) for i in range(num_obj) for j in range(num_obj) if i != j]

    # positives
    pos = pos_pairs
    pos_set = set(pos)

    # negatives
    neg = [p for p in all_pairs if p not in pos_set]
    num_neg = min(len(neg), len(pos) * neg_ratio if len(pos) > 0 else 256)
    neg = random.sample(neg, num_neg) if len(neg) > num_neg else neg

    pairs = pos + neg
    labels = torch.tensor([label_mat[i, j].item() for (i, j) in pairs], dtype=torch.long)
    return pairs, labels
```

---

### 3.6 Relation head（MLP/FFN）

```python
class RelationHead(nn.Module):
    def __init__(self, emb_dim, geom_dim, num_predicates, hidden=512, dropout=0.1):
        super().__init__()
        in_dim = emb_dim * 2 + geom_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_predicates),
        )

    def forward(self, z):  # z: [P, in_dim]
        return self.net(z)  # [P, num_predicates]
```

---

### 3.7 单图 forward：从 SAM3 到 logits（训练核心）

```python
def forward_one_image(
    sam3: FrozenSAM3,
    rel_head: RelationHead,
    image, gt_boxes, gt_obj_labels, gt_rels,
    max_props=200, iou_thr=0.5, bg_class=0, neg_ratio=3
):
    # 1) SAM3 proposals
    masks, embs, scores = sam3.forward(image)  # masks:[N,H,W], embs:[N,D]
    N = embs.size(0)
    if N > max_props:
        topk = torch.topk(scores, k=max_props).indices
        masks, embs = masks[topk], embs[topk]

    # 2) mask->box
    pred_boxes = []
    valid = []
    for n in range(masks.size(0)):
        box = mask_to_box(masks[n])
        if box is None:
            continue
        pred_boxes.append(box)
        valid.append(n)
    pred_boxes = torch.stack(pred_boxes, dim=0)          # [N',4]
    embs = embs[torch.tensor(valid, device=embs.device)] # [N',D]

    # 3) match GT->pred
    gt_to_pred = match_by_iou(pred_boxes, gt_boxes, iou_thr=iou_thr)  # [G]

    # 4) gather matched GT objects (drop unmatched)
    matched_g = (gt_to_pred >= 0).nonzero(as_tuple=False).squeeze(1)
    if matched_g.numel() < 2:
        return None  # skip

    # remap indices to compact [0..M-1]
    old_to_new = {int(g.item()): k for k, g in enumerate(matched_g)}
    M = matched_g.numel()

    obj_emb = []
    obj_box = []
    for g in matched_g.tolist():
        n = gt_to_pred[g].item()
        obj_emb.append(embs[n])
        obj_box.append(gt_boxes[g])
    obj_emb = torch.stack(obj_emb, dim=0)  # [M,D]
    obj_box = torch.stack(obj_box, dim=0)  # [M,4]

    # 5) filter / remap relations to matched set
    rels = []
    for s, o, p in gt_rels.tolist():
        if s in old_to_new and o in old_to_new and s != o:
            rels.append((old_to_new[s], old_to_new[o], p))
    if len(rels) == 0:
        return None

    gt_rels_m = torch.tensor(rels, device=image.device, dtype=torch.long)  # [R',3]

    # 6) build labels & sample pairs
    label_mat, pos_pairs = build_pair_labels(M, gt_rels_m, bg_class=bg_class)
    pairs, labels = sample_pairs(label_mat, pos_pairs, neg_ratio=neg_ratio)
    P = len(pairs)

    # 7) build pair features
    z_list = []
    for (i, j) in pairs:
        geom = box_geom_feat(obj_box[i], obj_box[j])     # [geom_dim]
        z = torch.cat([obj_emb[i], obj_emb[j], geom], dim=0)
        z_list.append(z)
    z = torch.stack(z_list, dim=0)  # [P, 2D+geom]

    # 8) logits
    logits = rel_head(z)  # [P, C]
    return logits, labels
```

---

### 3.8 训练 loop（只更新 relation head）

```python
def train_step(batch, sam3, rel_head, optimizer, ce_loss):
    rel_head.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    valid = 0

    for sample in batch:
        image = sample["image"].to(device)
        gt_boxes = sample["gt_boxes"].to(device)
        gt_obj_labels = sample["gt_obj_labels"].to(device)  # PredCls 阶段不必用
        gt_rels = sample["gt_rels"].to(device)

        out = forward_one_image(
            sam3, rel_head,
            image, gt_boxes, gt_obj_labels, gt_rels,
            bg_class=0, neg_ratio=3
        )
        if out is None:
            continue
        logits, labels = out
        loss = ce_loss(logits, labels)
        loss.backward()
        total_loss += loss.item()
        valid += 1

    if valid > 0:
        optimizer.step()
    return total_loss / max(valid, 1)
```

---

## 4) 推理（生成一个 scene graph 的最小输出）

推理时（如果你不做检测/分类），最简单是 PredCls setting 下输出关系：

* 先匹配得到 objects（同训练）
* 对所有 pairs 计算 predicate 概率
* 对每个 pair 取 top-1 或 top-k predicate（排除 BG）
* 输出 edges：`(sub_idx, obj_idx, pred_id, score)`

```python
def infer_scene_graph(obj_emb, obj_box, rel_head, bg_class=0, topk_per_pair=1):
    M = obj_emb.size(0)
    edges = []
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            geom = box_geom_feat(obj_box[i], obj_box[j])
            z = torch.cat([obj_emb[i], obj_emb[j], geom], dim=0).unsqueeze(0)
            logits = rel_head(z)[0]
            prob = logits.softmax(dim=0)

            # exclude bg
            prob_bg0 = prob.clone()
            prob_bg0[bg_class] = 0.0
            vals, idxs = torch.topk(prob_bg0, k=topk_per_pair)
            for v, p in zip(vals.tolist(), idxs.tolist()):
                edges.append((i, j, p, v))
    return edges
```

---

## 5) 你交给 Cursor 的“实现要点清单”（它最容易一次性补齐）

1. 你选择 PredCls setting：先别做 object 分类
2. 数据集侧要提供统一格式：

   * `image`
   * `gt_boxes`
   * `gt_obj_labels`
   * `gt_rels (s,o,p)`
3. SAM3 wrapper：只要能输出 `masks, embs, scores`
4. 匹配策略：先 greedy IoU，跑通后再换 Hungarian
5. Pair 采样：全正 + 随机负（控制 O(N²)）
6. Relation head：2-layer MLP 足够起步

---

如果你希望我再进一步把它“工程化成文件结构”（例如 `models/rel_head.py`, `datasets/vg.py`, `train.py` 的接口约定），我也可以按你现有 repo 风格给出更贴近你工程的版本（仍然保持只包含核心实现，不写测试/调用样例）。
