# SGG 训练快速开始指南

## 1. 环境准备

```bash
# 激活 conda 环境
conda activate sam3

# 安装依赖（如果需要）
pip install h5py numpy
```

## 2. 数据集检查

确保你的数据集路径 `/home/shi/abschluss/dataset/vg150` 包含：

```
vg150/
├── VG-SGG-dicts-with-attri.json
├── VG-SGG-with-attri.h5
├── image_data.json
├── images/          # 或 images2/
└── generate_attribute_labels.py
```

## 3. 测试数据集加载

```bash
conda activate sam3
python sgg/test_dataset.py
```

如果看到 "✅ All tests passed!"，说明数据集加载正常。

## 4. 构建缓存（离线预处理）

首先需要运行离线预处理，生成几何特征缓存：

```bash
bash sgg/run_build_cache.sh
```

## 5. 开始训练

```bash
# 方式1: 使用脚本
bash sgg/run_train_fast.sh

# 方式2: 直接运行
conda activate sam3
python sgg/train/train_fast.py \
    --cache_dir sgg/cache/train \
    --out_dir checkpoints/sgghead \
    --num_predicates 51 \
    --epochs 12 \
    --batch_size 32 \
    --lr 1e-3
```

## 5. 查看训练日志

训练日志保存在：
- JSON 日志：`sgg/logfiles/train_log_{timestamp}.json`
- TensorBoard：`sgg/logfiles/tensorboard/{timestamp}/`

查看 TensorBoard：
```bash
tensorboard --logdir sgg/logfiles/tensorboard
```

## 常见问题

### Q: h5py 导入错误
```bash
pip install --upgrade h5py numpy
```

### Q: 找不到图像文件
检查 `images/` 或 `images2/` 目录是否存在，以及 `image_data.json` 中的路径是否正确。

### Q: 数据集为空
检查 `VG-SGG-with-attri.h5` 文件是否完整，以及过滤条件（max_objects, max_relations）是否太严格。

## 下一步

训练完成后，检查点保存在 `sgg/checkpoints/` 目录下。

###
Valid pairs: 420/1280 (32.8%)
valid_pairs 的界定和 label 的来源?

根据代码，回答如下：

## 1. valid_pairs 的界定

**valid 的定义**（在 `train_fast.py` 中）：
```python
m = m & (y >= 0)  # valid = pair_mask==1 AND label>=0
```

**valid_pairs 减少的原因**（不是 SAM3 失败）：

1. Padding：固定 P=128 对/图，如果实际关系少于 128，会用 `label=-1, mask=0` 填充
2. 负样本采样：`neg_ratio=3` 会采样大量背景对（label=0）
   - 例如：10 个正样本 → 30 个负样本 → 40 个有效对
   - 如果实际关系很少，有效对比例会更低
3. SAM3 不影响 pair 有效性：从 `build_cache.py` 看，pairs 直接从 GT relations (`sample.rels`) 采样，不依赖 SAM3 是否成功生成 mask

**代码流程**：
```python
# 1. SAM3 生成 mask（用于几何特征计算）
sam3_out = sam3.predict(img, boxes)
masks = sam3_out.masks

# 2. 从 GT relations 采样 pairs（不依赖 SAM3）
ps = sample_pairs_fixed_P(
    num_obj=G,
    rels=sample.rels,  # ← 直接来自 GT，不是 SAM3
    P=P,
    neg_ratio=neg_ratio,
)
```

即使 SAM3 失败（mask 为空），pairs 仍然会被采样和保存，只是几何特征可能不准确。

## 2. Classification 的 label 来源

**Label 是 GT 结果，不是 SAM3 结果**：

从 `pair_sampler.py` 看：
```python
# 正样本：来自 GT relations
pos_edges.append((s, o, p))  # p 是 GT predicate ID

# 负样本：背景对
neg_edges = [(i, j, 0) for (i, j) in neg_pairs]  # 0 = background
```

从 `build_cache.py` 看：
```python
rels=sample.rels  # ← 来自 VG150 数据集的 GT 关系
```

**SAM3 的作用**：
- 仅用于生成 mask，用于计算 mask-aware 几何特征
- 不用于分类，不提供 label
- 不匹配 SAM3 的类别标签

**总结**：
- valid_pairs 减少是因为固定 P 的 padding 和负样本采样，不是 SAM3 失败
- Label 来自 GT（VG150 的 predicate），不是 SAM3
- SAM3 只提供 mask，用于几何特征计算

如果需要提高 valid_pairs 比例，可以：
1. 减少 `neg_ratio`（例如从 3 降到 1）
2. 根据实际关系数量动态调整 P
3. 过滤掉没有关系的图像