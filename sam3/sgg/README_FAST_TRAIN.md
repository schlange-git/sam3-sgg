# GT-driven + Mask-aware Geometry 快速训练方案
TP：correct 
FP：wrong relation
FN：miss prediction

## 概述

本方案实现了文档中描述的 "GT-driven + mask-aware geometry（离线缓存）" 方案，主要特点：

1. **离线预处理**：SAM3 只在离线阶段运行一次，生成 mask 和几何特征
2. **固定 pair 数量**：每个图像固定 P=128 个 pairs，避免动态形状
3. **Mask-aware 几何特征**：除了 box 几何特征（6维），还加入 mask 几何特征（5维）
4. **快速训练**：训练时只读缓存，不做 SAM3 推理，速度大幅提升

## 目录结构

```
sgg/
├── datasets/
│   ├── vg150_dataset.py      # 原有的 VG150 数据集
│   ├── vg150_reader.py        # 简化的 Reader（用于离线预处理）
│   └── cached_pairs.py        # 缓存数据集（训练时使用）
├── precompute/
│   ├── sam3_adapter.py        # SAM3 mask 生成器适配器
│   ├── pair_sampler.py        # 固定 P 的 pair 采样
│   ├── geom_extractor.py      # 几何特征提取（box + mask）
│   └── build_cache.py         # 离线预处理脚本
├── models/
│   ├── relation_head.py       # 原有的 relation head（embedding + geom）
│   └── relation_head_geom.py  # 新的 relation head（仅 geom）
├── train/
│   ├── loss.py                # Masked Cross Entropy Loss
│   └── train_fast.py          # 快速训练脚本
├── run_build_cache.sh         # 构建缓存脚本
└── run_train_fast.sh          # 快速训练脚本
```

## 使用流程

### 步骤 1: 构建缓存（离线预处理）

```bash
bash sgg/run_build_cache.sh
```

或者手动运行：

cd /home/shi/桌面/abschluss/sam3
python sgg/precompute/build_cache.py \
    --vg_root /home/shi/桌面/abschluss/dataset/vg150 \
    --out_dir sgg/cache/train \
    --split train \
    --P 128 \
    --neg_ratio 3 \
    --mask_size 256 \
    --sam3_impl real

```bash
python sgg/precompute/build_cache.py \
    --vg_root /home/shi/桌面/abschluss/dataset/vg150 \
    --out_dir sgg/cache/train \
    --split train \
    --P 128 \
    --neg_ratio 3 \
    --mask_size 256 \
    --sam3_impl real  # 或 "dummy" 用于测试
```

**参数说明**：
- `--vg_root`: VG150 数据集根目录
- `--out_dir`: 缓存输出目录
- `--split`: 数据集分割（train/val/test）
- `--P`: 固定 pair 数量（默认 128）
- `--neg_ratio`: 负样本比例（默认 3）
- `--mask_size`: Mask 分辨率（默认 256x256）
- `--sam3_impl`: "real" 使用真实 SAM3，"dummy" 使用矩形 mask（测试用）

**输出**：
- 每个图像一个 `.pt` 文件，包含：
  - `geom_feat`: [P, 11] 几何特征（6 box + 5 mask）
  - `pair_label`: [P] 标签（0=background, >0=predicate, -1=padding）
  - `pair_mask`: [P] 有效 pair 掩码
- `metadata.json`: 元数据信息

### 步骤 2: 快速训练

```bash
bash sgg/run_train_fast.sh
```

或者手动运行：

```bash
python sgg/train/train_fast.py \
    --cache_dir sgg/cache/train \
    --out_dir sgg/checkpoints/fast \
    --num_predicates 51 \
    --epochs 4 \
    --batch_size 32 \
    --lr 1e-3 \
    --amp
```

**参数说明**：
- `--cache_dir`: 缓存目录（步骤1的输出）
- `--out_dir`: 模型检查点输出目录
- `--num_predicates`: Predicate 数量（50 + 1 background = 51）
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率
- `--amp`: 使用混合精度训练

## 几何特征说明

### Box 几何特征（6维）
1. `dx`: 中心点 x 坐标差（归一化）
2. `dy`: 中心点 y 坐标差（归一化）
3. `dw`: 宽度比的对数
4. `dh`: 高度比的对数
5. `da`: 面积比的对数
6. `bbox_iou`: Bounding box IoU

### Mask 几何特征（5维）
1. `mask_iou`: Mask IoU
2. `inter_over_min`: 交集/最小面积
3. `contain_i_in_j`: i 被 j 包含的比例
4. `contain_j_in_i`: j 被 i 包含的比例
5. `rel_area_log`: 面积比的对数

**总特征维度**: 11 (6 + 5)

## 与原有方案的区别

### 原有方案（慢）
- 训练时每个 step 运行 SAM3
- 动态 pair 数量（O(N²)）
- 在线计算几何特征
- 速度：~10s/step

### 新方案（快）
- SAM3 只在离线阶段运行一次
- 固定 pair 数量（P=128）
- 离线计算并缓存几何特征
- 速度：<0.5s/step（提升 20x+）

## 注意事项

1. **首次运行需要构建缓存**：构建缓存可能需要较长时间（取决于数据集大小和 SAM3 速度）
2. **缓存空间**：每个图像约 10-50KB，10万图像约 1-5GB
3. **SAM3 实现**：`RealSAM3MaskGenerator` 需要根据你的 SAM3 接口调整，目前使用 mask 特征作为 embedding 的代理
4. **测试模式**：可以使用 `--sam3_impl dummy` 快速测试流程（使用矩形 mask）

## 下一步优化

1. **改进 SAM3 mask 提取**：如果 SAM3 提供真实的 mask，直接使用而不是矩形
2. **增加边界接触特征**：用于 "attached/holding" 类 predicate
3. **消融实验**：对比 box-only vs box+mask 的效果

