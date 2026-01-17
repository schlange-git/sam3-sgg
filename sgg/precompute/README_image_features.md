# SAM3 图像特征生成工具

## 概述

本工具用于使用 SAM3 作为 backbone 提取**整个图像的 feature**（区别于针对具体 GT box 的 mask 生成工具）。

### 与现有工具的区别

- **现有工具** (`build_cache.py` + `sam3_adapter.py`):
  - 针对每个 GT box 生成 mask
  - 输出：`masks[G, Hm, Wm]` 和 `obj_emb[G, D]`
  - 用途：离线生成 mask-aware geometry features

- **本工具** (`image_feature_generator.py`):
  - 提取整个图像的 feature（不针对具体 box）
  - 输出：`image_feat[D]` (D=256 默认)
  - 用途：作为下游任务的 backbone feature

## 使用方法

### 0. 准备本地 Checkpoint

由于 SAM3 模型需要访问权限，建议使用本地 checkpoint 文件 `sam3.pt`。

**推荐位置（按优先级查找）：**
1. `./sam3.pt`（项目根目录）
2. `./checkpoints/sam3.pt`
3. `./sgg/sam3.pt`
4. `~/.cache/huggingface/hub/models--facebook--sam3/snapshots/main/sam3.pt`

或者通过命令行参数指定：
```bash
--checkpoint_path path/to/sam3.pt  # 支持相对路径或绝对路径
```

### 1. 命令行使用

```bash
# 使用运行脚本（推荐）
cd sgg
./run_generate_image_features.sh

# 或直接使用 Python（使用自动查找 checkpoint）
python sgg/precompute/image_feature_generator.py \
    --data_root dataset/vg150 \
    --split train \
    --output_dir sgg/cache/image_features \
    --device cuda \
    --feature_dim 256 \
    --image_size 1024 \
    --max_samples 100  # 可选：限制处理数量

# 指定 checkpoint 路径（相对路径）
python sgg/precompute/image_feature_generator.py \
    --data_root dataset/vg150 \
    --split train \
    --output_dir sgg/cache/image_features \
    --checkpoint_path sam3.pt  # 相对路径，相对于项目根目录

# 指定 checkpoint 路径（绝对路径）
python sgg/precompute/image_feature_generator.py \
    --data_root /absolute/path/to/vg150 \
    --split train \
    --output_dir sgg/cache/image_features \
    --checkpoint_path /absolute/path/to/sam3.pt
```

### 2. Python API 使用

```python
from sgg.precompute.image_feature_generator import SAM3ImageFeatureGenerator
from sgg.datasets.vg150_dataset import VG150Dataset

# 创建 feature generator（自动查找本地 checkpoint）
generator = SAM3ImageFeatureGenerator(
    device="cuda",
    feature_dim=256,
    image_size=1024,
    checkpoint_path=None,  # None 表示自动查找，或指定路径如 "sam3.pt"
)

# 或指定 checkpoint 路径（相对路径或绝对路径）
generator = SAM3ImageFeatureGenerator(
    device="cuda",
    feature_dim=256,
    image_size=1024,
    checkpoint_path="sam3.pt",  # 相对路径，相对于项目根目录
)

# 加载数据集
dataset = VG150Dataset(
    data_root="/path/to/vg150",
    split="train",
)

# 处理数据集并保存
generator.process_dataset(
    dataset=dataset,
    output_dir="sgg/cache/image_features",
    split="train",
    max_samples=100,  # 可选
)
```

### 3. 单图像特征提取

```python
from PIL import Image
from sgg.precompute.image_feature_generator import SAM3ImageFeatureGenerator

# 创建 generator
generator = SAM3ImageFeatureGenerator(device="cuda")

# 加载图像
image = Image.open("path/to/image.jpg").convert("RGB")

# 提取特征
feature = generator.extract_image_feature(image)  # [256] tensor
```

## 输出格式

### 文件结构

```
sgg/cache/image_features/
├── train/
│   ├── image_feat_00000001.pt  # 每个图像一个文件
│   ├── image_feat_00000002.pt
│   ├── ...
│   └── metadata.json  # 元数据
└── val/
    ├── image_feat_00000001.pt
    ├── ...
    └── metadata.json
```

### 单个特征文件格式 (`.pt`)

```python
{
    "image_id": 1,          # int: 图像 ID
    "feature": tensor,      # [D] float32: 图像特征向量
    "feature_dim": 256,     # int: 特征维度
}
```

### 元数据文件格式 (`metadata.json`)

```json
{
    "split": "train",
    "num_images": 1000,
    "feature_dim": 256,
    "image_ids": [1, 2, 3, ...],
    "filename_prefix": "image_feat"
}
```

## 参数说明

- `device`: 计算设备 ("cuda" / "cpu")
- `feature_dim`: 输出特征维度（默认 256）
- `image_size`: 输入图像尺寸（默认 1024，SAM3 标准尺寸）
- `max_samples`: 最大处理样本数（None 表示处理全部）

## 注意事项

1. **内存使用**：SAM3 模型较大，建议使用 GPU 运行
2. **处理时间**：整图特征提取比 box-specific mask 生成更快，但仍需一定时间
3. **特征维度**：默认 256 维，可通过 `feature_dim` 参数调整
4. **数据格式**：输出的特征已经 L2 归一化

## 后续集成

目前**不需要进行适配**（按用户要求）。生成的 feature 可以：

1. 作为全局 context feature 用于关系预测
2. 与 box-specific features 拼接增强表示
3. 用于图像级别的下游任务

## 技术细节

- 使用 SAM3 的 `forward_image` 方法提取 backbone features
- 通过 `SAM3InteractiveImagePredictor.set_image()` 计算图像 embeddings
- 提取 `image_embed`（低分辨率全局特征）并全局平均池化
- 进行 L2 归一化确保特征在单位球面上

