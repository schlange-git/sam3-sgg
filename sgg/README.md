# SGG (Scene Graph Generation) 训练系统

基于 SAM3 的 Scene Graph Generation 训练框架，专注于 **Predicate Classification (PredCls)** 任务。

## 核心改进

按照 `1229new_pipeline.md` 的指导，本系统实现了以下关键改进：

1. **移除 IoU Matching 和 SAM3 Proposals 依赖**
   - 不再使用 SAM3 的 text prompt "object" 来检测 proposals
   - 直接使用 GT boxes 作为 geometric prompt 提取 embedding
   - 避免了大量样本被跳过的问题

2. **使用 Edge List 而不是 Label Matrix**
   - 支持同一 pair 的多个关系（multi-label）
   - 更灵活的关系表示

3. **简化的 Loss 设计**
   - 单一的 softmax CrossEntropy Loss（含 background）
   - Background 类别降权（默认 weight=0.1）

4. **VG150 数据集支持**
   - 参照 [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) 的数据格式
   - 支持 VG-SGG-dicts-with-attri.json 映射文件

## 文件结构

```
sgg/
├── datasets/
│   ├── __init__.py
│   └── vg150_dataset.py          # VG150 数据集加载器
├── models/
│   ├── __init__.py
│   ├── frozen_sam3_gt.py         # 使用 GT box prompt 的 SAM3
│   └── relation_head.py          # 关系分类头
├── utils/
│   ├── __init__.py
│   ├── geometry.py                # 几何特征计算
│   └── edge_builder.py           # Edge list 构建和负采样
├── configs/
│   └── __init__.py
├── train_predcls.py              # 主训练脚本
├── run_train.sh                  # 训练运行脚本
└── README.md                      # 本文件
```

## 环境设置

1. **激活 conda 环境**：
```bash
conda activate sam3
```

2. **准备 Visual Genome 数据集**：
   - 下载 Visual Genome 数据集
   - 确保包含以下文件（VG150 格式）：
     - `VG-SGG-dicts-with-attri.json` (可从 [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) 获取)
     - `VG-SGG-with-attri.h5` (VG150 专用格式，包含所有标注)
     - `image_data.json` (图像元数据)
     - `images/` 或 `images2/` 目录（包含图像文件）
   
   **注意**：VG150 使用 h5 格式而不是原始的 JSON 格式，这是 Scene-Graph-Benchmark.pytorch 的标准格式。

3. **安装依赖**：
```bash
pip install h5py  # 用于读取 h5 文件
```

## 使用方法

### 1. 修改运行脚本

编辑 `sgg/run_train.sh`，设置你的数据集路径：

```bash
DATA_ROOT="/home/shi/abschluss/dataset/vg150"  # 修改为你的 VG150 数据集路径
```

**注意**：确保数据集路径包含：
- `VG-SGG-dicts-with-attri.json`
- `VG-SGG-with-attri.h5`
- `image_data.json`
- `images/` 或 `images2/` 目录

### 2. 运行训练

```bash
bash sgg/run_train.sh
```

或者直接使用 Python：

```bash
conda activate sam3
python sgg/train_predcls.py \
    --data_root /path/to/visual_genome \
    --batch_size 4 \
    --num_epochs 3 \
    --lr 1e-4 \
    --neg_ratio 3 \
    --bg_weight 0.1
```

### 3. 训练参数说明

- `--data_root`: Visual Genome 数据集根目录
- `--batch_size`: 批次大小（默认 4）
- `--num_epochs`: 训练轮数（默认 3）
- `--lr`: 学习率（默认 1e-4）
- `--neg_ratio`: 负样本与正样本的比例（默认 3）
- `--max_negs`: 没有正样本时的最大负样本数（默认 50）
- `--bg_weight`: Background 类别的权重（默认 0.1）
- `--log_interval`: 日志输出间隔（默认 10）

## 输出

训练过程会生成：

1. **日志文件**：`sgg/logfiles/train_log_{timestamp}.json`
   - 包含训练配置、predicate vocabulary、详细训练日志

2. **TensorBoard 日志**：`sgg/logfiles/tensorboard/{timestamp}/`
   - 实时可视化训练指标

3. **检查点**：`sgg/checkpoints/checkpoint_epoch_{epoch}.pt`
   - 模型权重和优化器状态

4. **Predicate Vocabulary**：`sgg/configs/predicate_vocab.json`
   - 关系类别映射表

## 训练日志格式

```
Step 90/26213 | grad=1.3612 | loss=6.0427 | 
valid=1/4 (avg_num_pairs=4.0 pos=1.0 neg=3.0) | 
skipped: no_valid_relations:3
```

- `grad`: 梯度范数
- `loss`: 平均 loss
- `valid`: 有效样本数 / 总样本数
- `avg_num_pairs`: 平均每个样本的 pair 数量
- `pos/neg`: 平均正/负样本数量

## 关键设计决策

1. **为什么使用 GT box prompt？**
   - PredCls 任务的前提是 GT objects 已知
   - 直接使用 GT boxes 提取 embedding，避免检测不稳定性
   - 几乎不会出现 skip 的情况

2. **为什么使用 Edge List？**
   - 支持同一 pair 的多个关系
   - 避免 label matrix 的覆盖问题

3. **为什么简化 Loss？**
   - 单一 softmax CE 更标准、更稳定
   - Background 降权处理类别不平衡

## 参考

- [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
- Visual Genome 数据集：https://visualgenome.org/
- VG150 predicates 列表见 `1229new_pipeline.md`

## 注意事项

1. **SAM3 Embedding 提取**：当前实现使用 mask 特征作为 embedding 的代理。如果需要更精确的 embedding，可以考虑：
   - 从 SAM3 的 decoder output tokens 中提取
   - 使用 RoI pooling 从 backbone features 中提取

2. **数据集格式**：
   - 使用 VG150 格式（h5 文件），与 Scene-Graph-Benchmark.pytorch 兼容
   - 确保 `VG-SGG-with-attri.h5` 文件存在且可读
   - 如果 h5py 报错，可能需要重新安装：`pip install --upgrade h5py numpy`

3. **GPU 内存**：如果遇到 OOM，可以减小 `batch_size` 或 `max_negs`

4. **图像路径**：代码会自动查找 `images/` 或 `images2/` 目录，如果图像路径不匹配，可能需要调整 `_get_image_path` 方法

## 故障排除

- **"no_gt_objects"**: 图像中没有 GT 对象（应该很少见）
- **"less_than_2_objects"**: 对象数量少于 2，无法构建关系对
- **"sam3_error"**: SAM3 处理出错，检查图像格式和 box 坐标
- **"no_edges"**: 没有构建出任何 edge（正样本或负样本）

如果遇到其他问题，请检查日志文件中的详细错误信息。
