# SpeaQ 全链路训练与评测流程

一键运行完整的训练评测流程：预训练评测 → Finetune → 评测 → 结果对比

## 🚀 快速开始

### 基本用法

```bash
# 默认配置（10张图，1000次迭代）
./run_full_pipeline.sh

# 自定义迭代次数
./run_full_pipeline.sh --iters 2000

# 使用更多 GPU 和更大批次
./run_full_pipeline.sh --gpus 2 --batch-size 4

# 完整参数示例
./run_full_pipeline.sh \
    --iters 5000 \
    --gpus 2 \
    --overfit 20 \
    --batch-size 4
```

### 查看帮助

```bash
./run_full_pipeline.sh --help
```

---

## 📋 流程说明

脚本会自动执行以下 4 个步骤：

### Step 1: 评测预训练权重 ✅
- 在指定数据集上评测预训练模型
- 保存评测结果到 `output_pipeline_*/pretrained_eval/`

### Step 2: Finetune 训练 🔄
- 从预训练权重开始 finetune
- 训练指定次数的迭代
- 保存模型到 `output_pipeline_*/finetune_*iter/`

### Step 3: 评测 Finetune 模型 ✅
- 评测训练后的模型
- 保存评测结果

### Step 4: 生成结果对比 📊
- 对比预训练 vs finetune 的结果
- 生成对比表格和总结报告

---

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--iters` | 1000 | Finetune 训练迭代次数 |
| `--gpus` | 1 | 使用的 GPU 数量 |
| `--overfit` | 10 | 过拟合数据集大小（图片数量） |
| `--batch-size` | 2 | 每个 GPU 的批次大小 |
| `--pretrained` | `vg_objectdetector_pretrained.pth` | 预训练权重路径 |
| `--dataset` | `/home/shi/abschluss/dataset/vg150` | 数据集根目录 |

---

## 📁 输出结构

运行后会生成时间戳命名的输出目录：

```
output_pipeline_20260122_083000/
├── pretrained_eval/              # 预训练评测结果
│   ├── eval_results.txt         # 格式化结果 ✅
│   ├── metrics.json             # JSON 格式指标
│   └── log.txt                  # 完整日志
│
├── finetune_1000iter/            # Finetune 训练结果
│   ├── model_final.pth          # 最终模型 ⭐
│   ├── eval_results.txt         # 评测结果 ✅
│   ├── metrics.json
│   ├── log.txt
│   └── model_*.pth              # 中间检查点
│
├── results_comparison.txt        # 结果对比表格 ✅
└── pipeline_summary.txt          # 流程总结报告 ✅
```

---

## 📊 查看结果

### 方式 1: 查看格式化结果

```bash
# 预训练结果
cat output_pipeline_*/pretrained_eval/eval_results.txt

# Finetune 结果
cat output_pipeline_*/finetune_*/eval_results.txt

# 对比结果
cat output_pipeline_*/results_comparison.txt

# 总结报告
cat output_pipeline_*/pipeline_summary.txt
```

### 方式 2: 使用对比工具

```bash
# 手动对比多个实验
python compare_results.py \
    output_pipeline_20260122_080000/finetune_1000iter \
    output_pipeline_20260122_090000/finetune_2000iter
```

---

## 💡 使用场景

### 场景 1: 快速验证（推荐用于测试）

```bash
# 10张图，500次迭代，快速验证流程
./run_full_pipeline.sh --iters 500 --overfit 10
```

### 场景 2: 小规模过拟合实验

```bash
# 20张图，2000次迭代
./run_full_pipeline.sh --iters 2000 --overfit 20
```

### 场景 3: 完整 10-clip 过拟合

```bash
# 10张图，5000次迭代，充分训练
./run_full_pipeline.sh --iters 5000 --overfit 10 --batch-size 4
```

### 场景 4: 多 GPU 加速训练

```bash
# 使用 2 个 GPU，更大批次
./run_full_pipeline.sh --gpus 2 --batch-size 4 --iters 10000
```

---

## 🔍 结果解读

### Detection 指标 (bbox)

| 指标 | 含义 | 期望值 |
|------|------|--------|
| **AP** | 平均精度 (IoU 0.5-0.95) | 过拟合应 > 50 |
| **AP50** | AP @ IoU=0.5 | 过拟合应 > 80 |
| **AP75** | AP @ IoU=0.75 | 过拟合应 > 60 |

### Scene Graph 指标 (SG)

| 指标 | 含义 | 期望值 |
|------|------|--------|
| **R@100** | Recall @ top-100 | 过拟合应 > 0.5 |
| **mR@100** | Mean Recall @ 100 | 关键指标，应逐步提升 |

**⚠️ 注意**: 如果训练后 SG 指标仍为 0：
1. 增加迭代次数 (`--iters 5000` 或更多)
2. 检查预训练权重是否包含关系检测模块
3. 查看 log.txt 中的损失曲线是否收敛

---

## ⏱️ 预计耗时

| 配置 | 预计时间 |
|------|---------|
| 10 图 × 500 iter | ~5 分钟 |
| 10 图 × 1000 iter | ~10 分钟 |
| 10 图 × 2000 iter | ~20 分钟 |
| 10 图 × 5000 iter | ~50 分钟 |
| 20 图 × 5000 iter | ~100 分钟 |

*基于单 GPU (RTX 5070) 估算*

---

## 🛠️ 高级用法

### 自定义数据集路径

```bash
./run_full_pipeline.sh \
    --dataset /path/to/your/vg150 \
    --iters 2000
```

### 使用自己的预训练权重

```bash
./run_full_pipeline.sh \
    --pretrained /path/to/your/pretrained.pth \
    --iters 1000
```

### 修改检查点保存频率

编辑脚本中的 `CHECKPOINT_PERIOD` 变量：

```bash
# 在脚本开头修改
CHECKPOINT_PERIOD=100  # 每 100 次迭代保存一次
```

---

## 🐛 故障排除

### 问题 1: `conda activate speaq` 失败

**解决方案:**
```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate speaq
```

### 问题 2: CUDA Out of Memory

**解决方案:**
```bash
# 减小批次大小
./run_full_pipeline.sh --batch-size 1

# 或减少过拟合图片数量
./run_full_pipeline.sh --overfit 5
```

### 问题 3: 端口已被占用

**解决方案:**  
脚本会自动使用不同端口 (29500, 29501)，如果仍有冲突，可编辑脚本中的 `PORT_PRETRAINED` 和 `PORT_FINETUNE` 变量。

---

## 📌 与其他工具配合

### 1. 单独提取结果

```bash
# 已有训练输出，单独提取结果
python save_eval_results.py output_my_experiment --name "My Exp"
```

### 2. 对比多个实验

```bash
# 对比不同时间点的实验
python compare_results.py \
    output_pipeline_20260122_080000/finetune_1000iter \
    output_pipeline_20260122_090000/finetune_2000iter \
    output_pipeline_20260122_100000/finetune_5000iter
```

### 3. 可视化训练曲线

```bash
# 启动 TensorBoard
tensorboard --logdir=output_pipeline_*/finetune_*
```

---

## 🎯 最佳实践

1. **首次使用**: 先运行默认配置验证流程
   ```bash
   ./run_full_pipeline.sh
   ```

2. **迭代优化**: 根据结果逐步增加迭代次数
   ```bash
   ./run_full_pipeline.sh --iters 2000
   ./run_full_pipeline.sh --iters 5000
   ```

3. **保存重要结果**: 将有价值的输出目录备份或重命名
   ```bash
   cp -r output_pipeline_* important_results/exp_baseline
   ```

4. **版本控制**: 将总结报告加入 git 追踪
   ```bash
   git add output_*/pipeline_summary.txt
   git add output_*/results_comparison.txt
   ```

---

## 📚 相关文档

- **评测工具**: `EVAL_TOOLS_README.md` - 结果提取和对比工具详细说明
- **项目 README**: `README.md` - SpeaQ 项目总体说明
- **配置文件**: `configs/speaq.yaml` - 模型训练配置

---

**作者**: Auto-generated  
**更新**: 2026-01-22  
**用途**: 一键运行完整训练评测流程
