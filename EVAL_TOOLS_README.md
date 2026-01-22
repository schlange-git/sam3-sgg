# Evaluation Results Tools

自动提取和对比训练评测结果的工具脚本。

## 📋 工具列表

### 1. `save_eval_results.py` - 结果提取与保存

从训练输出目录中提取评测结果并保存为格式化的文本文件。

**用法：**
```bash
python save_eval_results.py <output_dir> [--name <experiment_name>] [--output <filename>]
```

**参数：**
- `output_dir`: 包含 `metrics.json` 和 `log.txt` 的输出目录
- `--name`: 实验名称（用于结果标题，默认："Evaluation"）
- `--output`: 输出文件名（默认："eval_results.txt"）

**示例：**
```bash
# 提取预训练权重的评测结果
python save_eval_results.py output_pretrained_eval \
    --name "Pretrained Weights" \
    --output eval_results.txt

# 提取 finetune 后的结果
python save_eval_results.py output_finetune_1000 \
    --name "Finetuned 1000 iter" \
    --output eval_results.txt
```

**输出格式：**
```
============================================================
Pretrained Weights Results
============================================================
Generated: 2026-01-22 08:21:07

Detection Results (bbox):
------------------------------------------------------------
  AP     = 34.11
  AP50   = 58.32
  AP75   = 32.22
  APs    = 9.49
  APm    = 42.96
  APl    = 48.49

Scene Graph Results:
------------------------------------------------------------
  R@20/50/100      = nan/nan/nan
  ng-R@20/50/100   = nan/nan/nan
  zR@20/50/100     = nan/nan/nan
  mR@20/50/100     = 0.00/0.00/0.00

============================================================
```

---

### 2. `compare_results.py` - 多实验对比

对比多个实验的评测结果，生成对比表格。

**用法：**
```bash
python compare_results.py <output_dir1> <output_dir2> [<output_dir3> ...]
```

**示例：**
```bash
# 对比预训练 vs finetune
python compare_results.py output_pretrained_eval output_finetune_1000

# 对比多个 finetune 实验
python compare_results.py \
    output_finetune_500 \
    output_finetune_1000 \
    output_finetune_2000
```

**输出格式：**
```
========================================================
EVALUATION RESULTS COMPARISON
========================================================

Detection Results (bbox):
--------------------------------------------------------
Metric         Pretrained    Finetune_1000   
--------------------------------------------------------
AP             34.11         30.74           
AP50           58.32         45.07           
AP75           32.22         33.67           
...

Scene Graph Results:
--------------------------------------------------------
Metric         Pretrained    Finetune_1000   
--------------------------------------------------------
R@20           nan           nan             
mR@20          0.00          0.00            
...
```

对比结果会同时打印到终端和保存到 `results_comparison.txt`。

---

## 🔄 集成到训练流程

### 方法 1: 训练后手动运行

```bash
# 1. 运行评测
python train_iterative_model.py --eval-only --num-gpus 1 \
    --config-file configs/speaq.yaml \
    OUTPUT_DIR output_my_experiment \
    MODEL.WEIGHTS my_checkpoint.pth \
    ...

# 2. 提取结果
python save_eval_results.py output_my_experiment \
    --name "My Experiment"
```

### 方法 2: 创建评测+保存脚本

```bash
#!/bin/bash
# eval_and_save.sh

OUTPUT_DIR=$1
EXPERIMENT_NAME=$2
MODEL_WEIGHTS=$3

# Run evaluation
python train_iterative_model.py --eval-only --num-gpus 1 \
    --config-file configs/speaq.yaml \
    OUTPUT_DIR "$OUTPUT_DIR" \
    MODEL.WEIGHTS "$MODEL_WEIGHTS" \
    DATASETS.VISUAL_GENOME.IMAGES /path/to/VG_100K \
    DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY /path/to/VG-SGG-dicts-with-attri.json \
    DATASETS.VISUAL_GENOME.IMAGE_DATA /path/to/image_data.json \
    DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5 /path/to/VG-SGG-with-attri.h5 \
    SOLVER.IMS_PER_BATCH 2

# Extract and save results
python save_eval_results.py "$OUTPUT_DIR" \
    --name "$EXPERIMENT_NAME"

echo "✓ Results saved to: $OUTPUT_DIR/eval_results.txt"
```

**使用：**
```bash
chmod +x eval_and_save.sh
./eval_and_save.sh output_my_exp "My Experiment" my_model.pth
```

---

## 📊 支持的指标

### Detection (bbox)
- **AP**: Average Precision @ IoU=0.50:0.95
- **AP50**: Average Precision @ IoU=0.50
- **AP75**: Average Precision @ IoU=0.75
- **APs**: AP for small objects
- **APm**: AP for medium objects
- **APl**: AP for large objects

### Scene Graph (SG)
- **R@20/50/100**: Recall @ top-20/50/100 predictions
- **ng-R@20/50/100**: No Graph Constraint Recall
- **zR@20/50/100**: Zero-shot Recall
- **mR@20/50/100**: Mean Recall

---

## 🛠️ 故障排除

### 问题: "Could not extract results"

**原因：** 输出目录中没有 `metrics.json` 或 `log.txt` 文件

**解决：**
1. 确认评测已完成
2. 检查 `OUTPUT_DIR` 路径是否正确
3. 查看是否有权限访问文件

### 问题: 所有指标显示 "N/A"

**原因：** 日志格式不匹配或评测未包含该指标

**解决：**
1. 检查 `log.txt` 中是否包含 "copypaste:" 行
2. 确认评测配置正确（是否包含 Scene Graph 评测）

---

## 📝 示例工作流

### 完整的实验对比流程

```bash
# 1. 评测预训练权重
python train_iterative_model.py --eval-only --num-gpus 1 \
    --config-file configs/speaq.yaml \
    OUTPUT_DIR output_pretrained \
    MODEL.WEIGHTS vg_objectdetector_pretrained.pth \
    ...

# 2. Finetune 1000 次迭代
python train_iterative_model.py --num-gpus 1 \
    --config-file configs/speaq.yaml \
    OUTPUT_DIR output_ft_1000 \
    MODEL.WEIGHTS vg_objectdetector_pretrained.pth \
    SOLVER.MAX_ITER 1000 \
    ...

# 3. Finetune 2000 次迭代
python train_iterative_model.py --num-gpus 1 \
    --config-file configs/speaq.yaml \
    OUTPUT_DIR output_ft_2000 \
    MODEL.WEIGHTS vg_objectdetector_pretrained.pth \
    SOLVER.MAX_ITER 2000 \
    ...

# 4. 提取各自结果
python save_eval_results.py output_pretrained --name "Pretrained"
python save_eval_results.py output_ft_1000 --name "Finetune 1K"
python save_eval_results.py output_ft_2000 --name "Finetune 2K"

# 5. 对比所有结果
python compare_results.py output_pretrained output_ft_1000 output_ft_2000

# 6. 查看对比结果
cat results_comparison.txt
```

---

## 📌 提示

1. **自动化**: 可以在训练脚本中集成这些工具，训练完成后自动生成结果摘要
2. **版本控制**: 将 `eval_results.txt` 和 `results_comparison.txt` 加入 git，方便追踪实验进展
3. **结果归档**: 定期将重要实验的结果文件归档保存
4. **可视化**: 可以基于这些脚本的输出创建可视化图表

---

**作者**: Auto-generated  
**更新**: 2026-01-22
