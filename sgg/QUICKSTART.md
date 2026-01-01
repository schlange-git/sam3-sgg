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

## 4. 开始训练

```bash
# 方式1: 使用脚本
bash sgg/run_train.sh

# 方式2: 直接运行
conda activate sam3
python sgg/train_predcls.py \
    --data_root /home/shi/abschluss/dataset/vg150 \
    --batch_size 4 \
    --num_epochs 3 \
    --lr 1e-4
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
