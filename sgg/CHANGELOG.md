# SGG 重构变更日志

## 重构完成（按照 1229new_pipeline.md）

### 核心变更

1. **移除 IoU Matching 和 SAM3 Proposals**
   - ❌ 旧流程：SAM3 text prompt "object" → proposals → IoU matching → GT boxes
   - ✅ 新流程：GT boxes → SAM3 box prompt → embeddings
   - 结果：几乎不会 skip 样本

2. **改用 Edge List**
   - ❌ 旧：label matrix [M, M]，会覆盖多关系
   - ✅ 新：edge list [(s, o, p), ...]，支持多关系

3. **简化 Loss**
   - ❌ 旧：existence loss + classification loss（双重 loss）
   - ✅ 新：单一 softmax CE loss（含 background，降权）

4. **VG150 数据集支持**
   - ✅ 支持 VG-SGG-dicts-with-attri.json
   - ✅ 50 个 predicate + background

### 文件组织

所有 SGG 相关代码现在都在 `sgg/` 目录下：

```
sgg/
├── datasets/          # 数据集加载器
├── models/            # 模型定义
├── utils/             # 工具函数
├── configs/           # 配置文件
├── train_predcls.py   # 主训练脚本
├── run_train.sh       # 运行脚本
└── README.md          # 使用说明
```

### 使用方法

1. 激活 conda 环境：`conda activate sam3`
2. 修改 `sgg/run_train.sh` 中的 `DATA_ROOT`
3. 运行：`bash sgg/run_train.sh`

### 关键改进点

- **不再依赖检测**：直接使用 GT boxes，避免检测不稳定性
- **更稳定的训练**：几乎不会 skip 样本，batch 利用率高
- **更清晰的代码结构**：模块化设计，易于维护和扩展

### 最新更新（支持 VG150 h5 格式）

- ✅ 重写数据集加载器以支持 `VG-SGG-with-attri.h5` 格式
- ✅ 兼容 Scene-Graph-Benchmark.pytorch 的数据格式
- ✅ 自动查找 `images/` 或 `images2/` 目录
- ✅ 更新运行脚本使用正确的数据路径

### 待优化项

1. **Embedding 提取**：当前使用 mask 特征作为代理，可以改进为：
   - 从 SAM3 decoder tokens 提取
   - 使用 RoI pooling 从 backbone features 提取

2. **批处理优化**：当前逐个处理 GT boxes，可以优化为批量处理

3. **多关系支持**：当前使用方式 A（展开为多条样本），可以改为方式 B（multi-label sigmoid）
