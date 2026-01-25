# SAM3 整图特征提取流水线说明

## Backbone 设计流程（SAM3 整图特征）
- **输入预处理**：`PIL.Image -> torch.Tensor (uint8)`，按 SAM3 规范 `Resize(1008)` + 归一化到 `[-1,1]`，再添加 batch 维。
- **前向主干**：调用 `sam3_model.backbone.forward_image`，获取 `backbone_out`。
- **特征选取**：优先取 `backbone_fpn` 的最高分辨率层；若缺失则退回 `vision_features`。
- **维度对齐**：惰性创建 1×1 Conv，将 `C_raw -> feature_dim`；若通道一致则 `Identity`，否则权重置零并做确定性通道拷贝（多余通道补零）。
- **输出形态**：得到 `dense_feat` 形状 `[C, Hf, Wf]`，并估计步幅 `stride = round(image_size / Hf)`（默认 16）。
- **内存管理**：逐步 `del` 中间张量并 `torch.cuda.empty_cache()`；失败则生成零特征图作为 fallback，保持尺寸与元数据一致。

> 关键代码参考：
> - 预处理与 backbone 前向、FPN 选取与投影：见下文代码片段。
> - 零特征 fallback 与 stride 计算：见代码片段。

## 核心思路（针对 2×2080Ti，内存紧张场景）
- 双进程静态切分：每个进程绑定一张 GPU（通过 `CUDA_VISIBLE_DEVICES`），各自只处理固定索引段，不共享模型/数据加载器/进程。
- 严格单图循环：一次只处理一张图，提取后立即落盘；`del` 中间变量并执行 `torch.cuda.empty_cache()`。
- 特征保存在 CPU，默认 FP32；若需要可开启 `--save_fp16`（当前脚本默认关闭）。
- 分段重启：可按固定间隔（例如 50 张）由外层脚本重启进程作为“内存垃圾回收器”（需自行在外层循环中控制）。

## 主要文件
- `sgg/precompute/image_feature_generator.py`：特征提取主逻辑（SAM3 Backbone、密集特征图、1×1 Conv 降维、保存元数据）。
- `sgg/run_generate_image_features.sh`：便捷运行脚本，默认双 GPU、`batch_size=1`、`num_workers=0`，FP16 关闭。

## 特征格式
- 输出目录：`<OUT_DIR>/<split>/`
- 文件：`image_feat_XXXXXXXX.pt`
- 内容：
  - `image_id`: int
  - `feature`: Tensor `[C, Hf, Wf]`（已搬到 CPU；若开启 FP16 则为 half）
  - `feature_dim`: C（默认 256，经 1×1 Conv 投影）
  - `stride`: 16
  - `img_size`: `(1008, 1008)`（与 SAM3 默认一致）
  - `spatial_shape`: `(Hf, Wf)`
  - `dtype`: `"fp32"` 或 `"fp16"`
- `metadata.json` 汇总：`image_ids`、`count`、`feature_dim`、`stride`、`image_size`、`spatial_shape`、`start_idx`、`max_samples`。

## 运行方式（推荐）
> 需求：每张卡独立进程、静态切分、批大小 1、`num_workers=0`、长时间稳定运行。

### 方案 A：单脚本自动并行（默认）
```bash
bash sgg/run_generate_image_features.sh
```
- 参数位于脚本顶部，可按需修改：
  - `VG_ROOT`：VG150 数据集根目录
  - `OUT_DIR`：输出目录
  - `SPLIT`：`train` / `val`
  - `FEATURE_DIM`：投影后通道数（默认 256）
  - `IMAGE_SIZE`：输入分辨率（默认 1008）
  - `NUM_GPUS`：默认 2
  - `BATCH_SIZE`：默认 1
  - `NUM_WORKERS`：默认 0（避免内存/多进程嵌套）
  - `LIMIT`：-1 或 0 表示全量；>0 表示只跑前 N 张
  - `START_IDX`：起始索引用于分段
  - `SAVE_FP16`：0 关闭，1 开启
  - `CHECKPOINT_PATH`：本地 checkpoint，相对路径如 `weights/sam3.pt`

### 方案 B：手动静态切分双进程（更贴合“单解决方案”）
```bash
# 进程 A（卡 0）：处理前半段
CUDA_VISIBLE_DEVICES=0 python sgg/precompute/image_feature_generator.py \
  --data_root "/home/shi/桌面/abschluss/sgg/dataset/vg150" \
  --split train \
  --output_dir "sgg/cache/image_features" \
  --device cuda \
  --feature_dim 256 \
  --image_size 1008 \
  --start_idx 0 \
  --max_samples 50000 \
  --batch_size 1 \
  --num_workers 0 \
  --num_gpus 1 \
  --checkpoint_path "weights/sam3.pt"

# 进程 B（卡 1）：处理后半段
CUDA_VISIBLE_DEVICES=1 python sgg/precompute/image_feature_generator.py \
  --data_root "/home/shi/桌面/abschluss/sgg/dataset/vg150" \
  --split train \
  --output_dir "sgg/cache/image_features" \
  --device cuda \
  --feature_dim 256 \
  --image_size 1008 \
  --start_idx 50000 \
  --max_samples 50000 \
  --batch_size 1 \
  --num_workers 0 \
  --num_gpus 1 \
  --checkpoint_path "weights/sam3.pt"
```
- 如需“分段重启”，可在外层包一层循环，每处理 N 张（如 50）后退出进程并重启下一段。
- 若要开启 FP16 保存，额外加 `--save_fp16`（当前脚本默认关闭）。

## 关键实现点
- **SAM3 Backbone**：直接调用 `self.sam3_model.backbone.forward_image()` 获取密集特征图，取 FPN 最高分辨率层。
- **降维投影**：若原通道数 ≠ `feature_dim`，使用 1×1 Conv 投影到目标维度；必要时截断/零填充保持确定性。
- **数据加载**：自定义 `collate_fn_pil_images` 支持 `PIL.Image`；`batch_size=1`、`num_workers=0` 避免嵌套多进程内存膨胀。
- **内存管理**：单图处理后立即 `del` 中间张量并调用 `torch.cuda.empty_cache()`；保存前搬到 CPU。
- **健壮性**：失败样本落盘零特征占位，保证索引对齐；`start_idx`/`max_samples` 支持静态切分。

## 常见配置建议
- 内存紧张：保持 `BATCH_SIZE=1`、`NUM_WORKERS=0`，必要时改为单 GPU（`--num_gpus 1`）。
- 路径迁移：`CHECKPOINT_PATH` 支持相对路径（默认查找 `weights/sam3.pt` 等）；`VG_ROOT` 可写绝对路径。
- 校验：可先用 `LIMIT=100` 快速冒烟测试，再跑全量。

## 已知行为
- 多进程模式会各自加载一份模型与数据元信息，显存/内存占用约为单进程的 2×（符合“独立进程”策略）。
- 目前默认关闭 FP16 保存；开启需添加 `--save_fp16` 或脚本中将 `SAVE_FP16` 设为 1。

