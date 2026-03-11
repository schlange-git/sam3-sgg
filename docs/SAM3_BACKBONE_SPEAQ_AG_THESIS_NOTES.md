# SAM3 Backbone、SpeaQ 接口适配与 Action Genome 抽帧/GT 结构说明（论文可粘贴版）

> 代码基线与改动参照：`feb80f9e7172a0e0915fd045e21301c745764c9e`、`f12a6a26a44ac188ce69ed1c23c83adb65655eb8`。  
> 下述内容采用论文写作风格，正文为 LaTeX 结构，可直接复制到 thesis `.tex`。

---

## 可直接粘贴的 LaTeX 正文

```tex
\section{SAM3 作为 Backbone 的特征获取与多步长拆分机制}
\label{sec:sam3_backbone_feature_extraction}

\subsection{统一输入尺寸与特征提取入口}

本工作将 SAM3 封装为 Detectron2/SpeaQ 可调用的视觉主干，核心实现位于 \texttt{SpeaQ/modeling/backbone/sam3\_backbone.py}。输入图像首先经过统一预处理：
\begin{enumerate}
    \item 转换为 \texttt{uint8}；
    \item 强制 resize 到 \texttt{(IMAGE\_SIZE, IMAGE\_SIZE)}；
    \item 转换为 \texttt{float32}；
    \item 归一化到 \texttt{[-1,1]}（均值/方差均为 0.5）。
\end{enumerate}

因此，无论上游数据增广如何设置，进入 SAM3 视觉编码器的空间分辨率均固定为
\[
H_{in} = W_{in} = \texttt{IMAGE\_SIZE}.
\]
在当前配置中，\texttt{IMAGE\_SIZE=1008}，故有 $H_{in}=W_{in}=1008$。

\subsection{单尺度/多尺度特征组织}

设 SAM3 主干输出候选特征为 $\mathbf{F}\in \mathbb{R}^{B\times C\times H_f\times W_f}$。系统支持两类路径：
\begin{itemize}
    \item \textbf{单尺度路径}：优先取 \texttt{backbone\_fpn[-1]}，若不存在则回退到 \texttt{vision\_features}；
    \item \textbf{多尺度路径}：读取 \texttt{backbone\_fpn} 全部层级，并对每层执行通道投影与 stride 估计。
\end{itemize}

对于第 $l$ 层特征 $\mathbf{F}_l$，其步长估计为
\[
s_l=\max\left(1,\mathrm{round}\left(\frac{H_{in}}{H_l}\right)\right).
\]
系统按 $s_l$ 升序排序得到多尺度特征集合 $\{(s_l,\mathbf{F}_l)\}$，并可通过 \texttt{last/sum/concat} 三种策略输出。

\subsection{维度与参数变换分析（含固定 resize 情况）}

通道维度经 $1\times 1$ 投影统一到
\[
C_{out}=\texttt{FEATURE\_DIM}=256.
\]
记投影后特征为 $\hat{\mathbf{F}}\in\mathbb{R}^{B\times 256\times H_f\times W_f}$。随后依据配置 \texttt{TARGET\_STRIDE} 进行可选下采样：
\[
\text{if } s_f < s_t,\quad
\hat{\mathbf{F}} \leftarrow \mathrm{AvgPool2d}(\hat{\mathbf{F}}, \mathrm{kernel}=s_t/s_f,\mathrm{stride}=s_t/s_f).
\]
当前配置中 \texttt{TARGET\_STRIDE=32}。若 SAM3 原生输出约为 $s_f\approx16$，则会再降采样一次得到约 $s=32$ 的输出特征图。由于输入固定为 $1008\times1008$，可推得：
\[
H_f\approx \frac{1008}{16}=63,\qquad
H_{out}\approx \left\lfloor\frac{63}{2}\right\rfloor=31,
\]
即最终主干输入 Transformer 前的空间尺度通常约为 $31\times31$（实际值受 SAM3 内部 patch/neck 细节影响，但 stride 估计在运行时动态修正）。

\subsection{核心代码块}

\begin{lstlisting}[language=Python,caption={SAM3 输入统一 resize 与归一化}]
self.transform = v2.Compose(
    [
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(self.image_size, self.image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
\end{lstlisting}

\begin{lstlisting}[language=Python,caption={步长估计与目标步长对齐}]
h_in = image_batch.shape[2]
h_feat = proj_feat.shape[2]
self.feature_stride = max(1, int(round(h_in / float(h_feat))))
if self.target_stride and self.feature_stride < self.target_stride:
    factor = self.target_stride // self.feature_stride
    if factor > 1:
        proj_feat = F.avg_pool2d(proj_feat, kernel_size=factor, stride=factor)
        self.feature_stride = self.feature_stride * factor
\end{lstlisting}

\begin{lstlisting}[language=Python,caption={多尺度层级的 stride 拆分}]
stride = max(1, int(round(float(image_h) / float(proj_feat.shape[2]))))
features.append(proj_feat)
strides.append(stride)
return sorted(zip(strides, features), key=lambda x: x[0])
\end{lstlisting}


\section{SAM3 特征输出到 SpeaQ 主干网络的接口适配}
\label{sec:sam3_to_speaq_interface}

\subsection{接口目标与数据结构}

SpeaQ 的 DETR 变体期望 backbone 输出为 \texttt{Dict[str, NestedTensor]}，其中每个 \texttt{NestedTensor} 含：
\begin{itemize}
    \item \texttt{tensors}: $\mathbb{R}^{B\times C\times H\times W}$；
    \item \texttt{mask}: $\mathbb{R}^{B\times H\times W}$（padding 区域为 1）。
\end{itemize}
为与该接口兼容，SAM3 包装器输出键值对（如 \texttt{\{"sam3": NestedTensor(...)\}} 或 \texttt{fpn\_*}），随后由 \texttt{Joiner} 统一转为列表 \texttt{features, pos}。

\subsection{接入路径}

主干前向关键链路如下：
\[
\texttt{ImageList} \rightarrow \texttt{SAM3Backbone} \rightarrow \texttt{Dict[str, NestedTensor]}
\rightarrow \texttt{Joiner} \rightarrow (\texttt{features},\texttt{pos})
\rightarrow \texttt{DETR Transformer}.
\]

在 DETR 前向中，系统取最后一层特征并送入输入投影层：
\[
\mathbf{z} = \mathrm{InputProj}(\mathbf{F}_{last}),\quad
\mathbf{y} = \mathrm{Transformer}(\mathbf{z}, \mathbf{m}, \mathbf{q}, \mathbf{p}),
\]
其中 $\mathbf{m}$ 为 mask，$\mathbf{q}$ 为 query embedding，$\mathbf{p}$ 为位置编码。

\subsection{冻结/解冻与梯度流控制}

SAM3 的冻结状态由 \texttt{MODEL.SAM3.FREEZE} 控制。实现中通过以下机制保证“配置--行为”一致：
\begin{itemize}
    \item \texttt{freeze=True}：\texttt{sam3\_model.eval()} 且 \texttt{requires\_grad=False}；
    \item \texttt{freeze=False}：\texttt{sam3\_model.train()} 且启用梯度；
    \item 前向时使用 \texttt{torch.set\_grad\_enabled(not self.freeze)} 精确控制计算图构建。
\end{itemize}

\subsection{核心代码块}

\begin{lstlisting}[language=Python,caption={Joiner 将字典特征转为 DETR 可消费形式}]
xs = self[0](tensor_list)
out, pos = [], []
for name, x in xs.items():
    out.append(x)
    pos.append(self[1](x).to(x.tensors.dtype))
return out, pos
\end{lstlisting}

\begin{lstlisting}[language=Python,caption={DETR 接收 backbone 特征并进入 Transformer}]
features, pos = self.backbone(samples)
src, mask = features[-1].decompose()
output = self.transformer(
    self.input_proj(src), mask,
    self.query_embed.weight,
    self.object_query_embed.weight,
    self.relation_query_embed.weight,
    pos[-1]
)
\end{lstlisting}

\begin{lstlisting}[language=Python,caption={SAM3 冻结状态驱动的梯度开关}]
if self.freeze:
    self.sam3_model.eval()
    for p in self.sam3_model.parameters():
        p.requires_grad_(False)
else:
    self.sam3_model.train()
...
with torch.set_grad_enabled(not self.freeze):
    backbone_out = self.sam3_model.backbone.forward_image(image_batch)
\end{lstlisting}


\section{Action Genome 抽帧逻辑、评测帧匹配与 GT 读取结构}
\label{sec:ag_frame_dump_and_gt}

\subsection{抽帧脚本行为}

\texttt{prepare\_actiongenome\_frames.sh} 调用
\texttt{data/ActionGenome/tools/dump\_frames.py} 完成抽帧。核心逻辑如下：
\begin{enumerate}
    \item 读取 \texttt{annotation\_dir/frame\_list.txt}，其元素形如 \texttt{video\_id/frame\_name}；
    \item 逐视频执行 \texttt{ffmpeg} 全量解码到 \texttt{frame\_dir/video\_id/\%06d.png}；
    \item 若未启用 \texttt{--all\_frames}，则仅保留 \texttt{frame\_list.txt} 中标注过的帧，其余帧删除。
\end{enumerate}
该设计保证了“训练/评测所需帧”与“注释可用帧”严格一致。

\subsection{评测帧匹配机制}

在 AG 数据适配器中（\texttt{SpeaQ/data/datasets/action\_genome.py}），系统按以下顺序构建样本：
\begin{enumerate}
    \item 从 \texttt{frame\_list.txt} 逐条读取键 \texttt{video/frame};
    \item 根据 split(video 级划分) 过滤视频集合；
    \item 在 \texttt{frames\_root/video/frame} 查找图像，若后缀不一致则尝试 \texttt{png/jpg/jpeg}；
    \item 读取图像宽高，加载 person/object 标注与关系，构建单帧样本记录。
\end{enumerate}

\subsection{标准 GT 数据结构（可复用）}

每个样本记录采用 Detectron2 标准字典，核心字段如下：

\begin{lstlisting}[language=Python,caption={Action Genome 单帧 GT 样本结构}]
record = {
    "file_name": frame_path,
    "image_id": image_id,
    "height": height,
    "width": width,
    "video_id": video,
    "frame_id": frame_name,
    "annotations": [
        {
            "bbox": [x1, y1, x2, y2],   # XYXY_ABS
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": cls_id,      # object_classes.txt 对应索引
            "attribute": np.zeros((1,), dtype=np.int64),
        },
        ...
    ],
    "relations": np.array([[sub_idx, obj_idx, rel_id], ...], dtype=np.int64),
}
\end{lstlisting}

其中：
\begin{itemize}
    \item \texttt{relations[:,0]} 与 \texttt{relations[:,1]} 是 \texttt{annotations} 下标；
    \item \texttt{rel\_id} 来源于 \texttt{relationship\_classes.txt}；
    \item 空关系样本采用 \texttt{shape=(0,3)} 的 \texttt{int64} 数组，保证下游张量类型稳定。
\end{itemize}

\subsection{GT 解析要点分析}

\begin{enumerate}
    \item \textbf{person 优先入表}：person 框先写入 \texttt{annotations}，后续 object 的关系常默认关联到 person 索引；
    \item \textbf{bbox 兼容解析}：支持 \texttt{\{x,y,w,h\}}、\texttt{\{x1,y1,x2,y2\}} 及长度为 4 的数组/列表；
    \item \textbf{关系源聚合}：\texttt{attention/spatial/contacting/relationships} 多字段统一汇聚为关系标签；
    \item \textbf{语义词表一致性}：类别和谓词索引严格由注释目录中的两份 class 文件定义，避免训练/评测索引漂移。
\end{enumerate}

\subsection{核心代码块}

\begin{lstlisting}[language=Python,caption={仅保留标注帧的抽帧逻辑}]
with open(os.path.join(annotation_dir, 'frame_list.txt'), 'r') as f:
    frame_list = [x.rstrip('\n') for x in f]
...
os.system('ffmpeg -loglevel panic -i %s/%s %s/%%06d.png' % (...))
if not all_frames:
    keep_frames = video2frames[v]
    frames_to_delete = set(os.listdir(curr_frame_dir)) - set(keep_frames)
    for frame in frames_to_delete:
        os.remove(os.path.join(curr_frame_dir, frame))
\end{lstlisting}

\begin{lstlisting}[language=Python,caption={Action Genome 中关系数组构建}]
relations = []
...
for rel_name in self._extract_rel_labels(obj):
    rel_id = self.predicate_to_idx.get(rel_name, None)
    if rel_id is not None:
        relations.append([0, obj_idx, rel_id])
...
"relations": np.array(relations, dtype=np.int64) \
    if len(relations) > 0 else np.zeros((0, 3), dtype=np.int64)
\end{lstlisting}

```

---

## 备注（写作建议）

- 若论文模板已加载 `listings` 宏包，以上 `lstlisting` 可直接编译。
- 若模板使用 `minted`，可将 `lstlisting` 替换为 `minted` 环境，正文逻辑不变。
- 若你需要，我可以继续补一版“图示版”（含流程图 TikZ 代码）：  
  1) 输入到 SAM3 到 SpeaQ 的张量流；  
  2) AG 抽帧与 GT 构建的数据流。


