\section{评估协议与评价指标}
\label{sec:evaluation_protocol_metrics_cn}

本节介绍本文实验中采用的评估协议与评价指标。由于本文框架同时涉及目标检测和场景图生成两个层面的任务，因此实验评估也从两个互补角度展开。首先，目标级检测指标用于衡量模型是否能够正确定位和分类图像中的目标实例。其次，三元组级场景图生成指标用于评估模型是否能够正确预测 subject--predicate--object 结构。区分这两个层面的指标是必要的，因为模型即使具有较好的目标检测性能，也不一定能够准确预测视觉关系；反过来，关系预测的性能也受到 subject 和 object 检测质量的直接限制。

\subsection{评估设置}

本文所有实验均在 Action Genome 数据集上进行，并采用 scene graph detection 设置。在该设置下，模型在推理阶段只接收图像或视频帧作为输入，需要同时预测目标实例和语义关系，而不能使用 ground-truth object boxes。因此，相比于给定目标框的 predicate classification，该任务更具挑战性，因为最终场景图结果同时依赖目标定位、目标分类和谓词分类。

对于视频实验，本文主要在带有标注的关键帧上进行评估。非关键帧可以作为时序上下文参与模型推理或特征传播，但如果没有对应的 ground-truth 标注，则不直接参与最终指标计算。这样的设置可以保证实验结果与 Action Genome 的帧级评估协议保持一致，同时允许模型利用相邻帧中的时序信息。

模型输出的场景图可以表示为一个按置信度排序的三元组集合：
\begin{equation}
\hat{\mathcal{T}}
=
\left\{
(\hat{s}_j, \hat{p}_j, \hat{o}_j)
\right\}_{j=1}^{N_r},
\end{equation}
其中，$\hat{s}_j$ 和 $\hat{o}_j$ 分别表示预测的 subject 实例和 object 实例，$\hat{p}_j$ 表示预测的 predicate 类别。每个 subject 或 object 实例都包含类别标签和边界框。只有当 subject 类别、object 类别、predicate 类别以及 subject/object 的定位结果同时满足匹配要求时，该预测三元组才会被认为是正确的。

\subsection{目标检测指标}

目标检测性能采用 COCO 风格的 Average Precision 和目标召回率进行评估。Average Precision，简称 AP，用于衡量模型在不同 intersection-over-union 阈值下的检测质量，同时考虑分类置信度和定位准确性。本文报告以下 AP 指标：
\begin{equation}
\mathrm{AP}, \quad
\mathrm{AP}_{50}, \quad
\mathrm{AP}_{75}, \quad
\mathrm{AP}_{s}, \quad
\mathrm{AP}_{m}, \quad
\mathrm{AP}_{l}.
\end{equation}
其中，$\mathrm{AP}$ 表示多个 IoU 阈值下的平均 AP，$\mathrm{AP}_{50}$ 和 $\mathrm{AP}_{75}$ 分别表示 IoU 阈值为 0.50 和 0.75 时的 AP。$\mathrm{AP}_{s}$、$\mathrm{AP}_{m}$ 和 $\mathrm{AP}_{l}$ 分别用于衡量小目标、中等目标和大目标上的检测性能。

除了 AP 之外，本文还报告 Object Recall@50。Object Recall@50 衡量 top 50 个预测目标框能够覆盖多少 ground-truth objects。当预测目标的类别正确，并且其边界框与对应 ground-truth box 具有足够高的重合度时，该 ground-truth object 被视为成功匹配。Object Recall@50 可定义为：
\begin{equation}
\mathrm{Object\ Recall@50}
=
\frac{N_{\mathrm{matched}}}{N_{\mathrm{GT}}},
\end{equation}
其中，$N_{\mathrm{matched}}$ 表示成功匹配的 ground-truth objects 数量，$N_{\mathrm{GT}}$ 表示 ground-truth objects 的总数。

Object recall 对场景图生成尤其重要。由于关系预测建立在检测到的 subject 和 object 实例之上，漏检目标会直接降低可被正确预测的关系数量上限。因此，本文在分析检测性能时不会只看 AP，而是将 AP 与 Object Recall@50 结合解释。AP 反映整体排序检测质量，而 Object Recall@50 更直接反映模型是否能够为后续关系预测提供足够的目标候选。

\subsection{场景图生成指标}

场景图生成性能在三元组级别进行评估。主要指标为 Recall@K，记作 R@K。对于每张图像或每一帧，模型会输出一个按置信度排序的预测三元组列表。R@K 衡量 ground-truth triplets 中有多少能够在 top $K$ 个预测三元组内被正确召回：
\begin{equation}
\mathrm{R@K}
=
\frac{
\left|
\mathcal{T}_{\mathrm{GT}}
\cap
\hat{\mathcal{T}}_{1:K}
\right|
}{
\left|
\mathcal{T}_{\mathrm{GT}}
\right|
},
\end{equation}
其中，$\mathcal{T}_{\mathrm{GT}}$ 表示 ground-truth triplets 集合，$\hat{\mathcal{T}}_{1:K}$ 表示模型预测的 top $K$ 个三元组。本文主要报告 R@20、R@50 和 R@100。

一个预测三元组只有在三个语义组成部分和两个实体定位结果均正确时，才会被判定为正确。具体而言，预测的 subject 类别和 object 类别必须与 ground truth 匹配，predicate 类别必须正确，并且预测的 subject/object boxes 必须与对应 ground-truth boxes 具有足够的空间重合度。因此，SGG Recall@K 比普通目标检测 recall 更严格，因为实体检测错误或谓词分类错误都会导致整个三元组预测失败。

除标准 Recall@K 外，本文还报告 no-graph-constraint Recall@K，记作 ng-R@K。在标准 graph-constrained evaluation 中，每个有序 subject--object pair 通常只允许保留一个 predicate prediction。而在 no-graph-constraint evaluation 中，同一个 subject--object pair 可以保留多个 predicate hypotheses。因此，ng-R@K 通常高于标准 R@K，并且可以额外反映模型是否学习到了合理的 predicate 分布，即使最终受到 graph constraint 后排序结果并不完全理想。

Mean Recall@K，记作 mR@K，用于缓解高频关系类别对整体指标的主导作用。标准 Recall@K 容易受到 head predicates 的影响，因为常见关系在数据集中出现频率远高于尾部关系。mR@K 则首先分别计算每个 predicate 类别的 recall，然后对所有 predicate classes 取平均：
\begin{equation}
\mathrm{mR@K}
=
\frac{1}{C_p}
\sum_{c=1}^{C_p}
\mathrm{R@K}_{c},
\end{equation}
其中，$C_p$ 表示 predicate classes 的数量，$\mathrm{R@K}_{c}$ 表示第 $c$ 个 predicate 类别的 recall。由于 Action Genome 中存在明显的 object 和 predicate 长尾分布，mR@K 对分析模型在低频类别上的表现尤其重要。

Zero-Shot Recall@K，记作 zR@K，用于评估模型对训练阶段未出现过的 subject--predicate--object 组合的预测能力。该指标关注的是模型对未见组合的组合泛化能力，而不仅仅是对训练集中高频三元组的记忆。尽管在数据有限且长尾偏置明显的场景下，zR@K 通常较难提升，但它仍然是衡量场景图模型组合泛化能力的重要参考指标。

\subsection{指标解释方式}

本文对上述指标进行联合解释，而不是孤立地比较单一数值。Object AP 反映检测预测的整体排序质量，但它对置信度校准、定位精度和重复预测较为敏感。Object Recall@50 则更关注预测集合是否覆盖了足够多的 ground-truth objects。对于场景图生成，R@K 衡量最终三元组检索质量，ng-R@K 反映在放宽 graph constraint 后的潜在 predicate 预测能力，而 mR@K 则强调模型在低频关系类别上的表现。

这种区分对于分析本文提出的不同模块非常重要。例如，backbone 替换可能提升 object recall，但并不一定立即提升 triplet-level R@K，因为关系预测仍然依赖 predicate classification 和 triplet matching。类似地，ROI refinement 主要改善小物体类别判别能力，因此它的作用更适合通过 per-category recall 和 classification-error analysis 进行观察，而不应只依赖全局 AP。Temporal query injection 可能通过相邻帧上下文提升关系召回，即使它对单独目标检测 AP 的提升并不显著。因此，后续实验分析将同时报告目标级指标和三元组级指标，以更完整地评估本文框架的有效性。