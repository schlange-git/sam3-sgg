\subsection{Detection Pretraining Results}
\label{subsec:detection_pretraining_results_cn}

在评估完整的场景图生成模型之前，本文首先分析在 Action Genome 数据集上进行 detection-only pretraining 后得到的检测 checkpoint。该阶段的目的并不是获得最终最优的独立目标检测器，而是为后续 subject--predicate--object 联合训练提供更稳定的目标检测分支初始化。由于场景图生成依赖于正确检测到的 subject 和 object 实例，因此具有更高目标覆盖率的检测器能够为后续关系学习提供更好的基础。

表~\ref{tab:detection_pretraining_comparison_cn} 对比了 ResNet-101 和 SAM3 两种 pretrained checkpoint 的目标检测性能。可以看到，SAM3-based checkpoint 在 AP、AP$_{50}$ 和 Object Recall@50 上均高于 ResNet-101 checkpoint。尤其是 Object Recall@50 从 54.565\% 提升到 65.200\%，说明 SAM3-based pretrained detector 能够覆盖更多 ground-truth objects。

\begin{table}[H]
\centering
\small
\caption{Action Genome 上不同 pretrained checkpoints 的目标检测性能对比。}
\label{tab:detection_pretraining_comparison_cn}
\begin{tabular}{lccc}
\toprule
模型 & AP & AP$_{50}$ & Object Recall@50 (\%) \\
\midrule
ResNet-101 pretrained checkpoint & 9.338 & 16.837 & 54.565 \\
SAM3 pretrained checkpoint       & 10.645 & 21.732 & 65.200 \\
\midrule
绝对提升                         & +1.307 & +4.895 & +10.635 \\
\bottomrule
\end{tabular}
\end{table}

表~\ref{tab:sam3_pretrain_detection_detail_cn} 进一步展示了 SAM3 pretrained checkpoint 的详细检测结果。虽然其绝对 AP 数值仍然不高，但 Object Recall@50 达到 65.2\%。这说明该模型已经能够召回相当一部分 ground-truth objects，只是在置信度排序和定位精度方面仍不足以获得较高的 AP。

\begin{table}[H]
\centering
\small
\caption{SAM3 pretrained checkpoint 在 Action Genome 上的详细检测性能。}
\label{tab:sam3_pretrain_detection_detail_cn}
\begin{tabular}{ccccccc}
\toprule
AP & AP$_{50}$ & AP$_{75}$ & AP$_s$ & AP$_m$ & AP$_l$ & Object Recall@50 \\
\midrule
10.645 & 21.732 & 8.916 & 1.305 & 7.639 & 16.113 & 65.200 \\
\bottomrule
\end{tabular}
\end{table}

这些指标需要结合起来解释，而不能孤立比较。AP 是一个较严格的检测指标，会受到定位质量、置信度校准以及预测排序等因素影响。相比之下，Object Recall@50 更直接反映模型是否能够为后续关系预测提供足够的目标候选。在场景图生成任务中，如果 subject 或 object 没有被检测出来，那么对应的关系三元组就不可能被正确预测。因此，SAM3 pretrained checkpoint 在 object recall 上的提升对于后续完整 SGG 训练尤其重要。

该结果也说明，在 detection pretraining 阶段，SAM3 backbone 相比原始 ResNet-101 backbone 提供了更强的目标级视觉表征能力。然而，这种目标检测层面的提升并不必然直接转化为最终场景图生成性能的提升。关系预测还依赖 predicate classification、triplet-level matching 以及 subject、object 和 relation queries 之间的交互。因此，detection pretraining results 更应被理解为一种更强 object initialization 的证据，而不是对最终 SGG 框架的完整评价。

总体而言，detection pretraining 实验证明，SAM3-based checkpoint 能够为目标级预测提供更强的初始状态。其更高的 object recall 和 AP$_{50}$ 表明，该 checkpoint 能够为后续关系推理提供更可靠的 subject/object candidates。因此，在后续 SAM3-based SpeaQ 实验中使用该 detection pretrained checkpoint 作为初始化是合理的。