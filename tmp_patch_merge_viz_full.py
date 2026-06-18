import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = "/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching"
RUN  = os.path.join(PROJ, "z_outputs/xsam_local_pretrain_bs16_iter160000")
CKPT = os.path.join(RUN, "model_0159999.pth")
OUT  = os.path.join(RUN, "patch_merge_analysis")
os.makedirs(OUT, exist_ok=True)

sd = torch.load(CKPT, map_location="cpu")
sd = sd.get("model", sd) if isinstance(sd, dict) else sd
keys = [k for k in sd if "_patch_merge_proj" in k]
W = sd[keys[0]].float().squeeze().numpy()      # [C, IN] = [256, 1024]
C, IN = W.shape
ff = IN // C
print(f"W shape={W.shape}  C={C} IN={IN} ff={ff}")

# 正确 channel-major avg_pool: W[o, o*ff + s] = 1/ff
Wavg = np.zeros((C, IN), np.float32)
for o in range(C):
    for s in range(ff):
        Wavg[o, o*ff + s] = 1.0/ff
# 代码 miscode (sub-major): W[c, c + s*C] = 1/ff
Wmis = np.zeros((C, IN), np.float32)
for c in range(C):
    for s in range(ff):
        Wmis[c, c + s*C] = 1.0/ff

# ---- 量化: 训练后 W 的能量分别落在 avg 对角 / miscode 4 条对角上 ----
tot = float((W**2).sum())
e_avg = float((W**2 * (Wavg > 0)).sum())
print(f"[trained] total energy = {tot:.3f}")
print(f"[trained] energy on AVG diagonal (slope4)   = {e_avg/tot:.2%}")
for s in range(ff):
    cols = (np.arange(C) + s*C)
    e = float((W[np.arange(C), cols]**2).sum())
    print(f"[trained] energy on MISCODE diag s={s} (offset {s*C}) = {e/tot:.2%}")
e_mis = float((W**2 * (Wmis > 0)).sum())
print(f"[trained] energy on MISCODE 4 diagonals total = {e_mis/tot:.2%}")

# ============ 全范围热力图 (不裁剪) ============
panels = [
    (Wavg, "1) avg_pool init (correct, channel-major)  ->  1 条 slope-4 对角", 0.25),
    (Wmis, "2) miscode init (as coded = 训练起点)  ->  4 条 slope-1 平行对角 @ col 0/256/512/768", 0.25),
    (W,    "3) trained W_t  (vmin/vmax = +-0.25, 和上面同尺度)", 0.25),
    (W,    "4) trained W_t  (vmin/vmax = +-0.05, 放大对比度 -> 残影更清楚)", 0.05),
]
fig, axes = plt.subplots(len(panels), 1, figsize=(16, 17))
for ax, (M, title, lim) in zip(axes, panels):
    im = ax.imshow(M, aspect="equal", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest", extent=[0, IN, C, 0])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("input subchannel j  (0..1023)   [j = 4*通道 + 子像素]")
    ax.set_ylabel("output channel o (0..255)")
    fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
fig.suptitle("B2_FULL. patch_merge weight  [256 x 1024] 全范围 (aspect=equal, 斜率为真实几何)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(os.path.join(OUT, "B2_full_heatmap.png"), dpi=130)
plt.close(fig)
print("saved B2_full_heatmap.png ->", OUT)
