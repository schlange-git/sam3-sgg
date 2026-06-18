import os, csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = "/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching"
RUN  = os.path.join(PROJ, "z_outputs/xsam_local_pretrain_bs16_iter160000")
CSVP = os.path.join(RUN, "diagnostics_log.csv")
CKPT = os.path.join(RUN, "model_0159999.pth")
OUT  = os.path.join(RUN, "patch_merge_analysis")
os.makedirs(OUT, exist_ok=True)

# ============ A. csv 标量趋势 ============
rows = []
with open(CSVP) as f:
    for row in csv.DictReader(f):
        rows.append(row)
col = lambda k: np.array([float(r[k]) for r in rows])
it    = col("iter")
dnorm = col("xsam_delta_norm")
wstd  = col("xsam_weight_std")
gnorm = col("xsam_grad_norm")
unorm = col("xsam_update_norm")
initn = col("xsam_init_norm")

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0,0].plot(it, dnorm, label="delta_norm ||W-init||")
ax[0,0].axhline(initn[-1], ls="--", c="r", label=f"init_norm={initn[-1]:.2f}")
ax[0,0].set_title(f"deviation from init   (end ratio = {dnorm[-1]/initn[-1]:.1%})")
ax[0,0].legend(); ax[0,0].set_xlabel("iter")
ax[0,1].plot(it, wstd, c="g"); ax[0,1].set_title("weight std (distribution width)"); ax[0,1].set_xlabel("iter")
ax[1,0].plot(it, gnorm, c="orange"); ax[1,0].set_title("grad_norm"); ax[1,0].set_xlabel("iter")
ax[1,1].plot(it, unorm, c="purple"); ax[1,1].set_yscale("log"); ax[1,1].set_title("update_norm (log)"); ax[1,1].set_xlabel("iter")
fig.suptitle("A. patch_merge conv — scalar trends (diagnostics_log.csv)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "A_csv_trends.png"), dpi=110); plt.close(fig)
print("[A] saved A_csv_trends.png")

# ============ B. 权重 3-way 对比 ============
sd = torch.load(CKPT, map_location="cpu")
sd = sd.get("model", sd) if isinstance(sd, dict) else sd
keys = [k for k in sd if "_patch_merge_proj" in k]
print("[B] patch_merge keys:", keys)
W = sd[keys[0]].float().squeeze().numpy()           # [C, IN]
C, IN = W.shape
ff = IN // C
f = int(round(ff ** 0.5))
assert f * f == ff, f"bad shape {W.shape}"
print(f"[B] W shape={W.shape}  C={C} IN={IN} factor={f}")

# (a) 正确 channel-major avg_pool init: W[o, o*ff + s] = 1/ff
Wavg = np.zeros((C, IN), dtype=np.float32)
for o in range(C):
    for s in range(ff):
        Wavg[o, o*ff + s] = 1.0/ff
# (b) 代码 sub-major 错位 init: W[c, c + s*C] = 1/ff
Wmis = np.zeros((C, IN), dtype=np.float32)
for c in range(C):
    for s in range(ff):
        Wmis[c, c + s*C] = 1.0/ff

nrm = lambda a: float(np.linalg.norm(a))
print(f"[B] ||W_t||={nrm(W):.3f}  ||W_avg||={nrm(Wavg):.3f}  ||W_mis||={nrm(Wmis):.3f}")
print(f"[B] ||W_t - W_avg||={nrm(W-Wavg):.3f}   ||W_t - W_mis||={nrm(W-Wmis):.3f}")
avg_mask = Wavg > 0
mis_mask = Wmis > 0
tot = float((W**2).sum())
e_avg = float((W[avg_mask]**2).sum())
e_mis = float((W[mis_mask]**2).sum())
union = avg_mask | mis_mask
e_off = float((W[~union]**2).sum())
print(f"[B] energy on avg-support = {e_avg/tot:.2%}")
print(f"[B] energy on mis-support = {e_mis/tot:.2%}")
print(f"[B] energy off both       = {e_off/tot:.2%}")
print(f"[B] avg/mis support overlap cells = {int((avg_mask & mis_mask).sum())}")

# B1 直方图
lim_lo, lim_hi = float(np.percentile(W, 0.05)), float(np.percentile(W, 99.95))
bins = np.linspace(min(lim_lo, -0.05), max(lim_hi, 0.30), 220)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(Wavg.ravel(), bins=bins, alpha=0.5, label="avg_pool init (correct)", log=True)
ax.hist(Wmis.ravel(), bins=bins, alpha=0.5, label="miscode init (as coded)", log=True)
ax.hist(W.ravel(),    bins=bins, alpha=0.5, label="trained W_t",            log=True)
ax.set_title("B1. weight value distribution (log count)")
ax.set_xlabel("weight value"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "B1_hist.png"), dpi=110); plt.close(fig)

# B2 热力图子块
sl = (slice(0, 64), slice(0, 160))
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, (M, t) in zip(ax, [(Wavg, "avg_pool (correct)"), (Wmis, "miscode (as coded)"), (W, "trained W_t")]):
    im = a.imshow(M[sl], aspect="auto", cmap="RdBu_r", vmin=-0.25, vmax=0.25)
    a.set_title(t); a.set_xlabel("input subchannel j"); a.set_ylabel("output channel o")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("B2. weight matrix [0:64, 0:160] — structure")
fig.savefig(os.path.join(OUT, "B2_heatmap.png"), dpi=110); plt.close(fig)

# B3 每输出通道能量占比 (互斥三集)
only_mis = mis_mask & ~avg_mask
off_mask = ~union
ea = (W**2 * avg_mask).sum(1)
em = (W**2 * only_mis).sum(1)
eo = (W**2 * off_mask).sum(1)
et = (W**2).sum(1)
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(C)
ax.stackplot(x, ea/et, em/et, eo/et, labels=["avg-support", "mis-support (only)", "off-both"], alpha=0.85)
ax.set_title("B3. per output-channel energy fraction")
ax.set_xlabel("output channel o"); ax.set_ylim(0, 1); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "B3_energy.png"), dpi=110); plt.close(fig)
print("[B] saved B1_hist.png B2_heatmap.png B3_energy.png")
print("ALL DONE ->", OUT)
