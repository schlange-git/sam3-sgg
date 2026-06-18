import os
import torch
import torch.nn.functional as F
import numpy as np

print("=" * 70)
print("PART A: pixel_unshuffle layout + 'is the code init really avg_pool?'")
print("=" * 70)
fd, factor = 2, 2
incc = fd * factor * factor  # 8
# md 里的例子: 通道A(c=0)=[[1,2],[5,6]], 通道B(c=1)=[[100,200],[500,600]]
x = torch.tensor([[[[1., 2.], [5., 6.]],
                   [[100., 200.], [500., 600.]]]])     # (1, 2, 2, 2)
u = F.pixel_unshuffle(x, factor).flatten()             # (8,)
print(f"input chA=[[1,2],[5,6]] chB=[[100,200],[500,600]]")
print(f"pixel_unshuffle output u = {u.tolist()}")
print(f"  -> 若是 [1,2,5,6,100,200,500,600] = channel-major (chA 4子像素连续在前)")
print(f"  -> 若是 [1,100,2,200,5,500,6,600] = sub-major   (按子像素交错)")

# 代码实际 init: W[c, c + s*fd] = 1/4
Wcode = torch.zeros(fd, incc)
for c in range(fd):
    for dy in range(factor):
        for dx in range(factor):
            s = dy * factor + dx
            Wcode[c, c + s * fd] = 1.0 / (factor * factor)
# channel-major 正确 init: W[o, o*4 + s] = 1/4
Wcm = torch.zeros(fd, incc)
for o in range(fd):
    for s in range(factor * factor):
        Wcm[o, o * factor * factor + s] = 1.0 / (factor * factor)

code_cols = [[c + s * fd for s in range(4)] for c in range(fd)]
cm_cols = [[o * 4 + s for s in range(4)] for o in range(fd)]
out_code = (Wcode @ u).tolist()
out_cm = (Wcm @ u).tolist()
print(f"\n真值 avg_pool          = [3.5, 350.0]")
print(f"代码 init   抓的列={code_cols}  ->  out={out_code}")
print(f"chan-major  抓的列={cm_cols}  ->  out={out_cm}")
print(f"\n结论: 代码 init {'== ' if np.allclose(out_code,[3.5,350.0]) else '!= '}avg_pool ; "
      f"chan-major {'== ' if np.allclose(out_cm,[3.5,350.0]) else '!= '}avg_pool")

print()
print("=" * 70)
print("PART B: 真实 checkpoint 上 3.51% 相对随机基线显著吗")
print("=" * 70)
PROJ = "/home/cfs/shizekun1_v/sam3-sgg-Auxiliary-Matching"
CKPT = os.path.join(PROJ, "z_outputs/xsam_local_pretrain_bs16_iter160000/model_0159999.pth")
sd = torch.load(CKPT, map_location="cpu")
sd = sd.get("model", sd) if isinstance(sd, dict) else sd
keys = [k for k in sd if "_patch_merge_proj" in k]
W = sd[keys[0]].float().squeeze().numpy()              # [256, 1024]
C, IN = W.shape
ff = IN // C
sq = W ** 2
tot = float(sq.sum())
rows = np.arange(C)

# avg 对角能量 (每行抓 4 个连续列 4o..4o+3)
avg_cells = sq[rows[:, None], (4 * rows[:, None] + np.arange(4)[None, :])].sum()
# miscode 4 条对角 (每行抓 o, o+256, o+512, o+768)
mis_cols = (rows[:, None] + np.arange(4)[None, :] * C)
mis_cells = sq[rows[:, None], mis_cols].sum()
print(f"avg 对角能量    = {avg_cells/tot:.4%}  (1024 cells)")
print(f"miscode 对角能量 = {mis_cells/tot:.4%}  (1024 cells)")

# 均匀基线: 1024/262144
n_cells = C * 4
print(f"\n均匀随机基线 (energy 若平均分布) = {n_cells/(C*IN):.4%}")

# 经验 null: 每行随机抓 4 个不同列, 重复 5000 次, 看能量分布
rng = np.random.default_rng(0)
N = 5000
null = np.empty(N)
for i in range(N):
    cols = rng.integers(0, IN, size=(C, 4))
    null[i] = sq[rows[:, None], cols].sum() / tot
print(f"经验 null (每行随机4列) 均值={null.mean():.4%}  std={null.std():.4%}  "
      f"99.9%分位={np.percentile(null,99.9):.4%}  max={null.max():.4%}")
print(f"avg 对角 {avg_cells/tot:.4%} 相当于 null 均值的 {(avg_cells/tot)/null.mean():.1f}x, "
      f"超出 null 最大值? {'是' if avg_cells/tot > null.max() else '否'}")
print(f"miscode  {mis_cells/tot:.4%} 相当于 null 均值的 {(mis_cells/tot)/null.mean():.1f}x")
print("DONE")
