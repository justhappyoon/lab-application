"""파지 타당성 비율 계산 + 인공 노이즈 스윕 + 그래프."""
import csv
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import OBJECTS, CONDITIONS, SIGMAS, REPEATS
from lib import make_reference, sample_grasps, validity_on_ref, add_noise, chamfer

rows = [["name", "sigma_mm", "n_valid_on_recon_mean", "validity_mean", "validity_std",
         "chamfer_mean_mm"]]
fig, ax = plt.subplots(figsize=(6, 4))
for obj, (kind, dims) in OBJECTS.items():
    ref = make_reference(kind, dims)
    for cond in CONDITIONS:
        name = f"{obj}_{cond}"
        base = trimesh.load(f"work/{name}_aligned.ply", force="mesh")
        means = []
        for sg in SIGMAS:
            vals, cnts, chs = [], [], []
            for k in range(REPEATS):
                m = add_noise(base, sg, seed=k) if sg > 0 else base
                p, q = sample_grasps(m, n=5000, seed=k)
                vals.append(validity_on_ref(p, q, ref))
                cnts.append(len(p))
                chs.append(chamfer(m, ref, n=5000, seed=k)[2])
            rows.append([name, sg, f"{np.mean(cnts):.0f}", f"{np.nanmean(vals):.3f}",
                         f"{np.nanstd(vals):.3f}", f"{np.mean(chs):.3f}"])
            means.append(np.nanmean(vals))
            print(rows[-1])
        ax.plot(SIGMAS, means, marker="o", label=name)

ax.set_xlabel("added noise sigma (mm)")
ax.set_ylabel("grasp validity ratio")
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/validity_vs_noise.png", dpi=150)
with open("results/grasp.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
print("저장: results/grasp.csv, results/validity_vs_noise.png")
