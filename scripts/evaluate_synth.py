"""
ACCURACY EVALUATION (the 'algorithm accuracy' part of the assignment)
---------------------------------------------------------------------
We know the TRUE direction of the synthetic warming front, so we can measure the error exactly.
Tests: (1) planar fronts: 8 directions x 3 noise levels, (2) body motion with/without registration,
       (3) radial pattern (warming spreading outward from one focus): is the focus found and where?

Run:  python scripts/evaluate_synth.py
Out : results/evaluation/synthetic_accuracy.csv  and  .png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.synth import make_synthetic
from src import pipeline, flow

OUT = "results/evaluation"
os.makedirs(OUT, exist_ok=True)
rows = []


def one(angle, noise, jitter=0.0, register=False, seed=0):
    T, mask, info = make_synthetic(angle, noise=noise, jitter_px=jitter, seed=seed)
    res = pipeline.analyze(T, info["fps"], None, pipeline.Config(register=register), verbose=False)
    pr = res["phase_results"]["rewarming"]
    return dict(true_angle=angle, noise=noise, jitter_px=jitter, registered=register,
                err_B=float(flow.angular_diff(pr["statsB"]["mean_deg"], angle)), R_B=pr["statsB"]["R"],
                err_A=float(flow.angular_diff(pr["statsA"]["mean_deg"], angle)), R_A=pr["statsA"]["R"],
                pattern=pr["pattern"],
                speed_B=pr["speedB_med"], speed_true=info["speed"])


def one_radial(noise, seed):
    T, mask, info = make_synthetic(0, noise=noise, pattern="radial", seed=seed)
    pr = pipeline.analyze(T, info["fps"], None, verbose=False)["phase_results"]["rewarming"]
    fx, fy = info["focus_xy"]
    err = min([np.hypot(x - fx, y - fy) for x, y, _ in pr["sources"]] or [np.nan])
    return dict(noise=noise, seed=seed, pattern=pr["pattern"], R=pr["statsB"]["R"],
                outward_alignment=pr["outward_alignment"], focus_error_px=float(err))


for noise in (0.005, 0.02, 0.05):
    for ang in range(0, 360, 45):
        rows.append(one(ang, noise))
        print(rows[-1])
for reg in (False, True):                       # body motion test
    for ang in (30, 210):
        rows.append(one(ang, 0.01, jitter=0.7, register=reg))
        print(rows[-1])

rad = pd.DataFrame([one_radial(nz, sd) for nz in (0.005, 0.02, 0.05) for sd in range(3)])
rad.to_csv(f"{OUT}/radial_accuracy.csv", index=False)
print("\nRadial pattern (focus found? error in pixels; image is 128 px wide):\n", rad.round(2).to_string(index=False))

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/synthetic_accuracy.csv", index=False)
print("\nPlanar fronts classified as 'uniform':", f"{(df[df.jitter_px == 0].pattern == 'uniform').mean() * 100:.0f}%")
base = df[df.jitter_px == 0]
tab = base.groupby("noise")[["err_B", "err_A", "R_B", "R_A"]].mean().round(2)
print("\nMean angular error (degrees) by noise level:\n", tab)
mot = df[df.jitter_px > 0].groupby("registered")[["err_B", "err_A"]].mean().round(2)
print("\nWith body motion (0.7 px jitter), mean error by registration:\n", mot)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
tab[["err_B", "err_A"]].plot.bar(ax=ax[0], color=["tab:red", "tab:blue"])
ax[0].set_ylabel("mean angular error (deg)"); ax[0].set_xlabel("noise sigma (normalized units)")
ax[0].set_title("Direction error vs noise\nB = arrival-time, A = gradient flow"); ax[0].grid(alpha=0.3)
mot[["err_B", "err_A"]].plot.bar(ax=ax[1], color=["tab:red", "tab:blue"])
ax[1].set_xlabel("motion correction (registration) used"); ax[1].set_title("Effect of body motion (0.7 px jitter)")
ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/synthetic_accuracy.png", dpi=110)
print("saved to", OUT)
