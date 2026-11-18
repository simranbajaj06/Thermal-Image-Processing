"""
RUN THE ASSIGNMENT'S "AT LEAST TWO SEQUENCES" EXPERIMENT
--------------------------------------------------------
Processes two sequences (e.g. two different subjects of DIRT-SR38) and writes a comparison.

Run (simplest, with your dataset layout):
  python scripts\\run_two_sequences.py --root "D:\\Thermal\\DIRT-SR38\\dataset_export\\8bit" --subjects P11 P12 --fps 9 --scale 0.5
  (uses <root>\\P11\\HR\\dynamic and <root>\\P12\\HR\\dynamic; change with --res LR / --kind static)
Or give two folders directly:  --seq1 "<folder>" --seq2 "<folder>"
Output: one result folder per sequence (video, figures, CSV, JSON) + comparison.csv + comparison.png
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io_utils, pipeline, dirt

ap = argparse.ArgumentParser()
ap.add_argument("--root", help="folder that contains P11, P12, ...")
ap.add_argument("--subjects", nargs=2, help="two subject folders, e.g. P11 P12")
ap.add_argument("--res", default="HR")
ap.add_argument("--kind", default="dynamic")
ap.add_argument("--seq1")
ap.add_argument("--seq2")
ap.add_argument("--fps", type=float, required=True, help="frame rate of the recording (see dataset paper)")
ap.add_argument("--scale", type=float, default=0.5)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--max_frames", type=int, default=None)
ap.add_argument("--out", default="results/two_sequences")
ap.add_argument("--video_stride", type=int, default=5)
ap.add_argument("--register", action="store_true")
ap.add_argument("--dark_body", action="store_true")
a = ap.parse_args()

if a.root and a.subjects:
    paths = [dirt.seq_dir(a.root, s_, a.res, a.kind) for s_ in a.subjects]
elif a.seq1 and a.seq2:
    paths = [a.seq1, a.seq2]
else:
    ap.error("give either --root and --subjects, or --seq1 and --seq2")
results, rows = {}, []
for k, path in enumerate(paths, 1):
    norm = os.path.normpath(path)
    name = f"seq{k}_" + "_".join(norm.split(os.sep)[-3:])
    print(f"\n######## {name} ########")
    T, files = io_utils.load_sequence(path, a.stride, a.max_frames, a.scale)
    cfg = pipeline.Config(register=a.register, body_is_brighter=not a.dark_body)
    res = pipeline.run(T, a.fps / a.stride, os.path.join(a.out, name), name, cfg=cfg, video_stride=a.video_stride)
    results[name] = res
    for phase, pr in res["phase_results"].items():
        rows.append(dict(sequence=name, phase=phase, frames=f"{pr['a']}-{pr['b']}", mean_change=pr["mean_change"],
                         dir_arrival_deg=pr["statsB"]["mean_deg"], coherence_arrival=pr["statsB"]["R"],
                         speed_arrival_px_s=pr["speedB_med"], dir_gradflow_deg=pr["statsA"]["mean_deg"],
                         coherence_gradflow=pr["statsA"]["R"], methods_differ_deg=pr["agreement_deg"]))
df = pd.DataFrame(rows).round(3)
os.makedirs(a.out, exist_ok=True)
df.to_csv(os.path.join(a.out, "comparison.csv"), index=False)
print("\n", df.to_string(index=False))

# side-by-side: ROI curves + rose plots of the rewarming direction
fig = plt.figure(figsize=(12, 4.5))
ax = fig.add_subplot(1, 2, 1)
for name, res in results.items():
    t = np.arange(len(res["curve"])) / res["fps"]
    ax.plot(t, res["curve"], label=name)
ax.set_xlabel("time (s)"); ax.set_ylabel("ROI mean (normalized)"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("ROI average over time")
axp = fig.add_subplot(1, 2, 2, projection="polar")
for (name, res), col in zip(results.items(), ("tab:red", "tab:blue")):
    pr = res["phase_results"].get("rewarming")
    if pr:
        ang = np.deg2rad(pr["B"]["angle"][np.isfinite(pr["B"]["angle"])])
        h, e = np.histogram(ang, bins=36, range=(0, 2 * np.pi))
        axp.bar(e[:-1], h / h.max(), width=2 * np.pi / 36, alpha=0.45, color=col, label=name, align="edge")
axp.legend(loc="lower left", bbox_to_anchor=(-0.3, -0.25), fontsize=8)
axp.set_title("Rewarming direction (0 deg = right, 90 = up)")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "comparison.png"), dpi=110)
print("saved", a.out)
