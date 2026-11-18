"""
LOOK AT A SEQUENCE BEFORE ANALYSING IT  (2 minutes, saves hours of confusion)
------------------------------------------------------------------------------
Shows first / middle / last frame, the automatically found ROI and the ROI-average curve, and prints checks:

  * number of frames, image size, value range
  * is every frame stretched to the full 0..255 range on its own?  (if YES, the dataset normalised each frame
    separately and slow temperature changes are partly erased - then the curve will look flat)
  * does the ROI-average curve show a dip (cooling) followed by a rise (rewarming)?

Run:  python scripts\\preview_sequence.py "D:\\Thermal\\DIRT-SR38\\dataset_export\\8bit\\P11\\HR\\dynamic"
Out : results\\preview\\<name>.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io_utils, roi

path = sys.argv[1]
dark_body = "--dark_body" in sys.argv
stride = 1
for i, a in enumerate(sys.argv):
    if a == "--stride":
        stride = int(sys.argv[i + 1])

T, files = io_utils.load_sequence(path, stride=stride, scale=0.5)
n = len(T)
print(f"\nFolder : {path}\nFrames : {n}  (first file: {files[0] if files else '-'}, last: {files[-1] if files else '-'})")
print(f"Size   : {T.shape[2]} x {T.shape[1]} px (after 0.5 resize)   value range: {T.min():.3f} .. {T.max():.3f}")

mins, maxs = T.reshape(n, -1).min(1), T.reshape(n, -1).max(1)
stretched = np.mean((mins <= 0.005) & (maxs >= 0.995)) * 100
print(f"Frames that use the full 0..1 range on their own: {stretched:.0f}%")
if stretched > 90:
    print("  -> WARNING: almost every frame is stretched to the full range. The dataset may have been normalised "
          "PER FRAME; absolute changes over time are then unreliable. Read the dataset paper (preprocessing section).")

mask = roi.auto_roi(T, body_is_brighter=not dark_body)
if mask.mean() < 0.35:
    print(f"\n!! The automatic ROI covers only {mask.mean() * 100:.0f}% of the image. If the skin fills the whole frame "
          "(close-up recording, no background) use the FULL-FRAME ROI: add  --roi full  to every analysis command.")
    mask_full = roi.full_roi(T.shape[1:])
    print(f"   Full-frame ROI curve: start={T[:, mask_full].mean(1)[:5].mean():.3f}  end={T[:, mask_full].mean(1)[-5:].mean():.3f}")
curve = T[:, mask].mean(1)
print(f"ROI    : {mask.mean() * 100:.0f}% of the image. ROI mean: start={curve[:5].mean():.3f}  "
      f"min={curve.min():.3f} (frame {curve.argmin()})  end={curve[-5:].mean():.3f}")
dip, rise = curve[:5].mean() - curve.min(), curve[-5:].mean() - curve.min()
if max(dip, rise) < 0.02:
    print("  -> The ROI curve is almost flat: no clear cooling/rewarming visible in this folder (is this the right "
          "folder? is the ROI correct? try --dark_body).")
else:
    print(f"  -> Cooling dip = {dip:.3f}, rewarming rise = {rise:.3f} (normalised units): looks like a cooling/rewarming recording.")

fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a, k, title in zip(ax[:3], (0, n // 2, n - 1), ("first", "middle", "last")):
    a.imshow(T[k], cmap="inferno", vmin=0, vmax=1)
    a.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    a.set_title(f"{title} frame (#{k})"); a.axis("off")
ax[3].plot(curve, "k"); ax[3].set_title("ROI average over frames"); ax[3].set_xlabel("frame"); ax[3].grid(alpha=0.3)
fig.tight_layout()
os.makedirs("results/preview", exist_ok=True)
name = "_".join(os.path.normpath(path).split(os.sep)[-3:])
out = f"results/preview/{name}.png"
fig.savefig(out, dpi=100)
print(f"\nSaved {out}  <- open it and check: is the cyan outline on the body? does the curve dip and recover?")
