"""
DOES SUPER-RESOLUTION HELP THE DIRECTION ESTIMATE?  (the research question that ties the trained model
to the assignment's main task)
------------------------------------------------------------------------------------------------------
For ONE subject we have the same recording at low resolution (LR) and high resolution (HR).
We run the identical propagation algorithm on three versions of the sequence:
    LR-bicubic  : LR frames simply enlarged x2
    LR-SR       : LR frames enlarged by the trained network
    HR          : the real high-resolution frames (treated as the reference)
and compare direction, speed and the arrival-time map against HR.

!! Needs PyTorch + the model from train_sr.py. Written for the real dataset; could not be executed in the
!! environment where the rest of the code was tested.

Run: python scripts/sr_downstream_test.py --lr_dir <LR folder> --hr_dir <HR folder> --model models/edsr_x2.pt --fps 9
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io_utils, pipeline, roi, preprocess, flow
from scripts.train_sr import EDSRLite

ap = argparse.ArgumentParser()
ap.add_argument("--lr_dir", required=True)
ap.add_argument("--hr_dir", required=True)
ap.add_argument("--model", default="models/edsr_x2.pt")
ap.add_argument("--fps", type=float, required=True)
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--max_frames", type=int, default=None)
a = ap.parse_args()

lr, _ = io_utils.load_sequence(a.lr_dir, a.stride, a.max_frames)
hr, _ = io_utils.load_sequence(a.hr_dir, a.stride, a.max_frames)
n = min(len(lr), len(hr))
lr, hr = lr[:n], hr[:n]
fps = a.fps / a.stride

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = EDSRLite().to(dev)
net.load_state_dict(torch.load(a.model, map_location=dev))
net.eval()
with torch.no_grad():
    sr = np.stack([net(torch.from_numpy(f)[None, None].to(dev))[0, 0].cpu().numpy() for f in lr])
bic = np.stack([cv2.resize(f, (hr.shape[2], hr.shape[1]), interpolation=cv2.INTER_CUBIC) for f in lr])

hr_norm, _ = preprocess.robust_normalize(hr)
mask = roi.auto_roi(hr_norm)                                  # SAME ROI for all three -> fair comparison
variants = {"HR (reference)": hr, "LR-bicubic": bic, "LR-SR (trained)": sr}
out = {}
for k, v in variants.items():
    print(f"\n--- {k} ---")
    out[k] = pipeline.analyze(v, fps, mask, verbose=False)["phase_results"]["rewarming"]

ref = out["HR (reference)"]
print("\n%-18s %10s %8s %12s %22s" % ("version", "dir (deg)", "R", "speed px/s", "arrival-map corr. vs HR"))
for k, pr in out.items():
    ta, tr = pr["B"]["ta_smooth"], ref["B"]["ta_smooth"]
    ok = np.isfinite(ta) & np.isfinite(tr)
    corr = np.corrcoef(ta[ok], tr[ok])[0, 1] if ok.sum() > 10 else np.nan
    print("%-18s %10.1f %8.2f %12.2f %22.3f" % (k, pr["statsB"]["mean_deg"], pr["statsB"]["R"], pr["speedB_med"], corr))
print("\nDirection difference to HR (deg): ",
      {k: round(float(flow.angular_diff(pr["statsB"]["mean_deg"], ref["statsB"]["mean_deg"])), 1) for k, pr in out.items()})
