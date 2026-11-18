"""
FIRST THING TO RUN AFTER DOWNLOADING DIRT-SR38
----------------------------------------------
Shows how the dataset is organised (which folders are sequences, how many frames, image size/type),
so you can choose the right folders for the pipeline.

Run:  python scripts/inspect_dataset.py data/DIRT-SR38
Tip : send the printed output to whoever helps you, it tells everything about the folder layout.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import io_utils

root = sys.argv[1] if len(sys.argv) > 1 else "data/DIRT-SR38"
print(f"\n=== Top-level content of {root} ===")
for name in sorted(os.listdir(root))[:40]:
    print("  ", name)

seqs = io_utils.list_sequences(root, min_frames=10)
print(f"\n=== {len(seqs)} folders with >= 10 images ===")
for i, (folder, n) in enumerate(seqs.items()):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(io_utils.IMG_EXT)], key=io_utils._natural_key)
    im = cv2.imread(os.path.join(folder, files[0]), cv2.IMREAD_UNCHANGED)
    print(f"[{i:3d}] {os.path.relpath(folder, root)}  | {n} images | first: {files[0]}  last: {files[-1]} "
          f"| shape={im.shape} dtype={im.dtype} min={im.min()} max={im.max()}")
    if i >= 60:
        print("   ... (list truncated)")
        break
print("\nNext: pick two HR sequence folders and run, e.g.\n"
      "  python -m src.pipeline --input <folder> --fps <FPS FROM THE PAPER> --out results/subject_A --scale 0.5 --video_stride 5")
