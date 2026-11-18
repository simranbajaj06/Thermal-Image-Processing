"""
MODEL TRAINING (optional but recommended for the "AI/ML" part)
---------------------------------------------------------------
DIRT-SR38 contains aligned LOW-resolution / HIGH-resolution thermal image pairs. That is the one thing in
this dataset we can train a model on with real labels:   LR image  ->  network  ->  HR image (x2).

Model : small EDSR-style CNN (residual blocks + PixelShuffle), predicts only the missing detail on top of a
        bicubic upscaling. Loss: L1. Metrics: PSNR (higher = better) compared with plain bicubic.
Split : BY SUBJECT (validation subjects are never seen in training) - otherwise the score is too optimistic.

Expected layout (yours):  <root>\\P11\\HR\\dynamic , <root>\\P11\\LR\\dynamic (+ \\static), <root>\\P12\\...
  <root> = D:\\Thermal\\DIRT-SR38\\dataset_export\\8bit   (the folder that contains P11, P12, ...)
!! NOTE: this script needs PyTorch and was written to run on the real dataset; it could not be executed in
!! the environment where the rest of the code was tested.

Run  : python scripts\\train_sr.py --root "D:\\Thermal\\DIRT-SR38\\dataset_export\\8bit" --kinds dynamic --epochs 20
Out  : models/edsr_x2.pt  +  printed PSNR table
"""
import os
import sys
import random
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.io_utils import read_image_gray
from src.dirt import find_pairs


class ResBlock(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(f, f, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(f, f, 3, padding=1))

    def forward(self, x):
        return x + 0.1 * self.body(x)


class EDSRLite(nn.Module):
    """1-channel thermal image (N,1,h,w) -> (N,1,2h,2w)."""
    def __init__(self, feats=48, blocks=6):
        super().__init__()
        self.head = nn.Conv2d(1, feats, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(feats) for _ in range(blocks)], nn.Conv2d(feats, feats, 3, padding=1))
        self.up = nn.Sequential(nn.Conv2d(feats, feats * 4, 3, padding=1), nn.PixelShuffle(2))
        self.tail = nn.Conv2d(feats, 1, 3, padding=1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        h = self.head(x)
        h = h + self.body(h)
        return (base + self.tail(self.up(h))).clamp(0, 1)


def subject_of(path, root):
    """Subject = first folder below root (e.g. 'P11')."""
    return os.path.relpath(path, root).split(os.sep)[0]


class PatchSet(Dataset):
    def __init__(self, pairs, patch=48, n=20000):
        self.pairs, self.patch, self.n = pairs, patch, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        lp, hp = random.choice(self.pairs)
        lr, hr = read_image_gray(lp), read_image_gray(hp)
        h, w = lr.shape
        p = self.patch
        y, x = random.randint(0, h - p), random.randint(0, w - p)
        a, b = lr[y:y + p, x:x + p], hr[2 * y:2 * y + 2 * p, 2 * x:2 * x + 2 * p]
        if random.random() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        if random.random() < 0.5:
            a, b = a[::-1], b[::-1]
        return torch.from_numpy(a.copy())[None], torch.from_numpy(b.copy())[None]


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse == 0 else -10 * np.log10(mse)


@torch.no_grad()
def evaluate(model, pairs, dev, n=60):
    model.eval()
    sel = random.Random(0).sample(pairs, min(n, len(pairs)))
    ps_model, ps_bic = [], []
    for lp, hp in sel:
        lr = torch.from_numpy(read_image_gray(lp))[None, None].to(dev)
        hr = torch.from_numpy(read_image_gray(hp))[None, None].to(dev)
        ps_model.append(psnr(model(lr), hr))
        ps_bic.append(psnr(F.interpolate(lr, scale_factor=2, mode="bicubic").clamp(0, 1), hr))
    return float(np.mean(ps_bic)), float(np.mean(ps_model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lr_name", default="LR")
    ap.add_argument("--hr_name", default="HR")
    ap.add_argument("--kinds", nargs="*", default=None, help="e.g. dynamic  (default: dynamic AND static)")
    ap.add_argument("--val_subjects", nargs="*", default=None, help="e.g. P37 P38 (default: last 15%% of subjects)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400, help="steps per epoch")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="models/edsr_x2.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = find_pairs(a.root, a.lr_name, a.hr_name, a.kinds)
    assert pairs, "No LR/HR pairs found - check --root (must contain P11, P12, ...) and the folder names"
    h0 = read_image_gray(pairs[0][0]).shape, read_image_gray(pairs[0][1]).shape
    print(f"{len(pairs)} pairs found. Example sizes LR/HR: {h0}")
    assert h0[1][0] == 2 * h0[0][0], "HR is not exactly 2x LR - adapt the scale in the model/dataset"

    subjects = sorted({subject_of(p[0], a.root) for p in pairs})
    val_sub = set(a.val_subjects) if a.val_subjects else set(subjects[-max(1, int(0.15 * len(subjects))):])
    train = [p for p in pairs if subject_of(p[0], a.root) not in val_sub]
    val = [p for p in pairs if subject_of(p[0], a.root) in val_sub]
    assert train and val, "need at least 2 subjects (train + validation)"
    print(f"subjects: {len(subjects)} total, validation subjects: {sorted(val_sub)}  "
          f"(train pairs {len(train)}, val pairs {len(val)})")

    model = EDSRLite().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    dl = DataLoader(PatchSet(train, n=a.steps * a.batch), batch_size=a.batch, num_workers=2, shuffle=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    best = 0
    for ep in range(a.epochs):
        model.train()
        tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = F.l1_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()
        pb, pm = evaluate(model, val, dev)
        print(f"epoch {ep + 1:3d}  train L1={tot / len(dl):.5f}  val PSNR bicubic={pb:.2f} dB  model={pm:.2f} dB")
        if pm > best:
            best = pm
            torch.save(model.state_dict(), a.out)
    print(f"best validation PSNR {best:.2f} dB  (bicubic baseline {pb:.2f} dB) -> saved {a.out}")


if __name__ == "__main__":
    main()
