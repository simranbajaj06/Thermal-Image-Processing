"""
SYNTHETIC THERMAL SEQUENCE WITH A KNOWN ANSWER
----------------------------------------------
Real datasets have NO ground truth for 'which way did the warming spread'. To measure the accuracy of
the algorithm we therefore simulate a cooling -> rewarming experiment where WE decide the direction:

  1) baseline  : body is warm and stable
  2) cooling   : the whole body cools down (smooth ramp)
  3) rewarming : a warming FRONT sweeps across the body in a chosen direction and speed
  + sensor noise, an optional small body motion (breathing jitter) and slow drift.

Returned info['angle_deg'] is the true direction (0 = right, 90 = up), info['speed'] in pixels/second.
"""
import numpy as np
import cv2


def make_synthetic(angle_deg=45.0, speed=6.0, noise=0.01, size=128, fps=5.0, seed=0,
                   jitter_px=0.0, depth=0.25, tau=1.0):
    rng = np.random.default_rng(seed)
    H = W = size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = H / 2, W / 2
    mask = (((xx - cx) / (0.42 * W)) ** 2 + ((yy - cy) / (0.38 * H)) ** 2) <= 1.0

    th = np.deg2rad(angle_deg)
    n = np.array([np.cos(th), -np.sin(th)], np.float32)          # (nx, ny) in IMAGE coordinates (y down)
    p = (xx - cx) * n[0] + (yy - cy) * n[1]                      # position along the travel direction
    pmin, pmax = p[mask].min(), p[mask].max()

    t_base, t_cool = 4.0, 8.0                                    # seconds
    t_rew0 = t_base + t_cool + 4.0 * tau                         # rewarming starts after cooling ends
    ta = t_rew0 + (p - pmin) / speed                             # TRUE arrival time of the front at each pixel
    t_end = t_rew0 + (pmax - pmin) / speed + 8.0
    t = np.arange(0, t_end, 1.0 / fps)

    smooth = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 12)
    base = 0.70 + 0.04 * smooth / (np.abs(smooth).max() + 1e-9)
    bg = 0.25

    frames = []
    for ti in t:
        c = 0.0 if ti < t_base else (1.0 if ti > t_base + t_cool
                                     else 0.5 - 0.5 * np.cos(np.pi * (ti - t_base) / t_cool))
        r = 1.0 / (1.0 + np.exp(-(ti - ta) / tau))               # 0 = still cold, 1 = recovered
        body = base - depth * c * (1.0 - r)
        img = np.where(mask, body, bg).astype(np.float32)
        img = cv2.GaussianBlur(img, (0, 0), 1.0)                  # optics blur
        img += rng.normal(0, noise, img.shape).astype(np.float32)
        img += 0.002 * np.sin(2 * np.pi * ti / 30.0)              # slow global drift
        if jitter_px > 0:                                         # breathing-like motion
            dx, dy = rng.normal(0, jitter_px, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        frames.append(img)
    T = np.clip(np.stack(frames), 0, 1).astype(np.float32)
    info = dict(angle_deg=float(angle_deg), speed=float(speed), fps=float(fps), noise=float(noise),
                t_rewarm_start_s=float(t_rew0), arrival_gt=np.where(mask, ta, np.nan))
    return T, mask, info
