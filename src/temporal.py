"""
STEP 4/5 - FRAME-BY-FRAME ANALYSIS
----------------------------------
Input : Ts [frames,H,W] (smoothed), mask [H,W]
Output: slope map  [frames,H,W]  = how fast each pixel warms (+) or cools (-), per second
        state      [H,W] per frame = +1 rising / -1 falling / 0 stable (computed by classify())
        curve      [frames]      = average value inside the ROI
        phases     (t_cool_start, t_min) = where cooling begins and where rewarming begins
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def slope_map(T, fps, window, order=2):
    """
    Per-pixel rate of change (units per second) using a Savitzky-Golay derivative.
    This is much less noisy than simply subtracting two neighbouring frames.
    """
    window = min(window, len(T) if len(T) % 2 == 1 else len(T) - 1)
    return savgol_filter(T, window, order, deriv=1, delta=1.0 / fps, axis=0).astype(np.float32)


def estimate_noise(T, mask):
    """
    Noise level of one pixel value, from frame-to-frame differences (robust MAD estimate).
    Real changes between two frames are tiny compared with noise, so the median absolute
    difference is almost purely noise. Dividing by sqrt(2) because a difference of two
    noisy values has sqrt(2) times the noise of one.
    """
    d = (T[1:] - T[:-1])[:, mask]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def slope_noise(sigma_T, window, fps):
    """
    Theoretical noise of the Savitzky-Golay slope for white noise sigma_T and an odd window w:
        sigma_slope = sigma_T * sqrt(12 / (w (w^2-1))) * fps
    -> lets us choose a threshold that is statistically meaningful (not a magic number).
    """
    w = window
    return sigma_T * np.sqrt(12.0 / (w * (w * w - 1))) * fps


def classify(slope_frame, thr):
    """+1 where clearly rising, -1 where clearly falling, 0 where stable (|slope| < thr)."""
    s = np.zeros(slope_frame.shape, np.int8)
    s[slope_frame > thr] = 1
    s[slope_frame < -thr] = -1
    return s


def roi_curve(T, mask):
    return T[:, mask].mean(axis=1)


def find_phases(curve, min_len=5):
    """
    Split the recording into: baseline -> COOLING [t_cool_start, t_min] -> REWARMING [t_min, end].
    t_min is the coldest moment of the ROI-average curve.
    """
    n = len(curve)
    c = np.convolve(curve, np.ones(5) / 5, mode="same")
    t_min = int(np.argmin(c[2:-2]) + 2)
    t_cool_start = int(np.argmax(c[: t_min + 1])) if t_min > min_len else 0
    ok_rewarm = (n - 1 - t_min) >= min_len
    ok_cool = (t_min - t_cool_start) >= min_len
    return dict(t_cool_start=t_cool_start, t_min=t_min, has_rewarm=ok_rewarm, has_cool=ok_cool)


def frame_table(slope, curve, mask, thr, fps):
    """One row per frame: time, mean ROI value, mean slope, % rising, % falling."""
    rows = []
    sm = slope[:, mask]
    for t in range(len(slope)):
        s = sm[t]
        rows.append(dict(frame=t, time_s=t / fps, roi_mean=float(curve[t]),
                         mean_slope_per_s=float(s.mean()),
                         pct_rising=float((s > thr).mean() * 100),
                         pct_falling=float((s < -thr).mean() * 100)))
    return pd.DataFrame(rows)
