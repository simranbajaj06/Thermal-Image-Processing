"""
STEP 2 - PRE-PROCESSING
-----------------------
Input : T [frames,H,W] raw values
Output: T [frames,H,W] cleaned values (scaled 0..1, optionally motion-corrected, de-noised)
"""
import numpy as np
import cv2
from scipy.signal import savgol_filter


def robust_normalize(T, lo_pct=1, hi_pct=99):
    """
    Scale the whole sequence to ~0..1 using robust percentiles of the WHOLE sequence.
    The SAME scale is used for every frame - otherwise real temperature changes would vanish.
    """
    sub = T[:: max(1, len(T) // 20)]
    lo, hi = np.percentile(sub, [lo_pct, hi_pct])
    return np.clip((T - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32), (float(lo), float(hi))


def spatial_denoise(T, sigma=1.0):
    """Gaussian blur every frame (removes pixel noise; sigma in pixels)."""
    if sigma <= 0:
        return T
    k = int(2 * round(3 * sigma) + 1)
    return np.stack([cv2.GaussianBlur(f, (k, k), sigma) for f in T])


def _grad_mag(f):
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    return g / (g.max() + 1e-9)


def register_frames(T, ref=0):
    """
    Motion correction (breathing / small movements): align every frame to a reference frame
    with ECC (translation only). We align on gradient magnitude (edges), because overall
    brightness changes a lot during cooling/rewarming but the body outline does not.
    Output: aligned T and the list of (dx, dy) shifts that were found.
    """
    h, w = T.shape[1:]
    ref_g = _grad_mag(cv2.GaussianBlur(T[ref], (5, 5), 1.5))
    out = np.empty_like(T)
    shifts = []
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    for i, f in enumerate(T):
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            g = _grad_mag(cv2.GaussianBlur(f, (5, 5), 1.5))
            _, warp = cv2.findTransformECC(ref_g, g, warp, cv2.MOTION_TRANSLATION, crit, None, 5)
            out[i] = cv2.warpAffine(f, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:  # ECC did not converge -> keep the frame unchanged
            out[i] = f
            warp = np.eye(2, 3, dtype=np.float32)
        shifts.append((float(warp[0, 2]), float(warp[1, 2])))
    return out, shifts


def odd_window(win_s, fps, min_frames=5, max_frames=None):
    """Convert a window length in seconds to an odd number of frames."""
    w = max(min_frames, int(round(win_s * fps)))
    if max_frames:
        w = min(w, max_frames)
    return w if w % 2 == 1 else w + 1


def temporal_smooth(T, window, order=2):
    """Savitzky-Golay smoothing along time for every pixel (keeps real trends, removes noise)."""
    window = min(window, len(T) if len(T) % 2 == 1 else len(T) - 1)
    return savgol_filter(T, window, order, axis=0).astype(np.float32)
