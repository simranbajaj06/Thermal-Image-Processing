"""
STEP 6 - DIRECTION AND PROPAGATION OF THE THERMAL PATTERN
---------------------------------------------------------
Two independent methods, so you can check one against the other:

METHOD B  "arrival-time map"  (the most robust one for cooling/rewarming experiments)
    For every pixel find the moment it reaches 50% of its own total change.
    The direction in which this arrival time grows = direction the thermal front travels.
    direction = grad(arrival_time) / |grad(arrival_time)|     speed = 1 / |grad(arrival_time)|

METHOD A  "gradient flow" (Lucas-Kanade on the temperature field)
    Assume the temperature pattern is carried along with velocity v:  dT/dt = - v . grad(T)
    and solve for v in a small window by least squares (works for warming AND cooling fronts).

Angle convention used everywhere: 0 deg = right, 90 deg = up, 180 = left, 270 = down.
"""
import numpy as np
import cv2
from scipy import ndimage


# ---------------------------------------------------------------- helpers
def vec_to_angle(vx, vy):
    """Image coordinates have y pointing DOWN, so flip y to get a normal 'up = 90 deg' angle."""
    return np.degrees(np.arctan2(-vy, vx)) % 360.0


def angle_to_text(a):
    names = ["right", "upper-right", "up", "upper-left", "left", "lower-left", "down", "lower-right"]
    return names[int(((a + 22.5) % 360) // 45)]


def angular_diff(a, b):
    """Smallest difference between two angles in degrees (0..180)."""
    d = np.abs((a - b + 180) % 360 - 180)
    return d


def circular_stats(angles_deg, weights=None):
    """
    Mean direction + coherence R of many angles.
    R = 1 -> all arrows point the same way;  R ~ 0 -> arrows point everywhere (no clear direction).
    """
    a = np.deg2rad(np.asarray(angles_deg, float))
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(a) & np.isfinite(w) & (w > 0)
    if ok.sum() == 0:
        return dict(mean_deg=np.nan, R=0.0, n=0)
    a, w = a[ok], w[ok]
    C, S = (w * np.cos(a)).sum(), (w * np.sin(a)).sum()
    return dict(mean_deg=float(np.degrees(np.arctan2(S, C)) % 360), R=float(np.hypot(C, S) / w.sum()), n=int(ok.sum()))


def masked_gaussian(img, valid, sigma):
    """Blur only over valid pixels (normalised convolution) so NaNs / outside-ROI do not leak in."""
    v = valid.astype(np.float32)
    num = cv2.GaussianBlur(np.where(valid, img, 0).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(v, (0, 0), sigma)
    out = num / np.maximum(den, 1e-6)
    out[den < 0.2] = np.nan
    return out


# ---------------------------------------------------------------- METHOD B
def arrival_time(Ts, fps, mask, t0, t1, frac=0.5, n_edge=3, min_amp=0.0):
    """
    Ts   : smoothed sequence [frames,H,W]       t0,t1 : frame range of the phase to analyse
    Returns arrival_time [H,W] in seconds (NaN where the pixel does not change enough).

    For each pixel:  v0 = value at start of phase,  v1 = value at the end,  amp = v1 - v0
    target = v0 + frac*amp.   The arrival time is the LAST moment the pixel was still short of the
    target (so a noisy early crossing is ignored), linearly interpolated between two frames.
    """
    seg = Ts[t0:t1 + 1]
    n = len(seg)
    v0 = seg[:n_edge].mean(0)
    v1 = seg[-n_edge:].mean(0)
    amp = v1 - v0
    sign = np.sign(amp)
    target = v0 + frac * amp
    short = (seg - target[None]) * sign[None] < 0            # True while target not yet reached
    last_short = (n - 1) - np.argmax(short[::-1], axis=0)    # index of last frame still short
    never = ~short.any(0)                                    # reached immediately
    always = short.all(0)                                    # never reached
    i0 = np.clip(last_short, 0, n - 2)
    a = np.take_along_axis(seg, i0[None], 0)[0]
    b = np.take_along_axis(seg, (i0 + 1)[None], 0)[0]
    f = np.clip((target - a) / np.where(np.abs(b - a) < 1e-9, 1e-9, b - a), 0, 1)
    ta = (t0 + i0 + f) / fps
    ta[never] = t0 / fps
    valid = mask & ~always & (np.abs(amp) > min_amp)
    ta[~valid] = np.nan
    return ta, amp, valid


def front_direction_from_arrival(ta, valid, sigma=2.0, max_speed=None):
    """
    Gradient of the arrival-time map -> per-pixel direction and speed.
    Returns dict with gx, gy (seconds per pixel), angle (deg), speed (px/s), ta_smooth.
    """
    ta_s = masked_gaussian(np.nan_to_num(ta), valid, sigma)
    ok = np.isfinite(ta_s)
    gy, gx = np.gradient(np.where(ok, ta_s, np.nan))
    mag = np.hypot(gx, gy)
    good = ok & np.isfinite(mag) & (mag > 1e-4)
    speed = np.where(good, 1.0 / np.where(mag > 1e-9, mag, np.nan), np.nan)
    if max_speed:
        good &= speed < max_speed
    angle = np.where(good, vec_to_angle(gx, gy), np.nan)   # front moves where arrival time INCREASES
    return dict(gx=gx, gy=gy, angle=angle, speed=np.where(good, speed, np.nan), ta_smooth=ta_s, good=good)


# ---------------------------------------------------------------- METHOD A
def lk_velocity(Ts_frame, slope_frame, win=15, reg=0.01):
    """
    Lucas-Kanade on temperature for ONE frame.
    Solves  dT/dt = -(vx*Tx + vy*Ty)  in a win x win neighbourhood (least squares).
    reg adds a tiny stabiliser: where the pattern only has ONE gradient direction the result
    gracefully becomes the 'normal velocity' (the part of motion we can really observe).
    Returns vx, vy in pixels/second and weight (= local gradient energy, how trustworthy).
    """
    Tx = cv2.Sobel(Ts_frame, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Ty = cv2.Sobel(Ts_frame, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    f = lambda a: cv2.boxFilter(a, -1, (win, win))
    Jxx, Jyy, Jxy = f(Tx * Tx), f(Ty * Ty), f(Tx * Ty)
    bx, by = f(Tx * slope_frame), f(Ty * slope_frame)
    lam = 1e-12 + reg * (Jxx + Jyy)
    a11, a22 = Jxx + lam, Jyy + lam
    det = a11 * a22 - Jxy ** 2
    vx = -(a22 * bx - Jxy * by) / det
    vy = -(-Jxy * bx + a11 * by) / det
    return vx.astype(np.float32), vy.astype(np.float32), (Jxx + Jyy).astype(np.float32)


def aggregate_lk(Ts, slope, mask, t0, t1, win=15, step=8, stride=1, sigma_space=1.5):
    """
    Run lk_velocity on every frame t0..t1 and average the vectors, weighted by gradient energy.
    Returns:
      mean_vx, mean_vy [H,W]     average velocity field of the phase (px/s)
      weight [H,W]               total evidence at each pixel
      samples                    dict(t=[...], u=[...], v=[...]) arrows on a coarse grid (for the video)
    """
    H, W = Ts.shape[1:]
    sx = np.zeros((H, W), np.float64)
    sy = np.zeros((H, W), np.float64)
    sw = np.zeros((H, W), np.float64)
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    samples = dict(t=[], u=[], v=[], grid=(xs, ys))
    energies = []
    for t in range(t0, t1 + 1, stride):
        im = cv2.GaussianBlur(Ts[t], (0, 0), sigma_space)
        vx, vy, w = lk_velocity(im, slope[t], win)
        energies.append(np.percentile(w[mask], 90))
        wthr = 0.05 * max(np.median(energies), 1e-12)          # ignore flat (gradient-free) areas
        wv = np.where(mask & (w > wthr), w, 0.0)
        sx += wv * vx
        sy += wv * vy
        sw += wv
        samples["t"].append(t)
        samples["u"].append(np.where(wv[ys, xs] > 0, vx[ys, xs], np.nan))
        samples["v"].append(np.where(wv[ys, xs] > 0, vy[ys, xs], np.nan))
    mvx = np.where(sw > 0, sx / np.maximum(sw, 1e-12), np.nan)
    mvy = np.where(sw > 0, sy / np.maximum(sw, 1e-12), np.nan)
    return mvx.astype(np.float32), mvy.astype(np.float32), sw.astype(np.float32), samples


# ---------------------------------------------------------------- foci (radial / multi-centre patterns)
def find_foci(ta_s, kind="min", size=15, max_n=5, frac=0.35):
    """
    Candidate recovery foci on the arrival-time map.
      kind="min": SOURCES  = places that recover EARLIEST (local minima of arrival time)
      kind="max": SINKS    = places that recover LATEST   (local maxima)
    Returns a list of (x, y, arrival_time_s). At most max_n foci, at least `size` pixels apart; a source must
    belong to the earliest `frac` of the map (a sink to the latest `frac`).
    """
    ok = np.isfinite(ta_s)
    if ok.sum() < 50:
        return []
    q = np.nanquantile(ta_s[ok], frac if kind == "min" else 1 - frac)
    fill = np.where(ok, ta_s, np.inf if kind == "min" else -np.inf)
    ext = ndimage.minimum_filter(fill, size=size) if kind == "min" else ndimage.maximum_filter(fill, size=size)
    cand = ok & (fill == ext) & ((fill <= q) if kind == "min" else (fill >= q))
    ys, xs = np.nonzero(cand)
    order = np.argsort(ta_s[ys, xs]) if kind == "min" else np.argsort(-ta_s[ys, xs])
    pts = []
    for i in order:
        p = (int(xs[i]), int(ys[i]))
        if all(np.hypot(p[0] - a[0], p[1] - a[1]) >= size for a in pts):
            pts.append(p)
        if len(pts) >= max_n:
            break
    return [(x, y, float(ta_s[y, x])) for x, y in pts]


def focal_alignment(gx, gy, good, foci, outward=True, min_r=3):
    """
    How well do the arrows point AWAY from the nearest source (outward=True) or TOWARD the nearest sink
    (outward=False)?  +1 = every arrow does, 0 = no relation, -1 = the opposite.
    """
    if not foci or good.sum() == 0:
        return np.nan
    ys, xs = np.nonzero(good)
    P = np.stack([xs, ys], 1).astype(float)
    S = np.array([[f[0], f[1]] for f in foci], float)
    d2 = ((P[:, None, :] - S[None]) ** 2).sum(-1)
    k = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(P)), k])
    keep = dist >= min_r
    if keep.sum() < 10:
        return np.nan
    r = (P - S[k])[keep] / dist[keep, None]
    g = np.stack([gx[ys, xs], gy[ys, xs]], 1)[keep]
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    a = (g * r).sum(1)
    return float(np.mean(a if outward else -a))


def classify_pattern(R, out_idx, conv_idx, r_thr=0.6, foc_thr=0.5):
    """'uniform' = one dominant direction; 'focal' = spreads from early foci to late foci; else 'mixed'."""
    if R >= r_thr:
        return "uniform"
    best = max([v for v in (out_idx, conv_idx) if np.isfinite(v)] or [-1])
    return "focal" if best >= foc_thr else "mixed"


# ---------------------------------------------------------------- summary
def summarize_direction(angle, weights=None):
    st = circular_stats(angle[np.isfinite(angle)], None if weights is None else weights[np.isfinite(angle)])
    st["text"] = angle_to_text(st["mean_deg"]) if np.isfinite(st["mean_deg"]) else "no clear direction"
    return st
