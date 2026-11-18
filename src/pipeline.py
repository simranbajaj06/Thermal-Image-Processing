"""
THE WHOLE PIPELINE IN ONE PLACE
-------------------------------
analyze(T, fps)  ->  results dict      (pure computation, no files written)
run(...)         ->  analyze + save CSV/JSON/figures/video into an output folder

Command line:
  python -m src.pipeline --input path/to/sequence_folder --fps 9 --out results/subject01
  python -m src.pipeline --synthetic 135 --out results/synth135       (demo without any dataset)
"""
import os
import json
import argparse
from dataclasses import dataclass, asdict
import numpy as np

from . import io_utils, preprocess, roi, temporal, flow


@dataclass
class Config:
    sigma_space: float = 1.0       # spatial blur (pixels)
    smooth_win_s: float = 2.0      # temporal smoothing / derivative window (seconds)
    k_sigma: float = 3.0           # a pixel is 'rising/falling' if |slope| > k_sigma * noise
    arrival_frac: float = 0.5      # arrival time = moment pixel reaches 50 % of its own change
    arrival_sigma: float = 2.0     # smoothing of the arrival-time map (pixels)
    lk_win: int = 15               # window of the gradient-flow method (pixels)
    min_amp_sigmas: float = 6.0    # a pixel must change by > this many noise sigmas to be used
    register: bool = False         # motion correction (slow, enable if the body moves)
    erode_px: int = 3              # shrink ROI from the border
    body_is_brighter: bool = True


def analyze(T_raw, fps, mask=None, cfg: Config = None, verbose=True):
    cfg = cfg or Config()
    log = print if verbose else (lambda *a, **k: None)
    res = dict(cfg=asdict(cfg), fps=fps)

    # ---- STEP 2: pre-processing -------------------------------------------------------------------
    T, (lo, hi) = preprocess.robust_normalize(T_raw)
    res["norm_range_raw"] = (lo, hi)
    if cfg.register:
        log("registering frames (motion correction) ...")
        T, shifts = preprocess.register_frames(T)
        res["shifts"] = shifts
    T = preprocess.spatial_denoise(T, cfg.sigma_space)

    # ---- STEP 3: ROI --------------------------------------------------------------------------------
    if mask is None:
        mask = roi.auto_roi(T, cfg.body_is_brighter, cfg.erode_px)
    res["mask"] = mask
    log(f"ROI covers {mask.mean() * 100:.1f}% of the image")

    # ---- STEP 4: temporal smoothing, slope maps, noise ---------------------------------------------
    win = preprocess.odd_window(cfg.smooth_win_s, fps, max_frames=len(T) - 1)
    sigma_T = temporal.estimate_noise(T, mask)
    Ts = preprocess.temporal_smooth(T, win)
    slope = temporal.slope_map(T, fps, win)
    thr = cfg.k_sigma * temporal.slope_noise(sigma_T, win, fps)
    res.update(T=T, Ts=Ts, slope=slope, thr=float(thr), sigma_T=sigma_T, window_frames=win)
    log(f"noise sigma={sigma_T:.4f} (normalized units) | slope threshold={thr:.4f} units/s | window={win} frames")

    # ---- STEP 5: per-frame statistics, phases ---------------------------------------------------------
    curve = temporal.roi_curve(Ts, mask)
    ph = temporal.find_phases(curve)
    res.update(curve=curve, phases=ph)
    res["table"] = temporal.frame_table(slope, curve, mask, thr, fps)
    log(f"phases: cooling {ph['t_cool_start']}->{ph['t_min']}, rewarming {ph['t_min']}->{len(T) - 1}")

    # ---- STEP 6: direction of propagation ---------------------------------------------------------------
    min_amp = cfg.min_amp_sigmas * sigma_T
    res["phase_results"] = {}
    for name, (a, b, ok) in {"rewarming": (ph["t_min"], len(T) - 1, ph["has_rewarm"]),
                             "cooling": (ph["t_cool_start"], ph["t_min"], ph["has_cool"])}.items():
        if not ok:
            continue
        ta, amp, valid = flow.arrival_time(Ts, fps, mask, a, b, cfg.arrival_frac, min_amp=min_amp)
        fb = flow.front_direction_from_arrival(ta, valid, cfg.arrival_sigma)
        stB = flow.summarize_direction(fb["angle"])
        mvx, mvy, w, samples = flow.aggregate_lk(Ts, slope, mask, a, b, cfg.lk_win)
        angA = flow.vec_to_angle(mvx, mvy)
        # only trust pixels that were crossed by a front in this phase (valid) and have evidence
        okA = valid & np.isfinite(angA) & (w > 0)
        stA = flow.summarize_direction(np.where(okA, angA, np.nan), np.where(okA, w, 0))
        spdA = np.hypot(mvx, mvy)
        pr = dict(a=a, b=b, arrival=ta, amp=amp, valid=valid, B=fb, A_vx=mvx, A_vy=mvy, A_w=w, A_angle=angA,
                  A_ok=okA, A_samples=samples, statsB=stB, statsA=stA,
                  speedB_med=float(np.nanmedian(fb["speed"])) if np.isfinite(fb["speed"]).any() else np.nan,
                  speedA_med=float(np.nanmedian(spdA[okA])) if okA.any() else np.nan,
                  mean_change=float(np.nanmean(amp[mask])))
        pr["agreement_deg"] = (float(flow.angular_diff(stA["mean_deg"], stB["mean_deg"]))
                               if np.isfinite(stA["mean_deg"]) and np.isfinite(stB["mean_deg"]) else np.nan)
        res["phase_results"][name] = pr
        log(f"[{name}] B arrival-time: {stB['mean_deg']:.0f} deg ({stB['text']}), R={stB['R']:.2f}, "
            f"speed={pr['speedB_med']:.2f} px/s | A gradient-flow: {stA['mean_deg']:.0f} deg, R={stA['R']:.2f}")
    res["summary_text"] = make_summary(res, len(T))
    return res


def make_summary(res, n_frames):
    fps, ph, tb = res["fps"], res["phases"], res["table"]
    lines = [f"Sequence of {n_frames} frames ({n_frames / fps:.1f} s at {fps:g} fps). "
             f"ROI = {res['mask'].mean() * 100:.0f}% of the image. Values are normalized (0-1), not degrees."]
    pc, pr = res["phase_results"].get("cooling"), res["phase_results"].get("rewarming")
    if pc:
        lines.append(f"COOLING (frames {pc['a']}-{pc['b']}): mean change {pc['mean_change']:+.3f}; "
                     f"max {tb.pct_falling.max():.0f}% of ROI falling at the same time.")
    if pr:
        s = pr["statsB"]
        coher = "clear" if s["R"] > 0.7 else "weak" if s["R"] > 0.4 else "no"
        lines.append(f"REWARMING (frames {pr['a']}-{pr['b']}): mean change {pr['mean_change']:+.3f}; "
                     f"max {tb.pct_rising.max():.0f}% of ROI rising at the same time.")
        lines.append(f"Warming front travels toward {s['text']} ({s['mean_deg']:.0f} deg), {coher} direction "
                     f"(coherence R={s['R']:.2f}), median speed ~{pr['speedB_med']:.1f} px/s.")
        a = pr["statsA"]
        lines.append(f"Second method (gradient flow): {a['mean_deg']:.0f} deg (R={a['R']:.2f}); "
                     f"the two methods differ by {pr['agreement_deg']:.0f} deg.")
    else:
        lines.append("No rewarming phase found in this recording.")
    lines.append("Interpretation: surface-temperature pattern related to perfusion; it is NOT a direct "
                 "measurement of blood flow.")
    return "\n".join(lines)


def run(T_raw, fps, out_dir, name="sequence", mask=None, cfg=None, make_video=True, video_stride=1):
    """analyze + save everything into out_dir."""
    from . import viz
    os.makedirs(out_dir, exist_ok=True)
    res = analyze(T_raw, fps, mask, cfg)
    res["table"].to_csv(os.path.join(out_dir, "frame_by_frame.csv"), index=False)
    viz.save_summary_figure(res, os.path.join(out_dir, "summary.png"), name)
    viz.save_states_figure(res, os.path.join(out_dir, "rising_falling_over_time.png"))
    if make_video:
        viz.save_video(res, os.path.join(out_dir, "demo.mp4"), stride=video_stride)
    summ = {"name": name, "fps": fps, "frames": int(len(res["T"])), "noise_sigma": res["sigma_T"],
            "slope_threshold": res["thr"], "phases": {k: (int(v) if not isinstance(v, (bool, np.bool_)) else bool(v))
                                                      for k, v in res["phases"].items()}, "text": res["summary_text"]}
    for ph, pr in res["phase_results"].items():
        summ[ph] = dict(arrival_time_method=pr["statsB"], gradient_flow_method=pr["statsA"],
                        speed_arrival_px_per_s=pr["speedB_med"], speed_gradient_px_per_s=pr["speedA_med"],
                        agreement_deg=pr["agreement_deg"], mean_change=pr["mean_change"])
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else float(o))
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(res["summary_text"])
    print("\n" + res["summary_text"])
    print(f"\nSaved results to {out_dir}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="folder of images / video / .npy")
    ap.add_argument("--synthetic", type=float, help="use a synthetic sequence with this true angle (deg)")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--scale", type=float, default=1.0, help="resize factor to save time/memory, e.g. 0.5")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th frame (fps is divided accordingly)")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--mask", default=None, help="optional mask image (white = ROI)")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--dark_body", action="store_true", help="body is DARKER than the background")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--video_stride", type=int, default=1)
    a = ap.parse_args()

    cfg = Config(register=a.register, body_is_brighter=not a.dark_body)
    if a.synthetic is not None:
        from .synth import make_synthetic
        T, mask, info = make_synthetic(a.synthetic, fps=a.fps)
        mask = None  # let the pipeline find the ROI by itself
        fps, name = a.fps, f"synthetic_{a.synthetic:g}deg"
        print(f"TRUE direction of the synthetic sequence: {info['angle_deg']} deg, speed {info['speed']} px/s")
    else:
        T, _ = io_utils.load_sequence(a.input, a.stride, a.max_frames, a.scale)
        fps, mask, name = a.fps / a.stride, None, "_".join(os.path.normpath(a.input).split(os.sep)[-3:])
        if a.mask:
            mask = roi.load_mask(a.mask, T.shape[1:])
    run(T, fps, a.out, name, mask, cfg, make_video=not a.no_video, video_stride=a.video_stride)


if __name__ == "__main__":
    main()
