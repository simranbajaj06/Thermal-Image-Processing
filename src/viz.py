"""
STEP 7 - VISUALIZATION
----------------------
summary figure   : mean image, change map + arrows, arrival-time map + isochrones, curve, rose plot, text
states figure    : % of ROI rising / falling over time
video            : frame-by-frame  [thermal frame | warming(red)/cooling(blue) map + arrows | ROI curve]
Colour code everywhere: RED = getting warmer, BLUE = getting colder, arrows = direction of propagation.
"""
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from . import flow


def _arrow_grid(shape, step):
    H, W = shape
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    return xs, ys


def _mask_outline(ax, mask):
    ax.contour(mask.astype(float), levels=[0.5], colors="white", linewidths=0.8)


def _draw_dir_arrows(ax, xs, ys, u, v, color="k", scale=22, width=0.006):
    """Draw unit-length arrows (direction only). u,v in image coordinates (y down)."""
    mag = np.hypot(u, v)
    ok = np.isfinite(mag) & (mag > 0)
    if not ok.any():
        return None
    return ax.quiver(xs[ok], ys[ok], (u / mag)[ok], (v / mag)[ok], color=color, angles="xy",
                     scale_units="xy", scale=1 / (scale / 2.0), width=width, headwidth=4, pivot="mid")


# ---------------------------------------------------------------------- summary figure
def save_summary_figure(res, path, name="sequence"):
    mask, T, fps = res["mask"], res["T"], res["fps"]
    pr = res["phase_results"].get("rewarming")
    ph = res["phases"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle(f"Thermal pattern analysis - {name}", fontsize=15, fontweight="bold")
    H, W = mask.shape
    step = max(6, min(H, W) // 16)
    xs, ys = _arrow_grid((H, W), step)

    ax = axs[0, 0]
    ax.imshow(T.mean(0), cmap="inferno")
    _mask_outline(ax, mask)
    ax.set_title("Average thermal image + ROI (white outline)", fontsize=10)

    ax = axs[0, 1]
    if pr:
        amp = np.where(mask, pr["amp"], np.nan)
        lim = np.nanpercentile(np.abs(amp), 98) + 1e-9
        im = ax.imshow(amp, cmap="RdBu_r", vmin=-lim, vmax=lim)
        plt.colorbar(im, ax=ax, fraction=0.046, label="change during rewarming (normalized)")
        okA = pr["A_ok"]
        _draw_dir_arrows(ax, xs, ys, np.where(okA, pr["A_vx"], np.nan)[ys, xs], np.where(okA, pr["A_vy"], np.nan)[ys, xs])
        ax.set_title("Rewarming change (red = warmer)\n+ arrows: METHOD A (gradient flow)", fontsize=10)
    else:
        ax.set_title("no rewarming phase found")

    ax = axs[0, 2]
    if pr:
        ta = pr["B"]["ta_smooth"]
        im = ax.imshow(np.where(mask, ta, np.nan), cmap="viridis")
        plt.colorbar(im, ax=ax, fraction=0.046, label="arrival time of the warming front (s)")
        try:
            ax.contour(np.where(np.isfinite(ta), ta, np.nan), levels=8, colors="white", linewidths=0.7)
        except Exception:
            pass
        g = pr["B"]
        _draw_dir_arrows(ax, xs, ys, g["gx"][ys, xs] * np.where(g["good"][ys, xs], 1, np.nan),
                         g["gy"][ys, xs] * np.where(g["good"][ys, xs], 1, np.nan), color="red")
        for (x, y, _t) in pr["sources"]:
            ax.plot(x, y, "*", ms=16, mfc="white", mec="k", mew=1.2)
        for (x, y, _t) in pr["sinks"]:
            ax.plot(x, y, "X", ms=11, mfc="k", mec="white", mew=1.0)
        ax.set_title("METHOD B: arrival-time of the warming front\n(white lines = same time, red arrows = direction,\n"
                     "star = earliest-recovery focus, X = latest-recovery focus)", fontsize=9)

    ax = axs[1, 0]
    t = np.arange(len(res["curve"])) / fps
    ax.plot(t, res["curve"], "k", lw=2)
    ax.axvspan(ph["t_cool_start"] / fps, ph["t_min"] / fps, color="tab:blue", alpha=0.2, label="cooling")
    ax.axvspan(ph["t_min"] / fps, t[-1], color="tab:red", alpha=0.2, label="rewarming")
    ax.set_xlabel("time (s)"); ax.set_ylabel("ROI mean (normalized)"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Average ROI value over time")

    axp = fig.add_subplot(2, 3, 5, projection="polar")
    axs[1, 1].axis("off")
    if pr:
        for key, col, lab in (("B", "tab:red", "B arrival-time"), ("A", "tab:blue", "A gradient flow")):
            ang = pr["B"]["angle"] if key == "B" else np.where(pr["A_ok"], pr["A_angle"], np.nan)
            a = np.deg2rad(ang[np.isfinite(ang)])
            if a.size:
                h, edges = np.histogram(a, bins=36, range=(0, 2 * np.pi))
                if key == "B":   # filled bars
                    axp.bar(edges[:-1], h / h.max(), width=2 * np.pi / 36, alpha=0.45, color=col, label=lab, align="edge")
                else:            # outline only, so both stay readable when they overlap
                    axp.bar(edges[:-1], h / h.max(), width=2 * np.pi / 36, fill=False, edgecolor=col,
                            linewidth=1.8, label=lab, align="edge")
        axp.legend(loc="lower left", bbox_to_anchor=(-0.2, -0.2), fontsize=8)
    axp.set_title("Direction histogram (0 deg = right, 90 = up)", fontsize=10)

    ax = axs[1, 2]
    ax.axis("off")
    ax.text(0, 1, res["summary_text"], va="top", ha="left", wrap=True, fontsize=10,
            transform=ax.transAxes, bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#999"))
    ax.set_title("Summary")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_states_figure(res, path):
    tb, fps, ph = res["table"], res["fps"], res["phases"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(tb.time_s, 0, tb.pct_rising, color="tab:red", alpha=0.6, label="% of ROI rising")
    ax.fill_between(tb.time_s, 0, -tb.pct_falling, color="tab:blue", alpha=0.6, label="% of ROI falling")
    ax.axvline(ph["t_min"] / fps, color="k", ls="--", label="coldest moment")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time (s)"); ax.set_ylabel("% of ROI pixels"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Increasing vs decreasing temperature regions over time")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------- video
def _smoothed_samples(samples, radius=3):
    """Average the arrow field over +-radius frames so arrows do not flicker."""
    U, V = np.array(samples["u"]), np.array(samples["v"])
    ts = np.array(samples["t"])
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, t in enumerate(ts):
            lo, hi = max(0, i - radius), min(len(ts), i + radius + 1)
            out[int(t)] = (np.nanmean(U[lo:hi], 0), np.nanmean(V[lo:hi], 0))
    return out


def save_video(res, path, stride=1, fps_out=15):
    import imageio.v2 as imageio
    T, slope, mask, thr, fps = res["T"], res["slope"], res["mask"], res["thr"], res["fps"]
    H, W = mask.shape
    arrows = {}
    grid = None
    for pr in res["phase_results"].values():
        arrows.update(_smoothed_samples(pr["A_samples"]))
        grid = pr["A_samples"]["grid"]
    vmax = max(4 * thr, np.percentile(np.abs(slope[:, mask]), 99))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), dpi=80)
    t_axis = np.arange(len(T)) / fps
    im1 = axs[0].imshow(T[0], cmap="inferno", vmin=0, vmax=1)
    _mask_outline(axs[0], mask)
    axs[0].set_title("thermal frame")
    axs[1].imshow(T.mean(0), cmap="gray", alpha=0.6)
    heat = axs[1].imshow(np.zeros((H, W)), cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.0)
    _mask_outline(axs[1], mask)
    axs[1].legend(handles=[Patch(color="tab:red", label="warming"), Patch(color="tab:blue", label="cooling")],
                  loc="lower right", fontsize=8)
    axs[2].plot(t_axis, res["curve"], "k")
    marker = axs[2].axvline(0, color="r")
    axs[2].set_xlabel("time (s)"); axs[2].set_ylabel("ROI mean"); axs[2].grid(alpha=0.3)
    axs[2].set_title("average ROI value")
    for a in axs[:2]:
        a.axis("off")
    fig.tight_layout()
    quiv = [None]
    writer = None
    try:
        writer = imageio.get_writer(path, fps=fps_out, codec="libx264", macro_block_size=1, quality=7)
        out_kind = "mp4"
    except Exception:
        path = path.replace(".mp4", ".gif")
        writer = imageio.get_writer(path, fps=fps_out)
        out_kind = "gif"
    for t in range(0, len(T), stride):
        im1.set_data(T[t])
        s = np.where(mask, slope[t], 0.0)
        heat.set_data(s)
        heat.set_alpha(np.where(np.abs(s) > thr, 0.85, 0.0))
        axs[1].set_title(f"frame {t}  t={t / fps:.1f}s  | red = warming, blue = cooling")
        if quiv[0] is not None:
            quiv[0].remove(); quiv[0] = None
        if t in arrows and grid is not None:
            xs, ys = grid
            u, v = arrows[t]
            active = np.abs(slope[t])[ys, xs] > thr
            quiv[0] = _draw_dir_arrows(axs[1], xs, ys, np.where(active, u, np.nan), np.where(active, v, np.nan))
        marker.set_xdata([t / fps, t / fps])
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    writer.close()
    plt.close(fig)
    return path
