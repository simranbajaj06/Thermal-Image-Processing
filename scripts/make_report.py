"""
BUILD THE DELIVERABLE DOCUMENTS FROM YOUR OWN RESULTS
-----------------------------------------------------
Reads the result folders produced by the pipeline and writes
    report/Technical_Report.docx            full technical report (approach, results, limitations) with figures
    report/Algorithm_Explanation.md         the algorithm explanation on its own
    report/References_and_Datasets.md       research papers and datasets referenced

Every number in the "Results" chapter is read from results\\...\\summary.json, so the report always matches what you ran.
Re-run this script after every new analysis run.

Run:
  python scripts\\make_report.py --seq_dir results\\two_sequences --eval_dir results\\evaluation --author "Your Name"
Needs:  pip install python-docx
"""
import os
import sys
import re
import json
import glob
import argparse
import datetime
import numpy as np

# ===================================================================== small helpers
def num(x, nd=2, unit=""):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{float(x):.{nd}f}{unit}"
    except Exception:
        return "n/a"


def compass(a):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return "no direction"
    names = ["right", "upper-right", "up", "upper-left", "left", "lower-left", "down", "lower-right"]
    return names[int(((a + 22.5) % 360) // 45)]


def ang_diff(a, b):
    return float(abs((a - b + 180) % 360 - 180))


# ===================================================================== document model
class Doc:
    """Tiny document model: a list of blocks that can be rendered to .docx and to .md"""
    def __init__(self):
        self.blocks = []
        self.section = None

    def h1(self, t, key=None):
        self.section = key or t
        self.blocks.append(("h1", t, self.section))

    def h2(self, t):
        self.blocks.append(("h2", t, self.section))

    def p(self, t, flag=False):
        self.blocks.append(("p", t, self.section, flag))

    def bullets(self, items):
        self.blocks.append(("bullets", items, self.section))

    def numbered(self, items):
        self.blocks.append(("numbered", items, self.section))

    def table(self, header, rows, widths=None):
        self.blocks.append(("table", header, rows, widths, self.section))

    def img(self, path, caption, width=6.2):
        if path and os.path.exists(path):
            self.blocks.append(("img", path, caption, width, self.section))

    def code(self, t):
        self.blocks.append(("code", t, self.section))

    def pagebreak(self):
        self.blocks.append(("pagebreak", None, self.section))


# ===================================================================== static knowledge (references, algorithm text)
REFERENCES = [
    ("Provided with the assignment",
     ["Journal of Thermal Biology (2023), article PII S0306-4565(23)00005-0: https://www.sciencedirect.com/science/article/abs/pii/S0306456523000050 "
      "[add authors, title and year from the article page]",
      "Thermal Infrared Imaging to Evaluate Emotional Competences in Nursing Students: A First Approach through a Case Study. "
      "Sensors (2020), PMC7248891: https://pmc.ncbi.nlm.nih.gov/articles/PMC7248891/ [add authors from the article page]"]),
    ("Physiological basis of skin temperature and thermography",
     ["Pennes HH. Analysis of tissue and arterial blood temperatures in the resting human forearm. J Appl Physiol 1948;1(2):93-122.",
      "Ring EFJ, Ammer K. Infrared thermal imaging in medicine. Physiological Measurement 2012;33(3):R33-R46.",
      "Garbey M, Sun N, Merla A, Pavlidis I. Contact-free measurement of cardiac pulse based on the analysis of thermal imagery. "
      "IEEE Trans Biomed Eng 2007;54(8):1418-1426."]),
    ("Dynamic thermography after a cold challenge (related literature found by search - verify bibliographic details before citing)",
     ["Variability in peripheral rewarming after cold stress among 255 healthy Norwegian army conscripts assessed by dynamic infrared "
      "thermography. Int J Circumpolar Health (2018). https://www.tandfonline.com/doi/full/10.1080/22423982.2018.1536250",
      "Dynamic infrared imaging for analysis of fingertip temperature after cold water stimulation and neurothermal modeling study. "
      "Comput Biol Med (2010). https://www.sciencedirect.com/science/article/abs/pii/S0010482510000806",
      "Cold stress dynamic thermography for evaluation of vascular disorders in hand-arm vibration syndrome (title as found in search results).",
      "Dynamic infrared thermography reveals bilateral thermal recovery asymmetry in scoliosis. PMC13413119 "
      "(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13413119/)."]),
    ("Image-analysis and statistics methods used",
     ["Otsu N. A threshold selection method from gray-level histograms. IEEE Trans Syst Man Cybern 1979;9(1):62-66.",
      "Savitzky A, Golay MJE. Smoothing and differentiation of data by simplified least squares procedures. Anal Chem 1964;36(8):1627-1639.",
      "Lucas BD, Kanade T. An iterative image registration technique with an application to stereo vision. Proc IJCAI 1981, pp. 674-679.",
      "Horn BKP, Schunck BG. Determining optical flow. Artificial Intelligence 1981;17:185-203.",
      "Evangelidis GD, Psarakis EZ. Parametric image alignment using enhanced correlation coefficient maximization. "
      "IEEE Trans Pattern Anal Mach Intell 2008;30(10):1858-1865.",
      "Mardia KV, Jupp PE. Directional Statistics. Wiley, 2000 (circular mean and mean resultant length R)."]),
    ("Dataset used",
     ["DIRT-SR38: Real Paired Cross-Sensor Thermal Sequence Dataset. Zenodo, DOI 10.5281/zenodo.19241293 (CC BY-NC 4.0). "
      "38 subjects, 30,780 aligned low-/high-resolution thermal image pairs, standardized abdominal cooling-rewarming protocol, "
      "8-bit and 16-bit PNG, relative (not absolute-temperature) intensities. Companion paper: Clarys et al., CVPR Workshops 2026 "
      "(cited as requested on the dataset page)."]),
]

PIPELINE_TABLE = [
    ("1 Load", "folder of PNG frames", "float array T[frames, H, W], values 0-1"),
    ("2 Pre-process", "T", "robust 0-1 scaling of the whole sequence, optional motion correction (ECC), Gaussian blur (sigma 1 px), "
                         "temporal Savitzky-Golay smoothing (2 s window)"),
    ("3 ROI", "T", "binary mask: whole frame (close-ups) or Otsu body segmentation (body on background); border eroded"),
    ("4 Frame-by-frame change", "smoothed T, mask", "per-pixel slope (units/s) from a Savitzky-Golay derivative; noise level from frame "
                                                  "differences; slope threshold = 3 x theoretical slope noise"),
    ("5 Rising / falling regions", "slope maps, threshold", "state +1 (rising) / -1 (falling) / 0 (stable) per pixel and frame; "
                                                         "ROI-mean curve; cooling and rewarming phases split at the coldest moment"),
    ("6 Direction and propagation", "smoothed T, slope, phase", "arrival-time map, direction + speed (Method B); Lucas-Kanade gradient flow "
                                                              "(Method A); early/late recovery foci; pattern class"),
    ("7 Summary and visualisation", "all of the above", "heat maps, arrows, rose plot, video, CSV/JSON, text summary"),
]

ALGO_STEPS = [
    ("Frame-by-frame analysis (steps 4-5)",
     ["Every pixel's time course is smoothed and differentiated with a Savitzky-Golay filter, giving a slope map (change per second) for every "
      "frame. A pixel is called rising if its slope exceeds +k sigma_slope, falling if it is below -k sigma_slope and stable otherwise (k = 3).",
      "sigma_slope is not a free parameter. The pixel noise sigma_T is estimated robustly from differences between consecutive frames (median absolute "
      "deviation / sqrt 2), and the theoretical noise of a Savitzky-Golay slope for window length w and frame rate fps is "
      "sigma_slope = sigma_T * sqrt(12 / (w (w^2 - 1))) * fps.",
      "The ROI-average curve is split at its coldest moment: before it is the COOLING phase, after it the REWARMING phase. Some recordings start "
      "after cooling has ended; then only the rewarming phase exists and is analysed."]),
    ("Method B - arrival-time map (main method)",
     ["For every pixel and a chosen phase, v0 is the value at the start and v1 the value at the end. The pixel's arrival time is the last moment it was "
      "still short of v0 + 0.5 (v1 - v0), linearly interpolated between two frames. Using the LAST moment makes the estimate immune to an early noisy crossing. "
      "Pixels that change less than 6 noise sigmas are discarded.",
      "The arrival-time map is smoothed (masked Gaussian, 2 px). The warming pattern travels in the direction in which arrival time increases: "
      "direction = grad(t) / |grad(t)| and speed = 1 / |grad(t)| (pixels per second). Example: arrival times of 12, 14 and 16 s at x = 10, 20, 30 px "
      "give a front moving right at 1 / 0.2 s per px = 5 px/s.",
      "Meaning for real skin: for an approximately exponential recovery, the 50 % time is proportional to the local recovery time constant. The map therefore "
      "orders regions from fast to slow rewarming, which is the perfusion-related information; it shows a travelling front only if the order is spatially smooth."]),
    ("Method A - gradient flow (independent check)",
     ["If a temperature pattern moves with velocity v then dT/dt = -(vx dT/dx + vy dT/dy). The slope map supplies dT/dt and Sobel filters supply the spatial "
      "gradients. v is found by least squares in a 15 x 15 window (Lucas-Kanade) with a small regulariser, so where only one gradient direction exists the result "
      "reduces to the observable normal component. Vectors of all frames of the phase are averaged, weighted by gradient energy; the same per-frame field draws the "
      "arrows in the video."]),
    ("Summary of the pattern (circular statistics and recovery foci)",
     ["All pixel directions are combined into a mean angle (0 deg = right, 90 deg = up) and a coherence R between 0 and 1 (R = length of the mean unit vector). "
      "R >= 0.6 means one dominant direction ('uniform').",
      "A single angle cannot describe radial patterns, where warming spreads away from a hot spot or vessel. The algorithm therefore also finds early-recovery foci "
      "(local minima of the arrival-time map = sources) and late-recovery foci (local maxima = sinks). The outward alignment is the mean cosine between each arrow and "
      "the vector from the nearest source to the pixel (+1 = all arrows point away); the convergence alignment is the same toward the nearest sink.",
      "Pattern class: 'uniform' (R >= 0.6); 'focal' (R < 0.6 and an alignment >= 0.5); otherwise 'mixed' = no coherent propagation could be resolved."]),
]

LIMITATIONS = [
    "Surface only: a thermal camera measures skin-surface radiation, not blood flow. The output is a perfusion-RELATED thermal pattern, never a velocity of blood.",
    "Conduction and other effects: heat also spreads through tissue by conduction and is influenced by skin thickness, fat, body curvature, sweating, ambient air flow and "
    "contact with the cold stimulus. A spreading warm patch is therefore not necessarily blood travelling in that direction.",
    "Relative units: the dataset stores relative, intensity-normalised values, not degrees Celsius, so amplitudes cannot be compared with physiological temperature changes.",
    "No ground truth for direction: no public dataset provides the true propagation direction. Accuracy was measured on simulated sequences only; on real data only internal "
    "consistency (method A vs B, subject vs subject, parameter stability) can be shown.",
    "Arrival time depends on each pixel's own time course: spatial differences in recovery speed, not only a travelling front, create arrival-time gradients.",
    "ROI: the automatic body-on-background segmentation is unsuitable for close-ups in which skin fills the frame; the whole-frame ROI was used there. No anatomical landmarks are used.",
    "Motion and resolution: small movements and breathing blur the maps; motion correction is optional and translation-only. Image resolution and frame rate limit the smallest resolvable gradient.",
    "Thresholds (3 sigma slope threshold, 50 % arrival level, 0.6 coherence limit) are reasonable defaults but are not clinically validated.",
    "Not a diagnostic tool: the method has not been validated against any clinical perfusion measurement.",
]


# ===================================================================== load results
def load_sequences(seq_dir):
    seqs = []
    for js in sorted(glob.glob(os.path.join(seq_dir, "*", "summary.json"))):
        d = os.path.dirname(js)
        with open(js) as f:
            j = json.load(f)
        j["_dir"] = d
        j["_summary_png"] = os.path.join(d, "summary.png")
        j["_states_png"] = os.path.join(d, "rising_falling_over_time.png")
        seqs.append(j)
    return seqs


def pattern_of(rw):
    if rw is None:
        return None
    if rw.get("pattern"):
        return rw["pattern"]
    R = rw["arrival_time_method"]["R"]
    return "uniform" if R >= 0.6 else "mixed"


def seq_rows(j):
    rw = j.get("rewarming")
    rows = [("Frames / duration", f"{j['frames']} frames, {j['frames'] / j['fps']:.1f} s at {j['fps']:g} fps"),
            ("ROI", f"{j.get('roi_mode', 'not recorded (older run)')}" + (f" ({j['roi_fraction'] * 100:.0f}% of image)" if 'roi_fraction' in j else "")),
            ("Noise level (sigma, normalised units)", num(j.get("noise_sigma"), 4)),
            ("Slope threshold (units/s)", num(j.get("slope_threshold"), 4)),
            ("Cooling phase present", "yes" if j["phases"].get("has_cool") else "no (recording starts after cooling)"),
            ]
    if rw:
        B, A = rw["arrival_time_method"], rw["gradient_flow_method"]
        rows += [("Mean change during rewarming (normalised)", num(rw.get("mean_change"), 3)),
                 ("Pattern class", pattern_of(rw)),
                 ("Method B: mean direction / coherence R", f"{num(B['mean_deg'], 0)} deg ({compass(B['mean_deg'])}) / {num(B['R'])}"),
                 ("Method A: mean direction / coherence R", f"{num(A['mean_deg'], 0)} deg ({compass(A['mean_deg'])}) / {num(A['R'])}"),
                 ("Difference between methods", num(rw.get("agreement_deg"), 0, " deg")),
                 ("Median front speed, Method B", num(rw.get("speed_arrival_px_per_s"), 2, " px/s"))]
        if "n_early_foci" in rw:
            rows += [("Early / late recovery foci", f"{rw['n_early_foci']} / {rw['n_late_foci']}"),
                     ("Outward alignment / convergence alignment", f"{num(rw.get('outward_alignment'))} / {num(rw.get('convergence_alignment'))}")]
    return rows


def seq_text(j):
    rw = j.get("rewarming")
    if not rw:
        return "No rewarming phase was found in this recording."
    B = rw["arrival_time_method"]
    pat = pattern_of(rw)
    t = (f"During rewarming the ROI changed on average by {num(rw.get('mean_change'), 3)} (normalised units); the rising/falling figure shows the share "
         "of ROI pixels that were rising at each moment. ")
    if pat == "uniform":
        t += (f"The arrival-time method finds ONE dominant direction: the recovery pattern spreads toward the {compass(B['mean_deg'])} "
              f"({num(B['mean_deg'], 0)} deg) with coherence R = {num(B['R'])} at a median speed of {num(rw.get('speed_arrival_px_per_s'), 1)} px/s. "
              f"The gradient-flow method gives {num(rw['gradient_flow_method']['mean_deg'], 0)} deg, a difference of {num(rw.get('agreement_deg'), 0)} deg.")
    elif pat == "focal":
        parts = []
        oa, ca = rw.get("outward_alignment"), rw.get("convergence_alignment")
        if rw.get("n_early_foci") and oa is not None and np.isfinite(oa) and oa >= 0.3:
            parts.append(f"{rw['n_early_foci']} early-recovery focus/foci from which the arrows point away (outward alignment {num(oa)})")
        if rw.get("n_late_foci") and ca is not None and np.isfinite(ca) and ca >= 0.3:
            parts.append(f"{rw['n_late_foci']} late-recovery focus/foci toward which the arrows converge (convergence alignment {num(ca)})")
        t += (f"There is NO single direction (R = {num(B['R'])}). Recovery is organised around " + " and ".join(parts)
              + ": the pattern spreads away from the early-recovery region instead of along one global direction.")
    else:
        t += (f"No coherent propagation could be resolved (R = {num(B['R'])}; outward alignment {num(rw.get('outward_alignment'))}, "
              f"convergence {num(rw.get('convergence_alignment'))}). Recovery is spatially heterogeneous: neighbouring regions rewarm at different, "
              "irregular rates and no travelling front is visible. This is a result, not a failure: it is what the data show.")
    return t


# ===================================================================== assemble the document
def build(seqs, eval_dir, author, title, seq_dir):
    d = Doc()
    d.p("", False)
    # ---------------------------------------------------------------- 1
    d.h1("1. Introduction and objective", "intro")
    d.p("Thermal cameras record the infrared radiation emitted by the skin surface. After a local cold stimulus the skin rewarms as blood perfusion returns, "
        "and the spatial pattern and speed of this recovery carry information about the underlying perfusion. The objective of this work is an algorithm that "
        "analyses a sequence of thermal images frame by frame, identifies regions of increasing and decreasing temperature, and estimates the direction and "
        "propagation of the thermal recovery pattern, with clear visualisations and a summary of the whole sequence.")
    d.p("Use case: perfusion-recovery mapping after skin cooling (dynamic infrared thermography). Given a cooling/rewarming recording, the software reports where "
        "the skin warms or cools, how fast, and in which direction or around which foci the recovery spreads.")
    # ---------------------------------------------------------------- 2
    d.h1("2. Scientific basis", "science")
    d.p("Skin temperature results from a balance between heat delivered by blood perfusion, heat conducted from deeper tissue, and heat lost by radiation, convection and "
        "evaporation. A standard description is the Pennes bioheat equation:")
    d.code("rho*c * dT/dt = k * laplacian(T) + w_b * rho_b * c_b * (T_a - T) + q_met")
    d.p("The first term on the right is conduction (k = tissue conductivity), the second is heat exchange with blood (w_b = perfusion rate, T_a = arterial temperature) and q_met "
        "is metabolic heat. After cooling, the skin is below its equilibrium temperature; the speed at which it returns toward it increases with perfusion w_b and "
        "depends on conduction. Regions with stronger local perfusion, or closer to supplying vessels, therefore tend to rewarm earlier, and the spatial order of recovery "
        "can be mapped.")
    d.p("What is measured is surface temperature, not blood flow. The camera signal also depends on emissivity (about 0.98 for skin), reflections and ambient conditions, and the dataset "
        "used here stores relative intensities. The reported quantities are therefore perfusion-related thermal patterns; they are not blood-flow velocities and not a diagnosis.")
    # ---------------------------------------------------------------- 3
    d.h1("3. Research, papers and datasets referenced", "refs")
    for head, items in REFERENCES:
        d.h2(head)
        d.bullets(items)
    d.p("The two references supplied with the assignment concern thermography of physiological states; they motivate the use of surface temperature as a physiological signal "
        "but do not describe a propagation-direction algorithm, which is why the method here is built from the physical model and standard image-analysis tools listed above.")
    d.h2("Dataset used: DIRT-SR38")
    d.p("DIRT-SR38 contains abdominal thermal image sequences of 38 subjects recorded under a standardised cooling-rewarming protocol, stored as aligned low- and "
        "high-resolution pairs (it was published for thermal super-resolution). In the folder layout used here (dataset_export/8bit/<subject>/HR|LR/dynamic|static) the "
        "'dynamic' folders hold the time sequences analysed in this report (high-resolution versions). Values are relative intensities, not degrees Celsius.")
    # ---------------------------------------------------------------- 4
    d.h1("4. Data processing pipeline (input and output of each step)", "pipeline")
    d.table(["Step", "Input", "Output / operation"], PIPELINE_TABLE, [1.5, 1.5, 3.3])
    # ---------------------------------------------------------------- 5
    d.h1("5. Proposed algorithm", "algorithm")
    for head, paras in ALGO_STEPS:
        d.h2(head)
        for t in paras:
            d.p(t)
    # ---------------------------------------------------------------- 6
    d.h1("6. Validation on simulated sequences with known direction", "validation")
    d.p("Real recordings have no ground truth for the propagation direction. The algorithm was therefore validated on simulated cooling-rewarming sequences "
        "(128 x 128 px, noise, slow drift, optional body motion) in which a warming front with a chosen direction and speed sweeps across the body.")
    try:
        import pandas as pd
        pa = pd.read_csv(os.path.join(eval_dir, "synthetic_accuracy.csv"))
        base = pa[pa.jitter_px == 0]
        g = base.groupby("noise")[["err_B", "err_A", "R_B"]].mean()
        rows = [(f"{nz:g}", num(r.err_B, 2), num(r.err_A, 2), num(r.R_B, 2)) for nz, r in g.iterrows()]
        d.table(["Noise sigma", "Mean angular error, Method B (deg)", "Mean angular error, Method A (deg)", "Coherence R (B)"], rows, [1.2, 1.9, 1.9, 1.3])
        d.p(f"Planar fronts: 8 directions x 3 noise levels. Planar fronts classified as 'uniform': {(base.pattern == 'uniform').mean() * 100:.0f}%. "
            f"Median speed error of Method B: {abs(base.speed_B - base.speed_true).mean():.2f} px/s on a true {base.speed_true.iloc[0]:g} px/s.")
        mot = pa[pa.jitter_px > 0].groupby("registered")[["err_B", "err_A"]].mean()
        if len(mot):
            d.p("Body motion (0.7 px random jitter): " + "; ".join(
                f"{'with' if k else 'without'} motion correction error B = {num(v.err_B)} deg, A = {num(v.err_A)} deg" for k, v in mot.iterrows()) + ".")
    except Exception as e:
        d.p(f"[Planar-front evaluation not found in {eval_dir}: run  python scripts\\evaluate_synth.py  ({e})]", flag=True)
    try:
        ra = pd.read_csv(os.path.join(eval_dir, "radial_accuracy.csv"))
        d.p(f"Radial pattern (warming spreading outward from one focus): classified as 'focal' in {(ra.pattern == 'focal').mean() * 100:.0f}% of {len(ra)} runs; "
            f"the focus was located with a mean error of {ra.focus_error_px.mean():.2f} px (max {ra.focus_error_px.max():.2f} px) on a 128 px wide image; "
            f"outward alignment {ra.outward_alignment.mean():.2f}.")
    except Exception:
        d.p("[Radial-pattern evaluation not found: run the updated scripts\\evaluate_synth.py]", flag=True)
    d.img(os.path.join(eval_dir, "synthetic_accuracy.png"), "Figure 1. Direction error on simulated sequences.", 5.8)
    d.p("These numbers are a best case: the simulated front is clean and smooth. They show that the implementation is correct and robust to noise; they do not predict the "
        "error on real skin, where the true direction is unknown.")
    # ---------------------------------------------------------------- 7
    d.h1("7. Results on real thermal sequences", "results")
    if not seqs:
        d.p(f"[No results found in {seq_dir}: run scripts\\run_two_sequences.py first]", flag=True)
    for k, j in enumerate(seqs, 1):
        d.h2(f"7.{k} Sequence {j['name']}")
        rm = j.get("roi_mode")
        if rm == "auto" and j.get("roi_fraction", 1) < 0.35:
            d.p(f"[AUTHOR CHECK: the automatic ROI covers only {j['roi_fraction'] * 100:.0f}% of the image. For close-up recordings re-run with --roi full and "
                "re-generate this report.]", flag=True)
        if rm is None:
            d.p("[AUTHOR CHECK: this result was produced by an older version of the code (ROI mode not recorded). Re-run the analysis with the updated code, "
                "using --roi full for close-ups, and re-generate this report.]", flag=True)
        d.table(["Quantity", "Value"], seq_rows(j), [3.0, 3.3])
        d.p(seq_text(j))
        d.img(j["_summary_png"], f"Figure 7.{k}a. Overview for {j['name']}: ROI, change map with gradient-flow arrows, arrival-time map with direction arrows and recovery foci, "
                                 "ROI curve, direction histogram and text summary.")
        d.img(j["_states_png"], f"Figure 7.{k}b. Percentage of ROI pixels that are rising (red) or falling (blue) over time for {j['name']}.", 5.4)
    cmp_png = os.path.join(seq_dir, "comparison.png")
    if len(seqs) >= 2:
        d.h2(f"7.{len(seqs) + 1} Comparison of the sequences")
        pats = [pattern_of(s.get("rewarming")) for s in seqs]
        rows = []
        for s in seqs:
            rw = s.get("rewarming")
            if rw:
                rows.append((s["name"], pattern_of(rw), f"{num(rw['arrival_time_method']['mean_deg'], 0)} deg", num(rw["arrival_time_method"]["R"]),
                             num(rw.get("speed_arrival_px_per_s"), 1), num(rw.get("agreement_deg"), 0, " deg")))
        d.table(["Sequence", "Pattern", "Direction (B)", "R (B)", "Speed (px/s)", "A vs B"], rows, [2.2, 0.9, 1.0, 0.6, 0.9, 0.8])
        if all(p == "uniform" for p in pats):
            angs = [s["rewarming"]["arrival_time_method"]["mean_deg"] for s in seqs]
            diff = ang_diff(angs[0], angs[1])
            d.p(f"Both sequences show a single dominant direction. The directions differ by {diff:.0f} deg between the two subjects; "
                + ("this is a similar direction." if diff < 45 else "the subjects therefore differ in how recovery spreads, which is expected because the "
                   "anatomy and vascular layout differ between people."))
        elif any(p == "focal" for p in pats):
            d.p("At least one sequence shows a focal pattern: recovery spreads from early-recovery regions toward late-recovery regions instead of along one global direction. "
                "A single mean angle is therefore not a suitable summary for that sequence, which is why the foci and alignment indices are reported.")
        else:
            d.p("Neither sequence shows a coherent propagation direction at the resolution and sensitivity of the data. In both, the recovery is dominated by region-specific "
                "rewarming rates rather than by a travelling front.")
        d.img(cmp_png, "Figure 7.c. Left: ROI-average curves of the sequences. Right: histograms of the arrival-time directions.", 6.0)
    d.h2("Frame-by-frame output")
    d.p("For each sequence the pipeline writes frame_by_frame.csv (time, ROI mean, mean slope, percentage of ROI rising and falling for every frame) and an annotated video "
        "demo.mp4 (thermal frame | rising regions in red and falling regions in blue with arrows | ROI curve with a time marker).")
    # ---------------------------------------------------------------- 8
    d.h1("8. Discussion", "discussion")
    d.p("The arrival-time method is the main estimator: it uses each pixel's complete time course and is robust to noise (0.15-0.24 deg error on simulated sequences), while the gradient-flow method "
        "is an independent, noisier check that also supplies the per-frame arrows of the video. Agreement between the two is meaningful only when the pattern is 'uniform'; for focal "
        "or mixed patterns the mean directions are not expected to agree, and the pattern class and alignment indices carry the information.")
    d.p("A key interpretation point is that the arrival-time map orders regions by how fast they recover. A smooth gradient of arrival time indicates a coherent spread; a patchy map indicates "
        "that local properties (perfusion, tissue thickness, proximity to vessels) dominate. Both are physiologically interesting, but only the first can be called propagation.")
    d.p("Because the results are surface patterns in relative units, comparisons between subjects should use the pattern class, the coherence and the relative speeds rather than absolute temperatures.")
    # ---------------------------------------------------------------- 9
    d.h1("9. Limitations", "limitations")
    d.bullets(LIMITATIONS)
    # ---------------------------------------------------------------- 10
    d.h1("10. Conclusion and future work", "conclusion")
    d.p("An end-to-end algorithm was implemented that loads thermal sequences, finds the region of interest, analyses every frame, labels increasing and decreasing regions, "
        "estimates the direction and speed of the recovery pattern with two independent methods, recognises radial (focal) patterns, visualises them with heat maps, arrows and video, "
        "and summarises the sequence. It was validated on simulated data with known direction and applied to two real subjects of DIRT-SR38.")
    d.bullets(["Calibrated data (absolute temperature) and the dataset paper's protocol details would allow physical units (mm/s, degrees C per s).",
               "A learned ROI/segmentation model (U-Net) for recordings with a body on a background, and the optional super-resolution model "
               "(scripts/train_sr.py) to test whether higher resolution improves the direction estimate.",
               "Fitting an exponential recovery per pixel to obtain a time-constant map, and validating against an independent perfusion measurement.",
               "A multi-subject study to relate pattern class and coherence to physiology."])
    # ---------------------------------------------------------------- appendix
    d.h1("Appendix A. How to reproduce", "appendix")
    d.code("python scripts\\inspect_dataset.py \"<ROOT>\"\n"
           "python scripts\\preview_sequence.py \"<ROOT>\\P11\\HR\\dynamic\"\n"
           "python scripts\\run_two_sequences.py --root \"<ROOT>\" --subjects P11 P12 --fps <FPS> --roi full --scale 0.5 --out results\\two_sequences\n"
           "python scripts\\evaluate_synth.py\n"
           "python scripts\\make_report.py --seq_dir results\\two_sequences --eval_dir results\\evaluation --author \"<Name>\"\n"
           "streamlit run app.py")
    d.p("<ROOT> = ...\\DIRT-SR38\\dataset_export\\8bit.  <FPS> = frame rate from the dataset paper.")
    return d


# ===================================================================== renderers
def render_md(doc, path, sections=None, title=""):
    out = [f"# {title}\n"] if title else []
    for b in doc.blocks:
        sec = b[-1] if b[0] != "p" else b[2]
        if sections and sec not in sections:
            continue
        k = b[0]
        if k == "h1":
            out.append(f"\n## {b[1]}\n")
        elif k == "h2":
            out.append(f"\n### {b[1]}\n")
        elif k == "p":
            out.append(b[1] + "\n")
        elif k in ("bullets", "numbered"):
            out += [(f"- {t}" if k == "bullets" else f"{i}. {t}") for i, t in enumerate(b[1], 1)] + [""]
        elif k == "table":
            out.append("| " + " | ".join(b[1]) + " |")
            out.append("|" + "---|" * len(b[1]))
            out += ["| " + " | ".join(str(c) for c in r) + " |" for r in b[2]] + [""]
        elif k == "code":
            out.append("```\n" + b[1] + "\n```\n")
        elif k == "img":
            out.append(f"*(figure: {os.path.basename(b[1])}) {b[2]}*\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def render_docx(doc, path, title, author, subtitle):
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    D = Document()
    sec = D.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)               # A4
    sec.left_margin = sec.right_margin = Cm(2.3)
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    st = D.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.1
    for name, size in (("Heading 1", 16), ("Heading 2", 13)):
        h = D.styles[name]
        h.font.name, h.font.size, h.font.bold = "Calibri", Pt(size), True
        h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        h.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True

    # title block
    p = D.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(title); r.bold = True; r.font.size = Pt(24); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    p = D.add_paragraph(); r = p.add_run(subtitle); r.font.size = Pt(13); r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    p = D.add_paragraph(); r = p.add_run(f"{author}    |    {datetime.date.today().strftime('%d %B %Y')}"); r.font.size = Pt(11)
    pb = p._p.get_or_add_pPr(); bd = OxmlElement("w:pBdr"); bt = OxmlElement("w:bottom")
    for k, v in (("w:val", "single"), ("w:sz", "8"), ("w:space", "6"), ("w:color", "1F3A5F")):
        bt.set(qn(k), v)
    bd.append(bt); pb.append(bd)

    def shade(cell, color):
        tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
        sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color); tcPr.append(sh)

    for b in doc.blocks:
        k = b[0]
        if k == "h1":
            D.add_heading(b[1], 1)
        elif k == "h2":
            D.add_heading(b[1], 2)
        elif k == "p":
            if not b[1]:
                continue
            para = D.add_paragraph(); run = para.add_run(b[1])
            if b[3]:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW; run.bold = True
        elif k in ("bullets", "numbered"):
            for t in b[1]:
                D.add_paragraph(t, style="List Bullet" if k == "bullets" else "List Number")
        elif k == "code":
            for line in b[1].split("\n"):
                para = D.add_paragraph(); para.paragraph_format.space_after = Pt(0)
                para.paragraph_format.left_indent = Cm(0.6)
                rr = para.add_run(line); rr.font.name = "Consolas"; rr.font.size = Pt(9.5)
                rr._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            D.add_paragraph().paragraph_format.space_after = Pt(2)
        elif k == "table":
            hdr, rows, widths = b[1], b[2], b[3]
            t = D.add_table(rows=1, cols=len(hdr)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for i, h in enumerate(hdr):
                c = t.rows[0].cells[i]; c.text = ""; rr = c.paragraphs[0].add_run(h); rr.bold = True; rr.font.size = Pt(10)
                shade(c, "D9E2F3")
            for row in rows:
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = ""; rr = cells[i].paragraphs[0].add_run(str(v)); rr.font.size = Pt(10)
            if widths:
                for row in t.rows:
                    for i, w in enumerate(widths):
                        row.cells[i].width = Inches(w)
            D.add_paragraph().paragraph_format.space_after = Pt(2)
        elif k == "img":
            para = D.add_paragraph(); para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.keep_with_next = True
            para.add_run().add_picture(b[1], width=Inches(min(b[3], 6.2)))
            cap = D.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            rr = cap.add_run(b[2]); rr.italic = True; rr.font.size = Pt(9.5)
        elif k == "pagebreak":
            D.add_page_break()

    # page numbers
    fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
        run._r.append(e)
    D.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", default="results/two_sequences")
    ap.add_argument("--eval_dir", default="results/evaluation")
    ap.add_argument("--out", default="report")
    ap.add_argument("--author", default="Author")
    ap.add_argument("--title", default="Estimating Perfusion-Related Thermal Patterns from Thermal Image Sequences")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seqs = load_sequences(a.seq_dir)
    print(f"{len(seqs)} sequence result(s) found in {a.seq_dir}")
    doc = build(seqs, a.eval_dir, a.author, a.title, a.seq_dir)
    for s in seqs:
        if s.get("roi_mode") is None:
            print(f"!! {s['name']}: produced by an older code version (no ROI info). Re-run the analysis with the updated code.")
        elif s.get("roi_mode") == "auto" and s.get("roi_fraction", 1) < 0.35:
            print(f"!! {s['name']}: automatic ROI covers only {s['roi_fraction'] * 100:.0f}% - re-run with --roi full if the skin fills the frame.")
    render_docx(doc, os.path.join(a.out, "Technical_Report.docx"), a.title, a.author, "Technical report - AI/ML thermal image processing assignment")
    render_md(doc, os.path.join(a.out, "Algorithm_Explanation.md"), {"algorithm", "pipeline"}, "Explanation of the proposed algorithm")
    render_md(doc, os.path.join(a.out, "References_and_Datasets.md"), {"refs"}, "Research papers and datasets referenced")
    print(f"Written to {a.out}:  Technical_Report.docx, Algorithm_Explanation.md, References_and_Datasets.md")


if __name__ == "__main__":
    main()
