# Thermal image sequence → perfusion-related pattern (DIRT-SR38)

Estimates **where the skin warms or cools, how fast, and in which direction the change spreads** from a
sequence of thermal images. Thermal cameras see *surface temperature*, not blood flow, so everything here is a
**perfusion-related thermal pattern**, never a direct blood-flow measurement.

## 0. Setup (once) - Windows
Unzip this project, e.g. to `D:\Thermal\thermal_flow`, then open **PowerShell** or **Command Prompt** there:
```
cd D:\Thermal\thermal_flow
python -m venv venv
venv\Scripts\activate
pip install numpy scipy opencv-python matplotlib pandas imageio imageio-ffmpeg streamlit
```
(PowerShell may refuse `activate`; if so run `Set-ExecutionPolicy -Scope Process Bypass` first. PyTorch is only
needed for the optional model training - see step 4.)
Always run the commands below **from inside `D:\Thermal\thermal_flow`**. Put each command on ONE line.

## 1. Try it with NO dataset (2 minutes) - proves the code works
```
python -m src.pipeline --synthetic 135 --out results\synth_135
python scripts\evaluate_synth.py
streamlit run app.py
```

## 2. Your dataset layout
```
D:\Thermal\DIRT-SR38\dataset_export\8bit\P11\HR\dynamic   <- time sequence (use this)
                                          \P11\HR\static
                                          \P11\LR\dynamic
                                          \P11\LR\static
                                     \P12\...   (one folder per subject)
```
Folder "ROOT" = `D:\Thermal\DIRT-SR38\dataset_export\8bit` (the folder that contains P11, P12, ...).
`dynamic` is assumed to be the cooling/rewarming recording; `static` is assumed to be separate still images - confirm
both in the dataset paper (I have not seen it). The analysis uses `HR\dynamic` only.

**Before anything else, find the frame rate (fps) in the dataset paper** (and how long the cooling lasts). Speeds and
time axes depend on it. If you cannot find it, use `--fps 1`: then times are in "frames" and speeds in pixels/frame.

### 2a. See how the dataset is organised
```
python scripts\inspect_dataset.py "D:\Thermal\DIRT-SR38\dataset_export\8bit"
```
### 2b. PREVIEW one sequence (do not skip!)
```
python scripts\preview_sequence.py "D:\Thermal\DIRT-SR38\dataset_export\8bit\P11\HR\dynamic"
```
Open `results\preview\P11_HR_dynamic.png` and check: (1) the cyan outline sits on the body, (2) the curve dips and
recovers, (3) the printed warning about "stretched frames" does not appear. If the body is DARKER than the
background add `--dark_body` to the preview and to all later commands.

## 3. Run the analysis
One subject:
```
python -m src.pipeline --input "D:\Thermal\DIRT-SR38\dataset_export\8bit\P11\HR\dynamic" --fps 9 --scale 0.5 --video_stride 5 --out results\P11
```
Two subjects (the assignment's "at least two sequences"):
```
python scripts\run_two_sequences.py --root "D:\Thermal\DIRT-SR38\dataset_export\8bit" --subjects P11 P12 --fps 9 --scale 0.5 --video_stride 5 --out results\two_sequences
```
* Replace `9` by the real fps. `--scale 0.5` halves the image size, `--stride 2` uses every 2nd frame (both save a lot of
  memory/time; fps is adjusted automatically). Remove them once everything works if you want full detail.
* Optional: `--register` (people move/breathe), `--dark_body`, `--mask my_mask.png` (white = ROI, to override the automatic ROI).
* Open `results\two_sequences\comparison.png` and `comparison.csv` for the side-by-side result.

## 4. (Optional) Train the super-resolution model and test it on the main task
Needs PyTorch (https://pytorch.org/get-started/locally/ gives the right `pip install` line for your PC; a GPU is strongly advised).
```
python scripts\train_sr.py --root "D:\Thermal\DIRT-SR38\dataset_export\8bit" --kinds dynamic --epochs 20
python scripts\sr_downstream_test.py --lr_dir "D:\Thermal\DIRT-SR38\dataset_export\8bit\P11\LR\dynamic" --hr_dir "D:\Thermal\DIRT-SR38\dataset_export\8bit\P11\HR\dynamic" --model models\edsr_x2.pt --fps 9
```
LR/HR images are paired by identical file name (or, if names differ but counts match, by order - a warning is printed).
The model trains on all subjects except the last 15 % (validation), so the score is on people it has never seen.
Use a validation subject for the downstream test (`--val_subjects P37 P38` fixes them explicitly).

## Troubleshooting
| Problem | Likely cause / fix |
|---|---|
| Preview says ROI curve is flat | wrong folder, wrong ROI (try `--dark_body` or a mask), or the dataset was normalised per frame (read paper) |
| Warning "frames stretched to full range" | per-frame normalisation: temporal changes are unreliable; mention it as a limitation and ask your supervisor |
| `MemoryError` | use `--scale 0.5 --stride 2` (or `--max_frames 400`) |
| Arrows point in odd directions | check the coherence R in the summary; R < 0.4 means no real direction |
| Video not created | the code falls back to a `.gif`; or `pip install imageio-ffmpeg` |
| `ModuleNotFoundError: src` | run from inside the project folder, and use `python -m src.pipeline ...` |

---

## What happens inside – every step, with input and output

| Step | File | Input | What it does | Output |
|---|---|---|---|---|
| 1 Load | `src/io_utils.py` | folder of PNGs / video / .npy | reads images in time order, 16-bit → 0..1 | `T[frames,H,W]` float32 |
| 2 Pre-process | `src/preprocess.py` | `T` | scales the whole sequence to 0..1 (robust percentiles), optional motion correction (ECC), Gaussian blur, temporal Savitzky–Golay smoothing | cleaned `T`, smoothed `Ts` |
| 3 ROI | `src/roi.py` | `T` | Otsu threshold of the average image → biggest blob → fill holes → erode border | `mask[H,W]` |
| 4 Frame-by-frame change | `src/temporal.py` | `Ts`, `mask` | per-pixel **slope** (units/second) by Savitzky–Golay derivative; noise level from frame differences; threshold = 3 × theoretical slope noise | `slope[frames,H,W]`, threshold |
| 5 Rising/falling regions | `src/temporal.py` | `slope`, threshold | `+1` rising, `-1` falling, `0` stable per pixel and frame; ROI-average curve; splits the recording into **cooling** and **rewarming** at the coldest moment | table (% rising, % falling per frame), phases |
| 6a Direction – METHOD B | `src/flow.py` | `Ts`, phase | per pixel: time at which it reaches 50 % of its own change (“arrival time”). The direction where arrival time grows is the direction the front travels; speed = 1/\|gradient\| | arrival map, direction, speed |
| 6b Direction – METHOD A | `src/flow.py` | `Ts`, `slope` | Lucas–Kanade on the temperature field: dT/dt = −v·∇T, solved in 15×15 windows, averaged over the phase | velocity field, direction |
| 7 Summary + visuals | `src/pipeline.py`, `src/viz.py` | everything | circular statistics: mean angle and **coherence R** (1 = all arrows agree, ≈0 = no direction); figures, video, text | `summary.png`, `rising_falling_over_time.png`, `demo.mp4`, `frame_by_frame.csv`, `summary.json`, `summary.txt` |

### How to read the output
* **Red = getting warmer, blue = getting colder.** White lines in the arrival map connect pixels that reached 50 % at the same time.
* **Coherence R** decides whether a direction is meaningful: R > 0.7 clear, 0.4–0.7 weak, < 0.4 *no real direction* – ignore
  the angle and speed then (e.g. a body that cools uniformly has R ≈ 0.05; its “speed” of 100+ px/s is meaningless).
* **Method B** is the main result; **Method A** is the independent check. Report how far apart they are.
* All values are **normalized intensity (0–1), not °C** because DIRT-SR38 stores relative intensity.
* Speeds are in **pixels/second**; to get mm/s you need the pixel size from the paper.

## Results of testing in this package (be honest in your report about what they mean)
Tested on **synthetic** sequences only (the real dataset could not be downloaded in the build environment):

| Test | Method B (arrival time) | Method A (gradient flow) |
|---|---|---|
| 8 directions × 3 noise levels, mean angular error | 0.15° – 0.24° | ≈ 4° (up to ~10° for some angles) |
| 0.7 px body jitter | 0.16° | ≈ 6° |

Caveat: the simulated warming front is clean and planar, so these numbers are the *best case*. On real skin
(curved body, blood-vessel structure, several recovery sites, sweat, breathing) expect lower coherence and larger
disagreement between methods – and report that. Real data has **no ground truth for direction**, so for real
sequences you can only show consistency (A vs B, subject vs subject, stability when parameters change).

## What was NOT tested
* Anything on the real DIRT-SR38 files. The folder layout was adapted from the path you gave; fps, sequence length and whether `static` is useful are unknown to the author of this code. A fake copy of your exact folder structure (P11/P12, HR/LR, dynamic/static, 8-bit PNG) was used to test the loader, pairing, preview and analysis scripts.
* `scripts/train_sr.py` and `scripts/sr_downstream_test.py` (need PyTorch; syntax-checked only).

## Tuning knobs (`Config` in `src/pipeline.py`)
`smooth_win_s` (2 s) longer = smoother but blurrier in time · `k_sigma` (3) higher = fewer pixels flagged ·
`arrival_frac` (0.5) 50 % recovery point · `arrival_sigma` (2 px) smoothing of the arrival map ·
`lk_win` (15 px) window of Method A · `min_amp_sigmas` (6) ignore pixels that barely change.

## Suggested report outline
1 Introduction & use case · 2 Scientific basis (Pennes bioheat equation; skin temperature = perfusion + conduction +
radiation/convection; why a cold challenge exposes perfusion) · 3 Dataset (DIRT-SR38: protocol, relative units) ·
4 Method (steps above, figures) · 5 Validation on synthetic data (table above) · 6 Results on two real subjects ·
7 Optional: SR model and its effect on the direction estimate · 8 Limitations (surface only, conduction, relative
units, motion, no ground truth, not a diagnosis) · 9 Future work · References (dataset DOI + paper, Pennes 1948,
Ring & Ammer 2012, Lucas–Kanade 1981, Horn–Schunck 1981, Savitzky–Golay 1964, ECC Evangelidis & Psarakis 2008).
