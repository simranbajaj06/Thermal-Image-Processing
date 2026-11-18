# Explanation of the proposed algorithm


## 4. Data processing pipeline (input and output of each step)

| Step | Input | Output / operation |
|---|---|---|
| 1 Load | folder of PNG frames | float array T[frames, H, W], values 0-1 |
| 2 Pre-process | T | robust 0-1 scaling of the whole sequence, optional motion correction (ECC), Gaussian blur (sigma 1 px), temporal Savitzky-Golay smoothing (2 s window) |
| 3 ROI | T | binary mask: whole frame (close-ups) or Otsu body segmentation (body on background); border eroded |
| 4 Frame-by-frame change | smoothed T, mask | per-pixel slope (units/s) from a Savitzky-Golay derivative; noise level from frame differences; slope threshold = 3 x theoretical slope noise |
| 5 Rising / falling regions | slope maps, threshold | state +1 (rising) / -1 (falling) / 0 (stable) per pixel and frame; ROI-mean curve; cooling and rewarming phases split at the coldest moment |
| 6 Direction and propagation | smoothed T, slope, phase | arrival-time map, direction + speed (Method B); Lucas-Kanade gradient flow (Method A); early/late recovery foci; pattern class |
| 7 Summary and visualisation | all of the above | heat maps, arrows, rose plot, video, CSV/JSON, text summary |


## 5. Proposed algorithm


### Frame-by-frame analysis (steps 4-5)

Every pixel's time course is smoothed and differentiated with a Savitzky-Golay filter, giving a slope map (change per second) for every frame. A pixel is called rising if its slope exceeds +k sigma_slope, falling if it is below -k sigma_slope and stable otherwise (k = 3).

sigma_slope is not a free parameter. The pixel noise sigma_T is estimated robustly from differences between consecutive frames (median absolute deviation / sqrt 2), and the theoretical noise of a Savitzky-Golay slope for window length w and frame rate fps is sigma_slope = sigma_T * sqrt(12 / (w (w^2 - 1))) * fps.

The ROI-average curve is split at its coldest moment: before it is the COOLING phase, after it the REWARMING phase. Some recordings start after cooling has ended; then only the rewarming phase exists and is analysed.


### Method B - arrival-time map (main method)

For every pixel and a chosen phase, v0 is the value at the start and v1 the value at the end. The pixel's arrival time is the last moment it was still short of v0 + 0.5 (v1 - v0), linearly interpolated between two frames. Using the LAST moment makes the estimate immune to an early noisy crossing. Pixels that change less than 6 noise sigmas are discarded.

The arrival-time map is smoothed (masked Gaussian, 2 px). The warming pattern travels in the direction in which arrival time increases: direction = grad(t) / |grad(t)| and speed = 1 / |grad(t)| (pixels per second). Example: arrival times of 12, 14 and 16 s at x = 10, 20, 30 px give a front moving right at 1 / 0.2 s per px = 5 px/s.

Meaning for real skin: for an approximately exponential recovery, the 50 % time is proportional to the local recovery time constant. The map therefore orders regions from fast to slow rewarming, which is the perfusion-related information; it shows a travelling front only if the order is spatially smooth.


### Method A - gradient flow (independent check)

If a temperature pattern moves with velocity v then dT/dt = -(vx dT/dx + vy dT/dy). The slope map supplies dT/dt and Sobel filters supply the spatial gradients. v is found by least squares in a 15 x 15 window (Lucas-Kanade) with a small regulariser, so where only one gradient direction exists the result reduces to the observable normal component. Vectors of all frames of the phase are averaged, weighted by gradient energy; the same per-frame field draws the arrows in the video.


### Summary of the pattern (circular statistics and recovery foci)

All pixel directions are combined into a mean angle (0 deg = right, 90 deg = up) and a coherence R between 0 and 1 (R = length of the mean unit vector). R >= 0.6 means one dominant direction ('uniform').

A single angle cannot describe radial patterns, where warming spreads away from a hot spot or vessel. The algorithm therefore also finds early-recovery foci (local minima of the arrival-time map = sources) and late-recovery foci (local maxima = sinks). The outward alignment is the mean cosine between each arrow and the vector from the nearest source to the pixel (+1 = all arrows point away); the convergence alignment is the same toward the nearest sink.

Pattern class: 'uniform' (R >= 0.6); 'focal' (R < 0.6 and an alignment >= 0.5); otherwise 'mixed' = no coherent propagation could be resolved.
