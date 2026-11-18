"""
STEP 3 - REGION OF INTEREST (ROI)
---------------------------------
Input : T [frames,H,W]
Output: mask [H,W] bool, True = skin/body pixels that will be analysed
Classic method (no training): Otsu threshold on the time-averaged image -> biggest blob -> clean up.
You can also pass your own mask (e.g. produced by a trained U-Net) with load_mask().
"""
import numpy as np
import cv2


def auto_roi(T, body_is_brighter=True, erode_px=3, min_area_frac=0.02):
    mean_img = T.mean(axis=0)
    img8 = cv2.normalize(cv2.GaussianBlur(mean_img, (7, 7), 2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not body_is_brighter:
        m = 255 - m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.ones(T.shape[1:], bool)
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])        # keep only the largest blob
    mask = (lab == big).astype(np.uint8)
    # fill holes inside the blob
    n2, lab2, _, _ = cv2.connectedComponentsWithStats(1 - mask)
    border = set(lab2[0, :]) | set(lab2[-1, :]) | set(lab2[:, 0]) | set(lab2[:, -1])
    for i in range(1, n2):
        if i not in border:
            mask[lab2 == i] = 1
    if erode_px > 0:   # stay away from the body edge: gradients there are not skin physiology
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2))
    mask = mask.astype(bool)
    if mask.mean() < min_area_frac:
        return np.ones(T.shape[1:], bool)
    return mask


def load_mask(path, shape):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127
