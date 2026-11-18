"""
STEP 1 - LOAD THE DATA
----------------------
Input : a folder of image files (PNG/TIFF...), a video file, or a .npy array
Output: T  -> numpy array of shape [frames, height, width], float32, values 0..1
        (0..1 because DIRT-SR38 stores *relative* intensity, not degrees Celsius)
"""
import os
import re
import numpy as np
import cv2

IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


def _natural_key(s):
    """Sort 'img2.png' before 'img10.png' (plain sorting would get this wrong)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_image_gray(path):
    """Read ONE image as float32 in 0..1 (8-bit -> /255, 16-bit -> /65535)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read {path}")
    if img.ndim == 3:  # colour (false-colour) image: convert to gray, values are only relative
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def list_sequences(root, min_frames=20, include=None, exclude=None):
    """
    Walk a dataset folder and report every sub-folder that contains >= min_frames images.
    Each such folder is treated as one sequence. Use include/exclude (substrings, case-insensitive)
    to pick e.g. only the high-resolution ('HR') folders.
    Returns a dict {folder_path: number_of_images}.
    """
    found = {}
    for dirpath, _, files in os.walk(root):
        n = sum(f.lower().endswith(IMG_EXT) for f in files)
        if n >= min_frames:
            p = dirpath.lower()
            if include and not all(s.lower() in p for s in include):
                continue
            if exclude and any(s.lower() in p for s in exclude):
                continue
            found[dirpath] = n
    return dict(sorted(found.items(), key=lambda kv: _natural_key(kv[0])))


def load_image_folder(folder, stride=1, max_frames=None, scale=1.0):
    """Load all images of a folder in time order -> T[frames,H,W]."""
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)]
    files.sort(key=_natural_key)
    files = files[::stride]
    if max_frames:
        files = files[:max_frames]
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    frames = []
    for f in files:
        im = read_image_gray(os.path.join(folder, f))
        if scale != 1.0:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(im)
    return np.stack(frames).astype(np.float32), files


def load_video(path, stride=1, max_frames=None, scale=1.0):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            if scale != 1.0:
                g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(g)
        i += 1
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.stack(frames).astype(np.float32), []


def load_sequence(path, stride=1, max_frames=None, scale=1.0):
    """One entry point: folder / video / .npy  ->  (T, file_names)."""
    if os.path.isdir(path):
        return load_image_folder(path, stride, max_frames, scale)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        T = np.load(path).astype(np.float32)[::stride]
        return (T[:max_frames] if max_frames else T), []
    if ext in (".mp4", ".avi", ".mov", ".mkv"):
        return load_video(path, stride, max_frames, scale)
    raise ValueError(f"Unsupported input: {path}")
