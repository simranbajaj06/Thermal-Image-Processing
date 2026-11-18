"""
DIRT-SR38 LAYOUT HELPERS
------------------------
Layout reported by the user:
    <root>\\dataset_export\\8bit\\<SUBJECT, e.g. P11>\\HR\\dynamic   (+ \\static)
    <root>\\dataset_export\\8bit\\<SUBJECT>\\LR\\dynamic             (+ \\static)
'root' below = the folder that CONTAINS the subject folders, e.g. D:\\Thermal\\DIRT-SR38\\dataset_export\\8bit
"""
import os
from .io_utils import IMG_EXT, _natural_key


def list_subjects(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))], key=_natural_key)


def seq_dir(root, subject, res="HR", kind="dynamic"):
    """Folder of one sequence, e.g. seq_dir(root, 'P11', 'HR', 'dynamic')."""
    p = os.path.join(root, subject, res, kind)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Folder not found: {p}")
    return p


def _images(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)], key=_natural_key)


def find_pairs(root, lr_name="LR", hr_name="HR", kinds=None, subjects=None, verbose=True):
    """
    Pair every LR image with its HR image.
    For each folder  <root>/<subject>/LR/<kind>  the partner is  <root>/<subject>/HR/<kind>.
    Pairing is by identical file name; if names differ but both folders hold the same number of images,
    by time order (a warning is printed). kinds=['dynamic'] restricts to the dynamic folders.
    """
    pairs, warned = [], False
    for dirpath, _, files in os.walk(root):
        rel = os.path.relpath(dirpath, root).split(os.sep)
        if lr_name not in rel:
            continue
        if subjects and rel[0] not in subjects:
            continue
        if kinds and not (set(rel) & set(kinds)):
            continue
        i = rel.index(lr_name)
        hr_dir = os.path.join(root, *rel[:i], hr_name, *rel[i + 1:])
        if not os.path.isdir(hr_dir):
            continue
        lr_files, hr_files = _images(dirpath), _images(hr_dir)
        if not lr_files:
            continue
        if set(lr_files) <= set(hr_files):
            pairs += [(os.path.join(dirpath, f), os.path.join(hr_dir, f)) for f in lr_files]
        elif len(lr_files) == len(hr_files):
            if verbose and not warned:
                print("WARNING: LR/HR file names differ -> pairing by sorted order (check this is correct!)")
                warned = True
            pairs += [(os.path.join(dirpath, a), os.path.join(hr_dir, b)) for a, b in zip(lr_files, hr_files)]
        elif verbose:
            print(f"WARNING: skipped {dirpath}: {len(lr_files)} LR vs {len(hr_files)} HR images")
    pairs.sort()
    return pairs
