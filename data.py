"""Data loading and augmentation.

Only train_images/ + train_labels.csv are used for training/validation.
test_images/ are loaded solely for producing predictions.
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT, "cache")


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_images(img_dir, filenames, cache_name):
    """Load PNGs into a uint8 array (N, 32, 32, 3), cached as .npy for speed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, cache_name)
    if os.path.exists(cache):
        arr = np.load(cache)
        if len(arr) == len(filenames):
            return arr
    arr = np.stack([np.array(Image.open(os.path.join(img_dir, fn)).convert("RGB")) for fn in filenames])
    np.save(cache, arr)
    return arr


def load_train():
    rows = _read_csv(os.path.join(ROOT, "train_labels.csv"))
    x = _load_images(os.path.join(ROOT, "train_images"), [r["filename"] for r in rows], "train_x.npy")
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    return x, y


def load_test():
    """Returns test images ordered to match sample_submission.csv, plus their ids."""
    fn_by_id = {r["id"]: r["filename"] for r in _read_csv(os.path.join(ROOT, "test.csv"))}
    ids = [r["id"] for r in _read_csv(os.path.join(ROOT, "sample_submission.csv"))]
    x = _load_images(os.path.join(ROOT, "test_images"), [fn_by_id[i] for i in ids], "test_x.npy")
    return x, ids


def split_train_val(x, y, val_frac, seed):
    """Stratified train/validation split of the training data."""
    if val_frac <= 0:
        return (x, y), None
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        val_idx.append(idx[: int(len(idx) * val_frac)])
    val_idx = np.concatenate(val_idx)
    mask = np.ones(len(y), dtype=bool)
    mask[val_idx] = False
    return (x[mask], y[mask]), (x[~mask], y[~mask])


def to_tensor(x, mean, std, device):
    """uint8 NHWC numpy -> normalized float NCHW tensor on device."""
    t = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float().div_(255)
    return (t - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def augment(x, pad=4, cutout=8):
    """Batched GPU augmentation: random crop (reflect pad), horizontal flip, cutout."""
    n, _, h, w = x.shape
    dev = x.device
    # random horizontal flip
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)
    # random crop with padding
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    dx = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    dy = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    rows = (dy.view(n, 1) + torch.arange(h, device=dev).view(1, h))  # n,h
    cols = (dx.view(n, 1) + torch.arange(w, device=dev).view(1, w))  # n,w
    idx_n = torch.arange(n, device=dev).view(n, 1, 1)
    x = xp.permute(0, 2, 3, 1)[idx_n, rows.view(n, h, 1), cols.view(n, 1, w)].permute(0, 3, 1, 2)
    # cutout (DeVries & Taylor, 2017)
    if cutout > 0:
        cy = torch.randint(0, h, (n, 1, 1), device=dev)
        cx = torch.randint(0, w, (n, 1, 1), device=dev)
        yy = torch.arange(h, device=dev).view(1, h, 1)
        xx = torch.arange(w, device=dev).view(1, 1, w)
        hole = ((yy - cy).abs() <= cutout // 2) & ((xx - cx).abs() <= cutout // 2)
        x = x * (~hole).unsqueeze(1)
    return x.contiguous()
