#!/usr/bin/env python3
"""
Data for the locked Split CIFAR-100 protocol. Replaces the data half of the
missing `exp17_resnet_circuits.py` (exp18 imported `class_split`, `augment`
and `prepare_data` from it; that file was never checked in).

Protocol, fixed here and nowhere else:
    CIFAR-100, cold start, n_tasks equal tasks of cpt classes.
    Class order: torch.randperm(100) under `split_seed` (default 1234).
    Validation: `val_per_task` images per task (default 500 = 50 per class)
    taken from the TRAINING set, never from test. Chosen with the same seed.
    Normalisation: CIFAR-100 channel mean/std.
    Augmentation: random crop with 4 pixel zero padding + horizontal flip,
    applied on device from a CPU torch.Generator so runs are reproducible.
Every method in cil_harness.py reads data only through prepare_data().
"""
import pickle
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def class_split(n_tasks: int, cpt: int, base_classes: int = 0) -> List[Tuple[int, int]]:
    """[(lo, hi), ...] in permuted-label space."""
    out, lo = [], 0
    if base_classes:
        out.append((0, base_classes))
        lo = base_classes
    for _ in range(n_tasks - (1 if base_classes else 0)):
        out.append((lo, lo + cpt))
        lo += cpt
    return out


def _find_cifar(data_dir: Path) -> Path:
    cands = [data_dir / "cifar-100-python",
             data_dir / "cifar100" / "cifar-100-python"]
    cands += list(data_dir.glob("**/cifar-100-python"))[:5]
    for c in cands:
        if (c / "train").exists() and (c / "test").exists():
            return c
    # kaggle datasets often ship the raw pickles at the top level
    if (data_dir / "train").exists() and (data_dir / "test").exists():
        return data_dir
    return None


def load_cifar100(data_dir: Path, download: bool = True):
    data_dir = Path(data_dir)
    root = _find_cifar(data_dir)
    if root is None:
        if not download:
            raise FileNotFoundError(f"cifar-100-python not under {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
        tgz = data_dir / "cifar-100-python.tar.gz"
        if not tgz.exists():
            print(f"downloading CIFAR-100 to {tgz}", flush=True)
            urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(data_dir)
        root = _find_cifar(data_dir)
    out = {}
    for split in ("train", "test"):
        with open(root / split, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        x = d["data"].reshape(-1, 3, 32, 32).astype(np.uint8)
        y = np.asarray(d["fine_labels"], dtype=np.int64)
        out[split] = (x, y)
    return out


def _to_tensor(x_u8: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(x_u8).float().div_(255.0)
    m = torch.tensor(MEAN).view(1, 3, 1, 1)
    s = torch.tensor(STD).view(1, 3, 1, 1)
    return ((x - m) / s).to(device)


def prepare_data(data_dir, n_tasks, cpt, split_seed, val_per_task, device,
                 base_classes=0, download=True):
    """Returns (tasks, perm). tasks[t] = {"train": (x, y), "val": (x, y),
    "test": (x, y), "classes": (lo, hi)} with y already in permuted label
    space and tensors on `device`. perm[k] is the original CIFAR label that
    became class k."""
    raw = load_cifar100(data_dir, download)
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(100, generator=g).tolist()
    inv = np.zeros(100, dtype=np.int64)
    for new, old in enumerate(perm):
        inv[old] = new
    xtr, ytr = raw["train"]
    xte, yte = raw["test"]
    ytr, yte = inv[ytr], inv[yte]
    spans = class_split(n_tasks, cpt, base_classes)
    tasks = []
    for lo, hi in spans:
        ncls = hi - lo
        per_cls = val_per_task // ncls
        tr_idx, va_idx = [], []
        for c in range(lo, hi):
            idx = np.nonzero(ytr == c)[0]
            idx = idx[torch.randperm(len(idx), generator=g).numpy()]
            va_idx.append(idx[:per_cls])
            tr_idx.append(idx[per_cls:])
        tr_idx = np.concatenate(tr_idx)
        va_idx = np.concatenate(va_idx)
        te_idx = np.nonzero((yte >= lo) & (yte < hi))[0]
        tasks.append({
            "train": (_to_tensor(xtr[tr_idx], device),
                      torch.from_numpy(ytr[tr_idx]).to(device)),
            "val": (_to_tensor(xtr[va_idx], device),
                    torch.from_numpy(ytr[va_idx]).to(device)),
            "test": (_to_tensor(xte[te_idx], device),
                     torch.from_numpy(yte[te_idx]).to(device)),
            "classes": (lo, hi)})
    return tasks, perm


def augment(x: torch.Tensor, g: torch.Generator, pad: int = 4) -> torch.Tensor:
    """Random crop (zero pad) + horizontal flip, on device, seeded by the CPU
    generator `g`. No per-step host sync beyond the two small randint calls."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (B,), generator=g)
    j = torch.randint(0, 2 * pad + 1, (B,), generator=g)
    flip = torch.rand(B, generator=g) < 0.5
    ar = torch.arange(H)
    rows = (i[:, None] + ar[None, :]).to(x.device)            # B,H
    cols = (j[:, None] + ar[None, :]).to(x.device)            # B,W
    bidx = torch.arange(B, device=x.device)[:, None, None]
    out = xp[bidx, :, rows[:, :, None], cols[:, None, :]]      # B,H,W,C
    out = out.permute(0, 3, 1, 2)
    flip = flip.to(x.device)
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out.contiguous()


def subsample_train(tasks, n_per_task: int):
    """Toy runs only: keep n_per_task training images per task, spread
    evenly over the task's classes (the train arrays are class ordered, so a
    plain prefix would keep one class)."""
    for tk in tasks:
        x, y = tk["train"]
        lo, hi = tk["classes"]
        per = max(1, n_per_task // (hi - lo))
        keep = torch.cat([(y == c).nonzero(as_tuple=True)[0][:per] for c in range(lo, hi)])
        tk["train"] = (x[keep], y[keep])
    return tasks
