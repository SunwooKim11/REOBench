"""Why ISA fixes radiometric but not degradation corruptions: pixel statistics and affine invertibility (AID, severity 5).

For 200 clean AID test images and the same images under each corruption (paired by file name):
  mean / std ratio : per-channel pixel mean and std, corrupted / clean (what ISA can see and change)
  HF ratio         : energy of neighbour-pixel differences (high frequency), corrupted / clean
  affine R2        : per image and channel, the best map clean ~ a * corrupted + b (least squares) and the share of clean
                     pixel variance it explains = upper bound of what any per-channel affine fix (such as ISA) can recover

  ~/REOBench/.venv/bin/python scripts/analysis/pixel_affine.py      # from repro/, CPU, ~2 min
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import common  # noqa: E402

CORRS = ["brightness_contrast", "haze_update_new", "gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur",
         "motion_blur", "compression_artifacts"]
items = datasets.ImageFolder(common.TEST_DIR).samples
idx = torch.randperm(len(items), generator=torch.Generator().manual_seed(0))[:200].tolist()
cls = common.class_names()


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255


def corr_path(p, c):
    cn = os.path.basename(os.path.dirname(p))
    return f"{common.CORRUPT_DIR}/{cn}/{c}/severity_5/{cls.index(cn)}/{os.path.basename(p)}"


def hf(x):
    g = x.mean(2)
    return (np.diff(g, axis=0) ** 2).mean() + (np.diff(g, axis=1) ** 2).mean()


X0 = [load(items[i][0]) for i in idx]
m0 = np.mean([x.mean((0, 1)) for x in X0], 0)
s0 = np.sqrt(np.mean([x.var((0, 1)) for x in X0], 0))
h0 = np.mean([hf(x) for x in X0])
print(f"{'corruption':22s} {'mean ratio':>11s} {'std ratio':>10s} {'HF ratio':>9s} {'affine R2':>10s}")
for c in CORRS:
    X1 = [load(corr_path(items[i][0], c)) for i in idx]
    m1 = np.mean([x.mean((0, 1)) for x in X1], 0)
    s1 = np.sqrt(np.mean([x.var((0, 1)) for x in X1], 0))
    r2 = []
    for a, b in zip(X0, X1):
        for k in range(3):
            A = np.stack([b[..., k].ravel(), np.ones(b[..., k].size)], 1)
            y = a[..., k].ravel()
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            r2.append(1 - ((A @ w - y) ** 2).mean() / y.var())
    print(f"{c:22s} {np.mean(m1 / m0):11.2f} {np.mean(s1 / s0):10.2f} {np.mean([hf(x) for x in X1]) / h0:9.2f} "
          f"{np.mean(r2):10.3f}")
