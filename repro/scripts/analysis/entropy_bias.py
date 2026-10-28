"""Why entropy-based TTA barely helps on AID: is the training signal (low entropy = probably right) still reliable under
corruption, and do predictions collapse onto a few classes? Source models only (PN heads, no adaptation).

Per (model, corruption) at severity 5 (+ clean), over the 2,000 test images:
  acc        : accuracy of the source model
  auroc      : AUROC of confidence (-entropy) for separating correct from wrong predictions (0.5 = no signal)
  low_ent    : share of images with entropy < 0.4 ln C (the SAR / EATA "reliable" margin)
  low_ent_acc: accuracy among those images = how often the samples entropy-TTA trusts most are actually right
  bias_L1    : L1 distance between the mean softmax and the true label distribution (Panda's prediction-bias measure)
  top1_share : share of predictions taken by the most predicted class (uniform = 1/30 = 3.3%)
  top_classes: the 3 most predicted classes (name, % of predictions, % of true labels)

  ~/REOBench/.venv/bin/python scripts/analysis/entropy_bias.py      # from repro/, ~5 min on one GPU
Output: experiments/2026-10-28_entropy_bias/{results.json, summary.txt}
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import common, paths  # noqa: E402

NORM = {"satmae": ([0.4182007312774658, 0.4214799106121063, 0.3991275727748871],
                   [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]),
        "scalemae": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}           # = cls_tta.py (PN heads)
C, SEV = common.NUM_CLASSES, 5
OUT = paths.EXP / "2026-10-28_entropy_bias"
OUT.mkdir(exist_ok=True)
dev = torch.device("cuda")


class Files(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def auroc(score, pos):
    """Probability that a random positive scores higher than a random negative (ties count 1/2)."""
    s, p = np.asarray(score), np.asarray(pos, bool)
    if p.all() or not p.any():
        return float("nan")
    order = s.argsort()
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                     # average ranks over ties
        m = s == v
        ranks[m] = ranks[m].mean()
    return float((ranks[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum()))


streams = [("clean", datasets.ImageFolder(common.TEST_DIR).samples)]
cor = common.CorruptedAID(None)
for ni, noise in enumerate(common.NOISES):
    streams.append((noise, [(p, c) for p, c, n, sv in cor.items if n == ni and sv == SEV - 1]))

CLS_NAMES = common.class_names()
res, lines = {}, []
for model in ["satmae", "scalemae"]:
    tf = T.Compose([T.ToTensor(), T.Normalize(*NORM[model])])
    bb = common.build_backbone(model, drop_path=0.0).to(dev).eval()
    head = nn.Linear(1024, C)
    head.load_state_dict(torch.load(f"{paths.CLS_ABLATION}/outputs/{model}/PN_s0_head.pth", map_location="cpu"))
    head = head.to(dev).eval()
    res[model] = {}
    lines.append(f"\n### {model} PN, source model, severity {SEV}")
    lines.append(f"{'stream':24s} {'acc':>6s} {'auroc':>6s} {'low_ent':>8s} {'low_ent_acc':>12s} {'bias_L1':>8s} {'top1_share':>10s}")
    for name, items in streams:
        probs, ys = [], []
        with torch.no_grad():
            for x, y in DataLoader(Files(items, tf), batch_size=128, num_workers=16):
                probs.append(head(common.features(bb, model, x.to(dev))).softmax(1).cpu()); ys.append(y)
        P = torch.cat(probs).numpy(); Y = torch.cat(ys).numpy()
        pred = P.argmax(1); ok = pred == Y
        ent = -(P * np.log(P + 1e-12)).sum(1)
        low = ent < 0.4 * math.log(C)
        q = np.bincount(Y, minlength=C) / len(Y)
        r = {"acc": 100 * ok.mean(), "auroc": auroc(-ent, ok), "low_ent": 100 * low.mean(),
             "low_ent_acc": 100 * ok[low].mean() if low.any() else float("nan"),
             "bias_L1": float(np.abs(P.mean(0) - q).sum()),
             "top1_share": 100 * np.bincount(pred, minlength=C).max() / len(pred),
             "top_classes": [(CLS_NAMES[k], 100 * np.mean(pred == k), 100 * np.mean(Y == k))
                             for k in np.bincount(pred, minlength=C).argsort()[::-1][:3]]}
        res[model][name] = r
        lines.append(f"{name:24s} {r['acc']:6.1f} {r['auroc']:6.3f} {r['low_ent']:7.1f}% {r['low_ent_acc']:11.1f}% "
                     f"{r['bias_L1']:8.3f} {r['top1_share']:9.1f}%")
        print(lines[-1], flush=True)
json.dump(res, open(OUT / "results.json", "w"), indent=1)
open(OUT / "summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
