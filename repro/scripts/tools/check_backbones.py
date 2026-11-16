"""Dry run of src/reobench_tta/backbones.py: do the downloaded checkpoints load and give meaningful features?

Per backbone (224 px, own normalization):
  load    : key check inside backbones.build (fails loudly on unexpected gaps)
  kNN AID : 20-NN cosine accuracy, 20 train / 10 test images per class from the clean AID splits (600 / 300 images),
            pretrained vs the same architecture with random weights (random must be far lower)
  IN top-1: supervised anchors only, their own ImageNet head on the first N val images (published V1 top-1 in brackets)

  ~/REOBench/.venv/bin/python scripts/tools/check_backbones.py [--n_in 1000]     # from repro/, one GPU, a few minutes
Output: experiments/2026-11-16_backbone_check/summary.txt
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import backbones, common, paths  # noqa: E402

PUBLISHED = {"sup_in_r50": 76.13, "sup_in_vitb16": 81.07, "sup_in_vitl16": 79.66}
IN_VAL = "/data/tta/ImageNet_Family/ImageNet_v1"

ap = argparse.ArgumentParser()
ap.add_argument("--n_in", type=int, default=1000, help="ImageNet val images for the supervised top-1 check")
ap.add_argument("--models", nargs="*", default=list(backbones.SPECS))
args = ap.parse_args()
OUT = paths.EXP / "2026-11-16_backbone_check"
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


def per_class(items, k, skip=0):
    out, seen = [], {}
    for p, y in items:
        seen[y] = seen.get(y, 0) + 1
        if skip < seen[y] <= skip + k:
            out.append((p, y))
    return out


tr_items = per_class(datasets.ImageFolder(common.TRAIN_DIR).samples, 20)
te_items = per_class(datasets.ImageFolder(common.TEST_DIR).samples, 10)
in_items = datasets.ImageFolder(IN_VAL).samples
in_items = [in_items[i] for i in torch.randperm(len(in_items), generator=torch.Generator().manual_seed(0))[:args.n_in]]


@torch.no_grad()
def embed(model, items, norm, resize):
    tf = T.Compose([T.Resize(resize), T.CenterCrop(224), T.ToTensor(), T.Normalize(*norm)])
    fs, ys = [], []
    for x, y in DataLoader(Files(items, tf), batch_size=64, num_workers=8):
        fs.append(model(x.to(dev)).float().cpu()); ys.append(y)
    return torch.cat(fs), torch.cat(ys)


def knn(model, norm, k=20):
    a, ya = embed(model, tr_items, norm, 224)  # AID 256 px -> 224 (whole image, no crop loss)
    b, yb = embed(model, te_items, norm, 224)
    sim = F.normalize(b, dim=1) @ F.normalize(a, dim=1).T
    idx = sim.topk(k, dim=1).indices
    votes = torch.zeros(len(b), 30).scatter_add_(1, ya[idx], torch.ones_like(idx, dtype=torch.float))
    return 100 * (votes.argmax(1) == yb).float().mean().item(), a.std().item()


lines = [f"{'model':14s} {'data':12s} {'method':32s} {'dim':>5s} {'kNN AID':>8s} {'random':>7s} {'feat std':>9s} {'IN top-1':>16s}"]
print(lines[0], flush=True)
for name in args.models:
    fam, data, method, _, _ = backbones.SPECS[name]
    m, norm, dim = backbones.build(name)
    m = m.to(dev)
    acc, std = knn(m, norm)
    top1 = ""
    if hasattr(m, "classify"):
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(*norm)])
        ok = n = 0
        with torch.no_grad():
            for x, y in DataLoader(Files(in_items, tf), batch_size=64, num_workers=8):
                ok += (m.classify(x.to(dev)).argmax(1).cpu() == y).sum().item(); n += len(y)
        top1 = f"{100 * ok / n:.1f} [{PUBLISHED[name]}]"
    del m
    r, _, _ = backbones.build(name, pretrained=False)
    racc, _ = knn(r.to(dev), norm)
    del r
    torch.cuda.empty_cache()
    lines.append(f"{name:14s} {data:12s} {method:32s} {dim:5d} {acc:8.1f} {racc:7.1f} {std:9.3f} {top1:>16s}")
    print(lines[-1], flush=True)
open(OUT / "summary.txt", "w").write("\n".join(lines) + "\n")
