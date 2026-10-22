"""Diagnostic for the depth sweep: how a corruption moves the backbone features, block by block (AID, frozen backbones).

For 400 clean AID test images and the same 400 images under each corruption (severity 5; files are paired by name):
  d_mu   : mean over channels of |mu_corr - mu_clean| / sd_clean      (shift of the per-channel mean)
  d_sd   : mean over channels of |log(sd_corr / sd_clean)|            (shift of the per-channel scale)
  cos    : mean cosine similarity between the clean and corrupted token of the same image and position
  cos_al : the same after aligning the corrupted features to the clean statistics (per channel, population stats)
           -> cos_al - cos = how much of the damage at this depth is a per-channel affine shift that alignment undoes
  outlier: on clean features, max / median of the per-channel RMS, and the share of the total energy held by the
           top 1% channels (a few huge channels make per-channel standardization fragile)
Positions: input of block 0 (patch embedding + pos), outputs of blocks 0..23, final norm. Tokens: all (CLS + patches).

  ~/REOBench/.venv/bin/python scripts/analysis/feature_shift.py      # from repro/, ~10 min on one GPU
Output: experiments/2026-10-22_feature_shift/{<model>.json, summary.txt}
"""
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import common, paths  # noqa: E402

NORM = {"satmae": ([0.4182007312774658, 0.4214799106121063, 0.3991275727748871],
                   [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]),
        "scalemae": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}           # = cls_tta.py (PN heads)
GROUPS = {"occlusion": ["clouds", "data_gaps"], "radiometric": ["brightness_contrast", "haze_update_new"],
          "degradation": ["gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
                          "compression_artifacts"]}          # geometry left out: tokens are not spatially paired
CORRS = [c for v in GROUPS.values() for c in v]
N_IMG, BS, SEV = 400, 50, 5
OUT = paths.EXP / "2026-10-22_feature_shift"
OUT.mkdir(exist_ok=True)
dev = torch.device("cuda")

items = datasets.ImageFolder(common.TEST_DIR).samples
idx = torch.randperm(len(items), generator=torch.Generator().manual_seed(0))[:N_IMG].tolist()
cls = common.class_names()
clean_paths = [items[i][0] for i in idx]


def corr_path(p, c):
    cname = os.path.basename(os.path.dirname(p))
    return f"{common.CORRUPT_DIR}/{cname}/{c}/severity_{SEV}/{cls.index(cname)}/{os.path.basename(p)}"


class Taps:
    """Collect the token features of every position for one forward pass."""
    def __init__(self, bb):
        self.out = {}
        self.names = ["embed"] + [f"block{i}" for i in range(len(bb.blocks))] + ["norm"]
        bb.blocks[0].register_forward_pre_hook(lambda m, i: self.out.__setitem__("embed", i[0].float()))
        for i, b in enumerate(bb.blocks):
            b.register_forward_hook(lambda m, i_, o, i=i: self.out.__setitem__(f"block{i}", o.float()))
        bb.norm.register_forward_hook(lambda m, i, o: self.out.__setitem__("norm", o.float()))


def batches(paths_, tf):
    for i in range(0, len(paths_), BS):
        yield torch.stack([tf(Image.open(p).convert("RGB")) for p in paths_[i:i + BS]]).to(dev)


@torch.no_grad()
def run(model):
    tf = T.Compose([T.ToTensor(), T.Normalize(*NORM[model])])
    bb = common.build_backbone(model, drop_path=0.0).to(dev).eval()
    taps = Taps(bb)
    fwd = (lambda x: bb.forward_features(x)) if model == "satmae" else \
        (lambda x: bb.forward_features(x, input_res=torch.ones(len(x), device=dev)))

    def stats(paths_):  # population per-channel mean / std over all tokens of the 400 images
        s1, s2, n = {}, {}, 0
        for x in batches(paths_, tf):
            fwd(x)
            for k, h in taps.out.items():
                f = h.reshape(-1, h.shape[-1]).double()
                s1[k] = s1.get(k, 0) + f.sum(0); s2[k] = s2.get(k, 0) + (f ** 2).sum(0)
            n += h.shape[0] * h.shape[1]
        return {k: (s1[k] / n, (s2[k] / n - (s1[k] / n) ** 2).clamp_min(1e-12).sqrt()) for k in s1}

    st0 = stats(clean_paths)
    res = {"outlier": {}, "shift": {}}
    for k, (mu, sd) in st0.items():
        rms = (mu ** 2 + sd ** 2).sqrt()
        e = (rms ** 2).sort(descending=True)[0]
        res["outlier"][k] = {"max_over_median": (rms.max() / rms.median()).item(),
                             "top1pct_energy": (e[:max(1, len(e) // 100)].sum() / e.sum()).item()}
    for c in CORRS:
        cp = [corr_path(p, c) for p in clean_paths]
        stc = stats(cp)
        cos, cos_al = {k: 0.0 for k in st0}, {k: 0.0 for k in st0}
        for x0, x1 in zip(batches(clean_paths, tf), batches(cp, tf)):
            fwd(x0); h0 = dict(taps.out)
            fwd(x1); h1 = taps.out
            for k in st0:
                (m0, s0), (m1, s1_) = st0[k], stc[k]
                a = ((h1[k] - m1.float()) / s1_.float() * s0.float() + m0.float())
                cos[k] += torch.nn.functional.cosine_similarity(h0[k], h1[k], dim=-1).mean().item() * len(x0)
                cos_al[k] += torch.nn.functional.cosine_similarity(h0[k], a, dim=-1).mean().item() * len(x0)
        res["shift"][c] = {k: {"d_mu": ((stc[k][0] - st0[k][0]).abs() / st0[k][1]).mean().item(),
                               "d_sd": (stc[k][1] / st0[k][1]).log().abs().mean().item(),
                               "cos": cos[k] / N_IMG, "cos_al": cos_al[k] / N_IMG} for k in st0}
        print(f"[{model}] {c} done", flush=True)
    json.dump(res, open(OUT / f"{model}.json", "w"), indent=1)
    return taps.names, res


lines = []
for model in ["satmae", "scalemae"]:
    names, res = run(model)
    lines.append(f"\n### {model} (AID test, {N_IMG} images, severity {SEV}; group means)")
    lines.append(f"{'position':9s} {'outlier':>15s} | " + " | ".join(f"{g:^27s}" for g in GROUPS))
    lines.append(f"{'':9s} {'max/med top1%':>15s} | " + " | ".join(f"{'d_mu':>6s}{'d_sd':>6s}{'cos':>7s}{'+al':>7s}" for _ in GROUPS))
    for k in names:
        o = res["outlier"][k]
        cells = []
        for g, cs in GROUPS.items():
            v = {m: np.mean([res["shift"][c][k][m] for c in cs]) for m in ("d_mu", "d_sd", "cos", "cos_al")}
            cells.append(f"{v['d_mu']:6.2f}{v['d_sd']:6.2f}{v['cos']:7.3f}{v['cos_al'] - v['cos']:+7.3f}")
        lines.append(f"{k:9s} {o['max_over_median']:7.1f}{o['top1pct_energy']:8.2f} | " + " | ".join(cells))
txt = "\n".join(lines)
print(txt)
open(OUT / "summary.txt", "w").write(txt + "\n")
