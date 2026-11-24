"""Label-free signals for Potsdam segmentation: does the pixel flip rate between the source prediction and an aligned
prediction follow the accuracy, as the flip does for classification (scripts/analysis/labelfree_signals.py)?

No learning. Same streams as seg_tta.py (one per corruption x severity + clean, the 2,016 test patches, seed-0 shuffle,
batch 8). Every batch is passed through the source model in four ways:
  none   : as is
  bn     : decode-head BN layers on the statistics of the test batch (= seg_tta.py bnadapt)
  isa    : input statistics aligned to the train pixel statistics (= seg_tta.py isa)
  isa_bn : both
Per stream and variant: confusion matrix over the valid pixels (-> mIoU, pixel accuracy), predicted-class pixel counts.
Per stream, from the none forward: mean max-softmax confidence, share of pixels with entropy < 0.4 ln C and their accuracy.
Per stream and aligned variant a: flip = share of valid pixels whose prediction differs between none and a,
  accuracy of the none prediction on the flipped / the not-flipped pixels.
All shares are over valid pixels (label not clutter / 255); the flip over all pixels is stored as flip_all.

  TIMM_FUSED_ATTN=0 /opt/conda/envs/segtta/bin/python scripts/seg/seg_signals.py --model scalemae   # from repro/
Output: experiments/2026-11-23_seg_signals/signals/<model>_<set>.json
"""
import argparse
import json
import math
import os
import random
import sys
import time

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402
os.chdir(paths.REOBENCH_SEG)
sys.path.insert(0, str(paths.REOBENCH_SEG))
from mmseg.models import build_segmentor  # noqa: E402

T = "/data/rs/reobench/Potsdam_extracted/test"
TRAIN = "/data/rs/reobench/Potsdam_extracted/images/train"
CORR = ["brightness_contrast", "clouds", "compression_artifacts", "gaussian_blur", "gaussian_noise", "haze",
        "motion_blur", "salt_and_pepper_noise"]
C, IGNORE = 5, 5
VARS = ["none", "bn", "isa", "isa_bn"]

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=["scalemae", "satmae"])
p.add_argument("--set", default="B")
p.add_argument("--bs", type=int, default=8)
p.add_argument("--corruptions", nargs="*", default=CORR)
p.add_argument("--sevs", nargs="*", type=int, default=[1, 2, 3, 4, 5])
p.add_argument("--max_images", type=int, default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--workers", type=int, default=5)
p.add_argument("--tag", default="")
args = p.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
dev = torch.device("cuda")
cfg = mmcv.Config.fromfile(f"configs/repro/{args.model}_{args.set}_12ep.py")
MEAN = torch.tensor(cfg.img_norm_cfg["mean"]).view(1, 3, 1, 1).to(dev)
STD = torch.tensor(cfg.img_norm_cfg["std"]).view(1, 3, 1, 1).to(dev)
OUT = paths.EXP / "2026-11-23_seg_signals" / "signals"
os.makedirs(OUT, exist_ok=True)
FILE = OUT / f"{args.model}_{args.set}{args.tag}.json"


class Files(Dataset):
    def __init__(self, img_dir, names):
        self.img_dir, self.names = img_dir, names
        # geometric corruptions have their own labels (<corruption>_ann/severity_<s>); everything else uses the clean ones
        ann = f"{T}/{os.path.basename(os.path.dirname(img_dir))}_ann/{os.path.basename(img_dir)}"
        self.ann_dir = ann if os.path.isdir(ann) else f"{T}/clean_ann"

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = np.asarray(Image.open(f"{self.img_dir}/{n}").convert("RGB"), dtype=np.float32)
        lab = np.asarray(Image.open(f"{self.ann_dir}/{n}").convert("L")).astype(np.int64)
        lab[lab == 255] = IGNORE
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(lab)


NAMES = sorted(os.listdir(f"{T}/clean"))
streams = [("clean", 0, f"{T}/clean")] + [(c, s, f"{T}/{c}/severity_{s}") for c in args.corruptions for s in args.sevs]
model = build_segmentor(cfg.model)
load_checkpoint(model, f"{paths.SEG_BASELINE}/work_dirs/{args.model}_{args.set}_12ep/latest.pth", map_location="cpu", logger="silent")
model = model.to(dev).eval().requires_grad_(False)
META = [dict(img_shape=(512, 512, 3), ori_shape=(512, 512, 3), pad_shape=(512, 512, 3), flip=False)]


def logits_of(x):
    return model.encode_decode((x - MEAN) / STD, META * x.shape[0])


def set_bn(batch):  # momentum 0: batch statistics are used, running statistics stay as they are
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.0
            m.train(batch)


# train pixel statistics for ISA (same 500 patches as seg_tta.py)
sub = random.Random(0).sample(sorted(os.listdir(TRAIN)), 500)
s1 = torch.zeros(3, dtype=torch.float64); s2 = torch.zeros(3, dtype=torch.float64); npx = 0
for n in sub:
    x = torch.from_numpy(np.asarray(Image.open(f"{TRAIN}/{n}").convert("RGB"), dtype=np.float64))
    s1 += x.sum((0, 1)); s2 += (x ** 2).sum((0, 1)); npx += x.shape[0] * x.shape[1]
SRC_MEAN = (s1 / npx).float().view(1, 3, 1, 1).to(dev)
SRC_STD = (s2 / npx - (s1 / npx) ** 2).sqrt().float().view(1, 3, 1, 1).to(dev)
print(f"train pixel mean {SRC_MEAN.flatten().tolist()} std {SRC_STD.flatten().tolist()}", flush=True)


def isa(x):
    mu = x.mean((0, 2, 3), keepdim=True); sd = x.std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd * SRC_STD + SRC_MEAN


res = json.load(open(FILE)) if FILE.exists() else {"config": vars(args), "streams": {}}
E0 = 0.4 * math.log(C)
t0 = time.time()
for si, (cond, sev, d) in enumerate(streams):
    key = "clean" if cond == "clean" else f"{cond}_s{sev}"
    if key in res["streams"]:
        continue
    names = NAMES if args.max_images is None else random.Random(si).sample(NAMES, args.max_images)
    g = torch.Generator().manual_seed(args.seed * 1000 + si)
    dl = DataLoader(Files(d, names), batch_size=args.bs, shuffle=True, generator=g, num_workers=args.workers, pin_memory=True)
    conf = {v: torch.zeros(C * C, dtype=torch.long, device=dev) for v in VARS}
    hist = {v: torch.zeros(C, dtype=torch.long, device=dev) for v in VARS}
    acc = {k: 0.0 for k in ["n", "n_all", "conf_sum", "ent_keep", "ent_keep_ok"] +
           [f"{k}_{v}" for v in VARS[1:] for k in ("flip", "flip_all", "flip_ok", "noflip_ok")]}
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            v = y != IGNORE
            pred = {}
            for name in VARS:
                set_bn("bn" in name)
                lg = logits_of(isa(x) if "isa" in name else x)
                pred[name] = lg.argmax(1)
                if name == "none":
                    pr = lg.softmax(1)
                    ent = -(pr * lg.log_softmax(1)).sum(1)
                    ok = pred["none"] == y
                    acc["conf_sum"] += pr.max(1)[0][v].sum().item()
                    k = (ent < E0) & v
                    acc["ent_keep"] += k.sum().item(); acc["ent_keep_ok"] += (k & ok).sum().item()
                conf[name] += torch.bincount(C * y[v] + pred[name][v], minlength=C * C)
                hist[name] += torch.bincount(pred[name][v], minlength=C)
            set_bn(False)
            acc["n"] += v.sum().item(); acc["n_all"] += v.numel()
            for name in VARS[1:]:
                fl = pred[name] != pred["none"]
                acc[f"flip_{name}"] += (fl & v).sum().item(); acc[f"flip_all_{name}"] += fl.sum().item()
                acc[f"flip_ok_{name}"] += (fl & v & ok).sum().item(); acc[f"noflip_ok_{name}"] += (~fl & v & ok).sum().item()
    r = {"n": acc["n"], "n_all": acc["n_all"], "mean_conf": acc["conf_sum"] / acc["n"],
         "ent_keep": acc["ent_keep"] / acc["n"], "ent_keep_acc": acc["ent_keep_ok"] / max(1, acc["ent_keep"])}
    for name in VARS:
        cm = conf[name].view(C, C).cpu().numpy().astype(np.float64)
        iou = np.diag(cm) / (cm.sum(0) + cm.sum(1) - np.diag(cm) + 1e-12)
        r[name] = {"miou": float(100 * iou[cm.sum(1) > 0].mean()), "pix_acc": float(100 * np.diag(cm).sum() / cm.sum()),
                   "iou": iou.tolist(), "pred_share": (hist[name].float() / hist[name].sum()).tolist()}
    r["label_share"] = (conf["none"].view(C, C).sum(1).float() / acc["n"]).tolist()
    for name in VARS[1:]:
        f = acc[f"flip_{name}"]
        r[f"flip_{name}"] = f / acc["n"]
        r[f"flip_all_{name}"] = acc[f"flip_all_{name}"] / acc["n_all"]
        r[f"flip_acc_{name}"] = acc[f"flip_ok_{name}"] / max(1, f)
        r[f"noflip_acc_{name}"] = acc[f"noflip_ok_{name}"] / max(1, acc["n"] - f)
    res["streams"][key] = r
    json.dump(res, open(FILE, "w"), indent=1)
    print(f"[{si + 1}/{len(streams)}] {key}: mIoU none {r['none']['miou']:.2f} bn {r['bn']['miou']:.2f} isa {r['isa']['miou']:.2f} "
          f"isa_bn {r['isa_bn']['miou']:.2f} | flip bn {100 * r['flip_bn']:.1f}% acc {r['none']['pix_acc']:.1f} conf {100 * r['mean_conf']:.1f} "
          f"({time.time() - t0:.0f}s)", flush=True)
print("saved", FILE)
