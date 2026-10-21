"""Test-time adaptation for Potsdam segmentation (REOBench mmseg 0.21 models trained by scripts/seg/train.sh), mild protocol.

Protocol (mirrors scripts/tta/cls_tta.py for classification):
  - one stream per (corruption, severity) [+ clean]: the 2,016 test patches, shuffled, batch 8 (= training batch)
  - reset to the source model before every stream; online mIoU from the prediction made BEFORE the update
  - mIoU over the 5 evaluated classes (label 5 = clutter and 255 are ignored, as in eval_all.sh)
Methods (losses are per pixel, averaged over valid pixels):
  none   : source model
  tent   : pixel entropy (DequanWang/tent objective), affine of all LayerNorm/BatchNorm, Adam 6e-5/8
           (lr / optimizer / parameter set = CoTTA's ACDC Tent code, validated in G_seg_verify);
           --bn fixed (BN running stats, as the CoTTA code) | batch (BN on test-batch statistics, original Tent)
  sar    : Tent + SAR (pixel filter entropy < 0.4 ln C, SAM rho 0.05 around the same Adam, recovery when the
           entropy EMA < --sar_reset). Pixel-level port of mr-eggplant/SAR (no official segmentation code)
  bnadapt: BN layers normalize with the test-batch statistics, no parameter update (classic "BN adapt" baseline;
           only the decode/aux heads have BN, the ViT backbone has LayerNorm)
  isa    : input statistics alignment (src/reobench_tta/tta/isa.py), batch level, in 0-255 pixel space, source = train split
  cotta  : CoTTA ACDC code (G_seg_verify single_gpu_cotta) ported: EMA teacher 0.999, anchor model, 7 scales x flip
           teacher views, anchor confidence > --cotta_thr uses the single-scale teacher label, all parameters,
           Adam 6e-5/8, stochastic restore 0.01; prediction = the teacher pseudo-label
Outputs: experiments/2026-10-20_seg_tta/outputs/<model>_<set>/<method><tag>.json

  TIMM_FUSED_ATTN=0 /opt/conda/envs/segtta/bin/python scripts/seg/seg_tta.py --model scalemae --method tent   # from repro/
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
from copy import deepcopy

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402
from reobench_tta.tta.sam import SAM  # noqa: E402  (same SAM as the classification SAR)
os.chdir(paths.REOBENCH_SEG)  # mmseg configs (configs/repro/...) are relative to the vendored mmseg copy
sys.path.insert(0, str(paths.REOBENCH_SEG))
from mmseg.models import build_segmentor  # noqa: E402

T = "/data/rs/reobench/Potsdam_extracted/test"
TRAIN = "/data/rs/reobench/Potsdam_extracted/images/train"
CORR = ["brightness_contrast", "clouds", "compression_artifacts", "gaussian_blur", "gaussian_noise", "haze",
        "motion_blur", "salt_and_pepper_noise"]
C = 5          # evaluated classes (decode head has 5 outputs; clutter=5 is the ignore index)
IGNORE = 5

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=["scalemae", "satmae"])
p.add_argument("--set", default="B")
p.add_argument("--method", required=True, choices=["none", "bnadapt", "tent", "sar", "isa", "cotta"])
p.add_argument("--bn", default="fixed", choices=["fixed", "batch"])
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=0.00006 / 8)
p.add_argument("--sar_e0", type=float, default=0.4, help="SAR reliable-pixel margin, x ln C")
p.add_argument("--sar_reset", type=float, default=0.2, help="SAR recovery threshold on the entropy EMA (paper 0.2)")
p.add_argument("--cotta_thr", type=float, default=None, help="anchor confidence threshold; default = 5%% quantile - 0.05 on train")
p.add_argument("--corruptions", nargs="*", default=CORR)
p.add_argument("--sevs", nargs="*", type=int, default=[1, 2, 3, 4, 5])
p.add_argument("--no_clean", action="store_true")
p.add_argument("--max_images", type=int, default=None, help="images per stream (random subset), for slow methods")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="")
p.add_argument("--workers", type=int, default=8)
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
dev = torch.device("cuda")
cfg = mmcv.Config.fromfile(f"configs/repro/{args.model}_{args.set}_12ep.py")
CKPT = f"{paths.SEG_BASELINE}/work_dirs/{args.model}_{args.set}_12ep/latest.pth"
MEAN = torch.tensor(cfg.img_norm_cfg["mean"]).view(1, 3, 1, 1).to(dev)    # 0-255 scale (pipeline Normalize)
STD = torch.tensor(cfg.img_norm_cfg["std"]).view(1, 3, 1, 1).to(dev)
tag = args.tag + (f"_bn{args.bn}" if args.method in ("tent", "sar") and args.bn != "fixed" else "")
if args.method != "none" and args.method != "isa" and args.lr != 0.00006 / 8:
    tag += f"_lr{args.lr:g}"
OUT = f"{paths.SEG_TTA}/outputs/{args.model}_{args.set}"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- data
class Files(Dataset):
    def __init__(self, img_dir, names):
        self.img_dir, self.names = img_dir, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = np.asarray(Image.open(f"{self.img_dir}/{n}").convert("RGB"), dtype=np.float32)   # RGB 0-255
        lab = np.asarray(Image.open(f"{T}/clean_ann/{n}").convert("L")).astype(np.int64)
        lab[lab == 255] = IGNORE                                                              # as in eval
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(lab)


NAMES = sorted(os.listdir(f"{T}/clean"))
streams = ([] if args.no_clean else [("clean", 0, f"{T}/clean")]) + \
          [(c, s, f"{T}/{c}/severity_{s}") for c in args.corruptions for s in args.sevs]


# ---------------------------------------------------------------- model
def build():
    m = build_segmentor(cfg.model)
    load_checkpoint(m, CKPT, map_location="cpu", logger="silent")
    return m.to(dev).eval()


source = build()
META = [dict(img_shape=(512, 512, 3), ori_shape=(512, 512, 3), pad_shape=(512, 512, 3), flip=False)]


def norm(x):            # x: B,3,H,W in 0-255 RGB -> model input (cfg Normalize, to_rgb on an RGB-loaded image = RGB)
    return (x - MEAN) / STD


def logits_of(model, x):  # B,C,H,W at input resolution (= encode_decode; slide window is one window at 512)
    return model.encode_decode(norm(x), META * x.shape[0])


def entropy(lg):        # per pixel
    return -(lg.softmax(1) * lg.log_softmax(1)).sum(1)


def norm_params(model):
    ps = []
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
            ps += [q for q in (m.weight, m.bias) if q is not None]
    return ps


def configure_norm(model):
    model.eval()
    model.requires_grad_(False)
    for q in norm_params(model):
        q.requires_grad_(True)
    if args.bn == "batch":   # original Tent configure_model
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train(); m.track_running_stats = False; m.running_mean = None; m.running_var = None
    return model


# source-side quantities (train split, never test): ISA pixel statistics, CoTTA anchor-confidence threshold
if args.method in ("isa", "cotta"):
    tr = sorted(os.listdir(TRAIN))
    sub = random.Random(0).sample(tr, 500)
    s1 = torch.zeros(3, dtype=torch.float64); s2 = torch.zeros(3, dtype=torch.float64); npx = 0; confs = []
    with torch.no_grad():
        for i in range(0, len(sub), args.bs):
            x = torch.stack([torch.from_numpy(np.asarray(Image.open(f"{TRAIN}/{n}").convert("RGB"), dtype=np.float32))
                             .permute(2, 0, 1) for n in sub[i:i + args.bs]]).to(dev)
            s1 += x.double().sum((0, 2, 3)).cpu(); s2 += (x.double() ** 2).sum((0, 2, 3)).cpu(); npx += x.numel() // 3
            if args.method == "cotta":
                confs.append(logits_of(source, x).softmax(1).max(1)[0].flatten()[::97].float().cpu())
    SRC_MEAN = (s1 / npx).float().view(1, 3, 1, 1).to(dev)
    SRC_STD = (s2 / npx - (s1 / npx) ** 2).sqrt().float().view(1, 3, 1, 1).to(dev)
    print(f"train pixel mean {SRC_MEAN.flatten().tolist()} std {SRC_STD.flatten().tolist()}")
    if args.method == "cotta":
        q5 = torch.quantile(torch.cat(confs), 0.05).item()
        COTTA_THR = args.cotta_thr if args.cotta_thr is not None else q5 - 0.05   # CoTTA: 0.74 (5% q.) -> 0.69
        print(f"CoTTA anchor threshold {COTTA_THR:.3f} (train 5% quantile of confidence {q5:.3f})")


# ---------------------------------------------------------------- methods
class Adapter:
    def __init__(self):
        self.model = deepcopy(source)
        self.n_reset = 0
        if args.method == "bnadapt":
            args.bn = "batch"; configure_norm(self.model); self.model.requires_grad_(False)
        if args.method in ("tent", "sar"):
            configure_norm(self.model)
            ps = [q for q in self.model.parameters() if q.requires_grad]
            if args.method == "tent":
                self.opt = torch.optim.Adam(ps, lr=args.lr, betas=(0.9, 0.999))
            else:
                self.opt = SAM(ps, torch.optim.Adam, rho=0.05, lr=args.lr, betas=(0.9, 0.999))
                self.state0 = deepcopy(self.model.state_dict()); self.opt_state0 = deepcopy(self.opt.state_dict())
                self.ema = None
            print(f"{args.method}: adapting {sum(q.numel() for q in ps) / 1e6:.2f}M params, BN {args.bn}")
        if args.method == "cotta":
            self.model.requires_grad_(True)
            self.anchor_state = deepcopy(self.model.state_dict())
            self.anchor = deepcopy(self.model).eval()
            self.ema_model = deepcopy(self.model).eval()
            for q in self.ema_model.parameters():
                q.detach_()
            self.opt = torch.optim.Adam([q for q in self.model.parameters()], lr=args.lr, betas=(0.9, 0.999))

    @torch.no_grad()
    def isa(self, x):
        mu = x.mean((0, 2, 3), keepdim=True); sd = x.std((0, 2, 3), keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd * SRC_STD + SRC_MEAN

    def __call__(self, x):
        m = args.method
        if m in ("none", "bnadapt"):
            with torch.no_grad():
                return logits_of(self.model, x).argmax(1)
        if m == "isa":
            with torch.no_grad():
                return logits_of(self.model, self.isa(x)).argmax(1)
        if m == "tent":
            lg = logits_of(self.model, x)
            entropy(lg).mean().backward()
            self.opt.step(); self.opt.zero_grad()
            return lg.detach().argmax(1)
        if m == "sar":
            e0 = args.sar_e0 * math.log(C)
            lg = logits_of(self.model, x)
            ent = entropy(lg); keep = ent < e0
            if keep.any():
                ent[keep].mean().backward()
                self.opt.first_step(zero_grad=True)
                ent2 = entropy(logits_of(self.model, x))
                keep2 = keep & (ent2 < e0)
                if keep2.any():
                    loss2 = ent2[keep2].mean(); loss2.backward()
                    v = loss2.item(); self.ema = v if self.ema is None else 0.9 * self.ema + 0.1 * v
                self.opt.second_step(zero_grad=True)
                if self.ema is not None and self.ema < args.sar_reset:   # model recovery
                    self.model.load_state_dict(self.state0); self.opt.load_state_dict(self.opt_state0)
                    self.ema = None; self.n_reset += 1
            return lg.detach().argmax(1)
        if m == "cotta":
            return self.cotta(x)

    def cotta(self, x):
        H, W = x.shape[2:]
        with torch.no_grad():
            probs, std_prob = 0, None
            for r in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:      # MultiScaleFlipAug order: (no flip, flip) per scale
                xs = F.interpolate(x, scale_factor=r, mode="bilinear", align_corners=False) if r != 1.0 else x
                for flip in (False, True):
                    xi = xs.flip(3) if flip else xs
                    pr = F.interpolate(logits_of(self.ema_model, xi), size=(H, W), mode="bilinear",
                                       align_corners=False).softmax(1)
                    pr = pr.flip(3) if flip else pr
                    probs = probs + pr
                    if r == 1.0 and not flip:
                        std_prob = pr                                     # preds[4] in the CoTTA code
            aug_pred = probs.argmax(1)                                    # result[0]: averaged over the 14 views
            anchor_conf = logits_of(self.anchor, x).softmax(1).max(1)[0]  # probs_[4][0]
            mask = anchor_conf > COTTA_THR
            pseudo = torch.where(mask, std_prob.argmax(1), aug_pred)
        lg = logits_of(self.model, x)
        F.cross_entropy(lg, pseudo).backward()
        self.opt.step(); self.opt.zero_grad()
        with torch.no_grad():
            for e, q in zip(self.ema_model.parameters(), self.model.parameters()):
                e.mul_(0.999).add_(q, alpha=0.001)
            for nm, q in self.model.named_parameters():                  # stochastic restore (weight/bias, p=0.01)
                if nm.split(".")[-1] in ("weight", "bias"):
                    keep = (torch.rand_like(q) < 0.01).float()
                    q.mul_(1 - keep).add_(self.anchor_state[nm] * keep)
        return pseudo


# ---------------------------------------------------------------- run
def miou(conf):
    iou = np.diag(conf) / (conf.sum(0) + conf.sum(1) - np.diag(conf) + 1e-12)
    present = conf.sum(1) > 0
    return 100 * iou[present].mean()


res = {"config": vars(args), "streams": {}}
t0 = time.time()
for si, (cond, sev, d) in enumerate(streams):
    names = NAMES if args.max_images is None else random.Random(si).sample(NAMES, args.max_images)
    g = torch.Generator().manual_seed(args.seed * 1000 + si)
    dl = DataLoader(Files(d, names), batch_size=args.bs, shuffle=True, generator=g, num_workers=args.workers,
                    pin_memory=True)
    adapt = Adapter()
    conf = np.zeros((C, C), np.int64)
    for x, y in dl:
        pred = adapt(x.to(dev, non_blocking=True)).cpu().numpy()
        y = y.numpy(); v = y != IGNORE
        conf += np.bincount(C * y[v] + pred[v], minlength=C * C).reshape(C, C)
    r = miou(conf)
    key = "clean" if cond == "clean" else f"{cond}_s{sev}"
    res["streams"][key] = r
    extra = f" resets {adapt.n_reset}" if args.method == "sar" else ""
    print(f"[{si + 1}/{len(streams)}] {key}: mIoU {r:.2f}{extra} ({time.time() - t0:.0f}s)", flush=True)
    del adapt; torch.cuda.empty_cache()

per = {c: np.mean([res["streams"][f"{c}_s{s}"] for s in args.sevs]) for c in args.corruptions
       if all(f"{c}_s{s}" in res["streams"] for s in args.sevs)}
res["per_corruption"] = per
if len(per) == len(CORR):
    res["avg8"] = float(np.mean(list(per.values())))
path = f"{OUT}/{args.method}{tag}.json"
json.dump(res, open(path, "w"), indent=2)
print("saved", path, {k: res.get(k) for k in ("avg8",)}, res["streams"].get("clean"))
