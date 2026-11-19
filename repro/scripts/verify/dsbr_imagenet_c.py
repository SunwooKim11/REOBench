"""Check our DSBR implementation against the numbers of its paper on ImageNet-C (ViT-B/16, severity 5, "Default" setting).

Paper: Nielen et al., "Entropy Minimization without Model Collapse", arXiv 2606.02339, Table 3, ViT-B/16, Default column:
  no adaptation 39.8, Tent 51.9, DSBR (alpha 0.9) 64.4, DSBR (alpha 0.99) 64.9   (accuracy %, mean of 15 corruptions)
No public code. What the paper states and what we had to assume:
  stated  : batch 64, the full 50,000 images of a corruption, online accuracy, Algorithm 1, Adam (default betas, no weight
            decay) "with the lr of Tent per batch size" for DSBR, official defaults for Tent.
  assumed : the checkpoint (timm vit_base_patch16_224; both augreg_in21k_ft_in1k = the timm default when SAR was written and
            augreg2_in21k_ft_in1k = the current default are evaluated without adaptation, --ckpt picks the one to adapt);
            the adapted parameters (--params all = affine of every LayerNorm, as Tent; sar = SAR rule, blocks 9-11 and the
            final norm excluded); the lr (--lr; Tent's ImageNet rule 0.00025 * bs / 64, SAR uses 0.001 for ViT);
            the model is reset at the start of every corruption; shuffled order (seed).
So agreement can only be approximate; the run tells whether the size of the gain over Tent is of the same order.

  ~/REOBench/.venv/bin/python -u scripts/verify/dsbr_imagenet_c.py --method none --ckpt augreg2      # from repro/, GPU
  ... --method tent --opt sgd --lr 0.00025 | --method dsbr --alpha 0.9 --opt adam --lr 0.00025
Output: experiments/2026-11-19_dsbr/imagenet_c/<ckpt>_<method>[_a<alpha>]_<opt>_lr<lr>_<params>.json
"""
import argparse
import copy
import json
import os
import re
import sys
import time

import timm
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as trn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402

OUT = paths.EXP / "2026-11-19_dsbr" / "imagenet_c"
INC = "/data/tta/ImageNet_Family/ImageNet_C"
CORR = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost",
        "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
CKPT = {"augreg": "vit_base_patch16_224.augreg_in21k_ft_in1k", "augreg2": "vit_base_patch16_224.augreg2_in21k_ft_in1k"}


def params_of(model, which):
    ps = []
    for n, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            b = re.search(r"blocks\.(\d+)", n)
            if which == "sar" and (not b or int(b.group(1)) >= 9):
                continue
            ps += [m.weight, m.bias]
    return ps


def entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def run(args):
    name = f"{args.ckpt}_{args.method}" + (f"_a{args.alpha:g}" if args.method == "dsbr" else "") + \
           ("" if args.method == "none" else f"_{args.opt}_lr{args.lr:g}_{args.params}")
    f = OUT / f"{name}.json"
    res = json.load(open(f)) if f.exists() else {"args": vars(args), "acc": {}, "top_share": {}}
    model0 = timm.create_model(CKPT[args.ckpt], pretrained=True).cuda().eval()
    cfg = model0.pretrained_cfg
    tf = trn.Compose([trn.ToTensor(), trn.Normalize(cfg["mean"], cfg["std"])])
    for c in CORR:
        if c in res["acc"]:
            continue
        t0 = time.time()
        torch.manual_seed(args.seed)
        dl = torch.utils.data.DataLoader(dset.ImageFolder(f"{INC}/{c}/{args.sev}", transform=tf), batch_size=args.bs, shuffle=True,
                                         num_workers=args.workers, pin_memory=True)
        model = copy.deepcopy(model0)
        model.requires_grad_(False)
        if args.method != "none":
            ps = params_of(model, args.params)
            for p in ps:
                p.requires_grad_(True)
            opt = torch.optim.Adam(ps, args.lr) if args.opt == "adam" else torch.optim.SGD(ps, args.lr, momentum=0.9)
        K = 1000
        P = torch.full((K,), 1.0 / K, device="cuda")
        ok, n, hist = 0, 0, torch.zeros(K, device="cuda")
        for x, y in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            if args.method == "none":
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(x).argmax(1)
            else:
                logits = model(x)
                pred = logits.detach().argmax(1)
                ent = entropy(logits)
                if args.method == "dsbr":  # Algorithm 1: P updated with the batch, then the weighted loss
                    P = args.alpha * P + (1 - args.alpha) * torch.bincount(pred, minlength=K).float() / len(pred)
                    loss = (ent / (K * P[pred])).mean()
                else:
                    loss = ent.mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
            ok += (pred == y).sum().item(); n += len(y); hist += torch.bincount(pred, minlength=K)
        res["acc"][c], res["top_share"][c] = 100 * ok / n, hist.max().item() / n
        print(f"  {name} {c:18s} acc {res['acc'][c]:5.1f} top-class share {100 * res['top_share'][c]:5.1f}% ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(f, "w"), indent=1)
    res["mean"] = sum(res["acc"].values()) / len(res["acc"])
    json.dump(res, open(f, "w"), indent=1)
    print(f"{name}: mean {res['mean']:.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="none", choices=["none", "tent", "dsbr"])
    ap.add_argument("--ckpt", default="augreg2", choices=list(CKPT))
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--lr", type=float, default=0.00025)
    ap.add_argument("--params", default="all", choices=["all", "sar"])
    ap.add_argument("--sev", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    run(args)
