"""Real vs synthetic haze: RRSHID (real clear/hazy pairs of the same scene) through every backbone.

RRSHID (/data/rs/RRSHID/extracted): thin / moderate / thick fog, train / val / test, paired clear (folder gt | GT | clear)
and hazy images, 256 px. No class labels, so the questions are about the shift itself, not accuracy:
  pixel   : PSNR(hazy, clear), per-channel affine R^2 (best a*hazy + b for clear, = what ISA can undo), mean / std ratio
  feature : on all pairs of a level, standardized with that level's clear-image statistics:
            rel_drift, frac_common (share of the drift common to all images), cka, cos (= feat_metrics.py)
  head    : prediction agreement hazy vs clear under the AID and RESISC45 linear heads of stage 1 (no labels needed)
  repair  : on val + test pairs (source statistics from the *train* clear images of all levels):
            pixel ISA (batch of 64 hazy images re-standardized to the source pixel mean / std per channel) and
            head-input alignment (batch re-standardized to the source feature mean / std) -> agreement again
Synthetic counterparts on the same clear images: REOBench haze x(1-i)+i, i = 0.2 / 0.4 / 0.6 (severities 1 / 3 / 5).

  ~/REOBench/.venv/bin/python scripts/analysis/rrshid_analysis.py [--models ...]   # from repro/, after feat_metrics.py
Output: experiments/2026-11-16_backbone_feats/rrshid/<model>.json and summary.txt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import backbones, paths  # noqa: E402
from extract_feats import MODELS  # noqa: E402
from feat_metrics import cka  # noqa: E402

R = "/data/rs/RRSHID/extracted"
LEVELS = ["thin_fog", "moderate_fog", "thick_fog"]
SYN = {"syn_i0.2": 0.2, "syn_i0.4": 0.4, "syn_i0.6": 0.6}
OUT = paths.EXP / "2026-11-16_backbone_feats" / "rrshid"
dev = torch.device("cuda")


def pairs(level, splits):
    out = []
    for s in splits:
        d = f"{R}/{level}/{s}"
        clean = next(c for c in ("gt", "GT", "clear") if os.path.isdir(f"{d}/{c}"))
        out += [(f"{d}/{clean}/{f}", f"{d}/hazy/{f}") for f in sorted(os.listdir(f"{d}/hazy"))]
    return out


def load(p):  # -> float tensor 3 x 224 x 224 in [0, 1]
    return TF.to_tensor(Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR))


def pixel_stats(clean, hazy):
    psnr = (10 * torch.log10(1 / ((clean - hazy) ** 2).mean((1, 2, 3)).clamp_min(1e-10))).mean().item()
    # best affine fit of clean from hazy per image and channel has R^2 = squared Pearson correlation (closed form)
    x, y = hazy.flatten(2), clean.flatten(2)
    xc, yc = x - x.mean(2, keepdim=True), y - y.mean(2, keepdim=True)
    corr = (xc * yc).sum(2) / ((xc ** 2).sum(2) * (yc ** 2).sum(2)).sqrt().clamp_min(1e-10)
    r2 = corr ** 2
    return {"psnr": psnr, "affine_r2": r2.mean().item(),
            "mean_ratio": (hazy.mean((0, 2, 3)) / clean.mean((0, 2, 3))).mean().item(),
            "std_ratio": (hazy.std((0, 2, 3)) / clean.std((0, 2, 3))).mean().item()}


@torch.no_grad()
def feats(model, norm, imgs, bs=128):
    mu, sd = torch.tensor(norm[0]).view(1, 3, 1, 1), torch.tensor(norm[1]).view(1, 3, 1, 1)
    return torch.cat([model(((imgs[i:i + bs] - mu) / sd).to(dev)).float() for i in range(0, len(imgs), bs)])


def isa_pixel(x, mu_s, sd_s, bs=64):
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(x), generator=g)
    out = torch.empty_like(x)
    for i in range(0, len(x), bs):
        idx = order[i:i + bs]
        b = x[idx]
        m, s = b.mean((0, 2, 3), keepdim=True), b.std((0, 2, 3), keepdim=True).clamp_min(1e-5)
        out[idx] = ((b - m) / s * sd_s + mu_s).clamp(0, 1)
    return out


def align_feat(z, mu_s, sd_s, bs=64):
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(z), generator=g).to(z.device)
    out = torch.empty_like(z)
    for i in range(0, len(z), bs):
        idx = order[i:i + bs]
        b = z[idx]
        out[idx] = (b - b.mean(0)) / (b.std(0, unbiased=False) + 1e-5) * sd_s + mu_s
    return out


def drift(fc, fh):
    mu, sd = fc.mean(0), fc.std(0) + 1e-6
    zc, zh = (fc - mu) / sd, (fh - mu) / sd
    d = zh - zc
    dn = (d ** 2).sum(1)
    return {"rel_drift": (dn.mean() / (zc ** 2).sum(1).mean()).item(),
            "frac_common": ((d.mean(0) ** 2).sum() / dn.mean()).item(),
            "cka": cka(fc, fh), "cos": F.cosine_similarity(fc, fh, dim=1).mean().item()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # images (CPU, once)
    allp = {L: pairs(L, ["train", "val", "test"]) for L in LEVELS}
    evp = {L: pairs(L, ["val", "test"]) for L in LEVELS}
    trp = [c for L in LEVELS for c, _ in pairs(L, ["train"])]
    IM = {}
    for L in LEVELS:
        IM[L] = (torch.stack([load(c) for c, _ in allp[L]]), torch.stack([load(h) for _, h in allp[L]]))
        IM[L + "_ev"] = (torch.stack([load(c) for c, _ in evp[L]]), torch.stack([load(h) for _, h in evp[L]]))
    tr = torch.stack([load(c) for c in trp])
    mu_px, sd_px = tr.mean((0, 2, 3), keepdim=True), tr.std((0, 2, 3), keepdim=True)
    allclean = torch.cat([IM[L][0] for L in LEVELS])
    pix = {L: pixel_stats(*IM[L]) for L in LEVELS}
    for k, i in SYN.items():
        pix[k] = pixel_stats(allclean, (allclean * (1 - i) + i).clamp(0, 1))
    print({k: {a: round(b, 3) for a, b in v.items()} for k, v in pix.items()}, flush=True)

    rows = {}
    for name in args.models:
        f = OUT / f"{name}.json"
        if f.exists():
            rows[name] = json.load(open(f)); continue
        heads = {ds: torch.load(f"/data/rs/feats/{name}/head_{ds}.pt") for ds in ("aid", "r45")}
        model, norm, _ = backbones.build(name)
        model = model.to(dev)
        hd = {ds: (lambda x, h=h: ((x - h["mu"].to(dev)) / h["sd"].to(dev)) @ h["w"].to(dev) + h["b"].to(dev))
              for ds, h in heads.items()}
        agree = lambda a, b, ds: (hd[ds](a).argmax(1) == hd[ds](b).argmax(1)).float().mean().item() * 100
        res = {"model": name, "pixel": pix, "real": {}, "syn": {}}
        f_tr = feats(model, norm, tr)
        mu_f, sd_f = f_tr.mean(0), f_tr.std(0, unbiased=False)
        for L in LEVELS:
            fc, fh = feats(model, norm, IM[L][0]), feats(model, norm, IM[L][1])
            r = drift(fc, fh)
            r.update({f"agree_{ds}": agree(fc, fh, ds) for ds in hd})
            ec, eh = IM[L + "_ev"]
            fec, feh = feats(model, norm, ec), feats(model, norm, eh)
            fisa = feats(model, norm, isa_pixel(eh, mu_px, sd_px))
            falg = align_feat(feh, mu_f, sd_f)
            for ds in hd:
                r[f"ev_agree_{ds}"] = agree(fec, feh, ds)
                r[f"ev_agree_isa_{ds}"] = agree(fec, fisa, ds)
                r[f"ev_agree_align_{ds}"] = agree(fec, falg, ds)
            res["real"][L] = r
        fall = feats(model, norm, allclean)
        for k, i in SYN.items():
            fs = feats(model, norm, (allclean * (1 - i) + i).clamp(0, 1))
            r = drift(fall, fs)
            r.update({f"agree_{ds}": agree(fall, fs, ds) for ds in hd})
            res["syn"][k] = r
        json.dump(res, open(f, "w"), indent=1)
        rows[name] = res
        print(name, {L: round(res["real"][L]["frac_common"], 3) for L in LEVELS},
              {k: round(v["frac_common"], 3) for k, v in res["syn"].items()}, flush=True)
        del model; torch.cuda.empty_cache()

    lines = ["pixel: " + json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in pix.items()}), ""]
    keys = ["rel_drift", "frac_common", "cka", "cos", "agree_aid", "agree_r45"]
    lines.append(f"{'model':15s} {'set':13s} " + " ".join(f"{k:>11s}" for k in keys)
                 + f" {'ev_agr_aid':>10s} {'+isa':>6s} {'+align':>6s}")
    for name, res in rows.items():
        for L in LEVELS:
            r = res["real"][L]
            lines.append(f"{name:15s} {L:13s} " + " ".join(f"{r[k]:11.3f}" if k[:5] != "agree" else f"{r[k]:11.1f}" for k in keys)
                         + f" {r['ev_agree_aid']:10.1f} {r['ev_agree_isa_aid']:6.1f} {r['ev_agree_align_aid']:6.1f}")
        for k2, r in res["syn"].items():
            lines.append(f"{name:15s} {k2:13s} " + " ".join(f"{r[k]:11.3f}" if k[:5] != "agree" else f"{r[k]:11.1f}" for k in keys))
    open(OUT / "summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
