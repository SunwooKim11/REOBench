"""Feature-level TTA on the saved features: which loss / sample filter makes entropy minimisation safe (tent, entropy filter, flip filter, DSBR)?

Everything runs on the saved features and the stage-1 linear heads (feat_metrics.py); no backbone forward pass.
Adapted parameters: one affine per channel on the head input, z' = gamma * z + beta (gamma = 1, beta = 0 at the start of a
stream). For the ViT backbones the feature is the CLS token after the final LayerNorm, so this is exactly TENT restricted
to the last normalisation layer; for ResNet-50 (global average pool) it is an affine on the pooled feature. The head
stays fixed. z = head input standardised with the clean-train mean / std (source mean 0, std 1).

Streams: clean test + 12 corruptions x 5 severities (61 per dataset), each shuffled with the permutation feat_metrics.align
uses (seed 0), batches of --bs, episodic (parameters reset at the start of every stream). Online protocol of TENT: the
prediction of a batch comes from the forward pass before the update made with that batch.

Input (what the affine is applied to):  raw = z,  al = batch re-standardised z (= align_ms, no state across batches)
Loss = mean entropy over the selected samples of the batch:
  tent  : all samples
  entf  : entropy < 0.4 ln C                     (reliability filter of SAR / EATA; no SAM, no weighting)
  flipf : prediction on z equals prediction on the batch-aligned z, both under the current affine (no flip)
  bothf : entf and flipf
  dsbr, dsbr99 : DSBR (arXiv 2606.02339, Algorithm 1): every sample, entropy weighted by 1 / (C * P[predicted class]),
          P = EMA (decay 0.9 / 0.99) of the predicted-class frequencies of the stream, started at 1 / C. Not in the default
          losses; --losses tent dsbr dsbr99 --out <dir> writes to a separate folder.
Optimiser: Adam, lr in --lrs (all reported; nothing is selected with labels). A batch with no selected sample gives a
zero gradient. Baselines without learning: none, align_ms.

Per stream: acc, ECE, signed ECE, NLL, AUROC (feat_metrics.stream_metrics) and the share of samples used for the update.

  ~/REOBench/.venv/bin/python scripts/analysis/feat_tta.py [--models ...] [--datasets aid r45] [--bs 64] [--device cuda]
Output: experiments/2026-11-17_flip_tta/bs<bs>/<model>_<ds>.json
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths  # noqa: E402
from extract_feats import MODELS  # noqa: E402
import feat_metrics as fm  # noqa: E402

OUT = paths.EXP / "2026-11-17_flip_tta"
dev = torch.device("cpu")
fm.dev = dev
LOSSES = ["tent", "entf", "flipf", "bothf"]


def batch_align(zb):  # (S, B, D) -> re-standardised per stream and channel (= feat_metrics.align "ms" on one batch)
    return (zb - zb.mean(1, keepdim=True)) / (zb.std(1, unbiased=False, keepdim=True) + 1e-5)


def entropy(logits):
    lp = logits.log_softmax(-1)
    return -(lp.exp() * lp).sum(-1)


def adapt(Z, w, b, inp, loss, lr, bs):
    """Z: (S, N, D) streams already in stream order. Returns the online logits (S, N, C) and the used share per stream."""
    S, N, D = Z.shape
    g = torch.ones(S, 1, D, device=Z.device, requires_grad=True)
    be = torch.zeros(S, 1, D, device=Z.device, requires_grad=True)
    opt = torch.optim.Adam([g, be], lr=lr)  # element-wise, and the loss is a sum over streams -> streams are independent
    C = w.shape[1]
    e0 = 0.4 * math.log(C)
    P = torch.full((S, C), 1.0 / C, device=Z.device)
    alpha = 0.99 if loss == "dsbr99" else 0.9
    out, used = [], torch.zeros(S, device=Z.device)
    for i in range(0, N, bs):
        zb = Z[:, i:i + bs]
        za = batch_align(zb)
        lr_ = (g * zb + be) @ w + b   # logits of the raw input under the current affine
        la_ = (g * za + be) @ w + b   # logits of the aligned input under the current affine
        logits = lr_ if inp == "raw" else la_
        out.append(logits.detach())
        ent = entropy(logits)
        with torch.no_grad():
            m = torch.ones_like(ent, dtype=torch.bool)
            if loss in ("entf", "bothf"):
                m &= ent < e0
            if loss in ("flipf", "bothf"):
                m &= lr_.argmax(-1) == la_.argmax(-1)
            used += m.float().sum(1)
        if loss.startswith("dsbr"):
            with torch.no_grad():
                pred = logits.argmax(-1)
                P = alpha * P + (1 - alpha) * torch.nn.functional.one_hot(pred, C).float().mean(1)
                wgt = 1 / (C * P.gather(1, pred))
            L = (ent * wgt).mean(1).sum()
        else:
            L = ((ent * m).sum(1) / m.float().sum(1).clamp_min(1)).sum()
        opt.zero_grad()
        L.backward()
        opt.step()
    return torch.cat(out, 1), used / N


def metrics(logits, Y):  # (S, N, C), (S, N)
    res = []
    for s in range(len(logits)):
        r = fm.stream_metrics(logits[s], Y[s])
        res.append({k: r[k] for k in ("acc", "ece", "signed_ece", "nll", "auroc")})
    return res


def run(model, ds, bs, lrs, losses=LOSSES):
    h = torch.load(f"{fm.FEATS}/{model}/head_{ds}.pt", map_location="cpu")
    w, b, mu, sd = h["w"], h["b"], h["mu"], h["sd"]
    te_x, te = fm.load(model, f"{ds}_test")
    c_x, cd = fm.load(model, f"{ds}_c")
    n = len(te_x)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))  # the permutation feat_metrics.align uses
    names, Z, Y = ["clean"], [((te_x - mu) / sd)[perm]], [te["y"][perm]]
    for k, cname in enumerate(common.NOISES):
        for s in range(1, 6):
            sel = (cd["corr"] == k) & (cd["sev"] == s)
            assert int(sel.sum()) == n, (cname, s, int(sel.sum()))
            names.append(f"{cname}_s{s}")
            Z.append(((c_x[sel] - mu) / sd)[perm]); Y.append(cd["y"][sel][perm])
    Z, Y, w, b = torch.stack(Z).to(dev), torch.stack(Y).to(dev), w.to(dev), b.to(dev)
    del c_x
    res = {"model": model, "dataset": ds, "bs": bs, "streams": names, "methods": {}}
    with torch.no_grad():
        res["methods"]["none"] = {"m": metrics(Z @ w + b, Y)}
        al = torch.cat([batch_align(Z[:, i:i + bs]) for i in range(0, n, bs)], 1)
        res["methods"]["align_ms"] = {"m": metrics(al @ w + b, Y)}
        del al
    for inp in ("raw", "al"):
        for loss in losses:
            for lr in lrs:
                t0 = time.time()
                logits, used = adapt(Z, w, b, inp, loss, lr, bs)
                key = f"{inp}_{loss}_lr{lr:g}"
                res["methods"][key] = {"m": metrics(logits, Y), "used": used.tolist()}
                del logits
                mean = lambda k_: sum(x[k_] for x in res["methods"][key]["m"][1:]) / 60
                print(f"  {model} {ds} {key:22s} clean {res['methods'][key]['m'][0]['acc']:5.1f} corrupted {mean('acc'):5.1f} "
                      f"sECE {mean('signed_ece'):+.3f} used {100 * used[1:].mean():5.1f}% ({time.time() - t0:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--datasets", nargs="*", default=["aid", "r45"])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lrs", nargs="*", type=float, default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--losses", nargs="*", default=LOSSES)
    ap.add_argument("--out", default=None, help="output folder (default experiments/2026-11-17_flip_tta/bs<bs>)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    fm.dev = dev
    out = paths.EXP / args.out if args.out else OUT / f"bs{args.bs}"
    out.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        for ds in args.datasets:
            f = out / f"{m}_{ds}.json"
            if f.exists():
                continue
            r = run(m, ds, args.bs, args.lrs, args.losses)
            a = lambda k: sum(x["acc"] for x in r["methods"][k]["m"][1:]) / 60
            print(f"{m} {ds}: none {a('none'):.1f} align_ms {a('align_ms'):.1f}", flush=True)
            json.dump(r, open(f, "w"))
    print("DONE")
