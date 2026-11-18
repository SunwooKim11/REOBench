"""How much of the feature drift under corruption lies in the directions the linear head reads?

Saved features and stage-1 heads (feat_metrics.py). z = head input standardised with clean-train mean / std.
Per stream (corruption x severity), with d_i = z_i(corrupted) - z_i(clean) of the same image:
  c   = mean_i d_i          common shift (one vector shared by every image)
  r_i = d_i - c             per-sample part (mean 0)
  P   = projection on the head subspace = span of the class-difference directions of W (columns of W centred over the
        classes; rank C - 1). The logits depend on z only through P z (up to a constant added to every class).
Energy split (the four shares sum to 1; mean_i ||d_i||^2 = ||c||^2 + mean_i ||r_i||^2):
  common_in  = ||P c||^2 / E      common_out = ||(I - P) c||^2 / E
  resid_in   = mean ||P r_i||^2 / E   resid_out = mean ||(I - P) r_i||^2 / E         E = mean_i ||d_i||^2
  in_common  = ||P c||^2 / ||c||^2,  in_resid = mean ||P r_i||^2 / mean ||r_i||^2   (chance level = (C - 1) / D)
Logit space (centred over the classes):
  bias_sd  = std over classes of W^T c          (a constant vector added to every image's logits = a change of the bias)
  noise_sd = std over images and classes of W^T r_i
  margin   = mean (top-1 - top-2 logit) on the clean test set, for scale
Also per stream: accuracy drop and signed ECE (mean confidence - accuracy), linear CKA, frac_common.

  ~/REOBench/.venv/bin/python scripts/analysis/head_subspace.py [--models ...] [--datasets aid r45] [--device cuda]
Output: experiments/2026-11-18_head_subspace/<model>_<ds>.json and summary.txt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths  # noqa: E402
import feat_metrics as fm  # noqa: E402

OUT = paths.EXP / "2026-11-18_head_subspace"
ORDER = ["mae_in", "satmae", "scalemae", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16", "dino_in_s8",
         "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16", "sup_in_r50", "sup_in_vitb16",
         "sup_in_vitl16"]
KEYS = ["common_in", "common_out", "resid_in", "resid_out", "in_common", "in_resid", "bias_sd", "noise_sd", "signed_ece", "drop",
        "cka", "frac_common"]


def rank(a):
    a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(len(a)); return r


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def run(model, ds, dev):
    h = torch.load(f"{fm.FEATS}/{model}/head_{ds}.pt", map_location="cpu")
    w, b, mu, sd = (h[k].float().to(dev) for k in ("w", "b", "mu", "sd"))
    te_x, te = fm.load(model, f"{ds}_test")
    c_x, cd = fm.load(model, f"{ds}_c")
    te_x, yt = te_x.to(dev), te["y"].to(dev)
    zt = (te_x - mu) / sd
    D, C = w.shape
    Q = torch.linalg.svd(w - w.mean(1, keepdim=True), full_matrices=False)[0][:, :C - 1]  # orthonormal basis of the head subspace
    lt = zt @ w + b
    t2 = lt.topk(2, 1)[0]
    acc0 = (lt.argmax(1) == yt).float().mean().item()
    res = {"model": model, "dataset": ds, "D": D, "C": C, "chance": (C - 1) / D, "margin": (t2[:, 0] - t2[:, 1]).mean().item(),
           "clean_acc": 100 * acc0, "streams": {}}
    corr, sev, src, yc = cd["corr"], cd["sev"], cd["src"], cd["y"]
    for n, cname in enumerate(common.NOISES):
        for s in range(1, 6):
            sel = (corr == n) & (sev == s)
            xc = c_x[sel].to(dev); sr = src[sel].to(dev); y = yc[sel].to(dev)
            zc = (xc - mu) / sd
            d = zc - zt[sr]
            E = (d ** 2).sum(1).mean()
            if E <= 0:  # corrupted file identical to the clean one
                continue
            c = d.mean(0); r = d - c
            pc, pr = ((c @ Q) ** 2).sum(), ((r @ Q) ** 2).sum(1).mean()
            ec, er = (c ** 2).sum(), (r ** 2).sum(1).mean()
            lb = c @ w; lb = lb - lb.mean()
            ln = r @ w; ln = ln - ln.mean(1, keepdim=True)
            p = (zc @ w + b).softmax(1); cf, pred = p.max(1); ok = (pred == y).float()
            a, bb = te_x[sr] - te_x[sr].mean(0), xc - xc.mean(0)
            res["streams"][f"{cname}_s{s}"] = {
                "common_in": (pc / E).item(), "common_out": ((ec - pc) / E).item(), "resid_in": (pr / E).item(),
                "resid_out": ((er - pr) / E).item(), "in_common": (pc / ec).item(), "in_resid": (pr / er).item(),
                "bias_sd": lb.std().item(), "noise_sd": ln.std().item(), "signed_ece": (cf.mean() - ok.mean()).item(),
                "drop": 100 * (acc0 - ok.mean().item()), "frac_common": (ec / E).item(),
                "cka": ((bb.T @ a).norm() ** 2 / ((a.T @ a).norm() * (bb.T @ bb).norm())).item()}
    return res


def summarize():
    lines = []
    for ds in ("aid", "r45"):
        R = {m: json.load(open(OUT / f"{m}_{ds}.json")) for m in ORDER if (OUT / f"{m}_{ds}.json").exists()}
        if not R:
            continue
        mean = lambda m, k: float(np.mean([v[k] for v in R[m]["streams"].values()]))
        lines += [f"== {ds}: mean over the corrupted streams (energy shares sum to 1)",
                  f"{'model':15s} {'D':>5s} {'chance':>6s} | {'com_in':>6s} {'com_out':>7s} {'res_in':>6s} {'res_out':>7s} | {'in_com':>6s} {'in_res':>6s} | "
                  f"{'margin':>6s} {'bias_sd':>7s} {'noise_sd':>8s} | {'sECE':>6s} {'drop':>5s} {'CKA':>5s}"]
        for m, r in R.items():
            lines.append(f"{m:15s} {r['D']:5d} {r['chance']:6.3f} | {mean(m, 'common_in'):6.3f} {mean(m, 'common_out'):7.3f} "
                         f"{mean(m, 'resid_in'):6.3f} {mean(m, 'resid_out'):7.3f} | {mean(m, 'in_common'):6.3f} {mean(m, 'in_resid'):6.3f} | "
                         f"{r['margin']:6.1f} {mean(m, 'bias_sd'):7.2f} {mean(m, 'noise_sd'):8.2f} | {mean(m, 'signed_ece'):+6.3f} "
                         f"{mean(m, 'drop'):5.1f} {mean(m, 'cka'):5.3f}")
        ms = list(R)
        col = lambda k: [mean(m, k) for m in ms]
        lines.append("across backbones, Spearman with signed ECE: " + ", ".join(
            f"{k} {spear(col('signed_ece'), col(k)):+.2f}" for k in ("bias_sd", "noise_sd", "cka", "frac_common", "common_in", "resid_in")))
        lines.append("across backbones, Spearman with accuracy drop: " + ", ".join(
            f"{k} {spear(col('drop'), col(k)):+.2f}" for k in ("bias_sd", "noise_sd", "cka", "frac_common", "common_in", "resid_in")))
        within = lambda a, k: float(np.mean([spear([v[a] for v in R[m]["streams"].values()], [v[k] for v in R[m]["streams"].values()]) for m in ms]))
        lines.append("within a backbone over its streams (mean of 17 Spearman), with accuracy drop: " + ", ".join(
            f"{k} {within('drop', k):+.2f}" for k in ("bias_sd", "noise_sd", "cka", "frac_common")))
        lines.append("")
    open(OUT / "summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--datasets", nargs="*", default=["aid", "r45"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    fm.dev = dev
    with torch.no_grad():
        for m in args.models:
            for ds in args.datasets:
                f = OUT / f"{m}_{ds}.json"
                if not f.exists():
                    json.dump(run(m, ds, dev), open(f, "w"), indent=1)
                    torch.cuda.empty_cache()
    summarize()
