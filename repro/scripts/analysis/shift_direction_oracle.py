"""Oracle test before any SAE: does removing estimated "shift directions" from the head input recover accuracy?

Uses the saved features and the stage-1 linear heads (feat_metrics.py); the head is kept fixed unless noted.
Space: z = head input standardized with clean-train mean / std (source mean 0, std 1). delta = z(corrupted) - z(clean)
of the same image (pairs via the corrupted set's "src"). Evaluation on the odd-indexed test images only (AID 1,000,
RESISC45 3,150); the even-indexed half is used only by the same-corruption oracles. Every number is the accuracy of a
held-out corruption, mean over its 5 severities.

Directions ("project" = z' = z - z U U^T, i.e. the component along U is reset to the source mean 0):
  pca_loco   : top-k eigenvectors of the uncentred second moment of delta over the OTHER 11 corruptions (all severities,
               both halves) -> the corruption being scored never contributes (leave-one-corruption-out)
  pca_same   : the same from the held-out corruption itself, even half only (upper bound: the shift type is known)
  random     : k random orthonormal directions (seed 0)
  dim_loco   : the k coordinates with the largest mean delta_d^2 over the other 11 corruptions, set to 0
  pca_loco_refit : pca_loco with the head refitted on the projected clean-train features (same lambda); asks whether
               the class information that lives in the removed directions can be replaced by the rest
  pca_loco_align : align_ms first, then pca_loco
Label-free baselines: raw, align_ms / align_m (feat_metrics.py: batches of 64, seed 0), clip3 (z clamped to [-3, 3]).
Mean-shift oracle: mean_same = z - mean(delta of the same corruption and severity, even half).
k in {1, 4, 16, 64}. Clean accuracy after each transform (odd half) is reported too (mean over the 12 folds for LOCO).

  ~/REOBench/.venv/bin/python scripts/analysis/shift_direction_oracle.py [--models ...] [--datasets aid r45]
Output: experiments/2026-11-17_shift_oracle/<model>_<ds>.json and summary.txt
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths  # noqa: E402
from extract_feats import MODELS  # noqa: E402
import feat_metrics as fm  # noqa: E402

OUT = paths.EXP / "2026-11-17_shift_oracle"
dev = torch.device("cuda")
fm.dev = dev
KS = [1, 4, 16, 64]


def topk_eig(m, k):
    ev, vec = torch.linalg.eigh(m)
    return vec[:, ev.argsort(descending=True)[:k]]


def proj(z, u):
    return z - (z @ u) @ u.T


def run(model, ds):
    h = torch.load(f"{fm.FEATS}/{model}/head_{ds}.pt")
    w, b, mu, sd, lam = h["w"].to(dev), h["b"].to(dev), h["mu"].to(dev), h["sd"].to(dev), h["lambda"]
    tr_x, tr = fm.load(model, f"{ds}_train")
    te_x, te = fm.load(model, f"{ds}_test")
    c_x, cd = fm.load(model, f"{ds}_c")
    z = lambda a: (a.to(dev) - mu) / sd
    ztr, ytr = z(tr_x), tr["y"].to(dev)
    zte, yte = z(te_x), te["y"].to(dev)
    D, nc = zte.shape[1], int(ytr.max()) + 1
    acc = lambda logits, y: 100 * (logits.argmax(1) == y).float().mean().item()
    head = lambda a, W=w, B=b: a @ W + B
    odd = torch.arange(len(zte), device=dev) % 2 == 1

    # streams: per corruption n, severity s -> (z_corrupted, y, src)
    corr, sev, src = cd["corr"].to(dev), cd["sev"].to(dev), cd["src"].to(dev)
    zc_all = z(c_x)
    S = {}
    M = torch.zeros(len(common.NOISES), D, D, device=dev)   # uncentred second moment of delta, all images
    Mev = torch.zeros(len(common.NOISES), D, D, device=dev)  # even half only
    cnt, cnt_ev = torch.zeros(len(common.NOISES), device=dev), torch.zeros(len(common.NOISES), device=dev)
    for n in range(len(common.NOISES)):
        for s in range(1, 6):
            sel = (corr == n) & (sev == s)
            zs, sr = zc_all[sel], src[sel]
            d = zs - zte[sr]
            ev = sr % 2 == 0
            M[n] += d.T @ d; cnt[n] += len(d)
            Mev[n] += d[ev].T @ d[ev]; cnt_ev[n] += ev.sum()
            S[(n, s)] = {"z": zs[~ev], "y": cd["y"].to(dev)[sel][~ev], "mean_d_even": d[ev].mean(0)}
    del zc_all
    g = torch.Generator(device="cpu").manual_seed(0)
    R = torch.linalg.qr(torch.randn(D, max(KS), generator=g))[0].to(dev)

    res = {"model": model, "dataset": ds, "D": D, "n_eval_per_stream": int((~(src[(corr == 0) & (sev == 1)] % 2 == 0)).sum()),
           "clean": {}, "held_out": {}}
    zcl, ycl = zte[odd], yte[odd]
    res["clean"]["raw"] = acc(head(zcl), ycl)
    res["clean"]["align_ms"] = acc(head(fm.align(zcl, "ms")), ycl)
    res["clean"]["clip3"] = acc(head(zcl.clamp(-3, 3)), ycl)
    for k in KS:
        res["clean"][f"random_k{k}"] = acc(head(proj(zcl, R[:, :k])), ycl)
    clean_loco = {}
    for n, cname in enumerate(common.NOISES):
        t0 = time.time()
        other = [i for i in range(len(common.NOISES)) if i != n]
        m_loco = M[other].sum(0) / cnt[other].sum()
        m_same = Mev[n] / cnt_ev[n]
        dvar_loco = torch.diagonal(m_loco)
        r = {}

        def score(name, f):
            r[name] = sum(acc(head(f(S[(n, s)]["z"], s)), S[(n, s)]["y"]) for s in range(1, 6)) / 5

        score("raw", lambda a, s: a)
        score("align_ms", lambda a, s: fm.align(a, "ms"))
        score("align_m", lambda a, s: fm.align(a, "m"))
        score("clip3", lambda a, s: a.clamp(-3, 3))
        score("mean_same", lambda a, s: a - S[(n, s)]["mean_d_even"])
        U_loco, U_same = topk_eig(m_loco, max(KS)), topk_eig(m_same, max(KS))
        top_dims = dvar_loco.argsort(descending=True)
        for k in KS:
            ul, us = U_loco[:, :k], U_same[:, :k]
            score(f"pca_loco_k{k}", lambda a, s: proj(a, ul))
            score(f"pca_same_k{k}", lambda a, s: proj(a, us))
            score(f"random_k{k}", lambda a, s: proj(a, R[:, :k]))
            mask = torch.ones(D, device=dev); mask[top_dims[:k]] = 0
            score(f"dim_loco_k{k}", lambda a, s: a * mask)
            score(f"pca_loco_align_k{k}", lambda a, s: proj(fm.align(a, "ms"), ul))
            clean_loco.setdefault(f"pca_loco_k{k}", []).append(acc(head(proj(zcl, ul)), ycl))
            clean_loco.setdefault(f"dim_loco_k{k}", []).append(acc(head(zcl * mask), ycl))
            if k >= 4:
                w2, b2 = fm.fit_head(proj(ztr, ul), ytr, nc, lam)
                r[f"pca_loco_refit_k{k}"] = sum(acc(proj(S[(n, s)]["z"], ul) @ w2 + b2, S[(n, s)]["y"]) for s in range(1, 6)) / 5
                clean_loco.setdefault(f"pca_loco_refit_k{k}", []).append(acc(proj(zcl, ul) @ w2 + b2, ycl))
        res["held_out"][cname] = r
        print(f"  {model} {ds} {cname:24s} raw {r['raw']:5.1f} align {r['align_ms']:5.1f} | pca_loco k16 {r['pca_loco_k16']:5.1f} "
              f"refit k16 {r['pca_loco_refit_k16']:5.1f} | pca_same k16 {r['pca_same_k16']:5.1f} | mean_same {r['mean_same']:5.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    res["clean"].update({k: sum(v) / len(v) for k, v in clean_loco.items()})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--datasets", nargs="*", default=["aid", "r45"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        for ds in args.datasets:
            f = OUT / f"{m}_{ds}.json"
            if f.exists():
                continue
            json.dump(run(m, ds), open(f, "w"), indent=1)
            torch.cuda.empty_cache()
    print("DONE")
