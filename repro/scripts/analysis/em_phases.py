"""Two phases of entropy minimisation (Entropy Enigma, Press et al. 2024, section 3) at the feature level: does accuracy
first rise and then fall, and do Silhouette score / Shift distance behave as the paper says (clusters keep getting
tighter; in the failing phase they move away from the source class means)? Compared for tent and DSBR.

Setting = scripts/analysis/feat_tta.py with a long run: per stream one affine on the head input, z' = gamma * z + beta
(for ViT = the affine of the final LayerNorm), head fixed, Adam, batches of 64 in the seed-0 order, the stream repeated
until --steps updates (default 2,000; AID stream = 2,000 images = 32 updates per pass). Input = z (not batch-aligned).
  tent : mean entropy          dsbr : entropy weighted by 1 / (C * P[predicted class]), P = EMA 0.9 of predicted frequencies
At the checkpoints (--ckpts) the current affine is applied to the whole stream:
  acc, mean entropy, n_pred (classes predicted at least once),
  sil_pred  : Silhouette score with the predictions as clusters,
  shift_pred: mean distance between the mean of the samples predicted as c and the clean-train mean of class c (fixed, z space)
and at step 0, at the accuracy peak and at the end also the k-means versions of the paper (k = C, best of 2 runs):
  sil_km, shift_km (centroids matched to the clean-train class means with the Hungarian method).
Phase 1 = step 0 -> accuracy peak, phase 2 = peak -> end (per stream, as in the paper's Figure 4).

  /opt/conda/envs/segtta/bin/python scripts/analysis/em_phases.py [--models ...] [--datasets aid] [--lr 1e-3]   # repro/, GPU
  /opt/conda/envs/segtta/bin/python scripts/analysis/em_phases.py --summary
Output: experiments/2026-11-24_cluster_cot/em_phases/<model>_<ds>.json, RESULTS_em_phases.md
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as lsa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths  # noqa: E402
from cluster_cot import FEATS, ORDER, kmeans, silhouette, spear  # noqa: E402

OUT = paths.EXP / "2026-11-24_cluster_cot" / "em_phases"
CKPTS = [0, 10, 20, 50, 100, 200, 500, 1000, 2000]


def entropy(logits):
    lp = logits.log_softmax(-1)
    return -(lp.exp() * lp).sum(-1)


def adapt(Z, w, b, loss, lr, bs, ckpts):
    S, N, D = Z.shape
    C = w.shape[1]
    g = torch.ones(S, 1, D, device=Z.device, requires_grad=True)
    be = torch.zeros(S, 1, D, device=Z.device, requires_grad=True)
    opt = torch.optim.Adam([g, be], lr=lr)
    P = torch.full((S, C), 1.0 / C, device=Z.device)
    saved, i = {}, 0
    for t in range(max(ckpts) + 1):
        if t in ckpts:
            saved[t] = (g.detach().clone(), be.detach().clone())
        if t == max(ckpts):
            break
        zb = Z[:, i:i + bs]
        i = i + bs if i + bs < N else 0
        logits = (g * zb + be) @ w + b
        ent = entropy(logits)
        if loss == "dsbr":
            with torch.no_grad():
                pred = logits.argmax(-1)
                P = 0.9 * P + 0.1 * F.one_hot(pred, C).float().mean(1)
                wgt = 1 / (C * P.gather(1, pred))
            L = (ent * wgt).mean(1).sum()
        else:
            L = ent.mean(1).sum()
        opt.zero_grad()
        L.backward()
        opt.step()
    return saved


def run(model, ds, dev, lr, bs, ckpts):
    h = torch.load(f"{FEATS}/{model}/head_{ds}.pt")
    tr = torch.load(f"{FEATS}/{model}/{ds}_train.pt")
    te = torch.load(f"{FEATS}/{model}/{ds}_test.pt")
    cd = torch.load(f"{FEATS}/{model}/{ds}_c.pt")
    w, b, mu, sd = h["w"].to(dev), h["b"].to(dev), h["mu"], h["sd"]
    nc = w.shape[1]
    ztr, ytr = ((tr["feats"].float() - mu) / sd).to(dev).double(), tr["y"].to(dev)
    cm = torch.stack([ztr[ytr == k].mean(0) for k in range(nc)])
    n = len(te["y"])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    names, Z, Y = ["clean"], [((te["feats"].float() - mu) / sd)[perm]], [te["y"][perm]]
    for k, cname in enumerate(common.NOISES):
        for s in range(1, 6):
            sel = (cd["corr"] == k) & (cd["sev"] == s)
            names.append(f"{cname}_s{s}")
            Z.append(((cd["feats"][sel].float() - mu) / sd)[perm]); Y.append(cd["y"][sel][perm])
    Z, Y = torch.stack(Z).to(dev), torch.stack(Y).to(dev)
    res = {"model": model, "dataset": ds, "lr": lr, "ckpts": ckpts, "streams": names, "losses": {}}
    for loss in ("tent", "dsbr"):
        saved = adapt(Z, w, b, loss, lr, bs, ckpts)
        R = {k: [] for k in ("acc", "ent", "n_pred", "sil_pred", "shift_pred", "peak", "km")}
        with torch.no_grad():
            for s in range(len(names)):
                row = {k: [] for k in ("acc", "ent", "n_pred", "sil_pred", "shift_pred")}
                xs = {}
                for t in ckpts:
                    g, be = saved[t]
                    x = (g[s] * Z[s] + be[s]).double()
                    lg = x.float() @ w + b
                    pred = lg.argmax(1)
                    xc = x - x.mean(0)
                    D = torch.cdist(xc, xc)
                    present = [k for k in range(nc) if (pred == k).any()]
                    row["acc"].append(100 * (pred == Y[s]).float().mean().item())
                    row["ent"].append(entropy(lg).mean().item())
                    row["n_pred"].append(len(present))
                    row["sil_pred"].append(silhouette(D, pred, nc))
                    row["shift_pred"].append(float(np.mean([(x[pred == k].mean(0) - cm[k]).norm().item() for k in present])))
                    xs[t] = t
                peak = int(np.argmax(row["acc"]))
                km = {}
                for name, j in (("start", 0), ("peak", peak), ("end", len(ckpts) - 1)):
                    g, be = saved[ckpts[j]]
                    x = (g[s] * Z[s] + be[s]).double()
                    xc = x - x.mean(0)
                    lab, c, _ = min((kmeans(x, nc, sd_) for sd_ in range(2)), key=lambda t_: t_[2])
                    dist = torch.cdist(c, cm).cpu().numpy()
                    i, jj = lsa(dist)
                    km[name] = {"sil_km": silhouette(torch.cdist(xc, xc), lab, nc), "shift_km": float(dist[i, jj].mean())}
                for k in row:
                    R[k].append(row[k])
                R["peak"].append(peak); R["km"].append(km)
        res["losses"][loss] = R
    return res


def summary():
    out = ["# Entropy minimisation의 두 단계: feature 단계 재현 (2026-11-24, 수치만)", "",
           "`scripts/analysis/em_phases.py`. head 입력의 affine(z′ = γz + β)만 학습, Adam, 배치 64, stream을 반복해 2,000 step.",
           "checkpoint(0, 10, 20, 50, 100, 200, 500, 1000, 2000 step)에서 stream 전체의 정확도를 재고, 정확도 최고점을 기준으로",
           "1단계(0 → 최고점), 2단계(최고점 → 끝)를 나눈다. 60개 손상 stream 평균. Δ = 단계 끝 − 단계 시작.",
           "sil = Silhouette score, shift = source 클래스 평균까지의 거리. _km = k-means 기반(논문 정의), _pred = 예측 클래스 기반.", ""]
    for ds, title in (("aid", "AID-C"), ("r45", "RESISC45-C")):
        R = {m: json.load(open(OUT / f"{m}_{ds}.json")) for m in ORDER if (OUT / f"{m}_{ds}.json").exists()}
        if not R:
            continue
        for loss in ("tent", "dsbr"):
            out += [f"## {title}, {loss} (lr {next(iter(R.values()))['lr']:g}, 백본 {len(R)}개)", "",
                    "| model | acc 시작 | acc 최고 | acc 끝 | 최고점 step(중앙값) | 끝 n_pred | Δsil_km 1단계 | Δsil_km 2단계 | Δshift_km 1단계 | Δshift_km 2단계 "
                    "| Δsil_pred 1단계 | Δsil_pred 2단계 | Δshift_pred 1단계 | Δshift_pred 2단계 | ρ(Δsil_km, Δacc) 1단계 | ρ(Δshift_km, Δacc) 2단계 |",
                    "|---|" + "---:|" * 15]
            tab = []
            for m, r in R.items():
                L = r["losses"][loss]; ck = r["ckpts"]
                idx = range(1, len(r["streams"]))
                a = np.array([L["acc"][s] for s in idx]); pk = np.array([L["peak"][s] for s in idx])
                g = lambda key: np.array([L[key][s] for s in idx], float)
                at = lambda A: A[np.arange(len(pk)), pk]
                km = lambda key, w_: np.array([L["km"][s][w_][key] for s in idx])
                sp, hp = g("sil_pred"), g("shift_pred")
                d1a, d2a = at(a) - a[:, 0], a[:, -1] - at(a)
                d1s, d2s = km("sil_km", "peak") - km("sil_km", "start"), km("sil_km", "end") - km("sil_km", "peak")
                d1h, d2h = km("shift_km", "peak") - km("shift_km", "start"), km("shift_km", "end") - km("shift_km", "peak")
                rho = lambda x, y: spear(x, y) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
                tab.append([a[:, 0].mean(), at(a).mean(), a[:, -1].mean(), float(np.median([ck[p] for p in pk])),
                            g("n_pred")[:, -1].mean(), d1s.mean(), d2s.mean(), d1h.mean(), d2h.mean(),
                            (at(sp) - sp[:, 0]).mean(), (sp[:, -1] - at(sp)).mean(), (at(hp) - hp[:, 0]).mean(),
                            (hp[:, -1] - at(hp)).mean(), rho(d1s, d1a), rho(d2h, d2a)])
                fmt = lambda i, v: (f"{v:.1f}" if i < 3 else f"{v:.0f}" if i == 3 else f"{v:.1f}" if i == 4 else
                                    f"{v:+.3f}" if i in (5, 6, 9, 10) else f"{v:+.2f}")
                out.append(f"| {m} | " + " | ".join(fmt(i, v) for i, v in enumerate(tab[-1])) + " |")
            out += ["| **평균** | " + " | ".join(fmt(i, v) for i, v in enumerate(np.nanmean(tab, 0))) + " |", ""]
    (OUT.parent / "RESULTS_em_phases.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--datasets", nargs="*", default=["aid"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ckpts", nargs="*", type=int, default=CKPTS)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        sys.exit()
    torch.set_num_threads(1)
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        for m in args.models:
            f = OUT / f"{m}_{ds}.json"
            if f.exists():
                continue
            t0 = time.time()
            r = run(m, ds, dev, args.lr, args.bs, args.ckpts)
            json.dump(r, open(f, "w"))
            for loss, L in r["losses"].items():
                a = np.array(L["acc"][1:])
                print(f"{m:16s} {ds} {loss:5s} acc by ckpt " + " ".join(f"{v:.1f}" for v in a.mean(0)) +
                      f" | n_pred end {np.mean([x[-1] for x in L['n_pred'][1:]]):.1f} | {time.time() - t0:.0f}s", flush=True)
