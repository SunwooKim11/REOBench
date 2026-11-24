"""Cluster geometry of the head input (Entropy Enigma, Press et al. 2024) and the COT accuracy estimate (Lu et al. 2023)
on the saved features, per backbone x dataset x stream (clean test, 12 corruptions x 5 severities). No learning.

Space: z = (h - mu_src) / sd_src, the input of the stage-1 head (head_<ds>.pt). All cluster metrics are computed on the
whole stream; compare them across streams of one backbone, not across backbones (dimension and scale differ).
  sil_km     : Silhouette score (Euclidean) of k-means clusters, k = number of classes, best of 3 k-means++ runs
  sil_pred   : Silhouette score with the head's predictions as the clusters (no k-means)
  km_acc     : accuracy (%) of the k-means clusters after the best one-to-one matching to the true labels (needs labels;
               tells whether the classes are separate clusters at all)
  shift_km   : "Shift distance" of the paper: k-means centroids matched (Hungarian) to the clean-train class means,
               mean Euclidean distance of the matched pairs
  shift_pred : mean distance between the mean of the samples predicted as class c and the clean-train mean of class c
  sil_km_h, sil_pred_h, km_acc_h : the same in the raw backbone output h (no standardization; the paper's space)
COT (official code luyuzhe111/COT, run_estimation.py): exact optimal transport between the n softmax vectors of the stream
and n one-hot labels drawn in the source label proportions, cost = ||p - onehot||_1 / 2 = 1 - p[label]; the estimated error
is the mean matched cost. Source proportions = clean-train label frequencies. Variants:
  cot    : head as it is             cot_ts : temperature fitted on the labelled clean test stream (the paper calibrates
  cot_al : head input aligned to source statistics (batches of 64, as feat_metrics.align "ms"), estimates acc_al
Also stored for checking against labelfree_signals.py: acc, acc_al, flip.

Needs scipy (linear_sum_assignment): run with the segtta env.
  /opt/conda/envs/segtta/bin/python scripts/analysis/cluster_cot.py [--models ...] [--datasets aid r45]   # from repro/, GPU
  /opt/conda/envs/segtta/bin/python scripts/analysis/cluster_cot.py --summary
Output: experiments/2026-11-24_cluster_cot/<model>_<ds>.json, RESULTS.md
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
from reobench_tta import common, paths  # noqa: E402

FEATS = "/data/rs/feats"
OUT = paths.EXP / "2026-11-24_cluster_cot"
SIG = paths.EXP / "2026-11-16_backbone_feats" / "signals"
ORDER = ["mae_in", "satmae", "scalemae", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16", "dino_in_s8",
         "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16", "sup_in_r50", "sup_in_vitb16",
         "sup_in_vitl16"]


def align(z, bs=64):  # feat_metrics.align(z, "ms")
    order = torch.randperm(len(z), generator=torch.Generator().manual_seed(0)).to(z.device)
    out = torch.empty_like(z)
    for i in range(0, len(z), bs):
        idx = order[i:i + bs]
        out[idx] = (z[idx] - z[idx].mean(0)) / (z[idx].std(0, unbiased=False) + 1e-5)
    return out


def kmeans(x, k, seed, iters=100):
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randint(len(x), (1,), generator=g)]
    d2 = ((x - c) ** 2).sum(1)
    for _ in range(k - 1):  # k-means++
        i = torch.multinomial((d2 / d2.sum()).cpu(), 1, generator=g).item()
        c = torch.cat([c, x[i:i + 1]])
        d2 = torch.minimum(d2, ((x - x[i]) ** 2).sum(1))
    lab = None
    for _ in range(iters):
        new = torch.cdist(x, c).argmin(1)
        if lab is not None and (new == lab).all():
            break
        lab = new
        oh = F.one_hot(lab, k).to(x.dtype)
        cnt = oh.sum(0)
        c = torch.where(cnt[:, None] > 0, (oh.T @ x) / cnt.clamp(min=1)[:, None], c)
    return lab, c, ((x - c[lab]) ** 2).sum().item()


def silhouette(D, lab, k):
    oh = F.one_hot(lab, k).to(D.dtype)
    cnt = oh.sum(0)
    S = D @ oh
    own = cnt[lab]
    a = S.gather(1, lab[:, None]).squeeze(1) / (own - 1).clamp(min=1)
    Sm = S / cnt.clamp(min=1)
    Sm[:, cnt == 0] = float("inf")
    Sm.scatter_(1, lab[:, None], float("inf"))
    b = Sm.min(1).values
    s = (b - a) / torch.maximum(a, b)
    s[(own <= 1) | ~torch.isfinite(s)] = 0
    return s.mean().item()


def cot(p, q):  # p: n x C softmax (numpy), q: source label proportions -> estimated accuracy (%)
    n = len(p)
    cnt = np.floor(q * n).astype(int)
    cnt[np.argsort(-(q * n - cnt))[: n - cnt.sum()]] += 1  # remainder to the largest fractional parts
    M = 1 - p[:, np.repeat(np.arange(len(q)), cnt)]
    r, c = lsa(M)
    return 100 * (1 - M[r, c].mean())


def run(model, ds, dev):
    h = torch.load(f"{FEATS}/{model}/head_{ds}.pt")
    tr = torch.load(f"{FEATS}/{model}/{ds}_train.pt")
    te = torch.load(f"{FEATS}/{model}/{ds}_test.pt")
    cd = torch.load(f"{FEATS}/{model}/{ds}_c.pt")
    w, b, mu, sd = h["w"].to(dev), h["b"].to(dev), h["mu"], h["sd"]
    z = lambda a: ((a.float() - mu) / sd).to(dev)
    nc = w.shape[1]
    ztr, ytr = z(tr["feats"]).double(), tr["y"].to(dev)
    cm = torch.stack([ztr[ytr == k].mean(0) for k in range(nc)])  # clean-train class means
    q = np.bincount(tr["y"].numpy(), minlength=nc) / len(tr["y"])
    zte, yte = z(te["feats"]), te["y"].to(dev)
    lg = zte @ w + b  # temperature on the labelled clean test stream
    Ts = torch.exp(torch.linspace(-2, 2, 401, device=dev))
    T = Ts[torch.stack([F.cross_entropy(lg / t, yte) for t in Ts]).argmin()].item()

    def one(zs, ys):
        za = align(zs)
        lg, lga = zs @ w + b, za @ w + b
        pred, preda = lg.argmax(1), lga.argmax(1)
        r = {"acc": 100 * (pred == ys).float().mean().item(), "acc_al": 100 * (preda == ys).float().mean().item(),
             "flip": 100 * (pred != preda).float().mean().item()}
        for k, l in (("cot", lg), ("cot_ts", lg / T), ("cot_al", lga)):
            r[k] = cot(l.double().softmax(1).cpu().numpy(), q)
        x = zs.double()
        D = torch.cdist(x - x.mean(0), x - x.mean(0))
        lab, c, _ = min((kmeans(x, nc, s) for s in range(3)), key=lambda t: t[2])
        r["sil_km"], r["sil_pred"] = silhouette(D, lab, nc), silhouette(D, pred, nc)
        cont = (F.one_hot(lab, nc).double().T @ F.one_hot(ys, nc).double()).cpu().numpy()
        i, j = lsa(-cont)
        r["km_acc"] = 100 * cont[i, j].sum() / len(ys)
        dist = torch.cdist(c, cm).cpu().numpy()
        i, j = lsa(dist)
        r["shift_km"] = float(dist[i, j].mean())
        present = [k for k in range(nc) if (pred == k).any()]
        r["shift_pred"] = float(np.mean([(x[pred == k].mean(0) - cm[k]).norm().item() for k in present]))
        r["n_pred_classes"] = len(present)
        x = x * sd.to(dev).double()  # raw backbone output h (up to the mean), the space the paper clusters in
        D = torch.cdist(x - x.mean(0), x - x.mean(0))
        lab, c, _ = min((kmeans(x, nc, s) for s in range(3)), key=lambda t: t[2])
        r["sil_km_h"], r["sil_pred_h"] = silhouette(D, lab, nc), silhouette(D, pred, nc)
        cont = (F.one_hot(lab, nc).double().T @ F.one_hot(ys, nc).double()).cpu().numpy()
        i, j = lsa(-cont)
        r["km_acc_h"] = 100 * cont[i, j].sum() / len(ys)
        return r

    res = {"model": model, "dataset": ds, "T": T, "class_mean_dist": torch.pdist(cm).mean().item(), "streams": {}}
    res["streams"]["clean"] = one(zte, yte)
    for n, cname in enumerate(common.NOISES):
        for s in range(1, 6):
            sel = (cd["corr"] == n) & (cd["sev"] == s)
            res["streams"][f"{cname}_s{s}"] = one(z(cd["feats"][sel]), cd["y"][sel].to(dev))
    return res


def rank(a):
    a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(len(a))
    for v in np.unique(a):
        r[a == v] = r[a == v].mean()
    return r


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def summary():
    out = ["# Cluster geometry와 COT (2026-11-24, 수치만)", "",
           "`scripts/analysis/cluster_cot.py`. 공간 = stage-1 head 입력 z. 손상 = 12종 × 5단계 60개 stream. 학습 없음.",
           "COT = exact optimal transport(비용 1 − p[label], source 라벨 비율 = clean train). COT_ts = clean test에서 맞춘 temperature 적용.",
           "AC / DoC / ATC / flip은 `2026-11-16_backbone_feats/signals/`의 값. flip 추정 = ATC_al − k × (flip − clean flip), k는 다른 16개 백본으로 적합.", ""]
    data = {}
    for ds in ("aid", "r45"):
        for m in ORDER:
            if (OUT / f"{m}_{ds}.json").exists():
                data[m, ds] = (json.load(open(OUT / f"{m}_{ds}.json")), json.load(open(SIG / f"{m}_{ds}.json"))["streams"])
    keys = lambda S: [k for k in S if k != "clean"]
    # leave-one-backbone-out flip estimate, pooled over both datasets (as eval_labelfree.py)
    rows = [(m, ds, k, G[k]["acc"], G[k]["acc_al"], max(G[k]["flip"] - G["clean"]["flip"], 0), G[k]["atc_al"])
            for (m, ds), (_, G) in data.items() for k in keys(G)]
    ms = np.array([r[0] for r in rows]); acc, al, fl, atc_al = (np.array([r[i] for r in rows], float) for i in (3, 4, 5, 6))
    est = {}
    for m in set(ms):
        te = ms == m
        kk = np.linalg.lstsq(fl[~te, None], (al - acc)[~te], rcond=None)[0][0]
        for i in np.where(te)[0]:
            est[rows[i][0], rows[i][1], rows[i][2]] = atc_al[i] - kk * fl[i]
    pooled = {k: [] for k in ("ac", "doc", "atc", "cot", "cot_ts", "flipest", "atc_al", "cot_al")}
    for ds, title in (("aid", "AID-C"), ("r45", "RESISC45-C")):
        M = [m for m in ORDER if (m, ds) in data]
        if not M:
            continue
        out += [f"## {title}: 정확도 추정의 평균 절대 오차 (pt, 60개 stream, 백본 {len(M)}개)", "",
                "| model | 정확도 | AC | DoC | ATC | COT | COT_ts | flip 추정 | 정렬 후: ATC_al | 정렬 후: COT_al |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        tab = []
        for m in M:
            R, G = data[m, ds]; S = R["streams"]; K = keys(S)
            a = np.array([G[k]["acc"] for k in K]); aa = np.array([G[k]["acc_al"] for k in K])
            e = {"ac": [G[k]["ac"] for k in K] - a, "doc": [G[k]["doc"] for k in K] - a, "atc": [G[k]["atc"] for k in K] - a,
                 "cot": [S[k]["cot"] for k in K] - a, "cot_ts": [S[k]["cot_ts"] for k in K] - a,
                 "flipest": [est[m, ds, k] for k in K] - a, "atc_al": [G[k]["atc_al"] for k in K] - aa,
                 "cot_al": [S[k]["cot_al"] for k in K] - aa}
            for k in pooled:
                pooled[k] += list(e[k])
            tab.append([a.mean()] + [np.abs(e[k]).mean() for k in pooled])
            out.append(f"| {m} | " + " | ".join(f"{v:.1f}" for v in tab[-1]) + " |")
        out += ["| **평균** | " + " | ".join(f"{v:.1f}" for v in np.mean(tab, 0)) + " |", "",
                f"## {title}: 60개 stream에서 정확도와의 Spearman (부호는 그대로), clean stream의 값", "",
                "| model | conf | flip | sil_km | sil_pred | shift_km | shift_pred | clean sil_km | clean sil_pred | clean km_acc | clean acc | 최저 sil_km |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        tab = []
        for m in M:
            R, G = data[m, ds]; S = R["streams"]; K = keys(S)
            a = [G[k]["acc"] for k in K]
            tab.append([spear([G[k]["ac"] for k in K], a), spear([G[k]["flip"] for k in K], a)]
                       + [spear([S[k][c] for k in K], a) for c in ("sil_km", "sil_pred", "shift_km", "shift_pred")]
                       + [S["clean"]["sil_km"], S["clean"]["sil_pred"], S["clean"]["km_acc"], S["clean"]["acc"],
                          min(S[k]["sil_km"] for k in K)])
            out.append(f"| {m} | " + " | ".join(f"{v:+.2f}" if i < 6 else (f"{v:.3f}" if i in (6, 7, 10) else f"{v:.1f}")
                                                for i, v in enumerate(tab[-1])) + " |")
        mean = np.mean(tab, 0)
        out += ["| **평균** | " + " | ".join(f"{v:+.2f}" if i < 6 else (f"{v:.3f}" if i in (6, 7, 10) else f"{v:.1f}")
                                           for i, v in enumerate(mean)) + " |", ""]
    if pooled["ac"]:
        out += [f"## 전체 {len(pooled['ac'])}개 stream", "", "| 추정 | 평균 절대 오차 | 중앙값 | > 5pt (%) | > 10pt (%) | 최대 | 편향 |", "|---|---:|---:|---:|---:|---:|---:|"]
        names = {"ac": "AC", "doc": "DoC", "atc": "ATC", "cot": "COT", "cot_ts": "COT_ts", "flipest": "flip 추정 (ATC_al − k·flip)",
                 "atc_al": "ATC_al → 정렬 후 정확도", "cot_al": "COT_al → 정렬 후 정확도"}
        for k, n in names.items():
            e = np.array(pooled[k]); v = np.abs(e)
            out.append(f"| {n} | {v.mean():.1f} | {np.median(v):.1f} | {100 * (v > 5).mean():.1f} | {100 * (v > 10).mean():.1f} | {v.max():.1f} | {e.mean():+.1f} |")
    (OUT / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--datasets", nargs="*", default=["aid", "r45"])
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
            r = run(m, ds, dev)
            json.dump(r, open(f, "w"), indent=1)
            G = json.load(open(SIG / f"{m}_{ds}.json"))["streams"]
            dev_ = max(max(abs(r["streams"][k][c] - G[k][c]) for c in ("acc", "acc_al", "flip")) for k in G)
            S = r["streams"]; K = [k for k in S if k != "clean"]
            mae = lambda c: np.mean([abs(S[k][c] - S[k]["acc"]) for k in K])
            print(f"{m:16s} {ds} T {r['T']:.2f} clean sil {S['clean']['sil_km']:.3f} km_acc {S['clean']['km_acc']:.1f} | "
                  f"COT err {mae('cot'):.1f} ts {mae('cot_ts'):.1f} | max diff vs signals {dev_:.2f} | {time.time() - t0:.0f}s", flush=True)
