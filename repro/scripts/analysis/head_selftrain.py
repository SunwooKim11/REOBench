"""Feasibility check: how much of the labelled head upper bound (head_oracle.py) does label-free head adaptation reach?

Frozen features, head input aligned per batch of 64 (feat_metrics.align "ms"). The linear head is adapted on the whole
unlabelled stream (transductive), R rounds of: targets from the current head -> a few Adam steps on
cross-entropy(targets) + lam * ||W - W0||^2. Targets:
  st    : hard pseudo-labels (argmax); plain self-training
  bal   : Sinkhorn-balanced soft targets, class proportions forced to the source (clean-train) proportions
  balnf : bal, but only the samples whose prediction does not flip between the raw and the aligned input enter the loss
lam is not selected (no labels): all values of LAMS are reported. Evaluation on the odd-index test images, as head_oracle.py.
Natural shift (RESISC45 <-> UC Merced, 19 classes): head trained on the source train split as in natural_shift.py,
adapted on all target images, evaluated on the target test split; "target head" = head trained on the target train split.

  ~/REOBench/.venv/bin/python scripts/analysis/head_selftrain.py [--models ...]        # from repro/, GPU
Output: experiments/2026-11-24_cluster_cot/head_selftrain/<model>.json, RESULTS_head_selftrain.md (--summary)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths  # noqa: E402
import feat_metrics as fm  # noqa: E402
from head_subspace import ORDER  # noqa: E402

OUT = paths.EXP / "2026-11-24_cluster_cot" / "head_selftrain"
LAMS = [1e-2, 1e-3]
MODES = ("st", "bal", "balnf")
SEVS = (3, 5)


def sinkhorn(P, q, iters=30):
    Q = P.clone()
    for _ in range(iters):
        Q = Q * (q * Q.shape[-2] / Q.sum(-2, keepdim=True).clamp_min(1e-12))   # class totals -> q * n
        Q = Q / Q.sum(-1, keepdim=True)                                       # every sample sums to 1
    return Q


def adapt(Z, keep, w0, b0, q, mode, rounds=4, steps=40, lr=3e-3):
    """Z: (S, n, D) unlabelled, keep: (S, n) bool. Returns heads w (L, S, D, C), b (L, S, 1, C)."""
    L, S = len(LAMS), Z.shape[0]
    lam = torch.tensor(LAMS, device=Z.device).view(L, 1)
    w = w0.repeat(L, S, 1, 1).clone().requires_grad_(True)
    b = b0.repeat(L, S, 1, 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([w, b], lr=lr)
    m = keep.float() if mode == "balnf" else torch.ones_like(keep, dtype=torch.float)
    for _ in range(rounds):
        with torch.no_grad():
            P = (Z @ w + b).softmax(-1)
            T = F.one_hot(P.argmax(-1), P.shape[-1]).float() if mode == "st" else sinkhorn(P, q)
        for _ in range(steps):
            ce = -(T * (Z @ w + b).log_softmax(-1)).sum(-1)                      # (L, S, n)
            loss = ((ce * m).sum(-1) / m.sum(-1).clamp_min(1) + lam * ((w - w0) ** 2).sum((2, 3))).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return w.detach(), b.detach()


def evaluate(Z, Y, ev, w0, b0, q):
    """Z raw (S, n, D). Returns dict of accuracies per stream on the ev subset."""
    Za = torch.stack([fm.align(x, "ms") for x in Z])
    pr, pa = (Z @ w0 + b0).argmax(-1), (Za @ w0 + b0).argmax(-1)
    acc = lambda p: (100 * (p[:, ev] == Y[:, ev]).float().mean(1)).tolist()
    res = {"raw": acc(pr), "align": acc(pa)}
    for mode in MODES:
        w, b = adapt(Za, pr == pa, w0, b0, q, mode)
        a = 100 * ((Za @ w + b).argmax(-1)[:, :, ev] == Y[:, ev]).float().mean(2)   # (L, S)
        npred = torch.stack([torch.stack([torch.tensor(float(len(torch.unique(p)))) for p in pl]) for pl in (Za @ w + b).argmax(-1)])
        for i, l in enumerate(LAMS):
            res[f"{mode}_lam{l:g}"] = a[i].tolist(); res[f"{mode}_lam{l:g}_npred"] = npred[i].tolist()
    return res


def run(model):
    out = {"model": model}
    for ds in ("aid", "r45"):
        h = torch.load(f"{fm.FEATS}/{model}/head_{ds}.pt")
        tr_x, tr = fm.load(model, f"{ds}_train"); te_x, te = fm.load(model, f"{ds}_test"); c_x, cd = fm.load(model, f"{ds}_c")
        w0, b0, mu, sd = h["w"].to(fm.dev), h["b"].to(fm.dev), h["mu"], h["sd"]
        z = lambda a: ((a - mu) / sd).to(fm.dev)
        n, nc = len(te_x), w0.shape[1]
        q = torch.bincount(tr["y"], minlength=nc).float().to(fm.dev); q = q / q.sum()
        names, Z, Y = ["clean"], [z(te_x)], [te["y"].to(fm.dev)]
        for k, cname in enumerate(common.NOISES):
            for s in SEVS:
                sel = (cd["corr"] == k) & (cd["sev"] == s)
                o = cd["src"][sel].argsort()
                names.append(f"{cname}_s{s}"); Z.append(z(c_x[sel][o])); Y.append(cd["y"][sel][o].to(fm.dev))
        Z, Y = torch.stack(Z), torch.stack(Y)
        r = evaluate(Z, Y, torch.arange(n, device=fm.dev) % 2 == 1, w0, b0, q)
        r["streams"] = names
        out[ds] = r
    for src, tgt in (("r45", "ucm"), ("ucm", "r45")):
        f = lambda s: fm.load(model, s)
        tr_x, tr = f(f"es_{src}_train"); va_x, va = f(f"es_{src}_val")
        T = {sp: f(f"es_{tgt}_{sp}") for sp in ("train", "val", "test")}
        mu, sd = tr_x.mean(0), tr_x.std(0) + 1e-6
        z = lambda a: ((a - mu) / sd).to(fm.dev)
        nll = {}
        for lam in fm.LAMBDAS:
            w, b = fm.fit_head(z(tr_x), tr["y"].to(fm.dev), 19, lam)
            nll[lam] = F.cross_entropy(z(va_x) @ w + b, va["y"].to(fm.dev)).item()
        w0, b0 = fm.fit_head(z(tr_x), tr["y"].to(fm.dev), 19, min(nll, key=nll.get))
        q = torch.bincount(tr["y"], minlength=19).float().to(fm.dev); q = q / q.sum()
        Zt = torch.cat([z(T[sp][0]) for sp in ("train", "val", "test")])[None]
        Yt = torch.cat([T[sp][1]["y"] for sp in ("train", "val", "test")]).to(fm.dev)[None]
        ev = torch.zeros(Zt.shape[1], dtype=torch.bool, device=fm.dev); ev[-len(T["test"][0]):] = True
        r = evaluate(Zt, Yt, ev, w0, b0, q)
        # head trained on the target train split (labelled upper bound), same recipe
        mt, st = T["train"][0].mean(0), T["train"][0].std(0) + 1e-6
        zt = lambda a: ((a - mt) / st).to(fm.dev)
        nll = {}
        for lam in fm.LAMBDAS:
            w, b = fm.fit_head(zt(T["train"][0]), T["train"][1]["y"].to(fm.dev), 19, lam)
            nll[lam] = F.cross_entropy(zt(T["val"][0]) @ w + b, T["val"][1]["y"].to(fm.dev)).item()
        w, b = fm.fit_head(zt(T["train"][0]), T["train"][1]["y"].to(fm.dev), 19, min(nll, key=nll.get))
        r["target_head"] = [100 * ((zt(T["test"][0]) @ w + b).argmax(1) == T["test"][1]["y"].to(fm.dev)).float().mean().item()]
        out[f"{src}_to_{tgt}"] = r
    return out


def summary():
    R = {m: json.load(open(OUT / f"{m}.json")) for m in ORDER if (OUT / f"{m}.json").exists()}
    out = ["# 라벨 없는 head 적응의 가능성 확인 (2026-11-24, 수치만)", "",
           "`scripts/analysis/head_selftrain.py`. 정렬된 head 입력 위에서 선형 head만 stream 전체(라벨 없음)로 적응. st = hard pseudo-label,",
           "bal = Sinkhorn으로 클래스 비율을 source 비율에 맞춘 soft target, balnf = bal + 정렬 전후 예측이 같은 표본만 사용. λ = head를 초기값에 묶는 세기(선택하지 않고 전부 표기).",
           "합성 손상: 12종의 3·5단계 24개 stream 평균(홀수 번째 test 이미지). 상한 = `RESULTS_head_oracle.md`의 ft_align(라벨 사용).", ""]
    keys = ["raw", "align"] + [f"{m}_lam{l:g}" for m in MODES for l in LAMS]
    O = paths.EXP / "2026-11-24_cluster_cot" / "head_oracle"
    for ds, title in (("aid", "AID-C"), ("r45", "RESISC45-C"), ("r45_to_ucm", "RESISC45 → UC Merced"), ("ucm_to_r45", "UC Merced → RESISC45")):
        syn = ds in ("aid", "r45")
        out += [f"## {title} (백본 {len(R)}개)", "", "| model | " + " | ".join(keys) + (" | 상한 (라벨) |" if syn else " | target head (라벨) |"), "|---|" + "---:|" * (len(keys) + 1)]
        tab = []
        for m, r in R.items():
            d = r[ds]
            row = [float(np.mean(d[k][1:] if syn else d[k])) for k in keys]
            row.append(float(np.mean(json.load(open(O / f"{m}_{ds}.json"))["ft_align"][1:])) if syn else d["target_head"][0])
            tab.append(row); out.append(f"| {m} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
        out += ["| **평균** | " + " | ".join(f"{v:.1f}" for v in np.mean(tab, 0)) + " |", ""]
        if syn:
            out += [f"clean stream에서의 변화(17개 평균, align 대비): " + ", ".join(f"{k} {np.mean([r[ds][k][0] - r[ds]['align'][0] for r in R.values()]):+.1f}" for k in keys[2:]), ""]
    (OUT.parent / "RESULTS_head_selftrain.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(); sys.exit()
    torch.set_num_threads(2)
    fm.dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        f = OUT / f"{m}.json"
        if f.exists():
            continue
        r = run(m)
        json.dump(r, open(f, "w"))
        g = lambda ds, k: float(np.mean(r[ds][k][1:] if ds in ("aid", "r45") else r[ds][k]))
        print(f"{m:16s} " + " | ".join(f"{ds}: al {g(ds, 'align'):.1f} st {g(ds, 'st_lam0.01'):.1f} bal {g(ds, 'bal_lam0.01'):.1f} balnf {g(ds, 'balnf_lam0.01'):.1f}"
                                         for ds in ("aid", "r45", "r45_to_ucm", "ucm_to_r45")), flush=True)
