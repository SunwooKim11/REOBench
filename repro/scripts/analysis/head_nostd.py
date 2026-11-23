"""Does the gain of head-input alignment depend on the head having been trained on standardized features?

Two linear heads per backbone x dataset, same data, same procedure (multinomial logistic regression, full-batch L-BFGS,
L2 lambda chosen by validation NLL, refit on train; feat_metrics.run), differing only in the head input:
  std : z = (h - mu_src) / sd_src (the stage-1 head; refit here so both heads come from this script)
  raw : h as the backbone gives it (no standardization)
mu_src, sd_src = per-channel mean / std of h over the clean train set.

Alignment per batch of 64 (same order as feat_metrics.align: seed-0 permutation), no learning:
  std head: z_al = (z - mean_batch(z)) / std_batch(z)                       (= (h - mu_test) / sd_test)
  raw head: h'   = (h - mu_test) / sd_test * sd_src + mu_src               (general form; needs mu_src, sd_src)
The two alignments give the same head input up to the fixed affine z <-> h, so any difference in gain comes from the
learned weights (the L2 penalty acts on W in different spaces), not from the alignment.

Lambda grid is wider than feat_metrics.LAMBDAS (1e-10 .. 1e-1) because raw features are on very different scales;
the raw head also gets more L-BFGS iterations (2000 vs 500). A chosen lambda on the grid edge is flagged in the output.

  ~/REOBench/.venv/bin/python scripts/analysis/head_nostd.py [--models ...] [--datasets aid r45]     # from repro/, GPU
Output: experiments/2026-11-23_head_nostd/<model>_<ds>.json, RESULTS.md (summarize with --summary)
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths  # noqa: E402
import feat_metrics as fm  # noqa: E402
from head_subspace import ORDER  # noqa: E402

OUT = paths.EXP / "2026-11-23_head_nostd"
LAMBDAS = [10.0 ** k for k in range(-10, 0)]


def fit(x, y, c, lam, iters):
    w = torch.zeros(x.shape[1], c, device=x.device, requires_grad=True)
    b = torch.zeros(c, device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w + b, y) + lam * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def run(model, ds, dev):
    tr_x, tr = fm.load(model, f"{ds}_train")
    te_x, te = fm.load(model, f"{ds}_test")
    c_x, cd = fm.load(model, f"{ds}_c")
    tr_x, te_x, c_x = tr_x.to(dev), te_x.to(dev), c_x.to(dev)
    mu, sd = tr_x.mean(0), tr_x.std(0) + 1e-6
    ytr = tr["y"].to(dev)
    nc = int(ytr.max()) + 1
    if ds == "aid":  # stratified 10% of train as validation (as feat_metrics.run)
        g = torch.Generator().manual_seed(0)
        val = torch.zeros(len(ytr), dtype=torch.bool)
        for k in range(nc):
            ix = (tr["y"] == k).nonzero().flatten()
            val[ix[torch.randperm(len(ix), generator=g)[: max(1, len(ix) // 10)]]] = True
        val = val.to(dev)
        fit_x, fit_y, v_x, v_y = tr_x[~val], ytr[~val], tr_x[val], ytr[val]
    else:
        v_raw, vd = fm.load(model, "r45_val")
        fit_x, fit_y, v_x, v_y = tr_x, ytr, v_raw.to(dev), vd["y"].to(dev)

    heads = {}
    for kind, tf, iters in (("std", lambda a: (a - mu) / sd, 500), ("raw", lambda a: a, 2000)):
        nll = {}
        for lam in LAMBDAS:
            w, b = fit(tf(fit_x), fit_y, nc, lam, iters)
            nll[lam] = F.cross_entropy(tf(v_x) @ w + b, v_y).item()
        lam = min(LAMBDAS, key=lambda l: nll[l])
        w, b = fit(tf(tr_x), ytr, nc, lam, iters)
        heads[kind] = {"w": w, "b": b, "lambda": lam, "edge": lam in (LAMBDAS[0], LAMBDAS[-1]),
                       "train_acc": 100 * ((tf(tr_x) @ w + b).argmax(1) == ytr).float().mean().item()}

    n = len(te_x)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)
    streams = [("clean", te_x, te["y"])]
    for k, cname in enumerate(common.NOISES):
        for s in range(1, 6):
            sel = (cd["corr"] == k) & (cd["sev"] == s)
            streams.append((f"{cname}_s{s}", c_x[sel.to(dev)], cd["y"][sel]))

    def batch_stats(h):  # per batch of 64 in the permuted order, per channel
        hp = h[perm]
        m = torch.cat([hp[i:i + 64].mean(0, keepdim=True).expand(len(hp[i:i + 64]), -1) for i in range(0, n, 64)])
        s = torch.cat([hp[i:i + 64].std(0, unbiased=False, keepdim=True).expand(len(hp[i:i + 64]), -1) for i in range(0, n, 64)])
        return hp, m, s + 1e-5

    res = {"model": model, "dataset": ds,
           "heads": {k: {kk: v[kk] for kk in ("lambda", "edge", "train_acc")} for k, v in heads.items()}, "streams": {}}
    acc = lambda logits, y: 100 * (logits.argmax(1) == y).float().mean().item()
    for name, h, y in streams:
        assert len(h) == n
        y = y.to(dev)[perm]
        hp, m, s = batch_stats(h)
        Hs, Hr = heads["std"], heads["raw"]
        z = (hp - mu) / sd
        z_al = (hp - m) / s
        h_al = z_al * sd + mu
        res["streams"][name] = {"std_none": acc(z @ Hs["w"] + Hs["b"], y), "std_align": acc(z_al @ Hs["w"] + Hs["b"], y),
                                "raw_none": acc(hp @ Hr["w"] + Hr["b"], y), "raw_align": acc(h_al @ Hr["w"] + Hr["b"], y)}
    return res


def summary():
    import numpy as np
    out = ["# 표준화 없이 학습한 head에서의 정렬 (2026-11-23, 수치만)", "",
           "`scripts/analysis/head_nostd.py`. std = clean train 통계로 표준화한 입력으로 학습한 head(stage-1 방식),",
           "raw = 표준화 없이 백본 출력 h로 학습한 head. 정렬: std는 z_al = (h − μ_test) / σ_test, raw는 h′ = z_al · σ_src + μ_src.",
           "손상 = 12종 × 5단계 60개 stream 평균 정확도(%), 배치 64, 학습 없음. λ는 검증 NLL로 선택(격자 1e-10 ~ 1e-1, * = 격자 끝).", ""]
    for ds, title in (("aid", "AID / AID-C"), ("r45", "RESISC45 / RESISC45-C")):
        R = {m: json.load(open(OUT / f"{m}_{ds}.json")) for m in ORDER if (OUT / f"{m}_{ds}.json").exists()}
        if not R:
            continue
        out += [f"## {title} (백본 {len(R)}개)", "",
                "| model | λ std | λ raw | clean std | clean raw | none std | align std | 이득 std | none raw | align raw | 이득 raw |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        rows = []
        for m, r in R.items():
            S = r["streams"]
            cor = [k for k in S if k != "clean"]
            a = {k: float(np.mean([S[s][k] for s in cor])) for k in ("std_none", "std_align", "raw_none", "raw_align")}
            rows.append([S["clean"]["std_none"], S["clean"]["raw_none"], a["std_none"], a["std_align"], a["std_align"] - a["std_none"],
                         a["raw_none"], a["raw_align"], a["raw_align"] - a["raw_none"]])
            lam = lambda k: f"{r['heads'][k]['lambda']:.0e}" + ("*" if r["heads"][k]["edge"] else "")
            out.append(f"| {m} | {lam('std')} | {lam('raw')} | " + " | ".join(
                f"{v:+.1f}" if i in (4, 7) else f"{v:.1f}" for i, v in enumerate(rows[-1])) + " |")
        mean = np.mean(rows, 0)
        out.append("| **평균** | | | " + " | ".join(f"{v:+.1f}" if i in (4, 7) else f"{v:.1f}" for i, v in enumerate(mean)) + " |")
        g = np.array(rows)
        out += ["", f"raw 이득 > 0인 백본 {int((g[:, 7] > 0.05).sum())}/{len(g)}, raw 이득 < std 이득인 백본 "
                f"{int((g[:, 7] < g[:, 4] - 0.05).sum())}/{len(g)}.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--datasets", nargs="*", default=["aid", "r45"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        sys.exit()
    torch.set_num_threads(2)
    dev = torch.device(args.device)
    fm.dev = dev
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        for m in args.models:
            f = OUT / f"{m}_{ds}.json"
            if f.exists():
                continue
            r = run(m, ds, dev)
            json.dump(r, open(f, "w"), indent=1)
            a = lambda k: sum(v[k] for s, v in r["streams"].items() if s != "clean") / 60
            print(f"{m:16s} {ds}  lam std {r['heads']['std']['lambda']:.0e} raw {r['heads']['raw']['lambda']:.0e}  "
                  f"clean {r['streams']['clean']['std_none']:.1f} / {r['streams']['clean']['raw_none']:.1f}  "
                  f"std {a('std_none'):.1f} -> {a('std_align'):.1f}  raw {a('raw_none'):.1f} -> {a('raw_align'):.1f}", flush=True)
