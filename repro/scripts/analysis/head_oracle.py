"""How much could a head-level method recover? Labelled upper bound of adapting only the linear head on the frozen features.

Per backbone x dataset x stream (severity 3 and 5 of the 12 corruptions, plus clean): the test images with an even clean
index are "labelled" (2/3 fit, 1/3 to pick the regularisation), the images with an odd clean index are evaluated.
  raw        : stage-1 head, head input z as it is
  align      : head input re-standardised per batch of 64 (feat_metrics.align "ms", whole stream, no labels)
  ft_raw     : head fine-tuned on the labelled half of the corrupted stream (cross-entropy + lam * ||W - W0||^2,
               full-batch Adam 300 steps, lam picked per stream on the held-out third by NLL), input z
  ft_align   : the same on the aligned input
  bias_align : only the class bias b re-fitted on the labelled half, aligned input (what a perfect class-prior /
               logit-offset correction could give)
ft_* use labels of the corrupted stream, so they are upper bounds for label-free head adaptation, not methods.

  ~/REOBench/.venv/bin/python scripts/analysis/head_oracle.py [--models ...] [--datasets aid r45]     # from repro/, GPU
Output: experiments/2026-11-24_cluster_cot/head_oracle/<model>_<ds>.json, RESULTS_head_oracle.md (--summary)
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

OUT = paths.EXP / "2026-11-24_cluster_cot" / "head_oracle"
LAMS = [1e-1, 1e-2, 1e-3, 1e-4]
SEVS = (3, 5)


def finetune(Zf, Yf, Zv, Yv, Ze, Ye, w0, b0, bias_only=False, steps=300, lr=3e-3):
    """Z*: (S, n, D), Y*: (S, n). One head per (lam, stream), trained in parallel. Returns eval accuracy (S,)."""
    L, S = len(LAMS), Zf.shape[0]
    lam = torch.tensor(LAMS, device=Zf.device).view(L, 1)
    w = w0.repeat(L, S, 1, 1).clone().requires_grad_(not bias_only)
    b = b0.repeat(L, S, 1, 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([p for p in (w, b) if p.requires_grad], lr=lr)
    ce = lambda Z, Y: F.cross_entropy((Z @ w + b).flatten(0, 2), Y.repeat(L, 1, 1).flatten(), reduction="none").view(L, S, -1).mean(2)
    for _ in range(steps):
        loss = (ce(Zf, Yf) + lam * ((w - w0) ** 2).sum((2, 3))).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        best = ce(Zv, Yv).argmin(0)                                              # (S,)
        pred = (Ze @ w + b).argmax(-1)                                           # (L, S, n)
        acc = (pred == Ye).float().mean(2)                                       # (L, S)
        return 100 * acc.gather(0, best[None]).squeeze(0)


def run(model, ds):
    h = torch.load(f"{fm.FEATS}/{model}/head_{ds}.pt")
    te_x, te = fm.load(model, f"{ds}_test")
    c_x, cd = fm.load(model, f"{ds}_c")
    w0, b0, mu, sd = h["w"].to(fm.dev), h["b"].to(fm.dev), h["mu"], h["sd"]
    z = lambda a: ((a - mu) / sd).to(fm.dev)
    n = len(te_x)
    names, Z, Y = ["clean"], [z(te_x)], [te["y"].to(fm.dev)]
    for k, cname in enumerate(common.NOISES):
        for s in SEVS:
            sel = (cd["corr"] == k) & (cd["sev"] == s)
            o = cd["src"][sel].argsort()                    # stream in clean-index order
            assert (cd["src"][sel][o] == torch.arange(n)).all()
            names.append(f"{cname}_s{s}"); Z.append(z(c_x[sel][o])); Y.append(cd["y"][sel][o].to(fm.dev))
    Z, Y = torch.stack(Z), torch.stack(Y)
    Za = torch.stack([fm.align(x, "ms") for x in Z])
    idx = torch.arange(n, device=fm.dev)
    ev, lab = idx % 2 == 1, idx % 2 == 0
    fit, val = lab & (idx % 6 != 0), lab & (idx % 6 == 0)
    acc = lambda X: (100 * ((X[:, ev] @ w0 + b0).argmax(-1) == Y[:, ev]).float().mean(1)).tolist()
    res = {"model": model, "dataset": ds, "streams": names, "raw": acc(Z), "align": acc(Za)}
    for key, X, bo in (("ft_raw", Z, False), ("ft_align", Za, False), ("bias_align", Za, True)):
        res[key] = finetune(X[:, fit], Y[:, fit], X[:, val], Y[:, val], X[:, ev], Y[:, ev], w0, b0, bias_only=bo).tolist()
    return res


def summary():
    out = ["# head만 고쳐서 얻을 수 있는 상한 (2026-11-24, 수치만)", "",
           "`scripts/analysis/head_oracle.py`. 손상 12종의 3·5단계 24개 stream 평균 정확도(%), 평가는 clean 번호가 홀수인 test 이미지.",
           "raw = stage-1 head 그대로, align = head 입력 정렬(라벨 없음), ft_raw / ft_align = 손상 stream의 짝수 번째 이미지 **라벨로** head를 미세조정(상한),",
           "bias_align = 정렬 입력에서 클래스 bias만 라벨로 다시 맞춤(상한). clean = 손상 없는 stream에서의 raw.", ""]
    for ds, title in (("aid", "AID-C"), ("r45", "RESISC45-C")):
        R = {m: json.load(open(OUT / f"{m}_{ds}.json")) for m in ORDER if (OUT / f"{m}_{ds}.json").exists()}
        if not R:
            continue
        out += [f"## {title} (백본 {len(R)}개)", "", "| model | clean | raw | align | bias_align | ft_raw | ft_align | ft_align − align | clean − ft_align |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        tab = []
        for m, r in R.items():
            g = lambda k: float(np.mean(r[k][1:]))
            row = [r["raw"][0], g("raw"), g("align"), g("bias_align"), g("ft_raw"), g("ft_align")]
            row += [row[5] - row[2], row[0] - row[5]]
            tab.append(row)
            out.append(f"| {m} | " + " | ".join(f"{v:.1f}" for v in row[:6]) + f" | {row[6]:+.1f} | {row[7]:.1f} |")
        mean = np.mean(tab, 0)
        out += ["| **평균** | " + " | ".join(f"{v:.1f}" for v in mean[:6]) + f" | {mean[6]:+.1f} | {mean[7]:.1f} |", "",
                f"### {title}: 손상별 (17개 백본 평균, 3·5단계 평균)", "", "| 손상 | raw | align | bias_align | ft_raw | ft_align |", "|---|---:|---:|---:|---:|---:|"]
        names = next(iter(R.values()))["streams"]
        for c in common.NOISES:
            ix = [i for i, nme in enumerate(names) if nme.rsplit("_s", 1)[0] == c]
            out.append(f"| {c} | " + " | ".join(f"{np.mean([[r[k][i] for i in ix] for r in R.values()]):.1f}" for k in ("raw", "align", "bias_align", "ft_raw", "ft_align")) + " |")
        out.append("")
    (OUT.parent / "RESULTS_head_oracle.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ORDER)
    ap.add_argument("--datasets", nargs="*", default=["aid", "r45"])
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(); sys.exit()
    torch.set_num_threads(2)
    fm.dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        for m in args.models:
            f = OUT / f"{m}_{ds}.json"
            if f.exists():
                continue
            r = run(m, ds)
            json.dump(r, open(f, "w"))
            g = lambda k: float(np.mean(r[k][1:]))
            print(f"{m:16s} {ds} clean {r['raw'][0]:.1f} | raw {g('raw'):.1f} align {g('align'):.1f} bias {g('bias_align'):.1f} "
                  f"ft_raw {g('ft_raw'):.1f} ft_align {g('ft_align'):.1f}", flush=True)
