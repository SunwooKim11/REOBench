"""How large is the prediction bias (DSBR, arXiv 2606.02339) on AID-C / RESISC45-C, how much of it does head-input alignment
remove, and do the gains of DSBR follow it?

Saved features and stage-1 heads; streams and order as flip_filter_tta.py (seed-0 permutation, batches of 64).
Per stream, with q = predicted class distribution and p = true class distribution of the stream:
  tv  = 0.5 * sum_k |q_k - p_k|     prediction bias (0 = the predicted class shares equal the true ones)
  top = max_k q_k                   share of the most frequent predicted class
for none (head input as is) and align_ms (head input re-standardised per batch).
Accuracy of none / tent / dsbr / align_ms comes from experiments/2026-11-19_dsbr/feat_bs64 (feature-level TTA, lr 0.01).

  ~/REOBench/.venv/bin/python scripts/analysis/pred_bias.py [--device cuda]      # from repro/
Output: experiments/2026-11-19_dsbr/pred_bias.json and pred_bias.md
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
from flip_filter_tta import batch_align  # noqa: E402
from head_subspace import ORDER, spear  # noqa: E402

D = paths.EXP / "2026-11-19_dsbr"
LR = "0.01"


def bias(pred, y, C):
    q, p = torch.bincount(pred, minlength=C).float() / len(pred), torch.bincount(y, minlength=C).float() / len(y)
    return 0.5 * (q - p).abs().sum().item(), q.max().item()


def run(model, ds, dev, bs=64):
    h = torch.load(f"{fm.FEATS}/{model}/head_{ds}.pt", map_location="cpu")
    w, b, mu, sd = (h[k].float().to(dev) for k in ("w", "b", "mu", "sd"))
    te_x, te = fm.load(model, f"{ds}_test")
    c_x, cd = fm.load(model, f"{ds}_c")
    n, C = len(te_x), w.shape[1]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    streams = [("clean", te_x, te["y"])]
    for k, cname in enumerate(common.NOISES):
        for s in range(1, 6):
            sel = (cd["corr"] == k) & (cd["sev"] == s)
            streams.append((f"{cname}_s{s}", c_x[sel], cd["y"][sel]))
    R = json.load(open(D / "feat_bs64" / f"{model}_{ds}.json"))
    out = {}
    for i, (name, x, y) in enumerate(streams):
        z, y = ((x[perm].to(dev) - mu) / sd), y[perm].to(dev)
        za = torch.cat([batch_align(z[None, j:j + bs])[0] for j in range(0, n, bs)])
        tv0, top0 = bias((z @ w + b).argmax(1), y, C)
        tv1, top1 = bias((za @ w + b).argmax(1), y, C)
        acc = lambda k: R["methods"][k]["m"][i]["acc"]
        out[name] = {"tv_none": tv0, "top_none": top0, "tv_align": tv1, "top_align": top1, "acc_none": acc("none"),
                     "acc_align": acc("align_ms"), "acc_tent": acc(f"raw_tent_lr{LR}"), "acc_dsbr": acc(f"raw_dsbr_lr{LR}")}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_num_threads(2)
    dev = torch.device(args.device)
    fm.dev = dev
    res, md = {}, ["# 예측 쏠림(prediction bias)의 크기와 DSBR·정렬의 이득 (2026-11-19, 수치만)", "",
                   "tv = 0.5 Σ|예측 클래스 비율 − 실제 클래스 비율| (%, 0 = 쏠림 없음), top = 최다 예측 클래스 비율 (%).",
                   "none = head 입력 그대로, align = 배치 64로 head 입력 정렬. tent / dsbr = feature 단계 TTA(lr 0.01)의 none 대비 정확도 변화(pt).",
                   "정의는 `scripts/analysis/pred_bias.py` docstring.", ""]
    with torch.no_grad():
        for ds, title in (("aid", "AID-C"), ("r45", "RESISC45-C")):
            res[ds] = {m: run(m, ds, dev) for m in ORDER}
            R = res[ds]
            mean = lambda names, k, ms=ORDER: float(np.mean([R[m][s][k] for m in ms for s in names]))
            row = lambda label, names, ms=ORDER: [label] + [f"{100 * mean(names, k, ms):.1f}" for k in ("tv_none", "tv_align", "top_none", "top_align")] + \
                [f"{mean(names, 'acc_none', ms):.1f}"] + [f"{mean(names, k, ms) - mean(names, 'acc_none', ms):+.1f}" for k in ("acc_tent", "acc_dsbr", "acc_align")]
            head = ["", "tv none", "tv align", "top none", "top align", "acc none", "tent", "dsbr", "align"]
            fmt = lambda rows: ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]
            corr = [f"{c}_s{s}" for c in common.NOISES for s in range(1, 6)]
            md += [f"## {title}: 손상별 (17개 백본 × 5단계 평균)", ""]
            md += fmt([row("clean", ["clean"])] + [row(c, [f"{c}_s{s}" for s in range(1, 6)]) for c in common.NOISES] + [row("**손상 평균**", corr)])
            md += [f"## {title}: 백본별 (손상 60개 stream 평균)", ""]
            md += fmt([row(m, corr, [m]) for m in ORDER])
            # streams binned by the bias of the unadapted model
            allv = [(R[m][s]["tv_none"], R[m][s]) for m in ORDER for s in corr]
            tvs = np.array([a for a, _ in allv])
            edges = np.quantile(tvs, [0, 0.25, 0.5, 0.75, 1.0])
            md += [f"## {title}: stream을 none의 tv로 4등분 (17 × 60 = {len(allv)}개 stream)", "",
                   "| tv 구간 (%) | stream 수 | acc none | tent | dsbr | align | align 뒤 남는 tv |", "|---|---:|---:|---:|---:|---:|---:|"]
            for lo, hi in zip(edges[:-1], edges[1:]):
                g = [v for a, v in allv if lo <= a <= hi] if hi == edges[-1] else [v for a, v in allv if lo <= a < hi]
                a0 = np.mean([v["acc_none"] for v in g])
                md.append(f"| {100 * lo:.1f}–{100 * hi:.1f} | {len(g)} | {a0:.1f} | " + " | ".join(
                    f"{np.mean([v[k] for v in g]) - a0:+.1f}" for k in ("acc_tent", "acc_dsbr", "acc_align")) + f" | {100 * np.mean([v['tv_align'] for v in g]):.1f} |")
            d = lambda k: [v[k] - v["acc_none"] for _, v in allv]
            md += ["", f"stream 전체에서 none의 tv와의 Spearman 상관: tent 이득 {spear(tvs, d('acc_tent')):+.2f}, dsbr 이득 {spear(tvs, d('acc_dsbr')):+.2f}, "
                   f"align 이득 {spear(tvs, d('acc_align')):+.2f}, none 정확도 {spear(tvs, [v['acc_none'] for _, v in allv]):+.2f}", ""]
    json.dump(res, open(D / "pred_bias.json", "w"))
    open(D / "pred_bias.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
