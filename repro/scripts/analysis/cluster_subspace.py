"""Why is k-means far below the linear head on the clean AID / RESISC45 test features? (follow-up to cluster_cot.py)

On the clean test stream, head input z (standardized with clean-train statistics), per backbone:
  lin        : accuracy of the stage-1 linear head
  knn        : 5-nearest-neighbour accuracy, train -> test, Euclidean in z (local structure, no learning)
  km_full    : k-means (k = C, best of 3) accuracy after the best one-to-one matching, all D dimensions
  km_pca     : the same after projecting on the top C-1 principal directions of the test stream (label-free)
  km_cls     : the same after projecting on the span of the clean-train class means (C-1 dimensions, needs labels)
  between    : between-class share of the total variance of z (test labels), in all dimensions / inside the class-mean span
  D, pr      : feature dimension, participation ratio (effective dimension) of z

  /opt/conda/envs/segtta/bin/python scripts/analysis/cluster_subspace.py     # from repro/, GPU, about a minute
Output: experiments/2026-11-24_cluster_cot/RESULTS_subspace.md
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as lsa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_cot import FEATS, ORDER, OUT, kmeans  # noqa: E402

dev = torch.device("cuda")


def km_acc(x, y, nc):
    lab, _, _ = min((kmeans(x, nc, s) for s in range(3)), key=lambda t: t[2])
    cont = (F.one_hot(lab, nc).double().T @ F.one_hot(y, nc).double()).cpu().numpy()
    i, j = lsa(-cont)
    return 100 * cont[i, j].sum() / len(y)


def between(x, y, nc):
    xc = x - x.mean(0)
    tot = (xc ** 2).sum()
    b = sum((y == k).sum() * (xc[y == k].mean(0) ** 2).sum() for k in range(nc))
    return (b / tot).item()


out = ["# k-means와 선형 head의 차이는 어디서 오나 (2026-11-24, 수치만)", "",
       "`scripts/analysis/cluster_subspace.py`. clean test stream, head 입력 z. km = k-means 군집 정확도(%), knn = 5-NN(train → test).",
       "km_pca = test의 상위 C−1 주성분으로 투영(라벨 없음), km_cls = clean train 클래스 평균이 만드는 C−1차원으로 투영(라벨 필요).",
       "between = 전체 분산 중 클래스 간 분산의 비율(전체 차원 / 클래스 평균 부분공간 안).", ""]
for ds, title in (("aid", "AID"), ("r45", "RESISC45")):
    out += [f"## {title}", "", "| model | D | pr | lin | knn | km_full | km_pca | km_cls | between 전체 | between 부분공간 | 부분공간이 차지하는 분산 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    tab = []
    for m in ORDER:
        h = torch.load(f"{FEATS}/{m}/head_{ds}.pt"); tr = torch.load(f"{FEATS}/{m}/{ds}_train.pt"); te = torch.load(f"{FEATS}/{m}/{ds}_test.pt")
        z = lambda a: ((a.float() - h["mu"]) / h["sd"]).to(dev).double()
        ztr, ytr, x, y = z(tr["feats"]), tr["y"].to(dev), z(te["feats"]), te["y"].to(dev)
        nc = h["w"].shape[1]
        lin = 100 * ((x.float() @ h["w"].to(dev) + h["b"].to(dev)).argmax(1) == y).float().mean().item()
        nn_ = torch.cdist(x, ztr).topk(5, largest=False).indices
        knn = 100 * (F.one_hot(ytr[nn_], nc).sum(1).argmax(1) == y).float().mean().item()
        cm = torch.stack([ztr[ytr == k].mean(0) for k in range(nc)])
        Q = torch.linalg.qr((cm - cm.mean(0)).T)[0][:, : nc - 1]      # D x (C-1), span of the class means
        xc = x - x.mean(0)
        V = torch.linalg.svd(xc, full_matrices=False)[2][: nc - 1].T  # D x (C-1), top principal directions
        ev = torch.linalg.svdvals(xc) ** 2
        r = [x.shape[1], (ev.sum() ** 2 / (ev ** 2).sum()).item(), lin, knn, km_acc(x, y, nc), km_acc(xc @ V, y, nc),
             km_acc(xc @ Q, y, nc), between(x, y, nc), between(xc @ Q, y, nc), (((xc @ Q) ** 2).sum() / (xc ** 2).sum()).item()]
        tab.append(r)
        out.append(f"| {m} | {r[0]} | {r[1]:.0f} | " + " | ".join(f"{v:.1f}" for v in r[2:7]) + " | " + " | ".join(f"{v:.2f}" for v in r[7:]) + " |")
        print(out[-1], flush=True)
    mean = np.mean(tab, 0)
    out += [f"| **평균** | | {mean[1]:.0f} | " + " | ".join(f"{v:.1f}" for v in mean[2:7]) + " | " + " | ".join(f"{v:.2f}" for v in mean[7:]) + " |", ""]
(OUT / "RESULTS_subspace.md").write_text("\n".join(out) + "\n")
