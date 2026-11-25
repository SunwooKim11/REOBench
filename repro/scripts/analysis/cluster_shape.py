"""What shape do the classes have in the clean head input z? (follow-up to cluster_subspace.py: why k-means << linear head)

Two readings of "linearly separable but k-means is low" are tested per backbone on the clean AID / RESISC45 features:
  (A) each class is ONE cloud, but stretched (anisotropic) along directions that do not separate classes, so Euclidean
      k-means cuts along the stretch instead of between classes;
  (B) each class is SEVERAL separate clouds (multi-modal), so one centre per class is not enough.

Columns (accuracy in %, test stream unless noted):
  lin      : stage-1 linear head
  knn      : 5-NN, train -> test
  ncm      : nearest clean-train class mean (one centre per class, centres given by labels)
  ncm_white: ncm after the within-class whitening of km_white (one centre per class + ONE shape shared by all classes = LDA)
  km       : k-means on test, k = C, k-means++ best of 3, best one-to-one matching (label-free)
  km_init  : k-means on test started AT the train class means and run to convergence, cluster k read as class k
  J ratio  : k-means objective of km_init / objective of km (> 1: the label-free solution has the LOWER objective, i.e.
             k-means prefers a partition that is not the classes; it is not a bad local optimum)
  km_white : km after whitening z with the pooled within-class covariance of train (labels; PCA 256 + shrinkage 0.1):
             if (A) holds this should approach ncm / lin
  kC, 2C, 4C, 8C : k-means fitted on TRAIN with that many centres, each centre named by its train majority class,
             test classified by the nearest centre: how many centres per class are needed
  split2   : per class (train), share of the class variance removed by the best 2-way split (2-means, best of 3)
  null     : the same for a Gaussian sample with that class's mean and covariance and the same n (one stretched cloud)
  pair2    : the same for a "fake class" made of two real classes pooled (class k and k+1): what two separate clouds give
  multi    : share of classes with split2 > null + 0.03 (evidence for (B))
  top1     : per class, share of the class variance on its first principal direction (stretch), mean over classes
  out      : share of the pooled within-class variance that lies outside the span of the class means

  /opt/conda/envs/segtta/bin/python scripts/analysis/cluster_shape.py     # from repro/, GPU, a few minutes
Output: experiments/2026-11-24_cluster_cot/RESULTS_shape.md, shape.json
"""
import json
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
torch.set_num_threads(4)


def match_acc(lab, y, k, nc):
    cont = (F.one_hot(lab, k).double().T @ F.one_hot(y, nc).double()).cpu().numpy()
    i, j = lsa(-cont)
    return 100 * cont[i, j].sum() / len(y)


def lloyd(x, c, iters=100):
    lab = None
    for _ in range(iters):
        new = torch.cdist(x, c).argmin(1)
        if lab is not None and (new == lab).all():
            break
        lab = new
        oh = F.one_hot(lab, len(c)).to(x.dtype); cnt = oh.sum(0)
        c = torch.where(cnt[:, None] > 0, (oh.T @ x) / cnt.clamp(min=1)[:, None], c)
    return lab, c, ((x - c[lab]) ** 2).sum().item()


def best_km(x, k, n=3):
    return min((kmeans(x, k, s) for s in range(n)), key=lambda t: t[2])


def split2(x):
    j1 = ((x - x.mean(0)) ** 2).sum().item()
    return 1 - best_km(x, 2)[2] / j1


def one(m, ds):
    h = torch.load(f"{FEATS}/{m}/head_{ds}.pt"); tr = torch.load(f"{FEATS}/{m}/{ds}_train.pt"); te = torch.load(f"{FEATS}/{m}/{ds}_test.pt")
    z = lambda a: ((a.float() - h["mu"]) / h["sd"]).to(dev).double()
    ztr, ytr, x, y = z(tr["feats"]), tr["y"].to(dev), z(te["feats"]), te["y"].to(dev)
    nc = h["w"].shape[1]
    r = {"D": x.shape[1], "n_train": len(ytr), "n_test": len(y)}
    r["lin"] = 100 * ((x.float() @ h["w"].to(dev) + h["b"].to(dev)).argmax(1) == y).float().mean().item()
    nn_ = torch.cdist(x, ztr).topk(5, largest=False).indices
    r["knn"] = 100 * (F.one_hot(ytr[nn_], nc).sum(1).argmax(1) == y).float().mean().item()
    cm = torch.stack([ztr[ytr == k].mean(0) for k in range(nc)])
    r["ncm"] = 100 * (torch.cdist(x, cm).argmin(1) == y).float().mean().item()
    lab, _, j = best_km(x, nc)
    r["km"] = match_acc(lab, y, nc, nc)
    lab_i, _, j_i = lloyd(x, cm.clone())
    r["km_init"] = 100 * (lab_i == y).float().mean().item()
    r["j_ratio"] = j_i / j
    # within-class whitening (labels): PCA 256 of train, pooled within-class covariance, shrinkage 0.1
    mu = ztr.mean(0)
    V = torch.linalg.svd(ztr - mu, full_matrices=False)[2][: min(256, ztr.shape[1])].T
    ptr, px = (ztr - mu) @ V, (x - mu) @ V
    res = ptr - (cm - mu)[ytr] @ V
    sw = res.T @ res / len(res)
    sw = 0.9 * sw + 0.1 * sw.diag().mean() * torch.eye(len(sw), device=dev, dtype=sw.dtype)
    ev, U = torch.linalg.eigh(sw)
    W = U @ torch.diag(ev.clamp(min=1e-8) ** -0.5) @ U.T
    labw, _, _ = best_km(px @ W, nc)
    r["km_white"] = match_acc(labw, y, nc, nc)
    r["ncm_white"] = 100 * (torch.cdist(px @ W, ((cm - mu) @ V) @ W).argmin(1) == y).float().mean().item()
    for mult in (1, 2, 4, 8):
        k = mult * nc
        labt, c, _ = best_km(ztr, k, 2)
        name = (F.one_hot(labt, k).double().T @ F.one_hot(ytr, nc).double()).argmax(1)
        r[f"k{mult}C"] = 100 * (name[torch.cdist(x, c).argmin(1)] == y).float().mean().item()
    g = torch.Generator(device=dev).manual_seed(0)
    s2, nu, t1, p2 = [], [], [], []
    for k in range(nc):
        xk = ptr[ytr == k]
        xk = xk - xk.mean(0)
        s2.append(split2(xk))
        xp = ptr[(ytr == k) | (ytr == (k + 1) % nc)]
        p2.append(split2(xp - xp.mean(0)))
        sv = torch.linalg.svdvals(xk) ** 2
        t1.append((sv[0] / sv.sum()).item())
        # Gaussian with the same covariance and n: xk^T-coloured noise
        nul = []
        for _ in range(3):
            e = torch.randn(len(xk), len(xk), generator=g, device=dev, dtype=xk.dtype) / len(xk) ** 0.5
            nul.append(split2(e @ xk))
        nu.append(float(np.mean(nul)))
    r["split2"], r["null"], r["top1"], r["pair2"] = float(np.mean(s2)), float(np.mean(nu)), float(np.mean(t1)), float(np.mean(p2))
    r["multi"] = float(np.mean([a > b + 0.03 for a, b in zip(s2, nu)]))
    r["split2_cls"], r["null_cls"] = s2, nu
    Q = torch.linalg.qr((cm - cm.mean(0)).T)[0][:, : nc - 1]
    resf = ztr - cm[ytr]
    r["out"] = 1 - (((resf @ Q) ** 2).sum() / (resf ** 2).sum()).item()
    return r


COLS = ["lin", "knn", "ncm", "ncm_white", "km", "km_init", "j_ratio", "km_white", "k1C", "k2C", "k4C", "k8C", "split2", "null", "pair2", "multi", "top1", "out"]
FMT = {"j_ratio": "{:.3f}", "split2": "{:.3f}", "null": "{:.3f}", "pair2": "{:.3f}", "multi": "{:.2f}", "top1": "{:.3f}", "out": "{:.2f}"}
out = ["# 클래스의 모양: k-means가 선형 head보다 낮은 이유 (2026-11-26, 수치만)", "",
       "`scripts/analysis/cluster_shape.py`. clean head 입력 z. 열의 정의는 스크립트 docstring.",
       "ncm = train 클래스 평균 중 가장 가까운 것, ncm_white = 클래스 내 공분산으로 whitening한 뒤의 ncm(LDA와 같은 모형), km_init = 클래스 평균에서 시작한 k-means, J ratio = km_init 목적함수 / km 목적함수,",
       "km_white = 클래스 내 공분산으로 whitening한 뒤의 k-means(라벨 필요), kC–8C = train에 중심을 C–8C개 두고 다수결 이름을 붙인 분류,",
       "split2 = 클래스 하나를 둘로 쪼갰을 때 줄어드는 분산 비율, null = 같은 공분산의 Gaussian에서의 값, pair2 = 두 클래스를 합친 가짜 클래스에서의 값(덩어리가 둘일 때의 기준), multi = split2 > null + 0.03인 클래스 비율,",
       "top1 = 클래스 분산 중 첫 주성분의 몫, out = 클래스 내 분산 중 클래스 평균 부분공간 밖의 몫.", ""]
res = {}
for ds, title in (("aid", "AID"), ("r45", "RESISC45")):
    out += [f"## {title}", "", "| model | " + " | ".join(COLS) + " |", "|---|" + "---:|" * len(COLS)]
    tab = []
    for m in ORDER:
        r = one(m, ds); res[f"{m}_{ds}"] = r
        tab.append([r[c] for c in COLS])
        out.append(f"| {m} | " + " | ".join(FMT.get(c, "{:.1f}").format(r[c]) for c in COLS) + " |")
        print(out[-1], flush=True)
    mean = np.mean(tab, 0)
    out += ["| **평균** | " + " | ".join(FMT.get(c, "{:.1f}").format(v) for c, v in zip(COLS, mean)) + " |", ""]
(OUT / "RESULTS_shape.md").write_text("\n".join(out) + "\n")
json.dump(res, open(OUT / "shape.json", "w"))
print("saved", OUT / "RESULTS_shape.md")
