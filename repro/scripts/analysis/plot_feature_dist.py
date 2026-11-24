"""Pictures of the head-input features (AID): separable but not clustered, and what a corruption / the alignment does.

  fig1_clean_geometry.png : clean test stream of 4 backbones (lowest and highest k-means accuracy) in three 2-D views:
      PCA            top-2 principal directions of the clean test stream (largest variance, label-free)
      class plane    the plane through the clean-train means of the three highlighted classes (their best linear view)
      t-SNE          exact t-SNE (perplexity 30) on the top-50 principal components (local neighbourhoods)
  fig2_shift_align.png    : the same backbones, clean / corrupted / corrupted after head-input alignment (batches of 64),
      x = direction of the common shift (mean corrupted - mean clean), y = top principal direction of the clean stream
      orthogonal to x; all three panels of a row share the axes.
Highlighted: the three residential classes of AID (similar classes); every other class is grey (a scatter plot carries at
most three categorical hues, dataviz palette slots 1-3).

  /opt/conda/envs/segtta/bin/python scripts/analysis/plot_feature_dist.py [--corr motion_blur --sev 3]   # from repro/, GPU
Output: docs/figs/2026-11-24_feature_dist/
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.optimize import linear_sum_assignment as lsa  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths  # noqa: E402
from cluster_cot import FEATS, align, kmeans  # noqa: E402

OUT = paths.REPO / "docs" / "figs" / "2026-11-24_feature_dist"
MODELS = ["satmae", "moco_in", "satdino_s8", "dinov3_web_l16"]
HL = {8: ("DenseResidential", "#2a78d6"), 14: ("MediumResidential", "#eb6834"), 25: ("SparseResidential", "#1baf7a")}
GREY, INK, INK2, SURF = "#c4c3bd", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b9b8b2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.titlesize": 10, "legend.frameon": False})
dev = torch.device("cuda")


def tsne(x, perp=30.0, iters=1000, seed=0):
    """Exact t-SNE (van der Maaten & Hinton 2008): Gaussian P with per-point bandwidth, Student-t Q, momentum + gains."""
    n = len(x)
    D = torch.cdist(x, x) ** 2
    beta = torch.ones(n, 1, device=x.device, dtype=x.dtype)
    lo, hi = torch.zeros_like(beta), torch.full_like(beta, float("inf"))
    eye = torch.eye(n, device=x.device, dtype=torch.bool)
    for _ in range(60):  # bisection on the precision so that every row has entropy log(perp)
        P = torch.exp(-D * beta).masked_fill(eye, 0)
        P = P / P.sum(1, keepdim=True).clamp_min(1e-300)
        H = -(P * torch.log(P.clamp_min(1e-300))).sum(1, keepdim=True)
        up = H > np.log(perp)
        lo = torch.where(up, beta, lo); hi = torch.where(up, hi, beta)
        beta = torch.where(torch.isinf(hi), beta * 2, (lo + hi) / 2)
    P = ((P + P.T) / (2 * n)).clamp_min(1e-12)
    y = x[:, :2] / x[:, 0].std() * 1e-4  # PCA initialisation
    v, gains = torch.zeros_like(y), torch.ones_like(y)
    for it in range(iters):
        num = 1 / (1 + torch.cdist(y, y) ** 2)
        num = num.masked_fill(eye, 0)
        Q = (num / num.sum()).clamp_min(1e-12)
        PQ = ((P * (12 if it < 250 else 1)) - Q) * num
        grad = 4 * (PQ.sum(1, keepdim=True) * y - PQ @ y)
        gains = torch.where((grad > 0) != (v > 0), gains + 0.2, gains * 0.8).clamp_min(0.01)
        v = (0.5 if it < 250 else 0.8) * v - 200 * gains * grad
        y = y + v
        y = y - y.mean(0)
    return y


def scatter(ax, p, y, s=7):
    p = p.cpu().numpy(); y = y.cpu().numpy()
    oth = ~np.isin(y, list(HL))
    ax.scatter(p[oth, 0], p[oth, 1], s=s * 0.6, c=GREY, alpha=0.55, linewidths=0, rasterized=True)
    for k, (_, col) in HL.items():
        ax.scatter(p[y == k, 0], p[y == k, 1], s=s * 1.6, c=col, edgecolors=SURF, linewidths=0.4, rasterized=True)


def legend(fig):
    hs = [plt.Line2D([], [], marker="o", ls="", ms=7, mfc=c, mec=SURF, label=n) for n, c in HL.values()]
    hs.append(plt.Line2D([], [], marker="o", ls="", ms=5, mfc=GREY, mec=GREY, label="other 27 classes"))
    fig.legend(handles=hs, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.965), fontsize=10)


def load(model):
    h = torch.load(f"{FEATS}/{model}/head_aid.pt"); tr = torch.load(f"{FEATS}/{model}/aid_train.pt")
    te = torch.load(f"{FEATS}/{model}/aid_test.pt"); cd = torch.load(f"{FEATS}/{model}/aid_c.pt")
    z = lambda a: ((a.float() - h["mu"]) / h["sd"]).to(dev).double()
    return h, z, tr, te, cd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", default="motion_blur")
    ap.add_argument("--sev", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f1, A1 = plt.subplots(len(MODELS), 3, figsize=(11.5, 3.55 * len(MODELS)))
    f2, A2 = plt.subplots(len(MODELS), 3, figsize=(11.5, 3.3 * len(MODELS)))
    for r, m in enumerate(MODELS):
        h, z, tr, te, cd = load(m)
        ztr, ytr, x, y = z(tr["feats"]), tr["y"].to(dev), z(te["feats"]), te["y"].to(dev)
        w, b = h["w"].to(dev).double(), h["b"].to(dev).double()
        nc = w.shape[1]
        acc = lambda a, yy: 100 * ((a @ w + b).argmax(1) == yy).float().mean().item()
        knn = 100 * (F.one_hot(ytr[torch.cdist(x, ztr).topk(5, largest=False).indices], nc).sum(1).argmax(1) == y).float().mean().item()
        lab, _, _ = min((kmeans(x, nc, s) for s in range(3)), key=lambda t: t[2])
        cont = (F.one_hot(lab, nc).double().T @ F.one_hot(y, nc).double()).cpu().numpy()
        i, j = lsa(-cont)
        km = 100 * cont[i, j].sum() / len(y)
        xc = x - x.mean(0)
        V = torch.linalg.svd(xc, full_matrices=False)[2]
        cm = torch.stack([ztr[ytr == k].mean(0) for k in HL])
        Q = torch.linalg.qr((cm[1:] - cm[0]).T)[0]
        views = [("PCA: 2 directions of largest variance", xc @ V[:2].T),
                 ("plane through the 3 class means", (x - cm.mean(0)) @ Q),
                 ("t-SNE: local neighbourhoods", tsne(xc @ V[:50].T))]
        for c, (title, p) in enumerate(views):
            ax = A1[r, c]
            scatter(ax, p, y)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title if r == 0 else "", color=INK2)
            if c == 0:
                ax.set_ylabel(f"{m}\nlinear {acc(x, y):.1f} | 5-NN {knn:.1f} | k-means {km:.1f}", color=INK, fontsize=10)
        # shift and alignment
        sel = (cd["corr"] == common.NOISES.index(args.corr)) & (cd["sev"] == args.sev)
        xk, yk = z(cd["feats"][sel]), cd["y"][sel].to(dev)
        xa = align(xk)
        d = xk.mean(0) - x.mean(0)
        u1 = d / d.norm()
        res = xc - (xc @ u1)[:, None] * u1
        u2 = torch.linalg.svd(res, full_matrices=False)[2][0]
        B = torch.stack([u1, u2]).T
        pts = [("clean", (x - x.mean(0)) @ B, y, acc(x, y)), (f"{args.corr} s{args.sev}", (xk - x.mean(0)) @ B, yk, acc(xk, yk)),
               (f"{args.corr} s{args.sev}, aligned", (xa - x.mean(0)) @ B, yk, acc(xa, yk))]
        allp = torch.cat([p for _, p, _, _ in pts])
        lim = [(allp[:, k].quantile(0.002).item(), allp[:, k].quantile(0.998).item()) for k in (0, 1)]
        for c, (title, p, yy, a) in enumerate(pts):
            ax = A2[r, c]
            scatter(ax, p, yy)
            ax.axvline(0, color=INK2, lw=0.6, ls=(0, (3, 3)), zorder=0)
            ax.plot([p[:, 0].mean().item()], [lim[1][0]], marker="^", ms=8, color=INK, clip_on=False, zorder=5)
            ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
            ax.set_title(f"{title}: accuracy {a:.1f}%", color=INK, loc="left")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            if c == 0:
                ax.set_ylabel(f"{m}\ntop clean direction", color=INK)
            if r == len(MODELS) - 1:
                ax.set_xlabel("direction of the common shift (source std units)")
        print(f"{m}: lin {acc(x, y):.1f} knn {knn:.1f} km {km:.1f} | {args.corr}_s{args.sev} {acc(xk, yk):.1f} -> aligned {acc(xa, yk):.1f} "
              f"| shift {d.norm().item():.1f}", flush=True)
    for f, name, title in ((f1, "fig1_clean_geometry.png", "AID clean test features (head input z): linearly separable, locally consistent, not one blob per class"),
                           (f2, f"fig2_shift_align_{args.corr}_s{args.sev}.png",
                            "A corruption moves all features together; alignment moves the mean back\n(triangle = stream mean, dashed line = clean mean)")):
        legend(f)
        two = "\n" in title
        f.suptitle(title, x=0.5, y=0.995, fontsize=12, fontweight="bold")
        f.legends[0].set_bbox_to_anchor((0.5, 0.955 if two else 0.965))
        f.tight_layout(rect=(0, 0, 1, 0.93 if two else 0.945))
        f.savefig(OUT / name, dpi=130)
        print("saved", OUT / name)
