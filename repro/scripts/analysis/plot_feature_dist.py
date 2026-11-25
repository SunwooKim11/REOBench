"""Pictures of the head-input features (AID): linearly separable but not found by k-means, and what a corruption / head-input standardization does.

  fig0_projection.png     : schematic on toy 3-D data: a 2-D view is an orthographic projection onto a plane; the PCA plane and the
      plane through the class means are two choices of plane (the first two columns of fig1)
  fig1_clean_geometry.png : clean test stream of 4 backbones (lowest and highest k-means accuracy) in three 2-D views:
      PCA            orthographic projection onto the top-2 principal directions of the clean test stream (largest variance, label-free)
      class plane    orthographic projection onto the plane through the clean-train means of the three highlighted classes
      t-SNE          exact t-SNE (perplexity 30) on the top-50 principal components (local neighbourhoods; axes have no meaning)
  fig3_class_shape.png    : why k-means is far below the linear head: toy pictures of three class shapes with four read-outs,
      and the measured numbers of scripts/analysis/cluster_shape.py (--shape_only)
  fig2_shift_align.png    : the same backbones, clean / corrupted / corrupted after head-input standardization (batches of 64),
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


NOTE1 = ("Each dot = one AID clean test image, as its head input z (384-2,048 dimensions).\n"
         "Columns 1 and 2 are orthographic projections of the same dots onto two different planes (see fig0_projection.png): column 1 = the plane of\n"
         "largest spread (PCA), column 2 = the plane through the clean-train means of the three coloured classes. Column 3 (t-SNE) keeps who is near whom.\n"
         "linear = linear head accuracy, 5-NN = majority label of the 5 nearest train images, k-means = 30 clusters without labels, best match to classes.")
NOTE2 = ("Each dot = one image, as its head input z, projected orthographically onto two directions.\n"
         "x = the common shift: the direction from the clean mean to the corrupted-stream mean. The value is the distance from the clean mean along it,\n"
         "in source std units (0 = clean mean, dashed line). y = the direction of largest spread of the clean features, made perpendicular to x.\n"
         "All three panels of a row use the same x and y. Standardization uses the mean / std of each test batch of 64 images.")


def schematic():
    """Toy 3-D data: two orthographic projections of the same points, onto the PCA plane and onto the plane through the class means."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    rng = np.random.default_rng(0)
    means = np.array([[-2.2, -1.2, 0.0], [2.2, -1.2, 0.0], [0.0, 2.0, 0.0]])
    cols = [c for _, c in HL.values()]
    pts = [mu + rng.normal(size=(60, 3)) * [0.5, 0.5, 2.4] for mu in means]
    P = np.concatenate(pts); lab = np.repeat(np.arange(3), 60)
    fig = plt.figure(figsize=(14.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0], projection="3d")
    sq = np.array([[-4.2, -3.2, 0], [4.2, -3.2, 0], [4.2, 3.8, 0], [-4.2, 3.8, 0]])
    ax.add_collection3d(Poly3DCollection([sq], facecolor="#e7e6e1", edgecolor=INK2, alpha=0.5, linewidths=0.8))
    for k in range(3):
        q = pts[k]
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=9, color=cols[k], depthshade=False)
        for i in range(0, 60, 12):
            ax.plot([q[i, 0]] * 2, [q[i, 1]] * 2, [q[i, 2], 0], color=INK2, lw=0.6, ls=(0, (2, 2)))
            ax.scatter([q[i, 0]], [q[i, 1]], [0], s=12, color=cols[k], marker="x", depthshade=False)
        ax.scatter(*means[k], s=110, color=cols[k], edgecolors=INK, marker="*", depthshade=False, zorder=10)
    ax.set_title("feature space (shown in 3-D)\ngrey plane = plane through the 3 class means (stars)\ndotted = drop each point straight onto it",
                 fontsize=10, color=INK)
    ax.set_axis_off(); ax.set_box_aspect((1.2, 1, 1.3)); ax.view_init(elev=20, azim=-25)
    ax.set_zlim(-6, 6)
    Pc = P - P.mean(0)
    V = np.linalg.svd(Pc, full_matrices=False)[2]
    Q = np.linalg.qr((means[1:] - means[0]).T)[0]
    for c, (title, pr) in enumerate((("projection on the PCA plane\n(2 directions of largest spread, no labels)", Pc @ V[:2].T),
                                    ("projection on the plane through the 3 class means\n(uses labels)", (P - means.mean(0)) @ Q))):
        a = fig.add_subplot(gs[c + 1])
        for k in range(3):
            a.scatter(pr[lab == k, 0], pr[lab == k, 1], s=14, color=cols[k], edgecolors=SURF, linewidths=0.4)
        a.set_title(title, fontsize=10, color=INK); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("A 2-D view = orthographic projection onto a plane. PCA and the class-mean view differ only in which plane is used.\n"
                 "Toy data: the widest spread (vertical) is shared by all classes, so the PCA plane mixes them; the class-mean plane separates them.",
                 fontsize=11, fontweight="bold", y=1.08)
    fig.savefig(OUT / "fig0_projection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "fig0_projection.png")


def _toy_km(x, k, rng, n=10):
    best = None
    for _ in range(n):
        c = x[rng.choice(len(x), 1)]
        for _ in range(k - 1):
            d2 = ((x[:, None] - c[None]) ** 2).sum(-1).min(1)
            c = np.concatenate([c, x[rng.choice(len(x), 1, p=d2 / d2.sum())]])
        for _ in range(100):
            lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            c2 = np.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(c2, c):
                break
            c = c2
        j = ((x - c[lab]) ** 2).sum()
        if best is None or j < best[2]:
            best = (lab, c, j)
    return best


def class_shape():
    """fig3: toy pictures of three class shapes with four read-outs, and the same four read-outs measured on the real features.
    fig3b (backup): two further measurements from cluster_shape.py (shape.json)."""
    import json
    from cluster_cot import OUT as EXP
    R = json.load(open(EXP / "shape.json"))
    rng = np.random.default_rng(0)
    cols = [c for _, c in HL.values()]
    n = 150
    toys = []
    m = np.array([[-3.0, -1.5], [3.0, -1.5], [0.0, 3.2]])
    toys.append(("(a) one round cloud per class", np.concatenate([mu + rng.normal(size=(n, 2)) * 0.8 for mu in m])))
    m = np.array([[-1.7, 0.0], [0.0, 0.0], [1.7, 0.0]])
    toys.append(("(b) one cloud per class, all stretched the same way", np.concatenate([mu + rng.normal(size=(n, 2)) * [0.42, 3.0] for mu in m])))
    isl = [[(i, j) for i in range(3) for j in range(3) if (i + j) % 3 == k] for k in range(3)]
    toys.append(("(c) several separate islands per class",
                 np.concatenate([np.concatenate([np.array(q) * 2.6 - 2.6 + rng.normal(size=(n // 3, 2)) * 0.38 for q in isl[k]]) for k in range(3)])))
    y = np.repeat(np.arange(3), n)
    READ = (("linear head", "lin"), ("5-NN", "knn"), ("nearest class mean\n(1 centre per class)", "ncm"), ("k-means\n(no labels)", "km"))
    fig = plt.figure(figsize=(14.5, 9.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.42, wspace=0.3)
    toy_acc = []
    for c, (title, x) in enumerate(toys):
        ax = fig.add_subplot(gs[0, c])
        lab, cen, _ = _toy_km(x, 3, rng)
        cont = np.array([[((lab == a) & (y == b)).sum() for b in range(3)] for a in range(3)])
        i, j = lsa(-cont)
        km = 100 * cont[i, j].sum() / len(y)
        X1 = np.c_[(x - x.mean(0)) / x.std(0), np.ones(len(x))]
        W = np.zeros((3, 3))
        for _ in range(3000):                                   # softmax regression = a linear head
            pr = np.exp(X1 @ W); pr /= pr.sum(1, keepdims=True)
            W -= 0.5 * X1.T @ (pr - np.eye(3)[y]) / len(y)
        lin = 100 * ((X1 @ W).argmax(1) == y).mean()
        mu = np.stack([x[y == k].mean(0) for k in range(3)])
        ncm = 100 * (((x[:, None] - mu[None]) ** 2).sum(-1).argmin(1) == y).mean()
        d = ((x[:, None] - x[None]) ** 2).sum(-1); np.fill_diagonal(d, np.inf)
        nb = y[np.argsort(d, 1)[:, :5]]
        knn = 100 * (np.array([np.bincount(r, minlength=3).argmax() for r in nb]) == y).mean()
        toy_acc.append({"lin": lin, "knn": knn, "ncm": ncm, "km": km})
        lo, hi = x.min(0) - 1.0, x.max(0) + 1.0
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 400), np.linspace(lo[1], hi[1], 400))
        G = np.c_[gx.ravel(), gy.ravel()]
        reg = ((G[:, None] - cen[None]) ** 2).sum(-1).argmin(1).reshape(gx.shape)
        ax.contour(gx, gy, reg, levels=[0.5, 1.5], colors=[INK], linewidths=0.9, linestyles="-")
        for k in range(3):
            ax.scatter(x[y == k, 0], x[y == k, 1], s=11, color=cols[k], edgecolors=SURF, linewidths=0.3)
        ax.scatter(cen[:, 0], cen[:, 1], s=130, marker="P", color=INK, edgecolors=SURF, linewidths=1.0, zorder=5)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel(f"linear {lin:.0f}    5-NN {knn:.0f}\nnearest class mean {ncm:.0f}    k-means {km:.0f}", fontsize=11, color=INK, labelpad=8)
    fig.text(0.5, 0.935, "toy data, 3 classes (colours = true class).  black crosses and lines = the 3 centres and borders k-means finds without labels.  "
             "numbers = accuracy (%) of four read-outs", ha="center", fontsize=10, color=INK2)
    # the same four read-outs: toys vs measured
    ax = fig.add_subplot(gs[1, :])
    meas = {ds: {k: np.mean([r[k] for key, r in R.items() if key.endswith("_" + ds)]) for _, k in READ} for ds in ("aid", "r45")}
    series = [("toy (a) round", toy_acc[0], "#dcdbd6", INK2), ("toy (b) stretched", toy_acc[1], "#52514e", SURF), ("toy (c) islands", toy_acc[2], "#a9a8a2", INK),
              ("measured, AID (17 backbones)", meas["aid"], "#2a78d6", SURF), ("measured, RESISC45 (17 backbones)", meas["r45"], "#8fb8ea", INK)]
    w = 0.16
    for si, (name, v, col, _) in enumerate(series):
        xs = np.arange(len(READ)) + (si - 2) * w + (0.05 if si >= 3 else -0.03)
        ax.bar(xs, [v[k] for _, k in READ], width=w - 0.015, color=col, label=name)
        for a, (_, k) in zip(xs, READ):
            ax.text(a, v[k] + 1.5, f"{v[k]:.0f}", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(np.arange(len(READ))); ax.set_xticklabels([r[0] for r in READ], fontsize=11, color=INK)
    ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("accuracy (%)")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.2), fontsize=10)
    ax.set_title("the same four read-outs on the real features (blue) follow the pattern of toy (b): high, high, fairly high, low", fontsize=11.5,
                 color=INK, fontweight="bold", y=1.2)
    fig.suptitle("Why is k-means far below the linear head?  Each class is one cloud, spread widely in directions that do not separate classes",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.savefig(OUT / "fig3_class_shape.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "fig3_class_shape.png")

    # backup figure: two more measurements
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 4.6), gridspec_kw={"wspace": 0.3})
    rows = (("nearest class mean, as is", "ncm"), ("nearest class mean after squashing\nthe spread the classes share", "ncm_white"), ("linear head (reference)", "lin"))
    yy = np.arange(len(rows))[::-1]
    for di, (ds, name, col) in enumerate((("aid", "AID", "#52514e"), ("r45", "RESISC45", "#a9a8a2"))):
        v = [np.mean([r[k] for key, r in R.items() if key.endswith("_" + ds)]) for _, k in rows]
        ax.barh(yy + (0.19 if di == 0 else -0.19), v, height=0.34, color=col, label=name)
        for a, b in zip(yy, v):
            ax.text(b + 1, a + (0.19 if di == 0 else -0.19), f"{b:.1f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=INK); ax.set_xlim(0, 125); ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="y", length=0); ax.legend(loc="center right", fontsize=10)
    ax.set_xlabel("accuracy on clean test features (%), mean of 17 backbones")
    ax.set_title("test 1: squash the shared spread, then one centre per class is almost enough", fontsize=11, color=INK, fontweight="bold", loc="left")
    bars = (("one real class", "split2", "#2a78d6"), ("reference: ONE cloud\n(Gaussian, same spread)", "null", "#a9a8a2"),
            ("reference: TWO clouds\n(two real classes pooled)", "pair2", "#eb6834"))
    keys = [k for k in R if k.endswith("_aid")]
    means = []
    for bi, (name, k, col) in enumerate(bars):
        v = np.array([100 * R[key][k] for key in keys])
        ax2.bar(bi, v.mean(), width=0.55, color=col, alpha=0.55)
        ax2.scatter(bi + rng.uniform(-0.16, 0.16, len(v)), v, s=22, color=col, edgecolors=SURF, linewidths=0.5, zorder=3)
        means.append(v.mean())
    ax2.set_xticks(range(3)); ax2.set_xticklabels([f"{b[0]}\nmean {m:.1f}%" for b, m in zip(bars, means)], fontsize=10, color=INK)
    ax2.set_ylabel("how much tighter the points get when\nthe group is cut in two (% of its variance)"); ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_ylim(0, max(100 * R[k]["pair2"] for k in keys) + 5)
    ax2.set_title("test 2: cut one class in two. It behaves like ONE cloud (AID, dots = 17 backbones)", fontsize=11, color=INK, fontweight="bold", loc="left")
    fig.savefig(OUT / "fig3b_class_shape_tests.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "fig3b_class_shape_tests.png")


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
    ap.add_argument("--schematic_only", action="store_true")
    ap.add_argument("--shape_only", action="store_true", help="only fig3_class_shape.png (needs shape.json of cluster_shape.py)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.shape_only:
        class_shape(); sys.exit(0)
    schematic()
    if args.schematic_only:
        sys.exit(0)
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
        views = [("projection on the 2 widest directions\n(PCA, no labels)", xc @ V[:2].T),
                 ("projection on the plane through\nthe 3 class means (uses labels)", (x - cm.mean(0)) @ Q),
                 ("t-SNE: nearest neighbours kept\n(distances and axes have no meaning)", tsne(xc @ V[:50].T))]
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
               (f"{args.corr} s{args.sev}, standardized", (xa - x.mean(0)) @ B, yk, acc(xa, yk))]
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
                ax.set_ylabel(f"{m}\nwidest clean direction ⊥ x", color=INK)
            if r == len(MODELS) - 1:
                ax.set_xlabel("x: along the common shift (source std units)")
        print(f"{m}: lin {acc(x, y):.1f} knn {knn:.1f} km {km:.1f} | {args.corr}_s{args.sev} {acc(xk, yk):.1f} -> aligned {acc(xa, yk):.1f} "
              f"| shift {d.norm().item():.1f}", flush=True)
    for f, name, title in ((f1, "fig1_clean_geometry.png", "AID clean test features (head input z): linearly separable, locally consistent, but k-means does not find the classes"),
                           (f2, f"fig2_shift_align_{args.corr}_s{args.sev}.png",
                            "A corruption moves all features together; head-input standardization moves the mean back\n(triangle = stream mean, dashed line = clean mean)")):
        legend(f)
        two = "\n" in title
        f.suptitle(title, x=0.5, y=0.995, fontsize=12, fontweight="bold")
        f.legends[0].set_bbox_to_anchor((0.5, 0.955 if two else 0.965))
        f.tight_layout(rect=(0, 0, 1, 0.93 if two else 0.945))
        if f is f1:
            f.text(0.01, 0.002, NOTE1, ha="left", va="bottom", fontsize=9, color=INK2)
            f.subplots_adjust(bottom=0.06)
        else:
            f.text(0.01, 0.002, NOTE2, ha="left", va="bottom", fontsize=9, color=INK2)
            f.subplots_adjust(bottom=0.085)
        f.savefig(OUT / name, dpi=130, bbox_inches="tight")
        print("saved", OUT / name)
