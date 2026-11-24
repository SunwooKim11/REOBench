"""Figures for the 2026-11-26 mid-term presentation (static PNGs for slides), from results already on disk.

  fig1_methods.png        : corrupted accuracy by method, 17 backbones as dots, mean as bar (T1)
  fig2_estimation.png     : label-free accuracy estimates vs true accuracy on the 2,040 corrupted streams (T3)
  fig3_signals.png        : confidence vs accuracy and flip vs accuracy, three backbones on RESISC45-C
  fig4_em_phases.png      : entropy minimisation over 2,000 steps, tent vs DSBR: accuracy and predicted classes (T9)
  fig5_headroom.png       : per corruption: as is -> aligned -> head re-fitted with labels -> clean (T12)
  fig6_pairs.png          : satellite minus generic pretraining for the six matched pairs (T10)
  fig7_blur_aug.png       : drop under Gaussian blur, backbones with / without blur augmentation (T7)
  fig8_corruptions.png    : one AID image under the 12 REOBench corruptions, grouped by acquisition stage (T6)
  fig9_seg.png            : Potsdam mIoU per corruption for none / BN / input / input + BN alignment (T5)
  fig10_alignment.png     : what head-input alignment does: pipeline, one feature channel, prediction share per class
  fig11_feature_geometry.png : clean features of two backbones: accuracy by read-out and three 2-D views
  fig12_shift_alignment.png  : clean / corrupted / aligned features along the common shift, two backbones
Labels are in English (no Korean font on the server). Colours: dataviz reference palette, light surface.

  /opt/conda/envs/segtta/bin/python scripts/analysis/plot_midterm_figs.py [--only fig1 fig5 ...]     # from repro/
Output: docs/figs/midterm_2026-11-26/
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import common, paths  # noqa: E402

OUT = paths.REPO / "docs" / "figs" / "midterm_2026-11-26"
E = paths.EXP
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a9a8a2"
RC = {"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titleweight": "bold", "axes.titlesize": 12,
      "legend.frameon": False, "axes.grid": False}
plt.rcParams.update(RC)
ORDER = ["mae_in", "satmae", "scalemae", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16", "dino_in_s8",
         "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16", "sup_in_r50", "sup_in_vitb16",
         "sup_in_vitl16"]
DS = (("aid", "AID-C"), ("r45", "RESISC45-C"))
SHORT = {"rotate_new": "rotate", "scale_image": "scale", "translate_image_new": "translate", "clouds": "clouds", "data_gaps": "data gaps",
         "brightness_contrast": "brightness / contrast", "haze_update_new": "haze", "gaussian_noise_new": "Gaussian noise",
         "salt_and_pepper_noise": "salt & pepper", "gaussian_blur": "Gaussian blur", "motion_blur": "motion blur",
         "compression_artifacts": "JPEG"}
sig = lambda m, ds: json.load(open(E / "2026-11-16_backbone_feats" / "signals" / f"{m}_{ds}.json"))["streams"]


def rank(a):
    a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(len(a))
    return r


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / name)


def fig1():
    M = [("none", "no adaptation", GREY), ("tent", "Tent", ORANGE), ("sar", "SAR", ORANGE), ("eata", "EATA", ORANGE),
         ("dsbr", "DSBR", BLUE), ("flipf", "flip filter", BLUE), ("none_al", "alignment only (no learning)", AQUA),
         ("tent_al", "alignment + Tent", AQUA), ("dsbr_al", "alignment + DSBR", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (ds, title) in zip(axes, DS):
        V = np.array([[json.load(open(E / "2026-11-17_fm_tta" / ds / m / f"{k}.json"))["avg"] for m in ORDER] for k, _, _ in M])
        y = np.arange(len(M))[::-1]
        ax.barh(y, V.mean(1), height=0.5, color=[c for _, _, c in M], alpha=0.35, zorder=1)
        for i, (_, _, c) in enumerate(M):
            ax.scatter(V[i], np.full(len(ORDER), y[i]), s=14, color=c, edgecolors=SURF, linewidths=0.4, zorder=3)
            ax.text(101, y[i], f"{V[i].mean():.1f}", va="center", ha="left", color=INK, fontsize=10)
        ax.axvline(V[0].mean(), color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=2)
        ax.set_xlim(10, 100); ax.set_yticks(y); ax.set_yticklabels([n for _, n, _ in M])
        ax.set_xlabel("accuracy on corrupted streams (%), dots = 17 backbones"); ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6, label=n) for n, c in (("entropy-based", ORANGE), ("no confidence used", BLUE),
                                                                              ("statistics alignment", AQUA))]
    fig.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    save(fig, "fig1_methods.png")


def estimates():
    """per stream: true acc, atc, cot_ts, flip estimate (leave-one-backbone-out slope, as cluster_cot.summary)."""
    rows = []
    for ds, _ in DS:
        for m in ORDER:
            G = sig(m, ds); C = json.load(open(E / "2026-11-24_cluster_cot" / f"{m}_{ds}.json"))["streams"]
            for k in G:
                if k != "clean":
                    rows.append((m, ds, k, G[k]["acc"], G[k]["acc_al"], max(G[k]["flip"] - G["clean"]["flip"], 0), G[k]["atc_al"],
                                 G[k]["atc"], C[k]["cot_ts"], G[k]["ac"], G[k]["flip"]))
    ms = np.array([r[0] for r in rows]); A = np.array([r[3:] for r in rows], float)
    acc, al, fl, atc_al = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    est = np.empty(len(rows))
    for m in ORDER:
        te = ms == m
        est[te] = atc_al[te] - np.linalg.lstsq(fl[~te, None], (al - acc)[~te], rcond=None)[0][0] * fl[te]
    return rows, acc, A[:, 4], A[:, 5], est


def fig2():
    rows, acc, atc, cot, est = estimates()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
    for ax, (name, e) in zip(axes, (("ATC (confidence threshold)", atc), ("COT + temperature", cot), ("flip estimate (ours)", est))):
        bad = np.abs(e - acc) > 10
        ax.plot([0, 100], [0, 100], color=INK2, lw=0.8, zorder=1)
        ax.scatter(acc[~bad], e[~bad], s=7, color=BLUE, alpha=0.45, linewidths=0, zorder=2, rasterized=True)
        ax.scatter(acc[bad], e[bad], s=9, color=ORANGE, alpha=0.8, linewidths=0, zorder=3, rasterized=True)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
        ax.set_title(name, loc="left", pad=36); ax.set_xlabel("true accuracy (%)")
        ax.text(0, 1.02, f"mean abs. error {np.abs(e - acc).mean():.1f} pt\nabove 10 pt: {100 * bad.mean():.1f}% of streams, max {np.abs(e - acc).max():.0f} pt",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=INK2)
    axes[0].set_ylabel("estimated accuracy (%)")
    hs = [plt.Line2D([], [], marker="o", ls="", ms=6, mfc=BLUE, mec=BLUE, label="error within 10 pt"),
          plt.Line2D([], [], marker="o", ls="", ms=6, mfc=ORANGE, mec=ORANGE, label="error above 10 pt")]
    fig.legend(handles=hs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig2_estimation.png")


def fig3():
    models = ["moco_in", "sup_in_r50", "satdino_s16"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharey=True)
    for c, m in enumerate(models):
        G = sig(m, "r45"); K = [k for k in G if k != "clean"]
        acc = np.array([G[k]["acc"] for k in K])
        for r, (key, lab, col) in enumerate((("ac", "mean confidence (%)", ORANGE), ("flip", "flip rate after alignment (%)", BLUE))):
            x = np.array([G[k][key] for k in K]); ax = axes[r, c]
            ax.scatter(x, acc, s=22, color=col, edgecolors=SURF, linewidths=0.4)
            ax.set_xlabel(lab); ax.set_ylim(0, 100)
            if r == 0:
                ax.plot([0, 100], [0, 100], color=INK2, lw=0.8, zorder=0); ax.set_xlim(0, 100); ax.set_title(m, loc="left")
            ax.text(0.04 if r == 0 else 0.96, 0.92 if r == 0 else 0.92, f"Spearman {spear(x, acc):+.2f}", transform=ax.transAxes,
                    ha="left" if r == 0 else "right", va="top", fontsize=10)
        axes[0, 0].set_ylabel("accuracy (%)"); axes[1, 0].set_ylabel("accuracy (%)")
    fig.suptitle("RESISC45-C, one dot = one corrupted stream (12 corruptions x 5 severities)", x=0.01, ha="left", fontsize=11, color=INK2, fontweight="normal")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig3_signals.png")


def fig4():
    R = [json.load(open(E / "2026-11-24_cluster_cot" / "em_phases" / f"{m}_aid.json")) for m in ORDER]
    ck = R[0]["ckpts"]; x = np.arange(len(ck))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, key, ylab in ((axes[0], "acc", "accuracy on corrupted streams (%)"), (axes[1], "n_pred", "classes still predicted (of 30)")):
        for loss, col, name in (("tent", ORANGE, "Tent (entropy)"), ("dsbr", BLUE, "DSBR (class-balanced entropy)")):
            V = np.array([np.mean(r["losses"][loss][key][1:], 0) for r in R])   # backbone x ckpt (mean over 60 corrupted streams)
            for v in V:
                ax.plot(x, v, color=col, lw=0.7, alpha=0.3)
            ax.plot(x, V.mean(0), color=col, lw=2.4, marker="o", ms=4)
            ax.text(x[-1] + 0.12, V.mean(0)[-1], f"{name}\n{V.mean(0)[-1]:.1f}", color=INK, va="center", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels([f"{c // 1000}k" if c >= 1000 else str(c) for c in ck]); ax.set_xlabel("adaptation steps"); ax.set_ylabel(ylab); ax.set_xlim(-0.2, len(ck) + 2.2)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_title("AID-C, head-input affine only; thin lines = 17 backbones", loc="left", fontsize=11, color=INK2, fontweight="normal")
    fig.tight_layout()
    save(fig, "fig4_em_phases.png")


def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    order = None
    for ax, (ds, title) in zip(axes, DS):
        R = [json.load(open(E / "2026-11-24_cluster_cot" / "head_oracle" / f"{m}_{ds}.json")) for m in ORDER]
        names = R[0]["streams"]
        V = {}
        for c in common.NOISES:
            ix = [i for i, n in enumerate(names) if n.rsplit("_s", 1)[0] == c]
            V[c] = [np.mean([[r[k][i] for i in ix] for r in R]) for k in ("raw", "align", "ft_raw")]
        clean = np.mean([r["raw"][0] for r in R])
        if order is None:
            order = sorted(common.NOISES, key=lambda c: V[c][2] - V[c][1])
        for y, c in enumerate(order):
            raw, al, ft = V[c]
            ax.plot([raw, al], [y, y], color=AQUA, lw=5, alpha=0.55, solid_capstyle="butt", zorder=1)
            ax.plot([al, ft], [y, y], color=BLUE, lw=5, alpha=0.55, solid_capstyle="butt", zorder=1)
            ax.scatter([raw], [y], s=36, color=GREY, edgecolors=SURF, linewidths=0.6, zorder=3)
            ax.scatter([al], [y], s=36, color=AQUA, edgecolors=SURF, linewidths=0.6, zorder=3)
            ax.scatter([ft], [y], s=36, color=BLUE, edgecolors=SURF, linewidths=0.6, zorder=3)
            ax.text(101.5, y, f"+{ft - al:.0f}", va="center", ha="left", fontsize=10, color=INK, clip_on=False)
        ax.axvline(clean, color=INK, lw=1, ls=(0, (3, 3))); ax.text(clean, len(order) - 0.35, f"clean {clean:.1f}", ha="center", fontsize=10)
        ax.text(101.5, len(order) - 0.35, "headroom", ha="left", fontsize=10, color=INK2, clip_on=False)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([SHORT[c] for c in order]); ax.tick_params(axis="y", length=0)
        ax.set_xlim(28, 100); ax.set_xlabel("accuracy (%), mean of 17 backbones, severity 3 and 5"); ax.set_title(title, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=7, mfc=GREY, mec=SURF, label="as is"),
          plt.Line2D([], [], marker="o", ls="", ms=7, mfc=AQUA, mec=SURF, label="alignment (what we use; no labels)"),
          plt.Line2D([], [], marker="o", ls="", ms=7, mfc=BLUE, mec=SURF, label="upper bound, not a method: head re-fitted WITH labels of the corrupted stream")]
    fig.legend(handles=hs, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=10)
    save(fig, "fig5_headroom.png")


def fig6():
    pairs = [("MAE -> SatMAE", "mae_in", "satmae"), ("MAE -> Scale-MAE", "mae_in", "scalemae"), ("MoCo -> MoCo-fMoW", "moco_in", "moco_fmow"),
             ("DINO-S/16 -> SatDINO", "dino_in_s16", "satdino_s16"), ("DINO-S/8 -> SatDINO", "dino_in_s8", "satdino_s8"),
             ("DINO-B/16 -> SatDINO", "dino_in_b16", "satdino_b16"), ("DINOv3 web -> sat", "dinov3_web_l16", "dinov3_sat_l16")]
    cor = lambda m, ds: np.mean([v["acc"] for k, v in sig(m, ds).items() if k != "clean"])
    nat = lambda m, d: json.load(open(E / "2026-11-17_natural_shift" / f"{m}.json"))["directions"][d]["streams"]["ood_all"]["acc"]
    conds = [("AID clean", lambda m: sig(m, "aid")["clean"]["acc"]), ("AID-C (12 corruptions)", lambda m: cor(m, "aid")),
             ("RESISC45 -> UC Merced", lambda m: nat(m, "r45_to_ucm")), ("UC Merced -> RESISC45", lambda m: nat(m, "ucm_to_r45"))]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9), sharey=True)
    y = np.arange(len(pairs))[::-1]
    for ax, (title, f) in zip(axes, conds):
        d = np.array([f(s) - f(g) for _, g, s in pairs])
        ax.barh(y, d, height=0.55, color=[BLUE if v >= 0 else ORANGE for v in d])
        ax.axvline(0, color=INK2, lw=0.8)
        for yi, v in zip(y, d):
            ax.text(v + (1.2 if v >= 0 else -1.2), yi, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=10)
        ax.set_xlim(-44, 22); ax.set_title(title, loc="left", fontsize=11); ax.set_xlabel("satellite - generic (pt)")
        ax.set_yticks(y); ax.set_yticklabels([p[0] for p in pairs]); ax.tick_params(axis="y", length=0)
    hs = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="satellite pretraining better"), plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="generic pretraining better")]
    fig.legend(handles=hs, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
    save(fig, "fig6_pairs.png")


def fig7():
    blur = {"moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16", "dino_in_s8", "satdino_s8", "dino_in_b16", "satdino_b16",
            "dinov3_web_l16", "dinov3_sat_l16"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    rng = np.random.RandomState(0)
    for ax, (ds, title) in zip(axes, (("aid", "AID"), ("r45", "RESISC45"))):
        for yi, (grp, col, lab) in enumerate(((True, BLUE, "with Gaussian-blur\naugmentation (11)"), (False, ORANGE, "without (6)"))):
            ms = [m for m in ORDER if (m in blur) == grp]
            d = np.array([sig(m, ds)["clean"]["acc"] - np.mean([sig(m, ds)[f"gaussian_blur_s{s}"]["acc"] for s in range(1, 6)]) for m in ms])
            ax.scatter(d, yi + rng.uniform(-0.12, 0.12, len(d)), s=46, color=col, edgecolors=SURF, linewidths=0.6, zorder=3)
            ax.plot([d.mean()] * 2, [yi - 0.28, yi + 0.28], color=INK, lw=1.6, zorder=4)
            ax.text(d.mean(), yi + 0.36, f"mean {d.mean():.1f}", ha="center", fontsize=10)
            for m, v in zip(ms, d):
                if v == d.max():
                    ax.text(v + 0.6, yi - 0.02, m, va="center", fontsize=9, color=INK2)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["with Gaussian-blur\naugmentation (11)", "without (6)"]); ax.set_ylim(-0.6, 1.7)
        ax.tick_params(axis="y", length=0); ax.set_title(title, loc="left"); ax.set_xlabel("accuracy drop under Gaussian blur (pt)")
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlim(-1, 40)
    fig.tight_layout()
    save(fig, "fig7_blur_aug.png")


def fig8(cls="Port", sev=3):
    stages = [("sensor", ["gaussian_noise_new", "salt_and_pepper_noise", "data_gaps"]), ("optics / platform", ["gaussian_blur", "motion_blur"]),
              ("atmosphere", ["clouds", "haze_update_new"]), ("radiometric", ["brightness_contrast"]), ("transmission", ["compression_artifacts"]),
              ("geometry", ["scale_image", "rotate_new", "translate_image_new"])]
    rs = {"data_gaps", "clouds", "haze_update_new", "scale_image", "rotate_new"}
    d0 = sorted(glob.glob(f"{common.CORRUPT_DIR}/{cls}/gaussian_blur/severity_{sev}/*/*"))
    f0 = d0[len(d0) // 2]; ci = os.path.basename(os.path.dirname(f0)); name = os.path.basename(f0)
    clean = glob.glob(f"{common.TEST_DIR}/*/{name}") + glob.glob(f"{common.TEST_DIR}/*/{os.path.splitext(name)[0]}.*")
    items = [("clean", clean[0] if clean else None, False, "")]
    for st, cs in stages:
        for c in cs:
            items.append((SHORT[c], f"{common.CORRUPT_DIR}/{cls}/{c}/severity_{sev}/{ci}/{name}", c in rs, st))
    fig, axes = plt.subplots(2, 7, figsize=(14, 4.9))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (lab, f, is_rs, st) in zip(axes.ravel(), items):
        if f and os.path.exists(f):
            ax.imshow(Image.open(f).convert("RGB"))
        ax.set_title(lab, fontsize=10.5, color=INK, fontweight="bold" if is_rs else "normal", pad=3)
        tag = "" if not is_rs else ("  |  partly RS-specific" if lab == "haze" else "  |  RS-specific")
        ax.text(0.5, -0.07, st + tag, transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color=BLUE if is_rs else INK2)
    fig.subplots_adjust(wspace=0.05, hspace=0.32)
    save(fig, "fig8_corruptions.png")


def fig9():
    cor = ["brightness_contrast", "haze", "gaussian_noise", "salt_and_pepper_noise", "compression_artifacts", "gaussian_blur", "motion_blur", "clouds"]
    lab = ["brightness /\ncontrast", "haze", "Gaussian\nnoise", "salt &\npepper", "JPEG", "Gaussian\nblur", "motion\nblur", "clouds"]
    V = (("none", "no adaptation", GREY), ("bn", "BN alignment", BLUE), ("isa", "input alignment", ORANGE), ("isa_bn", "input + BN alignment", AQUA))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
    for ax, (m, title) in zip(axes, (("satmae_B", "SatMAE"), ("scalemae_B", "Scale-MAE"))):
        d = json.load(open(E / "2026-11-23_seg_signals" / "signals" / f"{m}.json"))
        S = d["streams"] if "streams" in d else d
        x = np.arange(len(cor)); w = 0.2
        for i, (k, name, col) in enumerate(V):
            v = [np.mean([S[f"{c}_s{s}"][k]["miou"] for s in range(1, 6)]) for c in cor]
            ax.bar(x + (i - 1.5) * w, v, width=w - 0.02, color=col, label=name)
        ax.axhline(S["clean"]["none"]["miou"], color=INK, lw=1, ls=(0, (3, 3)))
        ax.text(len(cor) - 0.45, S["clean"]["none"]["miou"] + 0.6, f"clean {S['clean']['none']['miou']:.1f}", ha="right", fontsize=10)
        ax.set_ylim(0, 72); ax.set_ylabel("mIoU (%)"); ax.set_title(title, loc="left"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[1].set_xticks(np.arange(len(cor))); axes[1].set_xticklabels(lab)
    axes[0].legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.32))
    fig.tight_layout()
    save(fig, "fig9_seg.png")


def fig10(model="mae_in", ds="aid"):
    """What head-input alignment does: schematic + one real stream (the severity-3 corruption with the largest gain)."""
    import torch
    from matplotlib.patches import FancyBboxPatch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cluster_cot import FEATS, align
    h = torch.load(f"{FEATS}/{model}/head_{ds}.pt"); te = torch.load(f"{FEATS}/{model}/{ds}_test.pt"); cd = torch.load(f"{FEATS}/{model}/{ds}_c.pt")
    z = lambda a: (a.float() - h["mu"]) / h["sd"]
    w, b = h["w"], h["b"]
    pred = lambda x: (x @ w + b).argmax(1)
    zc, yc = z(te["feats"]), te["y"]
    best = None
    for k, c in enumerate(common.NOISES):
        sel = (cd["corr"] == k) & (cd["sev"] == 3)
        zk, yk = z(cd["feats"][sel]), cd["y"][sel]
        za = align(zk)
        a0, a1 = 100 * (pred(zk) == yk).float().mean().item(), 100 * (pred(za) == yk).float().mean().item()
        if best is None or a1 - a0 > best[0]:
            best = (a1 - a0, c, zk, za, yk, a0, a1)
    _, cname, zk, za, yk, a0, a1 = best
    ac = 100 * (pred(zc) == yc).float().mean().item()
    nc = w.shape[1]
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.32, wspace=0.2)
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(-1, 101); ax.set_ylim(0, 42); ax.axis("off")

    def box(x, y, wd, text, fc, ec=INK2, bold=False, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), wd, 7.5, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1.1))
        ax.text(x + wd / 2, y + 3.75, text, ha="center", va="center", fontsize=10.5, color=tc, fontweight="bold" if bold else "normal")

    def arrow(x0, x1, y):
        ax.annotate("", xy=(x1, y + 3.75), xytext=(x0, y + 3.75), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))

    for y, tag, mid, fc, ec in ((28, "as trained and used without adaptation", "z = (h - mean_src) / std_src", "#eceae5", INK2),
                                (9, "test time, aligned (ours)", "z = (h - mean_batch) / std_batch", "#dbe9fb", BLUE)):
        ax.text(0, y + 10.2, tag, fontsize=11, fontweight="bold", color=INK, va="bottom")
        box(0, y, 13, "image", SURF); arrow(13.8, 17.2, y)
        box(18, y, 19, "frozen backbone", "#eceae5"); arrow(37.8, 41.2, y)
        ax.text(39.5, y + 6.2, "h", ha="center", fontsize=10.5, style="italic")
        box(42, y, 30, mid, fc, ec=ec, bold=True); arrow(72.8, 76.2, y)
        box(77, y, 13, "linear head", "#eceae5"); arrow(90.8, 94.2, y)
        ax.text(95, y + 3.75, "class", ha="left", va="center", fontsize=10.5)
    ax.text(57, 25.6, "mean / std of the clean training set, per channel (fixed)", ha="center", va="top", fontsize=9.5, color=INK2)
    ax.text(57, 6.2, "mean / std of the current test batch (64 images), per channel", ha="center", va="top", fontsize=9.5, color=BLUE)
    ax.text(57, 2.6, "no labels, no gradient step, nothing stored between batches", ha="center", va="top", fontsize=9.5, color=BLUE)

    # one channel: the largest mean shift among channels with an ordinary source spread
    ok = h["sd"] >= 0.5 * h["sd"].median()
    j = int((zk.mean(0).abs() * ok).argmax())
    ax1 = fig.add_subplot(gs[1, 0])
    lo, hi = np.percentile(torch.cat([zc[:, j], zk[:, j], za[:, j]]).numpy(), [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ax1.hist(zc[:, j].numpy(), bins=bins, color=GREY, alpha=0.55, label=f"clean (accuracy {ac:.1f}%)")
    ax1.hist(zk[:, j].numpy(), bins=bins, histtype="step", color=ORANGE, lw=2, label=f"{SHORT[cname]}, severity 3 ({a0:.1f}%)")
    ax1.hist(za[:, j].numpy(), bins=bins, histtype="step", color=BLUE, lw=2, label=f"same stream, aligned ({a1:.1f}%)")
    ax1.axvline(0, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax1.set_xlabel("value of one feature channel (source std units; 0 = source mean)"); ax1.set_ylabel("images")
    ax1.set_title(f"{model}, {ds.upper()}: the channel with the largest shift", loc="left", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9.5)
    # prediction share per class
    ax2 = fig.add_subplot(gs[1, 1])
    share = lambda p: np.bincount(p.numpy(), minlength=nc) / len(p) * 100
    sk, sa, st = share(pred(zk)), share(pred(za)), share(yk)
    o = np.argsort(-sk); x = np.arange(nc)
    ax2.bar(x, st[o], width=0.8, color=GREY, alpha=0.55, label="true share")
    ax2.plot(x, sk[o], color=ORANGE, lw=2, marker="o", ms=4, label="predicted, corrupted")
    ax2.plot(x, sa[o], color=BLUE, lw=2, marker="o", ms=4, label="predicted, aligned")
    names = sorted(os.listdir(common.TRAIN_DIR))
    ax2.annotate(f"{names[o[0]]}: {sk[o[0]]:.0f}% of predictions", xy=(0, sk[o[0]]), xytext=(4, sk[o[0]] * 0.92), fontsize=10,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax2.set_xlabel("classes, sorted by predicted share on the corrupted stream"); ax2.set_ylabel("share of images (%)")
    ax2.set_xticks([]); ax2.set_title("predictions collapse to a few classes; alignment spreads them again", loc="left", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9.5)
    save(fig, "fig10_alignment.png")
    print(f"fig10: {model} {ds} {cname}_s3 clean {ac:.1f} corrupted {a0:.1f} aligned {a1:.1f} channel {j}")


HL3 = {8: ("DenseResidential", BLUE), 14: ("MediumResidential", ORANGE), 25: ("SparseResidential", AQUA)}


def _feat(model, ds="aid"):
    import torch
    from cluster_cot import FEATS
    dev = torch.device("cuda")
    h = torch.load(f"{FEATS}/{model}/head_{ds}.pt"); tr = torch.load(f"{FEATS}/{model}/{ds}_train.pt")
    te = torch.load(f"{FEATS}/{model}/{ds}_test.pt"); cd = torch.load(f"{FEATS}/{model}/{ds}_c.pt")
    z = lambda a: ((a.float() - h["mu"]) / h["sd"]).to(dev).double()
    return h, z, tr, te, cd, dev


def _scatter3(ax, p, y, s=16):
    p = p.cpu().numpy(); y = y.cpu().numpy()
    oth = ~np.isin(y, list(HL3))
    ax.scatter(p[oth, 0], p[oth, 1], s=s * 0.45, c="#c4c3bd", alpha=0.5, linewidths=0, rasterized=True)
    for k, (_, col) in HL3.items():
        ax.scatter(p[y == k, 0], p[y == k, 1], s=s * 1.5, c=col, edgecolors=SURF, linewidths=0.5, rasterized=True)


def _legend3(fig, y=0.98):
    hs = [plt.Line2D([], [], marker="o", ls="", ms=9, mfc=c, mec=SURF, label=n) for n, c in HL3.values()]
    hs.append(plt.Line2D([], [], marker="o", ls="", ms=6, mfc="#c4c3bd", mec="#c4c3bd", label="other 27 classes"))
    fig.legend(handles=hs, loc="upper center", ncol=4, bbox_to_anchor=(0.5, y), fontsize=13)


def fig11(models=("satmae", "satdino_s8")):
    """Clean features: linearly separable, locally consistent, not one blob per class (two backbones, three 2-D views)."""
    import torch
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment as lsa
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cluster_cot import kmeans
    from plot_feature_dist import tsne
    plt.rcParams.update(RC); plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(len(models), 4, figsize=(16, 4.3 * len(models)), gridspec_kw={"width_ratios": [0.72, 1, 1, 1]})
    for r, m in enumerate(models):
        h, z, tr, te, cd, dev = _feat(m)
        ztr, ytr, x, y = z(tr["feats"]), tr["y"].to(dev), z(te["feats"]), te["y"].to(dev)
        w, b = h["w"].to(dev).double(), h["b"].to(dev).double(); nc = w.shape[1]
        lin = 100 * ((x @ w + b).argmax(1) == y).float().mean().item()
        knn = 100 * (F.one_hot(ytr[torch.cdist(x, ztr).topk(5, largest=False).indices], nc).sum(1).argmax(1) == y).float().mean().item()
        lab, _, _ = min((kmeans(x, nc, s_) for s_ in range(3)), key=lambda t: t[2])
        cont = (F.one_hot(lab, nc).double().T @ F.one_hot(y, nc).double()).cpu().numpy()
        i, j = lsa(-cont); km = 100 * cont[i, j].sum() / len(y)
        ax = axes[r, 0]
        vals = [("linear head", lin), ("5 nearest\nneighbours", knn), ("k-means\nclusters", km)]
        ax.barh([2, 1, 0], [v for _, v in vals], height=0.55, color=[INK2, INK2, ORANGE])
        for yy, (n, v) in zip([2, 1, 0], vals):
            ax.text(v + 2, yy, f"{v:.0f}%", va="center", fontsize=13, fontweight="bold")
        ax.set_yticks([2, 1, 0]); ax.set_yticklabels([n for n, _ in vals]); ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 118); ax.set_xticks([0, 50, 100]); ax.set_title(m, loc="left", fontsize=15)
        if r == len(models) - 1:
            ax.set_xlabel("accuracy on clean AID test (%)")
        xc = x - x.mean(0)
        V = torch.linalg.svd(xc, full_matrices=False)[2]
        cm = torch.stack([ztr[ytr == k].mean(0) for k in HL3])
        Q = torch.linalg.qr((cm[1:] - cm[0]).T)[0]
        views = [("directions of largest variance (PCA)", xc @ V[:2].T), ("plane through the 3 class means", (x - cm.mean(0)) @ Q),
                 ("t-SNE: who is near whom", tsne(xc @ V[:50].T))]
        for c, (title, pts) in enumerate(views):
            a = axes[r, c + 1]
            _scatter3(a, pts, y)
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_color("#b9b8b2")
            if r == 0:
                a.set_title(title, fontsize=13, color=INK2, fontweight="normal")
        print(f"fig11 {m}: lin {lin:.1f} knn {knn:.1f} km {km:.1f}", flush=True)
    _legend3(fig, 1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig11_feature_geometry.png")


def fig12(models=("mae_in", "dinov3_web_l16"), corr="motion_blur", sev=3):
    """A corruption moves the whole stream in one direction; alignment moves the mean back (two backbones)."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cluster_cot import align
    plt.rcParams.update(RC); plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4.4 * len(models)))
    for r, m in enumerate(models):
        h, z, tr, te, cd, dev = _feat(m)
        x, y = z(te["feats"]), te["y"].to(dev)
        w, b = h["w"].to(dev).double(), h["b"].to(dev).double()
        acc = lambda a, yy: 100 * ((a @ w + b).argmax(1) == yy).float().mean().item()
        sel = (cd["corr"] == common.NOISES.index(corr)) & (cd["sev"] == sev)
        xk, yk = z(cd["feats"][sel]), cd["y"][sel].to(dev)
        xa = align(xk)
        d = xk.mean(0) - x.mean(0); u1 = d / d.norm()
        xc = x - x.mean(0)
        u2 = torch.linalg.svd(xc - (xc @ u1)[:, None] * u1, full_matrices=False)[2][0]
        B = torch.stack([u1, u2]).T
        pts = [("clean", xc @ B, y, acc(x, y)), (f"{SHORT[corr]}, severity {sev}", (xk - x.mean(0)) @ B, yk, acc(xk, yk)),
               ("same stream, aligned", (xa - x.mean(0)) @ B, yk, acc(xa, yk))]
        allp = torch.cat([p for _, p, _, _ in pts])
        lim = [(allp[:, k].quantile(0.003).item(), allp[:, k].quantile(0.997).item()) for k in (0, 1)]
        for c, (title, p, yy, a) in enumerate(pts):
            ax = axes[r, c]
            _scatter3(ax, p, yy)
            ax.axvline(0, color=INK2, lw=1, ls=(0, (3, 3)), zorder=0)
            mx = p[:, 0].mean().item()
            ax.plot([mx], [lim[1][0]], marker="^", ms=13, color=INK, clip_on=False, zorder=5)
            if c == 1:
                ax.annotate("", xy=(mx, lim[1][1] * 0.93), xytext=(0, lim[1][1] * 0.93), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
                ax.text(mx / 2, lim[1][1] * 0.97, "shift", ha="center", va="bottom", fontsize=12)
            ax.set_xlim(*lim[0]); ax.set_ylim(lim[1][0], lim[1][1] * 1.12)
            ax.set_title(f"{title}\naccuracy {a:.1f}%", loc="left", fontsize=13)
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(m, fontsize=15, fontweight="bold")
            if r == len(models) - 1:
                ax.set_xlabel("direction of the common shift (source std units)")
        print(f"fig12 {m}: clean {pts[0][3]:.1f} {corr}_s{sev} {pts[1][3]:.1f} aligned {pts[2][3]:.1f} shift {d.norm().item():.1f}", flush=True)
    _legend3(fig, 1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig12_shift_alignment.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, f in (("fig1", fig1), ("fig2", fig2), ("fig3", fig3), ("fig4", fig4), ("fig5", fig5), ("fig6", fig6), ("fig7", fig7),
                    ("fig8", fig8), ("fig9", fig9), ("fig10", fig10), ("fig11", fig11), ("fig12", fig12)):
        if args.only is None or name in args.only:
            try:
                f()
            except Exception as e:  # keep going, report at the end
                print(f"FAILED {name}: {type(e).__name__}: {e}")
