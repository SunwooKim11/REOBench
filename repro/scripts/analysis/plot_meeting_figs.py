"""Figures for the 2026-10 advisor meeting (static PNGs for slides), from results already on disk.

  fig1_why_tta_fails.png  : source-model confidence vs accuracy and prediction collapse under corruption (entropy_bias.py)
  fig2_depth_sweep.png    : change vs no adaptation by alignment position, radiometric vs degradation (4 settings)
  fig3_combination.png    : ISA alone / block 7 alone / ISA + block 7, radiometric and degradation
  fig4_corruptions.png    : one AID test image under radiometric and degradation corruptions + affine R2 (pixel_affine.py)
  fig5_feature_shift.png  : per-channel mean / scale shift by depth (feature_shift.py)
  fig6_isa_example.png    : what ISA does to a real haze batch (pixel statistics, histograms)
  fig7_norm_axes.png      : which values input Normalize / LayerNorm / BatchNorm / ISA average over

  /opt/conda/envs/segtta/bin/python scripts/analysis/plot_meeting_figs.py      # from repro/ (needs matplotlib)
Output: docs/figs/meeting_2026-10/
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402

OUT = paths.REPO / "docs" / "figs" / "meeting_2026-10"
OUT.mkdir(parents=True, exist_ok=True)
# reference palette (dataviz skill, light mode): slot 1 blue, 2 orange, 3 aqua; text and grid tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "legend.frameon": False})

CLS_GROUPS = {"radiometric": ["brightness_contrast", "haze_update_new"],
              "degradation": ["gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
                              "compression_artifacts"]}
SEG_GROUPS = {"radiometric": ["brightness_contrast", "haze"],
              "degradation": ["gaussian_noise", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
                              "compression_artifacts"]}
BLOCKS = [0, 3, 7, 11, 15, 19, 23]


def cls_group(model, fn, g):
    r = json.load(open(paths.CLS_TTA / "outputs" / model / "PN" / f"{fn}.json"))
    return np.mean([r[c]["mean"] for c in CLS_GROUPS[g]])


def seg_group(model, fn, g):
    s = json.load(open(paths.SEG_TTA / "outputs" / model / f"{fn}.json"))["streams"]
    return np.mean([s[f"{c}_s{v}"] for c in SEG_GROUPS[g] for v in (3, 5)])


SETTINGS = [  # title, getter, none file, input file, block file pattern, extra (label, file) rows
    ("Classification · SatMAE", lambda fn, g: cls_group("satmae", fn, g), "none_s0", "isa_batch_s0", "fsa_s0_blocks.{}",
     [("final\nnorm", "fsa_s0_norm"), ("head\ninput", "fsa_s0_headin")]),
    ("Classification · ScaleMAE", lambda fn, g: cls_group("scalemae", fn, g), "none_s0", "isa_batch_s0", "fsa_s0_blocks.{}",
     [("final\nnorm", "fsa_s0_norm"), ("head\ninput", "fsa_s0_headin")]),
    ("Segmentation · SatMAE", lambda fn, g: seg_group("satmae_B", fn, g), "none", "isa", "fsa_blocks.{}",
     [("classifier\ninput", "fsa_dh.conv_segin"), ("decoder\nBN", "bnadapt")]),
    ("Segmentation · ScaleMAE", lambda fn, g: seg_group("scalemae_B", fn, g), "none", "isa", "fsa_blocks.{}",
     [("classifier\ninput", "fsa_dh.conv_segin"), ("decoder\nBN", "bnadapt")]),
]


# ---------------------------------------------------------------- fig 1: why entropy TTA barely helps
eb = json.load(open(paths.EXP / "2026-10-28_entropy_bias" / "results.json"))
show = ["clean", "rotate_new", "clouds", "haze_update_new", "gaussian_noise_new", "brightness_contrast",
        "gaussian_blur", "compression_artifacts", "motion_blur"]
names = {"clean": "clean", "rotate_new": "rotate", "clouds": "clouds", "haze_update_new": "haze",
         "gaussian_noise_new": "Gauss noise", "brightness_contrast": "bright/contrast", "gaussian_blur": "Gauss blur",
         "compression_artifacts": "JPEG", "motion_blur": "motion blur"}
fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
for row, model in enumerate(["satmae", "scalemae"]):
    r = eb[model]
    y = np.arange(len(show))[::-1]
    ax = axes[row, 0]
    ax.barh(y + 0.2, [r[s]["low_ent"] for s in show], 0.38, color=AQUA, label="share judged confident (entropy < 0.4 ln C)")
    ax.barh(y - 0.2, [r[s]["low_ent_acc"] for s in show], 0.38, color=BLUE, label="accuracy of those confident samples")
    ax.set_yticks(y, [names[s] for s in show]); ax.set_xlim(0, 100); ax.set_xlabel("%")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(f"{'SatMAE' if model == 'satmae' else 'ScaleMAE'}: entropy says 'confident', but it is often wrong")
    ax = axes[row, 1]
    ax.barh(y, [r[s]["top1_share"] for s in show], 0.6, color=ORANGE)
    ax.axvline(100 / 30, color=INK2, lw=1.2, ls="--")
    ax.text(100 / 30 + 0.8, y[-1] - 0.45, "uniform 3.3%", color=INK2, fontsize=9, va="center")
    for yi, s in zip(y, show):
        ax.text(r[s]["top1_share"] + 0.8, yi, f"{r[s]['top1_share']:.0f}%  {r[s]['top_classes'][0][0]}",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [names[s] for s in show]); ax.set_xlabel("% of predictions taken by the most predicted class")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(0, max(r[s]["top1_share"] for s in show) * 1.35)
    ax.set_title("predictions collapse onto one class (label = that class)")
fig.suptitle("Source models on AID (PN heads, severity 5): the signal entropy-TTA learns from is unreliable",
             fontweight="bold", fontsize=13)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower left", ncol=2, fontsize=10, bbox_to_anchor=(0.06, 0.0))
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "fig1_why_tta_fails.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- fig 2: depth sweep
fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), sharey=False)
for ax, (title, get, none, inp, blk, extra) in zip(axes.flat, SETTINGS):
    labels = ["input\n(ISA)"] + [f"b{b}" for b in BLOCKS] + [e[0] for e in extra]
    files = [inp] + [blk.format(b) for b in BLOCKS] + [e[1] for e in extra]
    x = np.arange(len(labels))
    ax.axhline(0, color=INK2, lw=1)
    ax.axvspan(1.5, 4.5, color="#f1f0ec", zorder=0)   # blocks 3-11
    for g, col in [("radiometric", ORANGE), ("degradation", BLUE)]:
        base = get(none, g)
        d = [get(f, g) - base for f in files]
        ax.plot(x[:8], d[:8], color=col, lw=2, marker="o", ms=6, label=g)
        ax.plot(x[8:], d[8:], color=col, lw=0, marker="o", ms=6)
    ax.set_xticks(x, labels, fontsize=9); ax.set_title(title)
    ax.set_ylabel("Δ vs no adaptation (%p)"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.text(3, ax.get_ylim()[1] * 0.92, "blocks 3–11", ha="center", color=INK2, fontsize=9)
fig.suptitle("Where to align: radiometric is fixed at the input, degradation in the middle blocks (ViT-L, 24 blocks)",
             fontweight="bold", fontsize=13)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(OUT / "fig2_depth_sweep.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- fig 3: combination
combo = [  # setting label, getter, none, ISA, block 7, ISA + block 7
    ("Cls SatMAE", SETTINGS[0][1], "none_s0", "isa_batch_s0", "fsa_s0_blocks.7", "fsa_s0_blocks.7_isa"),
    ("Cls ScaleMAE", SETTINGS[1][1], "none_s0", "isa_batch_s0", "fsa_s0_blocks.7", "fsa_s0_blocks.7_isa"),
    ("Seg SatMAE", SETTINGS[2][1], "none", "isa", "fsa_blocks.7", "fsa_blocks.7_isa"),
    ("Seg ScaleMAE", SETTINGS[3][1], "none", "isa", "fsa_blocks.7", "fsa_blocks.7_isa"),
]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
w = 0.26
for ax, g in zip(axes, ["radiometric", "degradation"]):
    x = np.arange(len(combo))
    for k, (lab, col) in enumerate([("ISA (input)", ORANGE), ("block 7", BLUE), ("ISA + block 7", AQUA)]):
        vals = [c[1](c[3 + k], g) - c[1](c[2], g) for c in combo]
        bars = ax.bar(x + (k - 1) * w, vals, w - 0.03, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.4 if v >= 0 else -1.6), f"{v:+.1f}", ha="center",
                    fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=1); ax.set_xticks(x, [c[0] for c in combo]); ax.set_title(g); ax.set_ylim(-7, 28)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("Δ vs no adaptation (%p)"); axes[0].legend(loc="upper left")
fig.suptitle("Combining input and block-7 alignment: each corruption group keeps (about) the better of the two",
             fontweight="bold", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "fig3_combination.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- fig 4: corruption examples + affine R2
TEST = "/data/rs/reobench/AID/AID_clean_renamed/test"
CORR = "/data/rs/reobench/AID/corrupted_renamed"
cls_names = sorted(os.listdir(TEST))
cname, fname = "Farmland", sorted(os.listdir(f"{TEST}/Farmland"))[3]
ci = cls_names.index(cname)
panels = [("clean", f"{TEST}/{cname}/{fname}", None),
          ("brightness/contrast", f"{CORR}/{cname}/brightness_contrast/severity_5/{ci}/{fname}", 0.975),
          ("haze", f"{CORR}/{cname}/haze_update_new/severity_5/{ci}/{fname}", 0.993),
          ("Gaussian noise", f"{CORR}/{cname}/gaussian_noise_new/severity_5/{ci}/{fname}", 0.801),
          ("motion blur", f"{CORR}/{cname}/motion_blur/severity_5/{ci}/{fname}", 0.562),
          ("JPEG", f"{CORR}/{cname}/compression_artifacts/severity_5/{ci}/{fname}", 0.849)]
fig, axes = plt.subplots(1, 6, figsize=(16, 3.6))
for ax, (lab, p, r2) in zip(axes, panels):
    ax.imshow(Image.open(p).convert("RGB")); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    group = "" if r2 is None else ("radiometric" if r2 > 0.9 else "degradation")
    ax.set_title(lab, fontsize=11)
    if r2 is not None:
        ax.set_xlabel(f"{group}\nbest affine fix R² = {r2:.3f}", fontsize=10,
                      color=ORANGE if r2 > 0.9 else BLUE, fontweight="bold")
fig.suptitle("Severity 5 on one AID image. R² = share of clean pixels a per-channel a·x+b can recover "
             "(mean over 200 images, pixel_affine.py)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig4_corruptions.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- fig 5: feature shift by depth
FS_GROUPS = {"radiometric": ["brightness_contrast", "haze_update_new"],
             "degradation": ["gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
                             "compression_artifacts"]}
pos = ["embed"] + [f"block{i}" for i in range(24)]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
for model, ls in [("satmae", "-"), ("scalemae", "--")]:
    fs = json.load(open(paths.EXP / "2026-10-22_feature_shift" / f"{model}.json"))["shift"]
    for g, col in [("radiometric", ORANGE), ("degradation", BLUE)]:
        for ax, m in zip(axes, ["d_sd", "d_mu"]):
            v = [np.mean([fs[c][p][m] for c in FS_GROUPS[g]]) for p in pos]
            ax.plot(range(len(pos)), v, color=col, ls=ls, lw=2, label=f"{g} · {'SatMAE' if model == 'satmae' else 'ScaleMAE'}")
for ax, t in zip(axes, ["scale shift |log σ_corr/σ_clean| (per channel, mean)",
                        "mean shift |μ_corr − μ_clean| / σ_clean (per channel, mean)"]):
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24], ["input\n(embed)", "b3", "b7", "b11", "b15", "b19", "b23"])
    ax.set_title(t, fontsize=11); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axvspan(4, 12, color="#f1f0ec", zorder=0)
axes[0].legend(fontsize=9)
fig.suptitle("Radiometric = scale change at the input; degradation = no mean shift at the input, "
             "a mean shift in the middle blocks (AID, severity 5)", fontweight="bold", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "fig5_feature_shift.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- fig 6: what ISA does, on a real batch
TRAIN_MU = np.array([0.39845189452171326, 0.4096486270427704, 0.3685896098613739])   # cls_tta.py log, 2,000 train images
TRAIN_SD = np.array([0.20769336819648743, 0.18474073708057404, 0.18328900635242462])
rng = np.random.RandomState(0)
items = [(c, f) for c in cls_names for f in sorted(os.listdir(f"{TEST}/{c}"))]
batch = [items[i] for i in rng.choice(len(items), 8, replace=False)]
load = lambda p: np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255
clean = np.stack([load(f"{TEST}/{c}/{f}") for c, f in batch])
hazy = np.stack([load(f"{CORR}/{c}/haze_update_new/severity_5/{cls_names.index(c)}/{f}") for c, f in batch])
mu_t, sd_t = hazy.mean((0, 1, 2)), hazy.std((0, 1, 2))
fixed = np.clip((hazy - mu_t) / sd_t * TRAIN_SD + TRAIN_MU, 0, 1)
fig = plt.figure(figsize=(16, 7.6))
gs = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 0.15, 2.6])
for j in range(4):
    for i, (img, lab) in enumerate([(clean, "clean"), (hazy, "haze (severity 5)"), (fixed, "haze after ISA")]):
        ax = fig.add_subplot(gs[i, j]); ax.imshow(img[j]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if j == 0:
            ax.set_ylabel(lab, fontsize=11, color=INK)
ax = fig.add_subplot(gs[:, 5])
bins = np.linspace(0, 1, 60)
for img, lab, col, ls in [(clean, "clean (same 8 images)", INK2, "-"), (hazy, "haze batch", ORANGE, "-"),
                          (fixed, "haze after ISA", AQUA, "--")]:
    h, _ = np.histogram(img[..., 0].ravel(), bins=bins, density=True)
    ax.plot((bins[1:] + bins[:-1]) / 2, h, color=col, lw=2, ls=ls, label=lab)
ax.set_xlabel("red-channel pixel value"); ax.set_yticks([]); ax.legend(loc="upper right", fontsize=9)
ax.set_title("red channel: the haze batch is squeezed and shifted;\nISA stretches it back", fontsize=11)
txt = ("per channel (R, G, B), over all pixels of the 8 test images:\n"
       f"  haze batch   mean {', '.join(f'{v:.2f}' for v in mu_t)}   std {', '.join(f'{v:.2f}' for v in sd_t)}\n"
       f"  train (2,000 clean)   mean {', '.join(f'{v:.2f}' for v in TRAIN_MU)}   std {', '.join(f'{v:.2f}' for v in TRAIN_SD)}\n"
       "  x' = (x - batch mean) / batch std * train std + train mean   (6 numbers per batch, no learning)")
fig.text(0.02, 0.005, txt, fontsize=10, family="monospace", color=INK, va="bottom")
fig.suptitle("ISA on a real AID test batch: match each RGB channel's mean / std of the test batch to the training set",
             fontweight="bold", fontsize=13)
fig.tight_layout(rect=(0, 0.11, 1, 0.97)); fig.savefig(OUT / "fig6_isa_example.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- fig 7: which numbers each normalization averages
import matplotlib.patches as mpatches  # noqa: E402
NI, NT, NC = 3, 4, 6          # images in the batch, tokens (or pixels) per image, channels


def grid(ax, title, sub, cells, color, hatch=None):
    for im in range(NI):
        for t in range(NT):
            for c in range(NC):
                y = -(im * (NT + 0.6) + t)
                on = (im, t, c) in cells
                ax.add_patch(mpatches.Rectangle((c, y), 0.92, 0.92, facecolor=color if on else "#ffffff",
                                                edgecolor=INK2 if not on else color, lw=0.8, hatch=hatch if on else None,
                                                alpha=1.0 if on else 1.0))
        ax.text(-0.3, -(im * (NT + 0.6)) - 1.0, f"image {im + 1}", ha="right", va="center", fontsize=9, color=INK2)
    ax.text(NC / 2, 1.5, "channels →", ha="center", fontsize=9, color=INK2)
    ax.set_xlim(-2.6, NC + 0.2); ax.set_ylim(-(NI * (NT + 0.6)) - 0.2, 2.2); ax.axis("off")
    ax.set_title(title, fontsize=11.5)
    ax.text(NC / 2 - 1.2, -(NI * (NT + 0.6)) - 0.9, sub, ha="center", va="top", fontsize=9.5, color=INK)


fig = plt.figure(figsize=(17, 9.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1])
col_all = {(im, t, 1) for im in range(NI) for t in range(NT)}
grid(fig.add_subplot(gs[0, 0]), "① input Normalize (pre-processing)",
     "one mean / std per channel,\nbut FIXED numbers from the\npretraining data - never computed\nfrom the test images",
     col_all, "#d9d8d3", hatch="//")
grid(fig.add_subplot(gs[0, 1]), "② ViT LayerNorm (every block)",
     "one mean / std per TOKEN, over\nits 1,024 channels, recomputed\nevery time - each token on its own,\nnothing shared across tokens",
     {(1, 2, c) for c in range(NC)}, BLUE)
grid(fig.add_subplot(gs[0, 2]), "③ UperNet BatchNorm (decoder)",
     "one mean / std per CHANNEL over\nthe batch; at test time the stored\ntraining values are used\n(bnadapt = recompute from test batch)",
     col_all, ORANGE)
grid(fig.add_subplot(gs[0, 3]), "ISA / feature alignment (ours)",
     "like ③ with test-batch values, then\nmapped to the SOURCE mean / std;\nISA: at the pixels (3 channels)\nfsa: at a block output (1,024 channels)",
     col_all, AQUA)
# bottom: why LayerNorm cannot undo a per-channel shift
ax = fig.add_subplot(gs[1, :2])
tok = np.array([1.0, -0.5, 2.0, -1.0, 0.3, 0.8])
cor = tok.copy(); cor[1] += 2.0                                       # corruption: +2 on channel 2, in every token
ln = lambda v: (v - v.mean()) / v.std()
xx = np.arange(NC)
ax.bar(xx - 0.2, ln(tok), 0.38, color=INK2, label="clean token after LayerNorm")
ax.bar(xx + 0.2, ln(cor), 0.38, color=BLUE, label="corrupted token after LayerNorm")
ax.axhline(0, color=INK2, lw=1); ax.set_xticks(xx, [f"ch {i + 1}" for i in xx])
ax.legend(fontsize=9, loc="upper right"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("a corruption adds +2 to channel 2 of every token: LayerNorm (per token) does not remove it", fontsize=11)
ax = fig.add_subplot(gs[1, 2:]); ax.axis("off")
ax.text(0, 0.95, "Why this matters", fontsize=12, fontweight="bold", va="top")
ax.text(0, 0.80,
        "• LayerNorm subtracts the token's own average over channels. A shift that hits the\n"
        "  same channel in every token barely moves that average, so it survives (ch 2 above).\n"
        "• A per-channel shift is exactly what corruptions cause inside the network (fig 5),\n"
        "  and ViT has no BatchNorm to absorb it - the backbone has only LayerNorm.\n"
        "• To remove it you must compare the SAME channel across many tokens / images\n"
        "  (column in ③ / ISA), i.e. use batch statistics: that is what bnadapt, ISA and\n"
        "  feature alignment do, at different depths.",
        fontsize=10.5, va="top", color=INK, linespacing=1.5)
fig.suptitle("Three normalizations, three different sets of numbers (highlighted = values that share one mean / std)",
             fontweight="bold", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "fig7_norm_axes.png", dpi=200); plt.close(fig)
print("saved to", OUT)
