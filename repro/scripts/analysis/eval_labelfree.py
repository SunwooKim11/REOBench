"""Evaluate the label-free signals of labelfree_signals.py against the true accuracy of each stream.

Estimators in accuracy units: ATC on the raw head (-> raw accuracy), ATC on the source-aligned head (-> aligned
accuracy), and ATC_al - k * excess_flip (-> raw accuracy; excess_flip = flip - flip on the clean stream, k = least-squares
slope of (aligned acc - raw acc) on excess_flip fitted on the *other* backbones). Rank signals: Spearman with accuracy.

  ~/REOBench/.venv/bin/python scripts/analysis/eval_labelfree.py      # from repro/, CPU, seconds
Output: experiments/2026-11-16_backbone_feats/signals/summary.txt
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import paths  # noqa: E402

ORDER = ["mae_in", "satmae", "scalemae", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16", "dino_in_s8",
         "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16", "sup_in_r50", "sup_in_vitb16",
         "sup_in_vitl16"]
SHORT = {"rotate_new": "rotate", "scale_image": "scale", "translate_image_new": "transl", "clouds": "clouds", "data_gaps": "gaps",
         "brightness_contrast": "br/con", "haze_update_new": "haze", "gaussian_noise_new": "gnoise",
         "salt_and_pepper_noise": "S&P", "gaussian_blur": "gblur", "motion_blur": "mblur", "compression_artifacts": "JPEG"}
CLS_C = list(SHORT)

D = paths.EXP / "2026-11-16_backbone_feats" / "signals"


def rank(a):
    a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(len(a))
    for v in np.unique(a):
        r[a == v] = r[a == v].mean()
    return r


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def stats(e):
    v = np.abs(e)
    return f"MAE {v.mean():4.1f}  median {np.median(v):4.1f}  >5pt {100 * (v > 5).mean():4.1f}%  >10pt {100 * (v > 10).mean():4.1f}%  max {v.max():4.1f}  bias {e.mean():+4.1f}"


L, rows = [], []
for ds in ("aid", "r45"):
    L += [f"\n### {ds}: Spearman of each signal with accuracy over 60 streams (sign set so + = tracks accuracy) | ATC error (pt)",
          f"{'model':15s} {'conf':>5s} {'flip':>5s} {'mean':>5s} {'std':>5s} | {'ATC':>5s} {'max':>5s} {'bias':>5s} | {'ATC_al':>6s} {'max':>5s} {'bias':>5s} | flip_clean"]
    per = {c: {"atc": [], "atc_al": []} for c in CLS_C}
    for m in ORDER:
        S = json.load(open(D / f"{m}_{ds}.json"))["streams"]
        K = [(c, s) for c in CLS_C for s in range(1, 6)]
        g = lambda k: np.array([S[f"{c}_s{s}"][k] for c, s in K])
        acc, e, ea = g("acc"), g("atc") - g("acc"), g("atc_al") - g("acc_al")
        for i, (c, s) in enumerate(K):
            per[c]["atc"].append(e[i]); per[c]["atc_al"].append(ea[i])
            rows.append((m, ds, c, acc[i], g("acc_al")[i], g("flip")[i] - S["clean"]["flip"], g("atc")[i], g("atc_al")[i]))
        L.append(f"{m:15s} {spear(g('ac'), acc):5.2f} {-spear(g('flip'), acc):5.2f} {-spear(g('mean_shift_c'), acc):5.2f} "
                 f"{-spear(g('std_shift_c'), acc):5.2f} | {np.abs(e).mean():5.1f} {np.abs(e).max():5.1f} {e.mean():+5.1f} | "
                 f"{np.abs(ea).mean():6.1f} {np.abs(ea).max():5.1f} {ea.mean():+5.1f} | {S['clean']['flip']:5.1f}")
    L += [f"--- {ds} per corruption (17 backbones x 5 severities): MAE, bias (+ = overestimates accuracy)",
          " " * 12 + " ".join(f"{SHORT[c][:7]:>7s}" for c in CLS_C)]
    for k, n in (("atc", "ATC"), ("atc_al", "ATC_al")):
        L.append(f"{n + ' MAE':12s}" + " ".join(f"{np.mean(np.abs(per[c][k])):7.1f}" for c in CLS_C))
        L.append(f"{n + ' bias':12s}" + " ".join(f"{np.mean(per[c][k]):+7.1f}" for c in CLS_C))

ms = np.array([r[0] for r in rows])
acc, al, fl, atc, atc_al = (np.array([r[i] for r in rows]) for i in (3, 4, 5, 6, 7))
fl = np.maximum(fl, 0)
est = np.empty(len(rows))
for m in ORDER:
    te = ms == m
    est[te] = atc_al[te] - np.linalg.lstsq(fl[~te, None], (al - acc)[~te], rcond=None)[0][0] * fl[te]
k = np.linalg.lstsq(fl[:, None], al - acc, rcond=None)[0][0]
L += ["\n### pooled over 17 backbones x 2 datasets x 60 streams (2,040 streams)",
      f"ATC -> raw accuracy                 {stats(atc - acc)}",
      f"ATC_al -> aligned accuracy          {stats(atc_al - al)}",
      f"ATC_al - k*excess_flip -> raw acc   {stats(est - acc)}",
      f"(aligned acc - raw acc) vs excess_flip: slope k = {k:.3f}, Pearson r = {np.corrcoef(fl, al - acc)[0, 1]:.3f}",
      "\nper backbone MAE (both datasets): ATC -> raw | ATC_al - k*flip -> raw"]
L += [f"  {m:15s} {np.abs(atc - acc)[ms == m].mean():5.1f} {np.abs(est - acc)[ms == m].mean():5.1f}" for m in ORDER]
open(D / "summary.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L[-24:]))
