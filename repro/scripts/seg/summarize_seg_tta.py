"""Potsdam segmentation TTA: mIoU change vs `none` per method (experiments/2026-10-20_seg_tta/outputs).

  all : mean over the 8 released corruptions x severities 1-5
  s35 : mean over severities 3 and 5 only (the subset CoTTA was run on), so every method can be compared to CoTTA
Per-corruption columns are the absolute mIoU averaged over the severities each method was run on.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402

CORR = ["brightness_contrast", "clouds", "compression_artifacts", "gaussian_blur", "gaussian_noise", "haze",
        "motion_blur", "salt_and_pepper_noise"]
SHORT = ["BC", "Cloud", "JPEG", "GBlur", "GNoise", "Haze", "MBlur", "S&P"]
METHODS = ["none", "bnadapt", "isa", "tent", "tent_bnbatch", "sar", "cotta"]

for m in sorted(os.listdir(paths.SEG_TTA / "outputs")):
    print(f"\n### {m}  (mIoU; none row absolute, others = change vs none)")
    print(f"{'method':13s} {'clean':>6s} {'all':>6s} {'s35':>6s}  " + " ".join(f"{s:>6s}" for s in SHORT))
    base = None
    for meth in METHODS:
        f = paths.SEG_TTA / "outputs" / m / f"{meth}.json"
        if not f.exists():
            continue
        r = json.load(open(f))["streams"]
        full = all(f"{c}_s1" in r for c in CORR)
        s35 = np.mean([r[f"{c}_s{s}"] for c in CORR for s in (3, 5)])
        allv = np.mean([r[f"{c}_s{s}"] for c in CORR for s in range(1, 6)]) if full else float("nan")
        per = [np.mean([r[f"{c}_s{s}"] for s in (range(1, 6) if full else (3, 5))]) for c in CORR]
        if meth == "none":
            base = (r["clean"], allv, s35)
        d = (lambda x, b: f"{x:6.2f}") if meth == "none" else (lambda x, b: f"{x - b:+6.2f}" if x == x else "     -")
        print(f"{meth:13s} {d(r['clean'], base[0])} {d(allv, base[1])} {d(s35, base[2])}  "
              + " ".join(f"{x:6.2f}" for x in per) + ("" if full else "   (sev 3,5 only)"))
