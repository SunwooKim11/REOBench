"""Depth sweep of statistics alignment: change vs no adaptation, per alignment position x corruption group.

Classification (AID, PN heads, severities 1-5): input = isa_batch, blocks.k = fsa at block k, norm / head:in.
Segmentation (Potsdam B models, severities 3 and 5): input = isa, blocks.k, classifier input, decoder BN = bnadapt.
Combination rows (scripts/archive/seg/run_combo.sh): batch ISA on the input + one feature alignment (`--with_isa`).
Groups (NOTES_tta.md §4.2): geometry, occlusion, radiometric, degradation.

  ~/REOBench/.venv/bin/python scripts/analysis/summarize_depth_sweep.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402

CLS_GROUPS = {"geometry": ["rotate_new", "scale_image", "translate_image_new"],
              "occlusion": ["clouds", "data_gaps"],
              "radiometric": ["brightness_contrast", "haze_update_new"],
              "degradation": ["gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
                              "compression_artifacts"]}
SEG_GROUPS = {"occlusion": ["clouds"], "radiometric": ["brightness_contrast", "haze"],
              "degradation": ["gaussian_noise", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
                              "compression_artifacts"]}
BLOCKS = [0, 3, 7, 11, 15, 19, 23]


def cls_scores(r):
    g = {k: np.mean([r[n]["mean"] for n in v]) for k, v in CLS_GROUPS.items()}
    return {"clean": r["clean"], "avg": r["avg"], **g}


def seg_scores(r, sevs=(3, 5)):
    s = r["streams"]
    g = {k: np.mean([s[f"{c}_s{v}"] for c in cs for v in sevs]) for k, cs in SEG_GROUPS.items()}
    allc = [c for cs in SEG_GROUPS.values() for c in cs]
    return {"clean": s["clean"], "avg": np.mean([s[f"{c}_s{v}"] for c in allc for v in sevs]), **g}


def table(title, rows, base_key="none"):
    keys = list(next(iter(rows.values())).keys())
    print(f"\n### {title}  ({base_key} absolute; other rows = change)")
    print(f"{'position':14s} " + " ".join(f"{k:>11s}" for k in keys))
    base = rows[base_key]
    for name, sc in rows.items():
        cells = [f"{sc[k]:11.2f}" if name == base_key else f"{sc[k] - base[k]:+11.2f}" for k in keys]
        print(f"{name:14s} " + " ".join(cells))


for m in ["satmae", "scalemae"]:
    d = paths.CLS_TTA / "outputs" / m / "PN"
    rows = {}
    for name, fn in [("none", "none_s0"), ("input (ISA)", "isa_batch_s0")] + \
            [(f"block {b}", f"fsa_s0_blocks.{b}") for b in BLOCKS] + [("final norm", "fsa_s0_norm"),
                                                                     ("head input", "fsa_s0_headin")] + \
            [(f"ISA + block {b}", f"fsa_s0_blocks.{b}_isa") for b in (7, 11)] + [("ISA + head in", "fsa_s0_headin_isa")]:
        f = d / f"{fn}.json"
        if f.exists():
            r = json.load(open(f))
            if "avg" in r:
                rows[name] = cls_scores(r)
    if len(rows) > 1:
        table(f"classification AID, {m} PN, severities 1-5", rows)

for m in ["scalemae_B", "satmae_B"]:
    d = paths.SEG_TTA / "outputs" / m
    rows = {}
    for name, fn in [("none", "none"), ("input (ISA)", "isa")] + [(f"block {b}", f"fsa_blocks.{b}") for b in BLOCKS] + \
            [("cls input", "fsa_dh.conv_segin"), ("decoder BN", "bnadapt")] + \
            [(f"ISA + block {b}", f"fsa_blocks.{b}_isa") for b in (3, 7, 11)] + [("ISA + dec BN", "bnadapt_isa")]:
        f = d / f"{fn}.json"
        if f.exists():
            r = json.load(open(f))
            if all(f"{c}_s{v}" in r["streams"] for cs in SEG_GROUPS.values() for c in cs for v in (3, 5)):
                rows[name] = seg_scores(r)
    if len(rows) > 1:
        table(f"segmentation Potsdam, {m}, severities 3 and 5", rows)
