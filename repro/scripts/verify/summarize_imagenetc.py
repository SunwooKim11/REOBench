"""Compare outputs/{setting}/{method}.json with the paper numbers in REFERENCES.md (accuracy, or error for D)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402
HERE = str(paths.VERIFY_IMAGENETC)
COR = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
       'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def row(s):
    v = [float(x) for x in s.split()]
    return v[:15], (v[15] if len(v) > 15 else sum(v[:15]) / 15)


REF = {  # setting -> method -> (source, per-corruption, avg); metric
    "deyo_mild_r50": ("acc", {
        "none": ("DeYO T1", "2.2 2.9 1.8 17.9 9.8 14.8 22.5 16.9 23.3 24.4 58.9 5.4 16.9 20.7 31.7 18.0"),
        "tent": ("DeYO T1", "29.2 31.2 30.1 28.1 27.7 41.4 49.4 47.2 41.5 57.7 67.4 29.2 54.8 58.5 52.4 43.1"),
        "eata": ("DeYO T1", "34.9 37.1 35.8 33.4 33.0 47.1 52.7 51.6 45.7 60.0 68.1 44.4 57.9 60.6 55.1 47.8"),
        "sar": ("DeYO T1", "30.6 30.6 31.3 28.5 28.5 41.9 49.4 47.1 42.2 57.5 67.3 37.8 54.6 58.4 52.1 43.9"),
        "deyo": ("DeYO T1", "35.6 37.9 37.1 33.8 34.1 48.5 52.8 52.7 46.4 60.6 68.0 46.1 58.4 61.5 55.7 48.6")}),
    "ls_vitb": ("acc", {
        "none": ("DeYO T5", "9.4 6.7 8.3 29.1 23.4 34.0 27.1 15.8 26.4 47.4 54.7 44.0 30.5 44.5 47.6 29.9"),
        "tent": ("DeYO T5", "33.9 1.8 27.2 54.8 52.9 58.6 54.3 12.4 11.7 69.7 76.3 66.3 59.6 69.7 66.6 47.7"),
        "eata": ("DeYO T5", "36.2 34.7 35.5 43.4 44.3 49.3 48.5 53.2 53.5 62.3 72.7 18.8 58.0 64.7 62.8 49.2"),
        "sar": ("DeYO T5", "42.3 34.9 44.1 50.0 50.5 55.6 53.1 59.7 47.2 66.2 75.2 50.3 60.1 67.3 65.0 54.8"),
        "deyo": ("DeYO T5", "53.5 36.0 54.6 57.6 58.7 63.7 46.2 67.6 66.0 73.2 77.9 66.7 69.0 73.5 70.3 62.3")}),
    "cont_r50": ("err", {
        "none": ("ViDA T1", "97.8 97.1 98.2 81.7 89.8 85.2 78.0 83.5 77.1 75.9 41.3 94.5 82.5 79.3 68.6 82.0"),
        "cotta": ("ViDA T1", "84.7 82.1 80.6 81.3 79.0 68.6 57.5 60.3 60.5 48.3 36.6 66.1 47.2 41.2 46.0 62.7")}),
    "cont_vitb": ("err", {
        "none": ("ViDA T1", "53.0 51.8 52.1 68.5 78.8 58.5 63.3 49.9 54.2 57.7 26.4 91.4 57.5 38.0 36.2 55.8"),
        "vida": ("ViDA T1", "47.7 42.5 42.9 52.2 56.9 45.5 48.9 38.9 42.7 40.7 24.3 52.8 49.1 33.5 33.1 43.4")}),
    "eatac_vitb": ("acc", {
        "none": ("EATA-C TA", "12.9 17.6 11.7 34.4 27.7 43.7 36.2 43.4 45.4 52.8 73.3 45.5 37.9 54.7 60.2 39.84"),
        "tent": ("EATA-C TA", "33.4 42.1 41.4 48.8 45.2 54.9 48.2 55.6 55.1 64.4 75.2 62.5 51.6 65.5 65.0 53.92"),
        "cotta": ("EATA-C TA", "45.6 46.8 28.1 45.3 42.0 57.6 49.8 59.5 57.6 65.2 74.9 58.2 55.5 66.3 65.7 54.54"),
        "sar": ("EATA-C TA", "43.1 44.0 44.3 50.4 48.2 55.1 49.0 56.9 56.0 64.7 75.3 62.4 53.5 65.3 64.9 55.53"),
        "eata": ("EATA-C TA", "50.5 51.3 51.3 55.9 56.0 60.7 58.0 64.2 62.2 70.1 77.4 66.7 63.9 70.5 68.9 61.84")}),
}
REF["eatac_src"] = REF["eatac_vitb"]  # candidate checkpoints for EATA-C's ViT (none only)
SRC_EXTRA = {"ls_vitb": {"sar": ("SAR T2", 58.0), "tent": ("SAR T2", 47.3), "eata": ("SAR T2", 49.9)}}

hdr = " ".join(c[:5] for c in COR)
for setting in sorted(os.listdir(f"{HERE}/outputs")):
    if setting == "smoke":
        continue
    metric, refs = REF.get(setting, ("acc", {}))
    print(f"\n### {setting} ({metric}, ours / paper)")
    print(f"{'method':16s} {'ours':>6s} {'paper':>6s} {'diff':>6s}  max|diff| per corruption")
    for fn in sorted(os.listdir(f"{HERE}/outputs/{setting}")):
        r = json.load(open(f"{HERE}/outputs/{setting}/{fn}"))
        name = fn[:-5]
        ours = [r["acc"][c] if metric == "acc" else 100 - r["acc"][c] for c in COR if c in r["acc"]]
        avg = r["avg_acc"] if metric == "acc" else r["avg_err"]
        meth = r["config"]["method"]
        if meth in refs and len(ours) == 15:
            src, s = refs[meth]
            per, pavg = row(s)
            d = [o - q for o, q in zip(ours, per)]
            worst = max(range(15), key=lambda i: abs(d[i]))
            extra = ""
            if meth in SRC_EXTRA.get(setting, {}):
                extra = f"  ({SRC_EXTRA[setting][meth][0]}: {SRC_EXTRA[setting][meth][1]})"
            print(f"{name:16s} {avg:6.2f} {pavg:6.2f} {avg - pavg:+6.2f}  {abs(d[worst]):.1f} ({COR[worst]}) [{src}]{extra}")
            print(f"{'':16s} ours : " + " ".join(f"{x:5.1f}" for x in ours))
            print(f"{'':16s} paper: " + " ".join(f"{x:5.1f}" for x in per))
        else:
            print(f"{name:16s} {avg:6.2f}   (no paper row)  " + " ".join(f"{x:5.1f}" for x in ours))
