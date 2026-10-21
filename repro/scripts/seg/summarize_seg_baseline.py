"""Potsdam mIoU per condition (mean over severities 1-5) vs REOBench Table 2. Only the 8 released corruptions."""
import glob, json, os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402
HERE = str(paths.SEG_BASELINE)
COLS = ["brightness_contrast", "clouds", "compression_artifacts", "gaussian_blur", "gaussian_noise", "haze",
        "motion_blur", "salt_and_pepper_noise"]
SHORT = ["BC", "Cloud", "JPEG", "GBlur", "GNoise", "Haze", "MBlur", "S&P"]
# Table 2 (clean + these 8 columns; paper Avg is over all 12 corruptions)
PAPER = {"scalemae": dict(clean=68.92, vals=[64.37, 65.43, 49.96, 64.86, 54.89, 63.89, 64.49, 52.12], avg12=60.02),
         "satmae": dict(clean=59.51, vals=[50.18, 37.39, 48.23, 57.89, 41.83, 44.95, 57.52, 36.07], avg12=49.59)}

def miou(d):
    f = sorted(glob.glob(f"{d}/eval_*.json"))
    return 100 * json.load(open(f[-1]))["metric"]["mIoU"] if f else None

for m in sorted(os.listdir(f"{HERE}/results")):
    R = f"{HERE}/results/{m}"
    clean = miou(f"{R}/clean")
    per = []
    for c in COLS:
        v = [miou(f"{R}/{c}_s{s}") for s in range(1, 6)]
        per.append(sum(v) / 5 if None not in v else None)
    print(f"\n### {m} (mIoU %, mean over severities)")
    print(f"{'':6s} {'clean':>6s} " + " ".join(f"{s:>6s}" for s in SHORT) + f" {'avg8':>6s}")
    fmt = lambda x: f"{x:6.2f}" if x is not None else "     -"
    done = [x for x in per if x is not None]
    print(f"{'ours':6s} {fmt(clean)} " + " ".join(fmt(x) for x in per) + f" {fmt(sum(done)/len(done) if len(done)==8 else None)}")
    if m.split("_")[0] in PAPER:
        p = PAPER[m.split("_")[0]]
        print(f"{'paper':6s} {p['clean']:6.2f} " + " ".join(f"{x:6.2f}" for x in p["vals"]) + f" {sum(p['vals'])/8:6.2f}"
              f"   (paper avg over 12 corruptions: {p['avg12']})")
