"""Tabulate C_ablation results: mean ± std over seeds, per model and config."""
import glob, json, os, sys, statistics as st
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import common  # noqa: E402
from reobench_tta import paths  # noqa: E402
HERE = str(paths.CLS_ABLATION)
ORDER = ["A0", "A0ep", "PN", "norm", "bnhead", "optim", "aug", "bbeval", "B"]
SHORT = {"gaussian_noise_new": "GN", "salt_and_pepper_noise": "SP", "gaussian_blur": "GB", "motion_blur": "MB",
         "brightness_contrast": "BC", "clouds": "Cld", "haze_update_new": "Haze", "data_gaps": "Gap",
         "compression_artifacts": "JPEG", "rotate_new": "Rot", "scale_image": "Scale", "translate_image_new": "Trans"}
def ms(v): return f"{st.mean(v):5.1f}±{st.stdev(v):3.1f}" if len(v) > 1 else f"{v[0]:5.1f}    "
for m in ["satmae", "scalemae"]:
    print(f"\n### {m}  (mean±std over seeds; n=seeds)")
    print(f"{'config':8s} n {'clean':>9s} {'avg':>9s} " + " ".join(f"{SHORT[n]:>5s}" for n in common.NOISES))
    p = common.PAPER[m]
    print(f"{'paper':8s} - {p['clean']:9.2f} {p['avg']:9.2f} " + " ".join(f"{p[n]:5.1f}" for n in common.NOISES))
    for c in ORDER:
        rs = [json.load(open(f)) for f in sorted(glob.glob(f"{HERE}/outputs/{m}/{c}_s*.json"))]
        if not rs: continue
        print(f"{c:8s} {len(rs)} {ms([r['clean'] for r in rs]):>9s} {ms([r['avg'] for r in rs]):>9s} "
              + " ".join(f"{st.mean([r[n]['mean'] for r in rs]):5.1f}" for n in common.NOISES))
