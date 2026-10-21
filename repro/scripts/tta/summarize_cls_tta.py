"""Tabulate D_tta results: per model x head config, rows = TTA method (accuracy %, Δ vs no adaptation)."""
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import common  # noqa: E402
from reobench_tta import paths  # noqa: E402
HERE = str(paths.CLS_TTA)
SHORT = {"gaussian_noise_new": "GN", "salt_and_pepper_noise": "SP", "gaussian_blur": "GB", "motion_blur": "MB",
         "brightness_contrast": "BC", "clouds": "Cld", "haze_update_new": "Haze", "data_gaps": "Gap",
         "compression_artifacts": "JPEG", "rotate_new": "Rot", "scale_image": "Scale", "translate_image_new": "Trans"}
for m in ["satmae", "scalemae"]:
    for cfg in ["P", "PN"]:
        rs = {}
        for meth in ["none", "tent", "sar", "deyo", "eata", "cotta", "lscd", "isa_batch", "isa_image"]:
            f = f"{HERE}/outputs/{m}/{cfg}/{meth}_s0.json"
            if os.path.exists(f):
                rs[meth] = json.load(open(f))
        if not rs:
            continue
        print(f"\n### {m} / head {cfg}   (mean over severities; 'S5' = severity-5 average; Δ vs none in [])")
        print(f"{'method':9s} {'clean':>13s} {'avg':>13s} {'S5':>13s} " + " ".join(f"{SHORT[n]:>5s}" for n in common.NOISES))
        base = rs.get("none")
        for meth, r in rs.items():
            if "avg" not in r:
                continue
            s5 = sum(r[n]["severity"][4] for n in common.NOISES) / len(common.NOISES)
            r["S5"] = s5
            def cell(k):
                v = r[k]
                if base is None or meth == "none" or k not in base:
                    return f"{v:6.2f}       "
                b = base[k] if k != "S5" else sum(base[n]["severity"][4] for n in common.NOISES) / len(common.NOISES)
                return f"{v:6.2f} [{v - b:+5.1f}]"
            print(f"{meth:9s} {cell('clean'):>13s} {cell('avg'):>13s} {cell('S5'):>13s} "
                  + " ".join(f"{r[n]['mean']:5.1f}" for n in common.NOISES))
