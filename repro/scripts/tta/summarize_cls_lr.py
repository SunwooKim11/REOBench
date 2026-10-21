"""lr sensitivity table: accuracy (%) and Δ vs no adaptation, per model x head config x method x lr."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import common  # noqa: E402
from reobench_tta import paths  # noqa: E402
HERE = str(paths.CLS_TTA)
LRS = [("0.001", ""), ("3e-4", "_lr0.0003"), ("1e-4", "_lr0.0001")]
def s5(r): return sum(r[n]["severity"][4] for n in common.NOISES) / len(common.NOISES)
for m in ["satmae", "scalemae"]:
    for cfg in ["P", "PN"]:
        d = f"{HERE}/outputs/{m}/{cfg}"
        base = json.load(open(f"{d}/none_s0.json"))
        print(f"\n### {m} / {cfg}   none: clean {base['clean']:.2f}  avg {base['avg']:.2f}  S5 {s5(base):.2f}")
        print(f"{'method':6s} {'lr':>6s} {'clean Δ':>8s} {'avg Δ':>8s} {'S5 Δ':>8s}")
        for meth in ["tent", "sar", "deyo"]:
            for lr, tag in LRS:
                f = f"{d}/{meth}_s0{tag}.json"
                if not os.path.exists(f) or "avg" not in json.load(open(f)):
                    continue
                r = json.load(open(f))
                print(f"{meth:6s} {lr:>6s} {r['clean'] - base['clean']:+8.1f} {r['avg'] - base['avg']:+8.1f} {s5(r) - s5(base):+8.1f}")
