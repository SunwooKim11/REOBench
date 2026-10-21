"""Rebuild outputs/{model}/{cfg}/{method}_s0.json from a finished tta.py log whose final save failed
(the folder was renamed temp -> repro while the run was going). Same format as tta.py."""
import json, re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import common  # noqa: E402
from reobench_tta import paths  # noqa: E402
os.chdir(paths.CLS_TTA)  # logs/ and outputs/ live in the experiment folder
model, cfg, method = sys.argv[1:4]
res = {"config": {"model": model, "head_cfg": cfg, "method": method, "seed": 0, "bs": 64,
                  "recovered_from_log": f"logs/{model}_{cfg}_{method}.log"}}
for line in open(f"logs/{model}_{cfg}_{method}.log"):
    m = re.match(r"\[(\d+)/61\] (\S+) s(\d): ([\d.]+)", line)
    if not m:
        continue
    noise, sev, acc = m.group(2), int(m.group(3)), float(m.group(4))
    if noise == "clean":
        res["clean"] = acc
    else:
        res.setdefault(noise, {"severity": [None] * 5})["severity"][sev - 1] = acc
assert "clean" in res and all(k in res and None not in res[k]["severity"] for k in common.NOISES), "incomplete log"
for k in common.NOISES:
    res[k]["mean"] = sum(res[k]["severity"]) / 5
res["avg"] = sum(res[k]["mean"] for k in common.NOISES) / len(common.NOISES)
out = f"outputs/{model}/{cfg}/{method}_s0.json"
assert not os.path.exists(out)
json.dump(res, open(out, "w"), indent=2)
print(out, res["clean"], round(res["avg"], 2))
