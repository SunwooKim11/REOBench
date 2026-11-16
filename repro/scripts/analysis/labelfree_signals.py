"""Label-free signals of accuracy under shift, from the saved features and stage-1 heads (no test labels, no clean pairs).

Question: when the head's confidence stops tracking accuracy, does a signal built from source statistics still track it?
Per backbone x dataset x stream (clean test, 12 corruptions x 5 severities), on the whole stream:
  confidence-based (accuracy units, %):
    ac      : average max-softmax confidence                                   (Average Confidence)
    doc     : clean accuracy - (clean confidence - stream confidence)            (Difference of Confidences)
    atc     : share of samples with confidence above t, t set on the labelled clean test stream so that the share
              equals the clean accuracy (Average Thresholded Confidence, Garg et al. 2022)
  source-statistics signals (z = head input standardized with clean-train mean / std, so source mean 0, std 1):
    mean_shift : ||mean z||^2 / D           (squared distance of the stream mean from the source mean)
    std_shift  : mean_d (std_d - 1)^2
    *_c        : the same with z clamped to [-10, 10] (MoCo ResNet channels reach |z| of several hundred under blur)
    flip       : share (%) of predictions that change when the head input is re-standardized to the source statistics
                 (align ms of feat_metrics.py, batches of 64)
  after alignment: acc_al, ac_al, atc_al (threshold set on the aligned clean stream)
Targets: acc (true accuracy of the stream), acc_al.

  ~/REOBench/.venv/bin/python scripts/analysis/labelfree_signals.py [--models ...]   # from repro/, GPU, seconds per model
Output: experiments/2026-11-16_backbone_feats/signals/<model>_<ds>.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths  # noqa: E402
from extract_feats import MODELS  # noqa: E402
import feat_metrics as fm  # noqa: E402

OUT = paths.EXP / "2026-11-16_backbone_feats" / "signals"
dev = torch.device("cuda")
fm.dev = dev


def atc_threshold(conf, acc):  # share(conf > t) = acc on the labelled source stream
    return torch.quantile(conf, 1 - acc).item()


def run(model, ds):
    h = torch.load(f"{fm.FEATS}/{model}/head_{ds}.pt")
    te_x, te = fm.load(model, f"{ds}_test")
    c_x, cd = fm.load(model, f"{ds}_c")
    w, b, mu, sd = h["w"].to(dev), h["b"].to(dev), h["mu"], h["sd"]
    z = lambda a: ((a - mu) / sd).to(dev)
    head = lambda a: a @ w + b

    def raw(zs, ys):
        p = head(zs).softmax(1)
        conf, pred = p.max(1)
        pa = head(fm.align(zs, "ms")).softmax(1)
        conf_a, pred_a = pa.max(1)
        zc = zs.clamp(-10, 10)
        return {"acc": (pred == ys).float().mean().item(), "conf": conf, "acc_al": (pred_a == ys).float().mean().item(),
                "conf_al": conf_a, "flip": 100 * (pred != pred_a).float().mean().item(),
                "mean_shift": (zs.mean(0) ** 2).mean().item(), "std_shift": ((zs.std(0) - 1) ** 2).mean().item(),
                "mean_shift_c": (zc.mean(0) ** 2).mean().item(), "std_shift_c": ((zc.std(0) - 1) ** 2).mean().item()}

    cl = raw(z(te_x), te["y"].to(dev))
    t, t_al = atc_threshold(cl["conf"], cl["acc"]), atc_threshold(cl["conf_al"], cl["acc_al"])

    def finish(r):
        c, ca = r.pop("conf"), r.pop("conf_al")
        r.update({"ac": 100 * c.mean().item(), "doc": 100 * (cl["acc"] - (cl_conf - c.mean().item())),
                  "atc": 100 * (c > t).float().mean().item(), "ac_al": 100 * ca.mean().item(),
                  "atc_al": 100 * (ca > t_al).float().mean().item(), "acc": 100 * r["acc"], "acc_al": 100 * r["acc_al"]})
        return r

    cl_conf = cl["conf"].mean().item()
    res = {"model": model, "dataset": ds, "streams": {}}
    streams = {}
    for n, cname in enumerate(common.NOISES):
        for s in range(1, 6):
            sel = (cd["corr"] == n) & (cd["sev"] == s)
            streams[f"{cname}_s{s}"] = finish(raw(z(c_x[sel]), cd["y"][sel].to(dev)))
    res["streams"] = {"clean": finish(dict(cl)), **streams}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        for ds in ("aid", "r45"):
            r = run(m, ds)
            json.dump(r, open(OUT / f"{m}_{ds}.json", "w"), indent=1)
            st = [v for k, v in r["streams"].items() if k != "clean"]
            mae = lambda k, tgt="acc": sum(abs(v[k] - v[tgt]) for v in st) / len(st)
            print(f"{m:15s} {ds} acc {sum(v['acc'] for v in st) / len(st):5.1f} | MAE ac {mae('ac'):5.1f} doc {mae('doc'):5.1f} "
                  f"atc {mae('atc'):5.1f} | aligned: ac {mae('ac_al', 'acc_al'):5.1f} atc {mae('atc_al', 'acc_al'):5.1f}", flush=True)
    print("DONE")
