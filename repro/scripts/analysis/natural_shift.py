"""Natural shift RESISC45 <-> UC Merced (EarthShift 19 shared classes), the stage-1 measurements on a real shift.

Per backbone and direction (r45 -> ucm, ucm -> r45), frozen features from extract_feats.py (sets es_*):
  head   : same as feat_metrics.py (standardize with source-train mean / std, multinomial LR by full-batch L-BFGS,
           lambda by source-val NLL, refit on source train)
  streams: id  = source test split
           ood = target test split (torchgeo split filtered to the 19 classes; EarthShift's OOD test set)
           ood_all = every target image (train + val + test; UC Merced 1,900, RESISC45 13,300)
  per stream:
    acc, ECE (15 bins), signed ECE (> 0 overconfident), NLL, AUROC of max-softmax confidence (correct vs wrong)
    confidence-based accuracy estimates: AC (mean confidence), DoC (id acc - (id conf - stream conf)),
      ATC with the threshold set on the labelled source val split (standard) and on the source test split (stage-1 choice)
    head-input alignment as feat_metrics.py (stream shuffled with seed 0, batches of 64, re-standardized to the
      source-train statistics): align_ms (mean and std), align_m (mean only) -> acc, ECE; flip = share of predictions
      changed by align_ms; ATC on the aligned head (threshold from the aligned source val split)
    flip estimate of the unaligned accuracy: ATC_al - k * max(0, flip - flip_id), k fitted on the stage-1 synthetic
      streams (signals/*.json, all 17 backbones x AID / RESISC45, least squares through the origin)
    unpaired feature statistics (standardized space, source train mean 0 / std 1):
      mean_shift = ||mean z||^2 / D, std_shift = mean_d (std_d - 1)^2 (both also with z clamped to [-10, 10]),
      frac_mean = ||mean z - mean z_id||^2 / mean ||z - mean z_id||^2, pr_ratio = participation ratio stream / id
    class balance: max / min class count of the stream

  ~/REOBench/.venv/bin/python scripts/analysis/natural_shift.py [--models ...]   # from repro/, GPU, seconds per model
Output: experiments/2026-11-17_natural_shift/{<model>.json, summary.txt}
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import paths  # noqa: E402
from extract_feats import MODELS  # noqa: E402
import feat_metrics as fm  # noqa: E402

OUT = paths.EXP / "2026-11-17_natural_shift"
SIG = paths.EXP / "2026-11-16_backbone_feats" / "signals"
dev = torch.device("cuda")
fm.dev = dev
DIRS = {"r45_to_ucm": ("r45", "ucm"), "ucm_to_r45": ("ucm", "r45")}


def stage1_k():
    """slope of (aligned acc - raw acc) on excess flip over all stage-1 synthetic streams (eval_labelfree.py)."""
    xs, ys = [], []
    for f in sorted(SIG.glob("*.json")):
        st = json.load(open(f))["streams"]
        for name, v in st.items():
            if name != "clean":
                xs.append(max(0.0, v["flip"] - st["clean"]["flip"])); ys.append(v["acc_al"] - v["acc"])
    x, y = torch.tensor(xs), torch.tensor(ys)
    return ((x * y).sum() / (x * x).sum()).item(), len(xs)


def feats(model, s):
    x, d = fm.load(model, s)
    return x, d["y"]


def run(model, src, tgt, k):
    tr_x, tr_y = feats(model, f"es_{src}_train")
    va_x, va_y = feats(model, f"es_{src}_val")
    te_x, te_y = feats(model, f"es_{src}_test")
    tg = {sp: feats(model, f"es_{tgt}_{sp}") for sp in ("train", "val", "test")}
    mu, sd = tr_x.mean(0), tr_x.std(0) + 1e-6
    z = lambda a: ((a - mu) / sd).to(dev)
    nc = 19
    val_nll = {}
    for lam in fm.LAMBDAS:
        w, b = fm.fit_head(z(tr_x), tr_y.to(dev), nc, lam)
        val_nll[lam] = F.cross_entropy(z(va_x) @ w + b, va_y.to(dev)).item()
    lam = min(fm.LAMBDAS, key=lambda l: val_nll[l])
    w, b = fm.fit_head(z(tr_x), tr_y.to(dev), nc, lam)
    head = lambda a: a @ w + b

    def conf_of(zs):
        return head(zs).softmax(1).max(1)[0]

    zva, zte = z(va_x), z(te_x)
    yva, yte = va_y.to(dev), te_y.to(dev)
    thr = lambda c, ok: torch.quantile(c, 1 - ok.float().mean()).item()  # share(conf > t) = accuracy
    t_val = thr(conf_of(zva), head(zva).argmax(1) == yva)
    t_test = thr(conf_of(zte), head(zte).argmax(1) == yte)
    zva_a = fm.align(zva, "ms")
    t_val_al = thr(conf_of(zva_a), head(zva_a).argmax(1) == yva)
    pr_id = fm.pr(zte)
    mu_id = zte.mean(0)
    id_acc = (head(zte).argmax(1) == yte).float().mean().item()
    id_conf = conf_of(zte).mean().item()

    def stream(zs, ys):
        m = fm.stream_metrics(head(zs), ys)
        conf, ok = m.pop("conf"), m.pop("ok")
        pred = head(zs).argmax(1)
        za = fm.align(zs, "ms")
        pred_a, conf_a = head(za).argmax(1), conf_of(za)
        ma = fm.stream_metrics(head(za), ys)
        mm = fm.stream_metrics(head(fm.align(zs, "m")), ys)
        zc = zs.clamp(-10, 10)
        d = zs - mu_id
        cnt = torch.bincount(ys, minlength=nc).float()
        r = dict(m)
        r.update({"n": len(ys), "class_ratio": (cnt.max() / cnt[cnt > 0].min()).item(),
                  "ac": 100 * conf.mean().item(), "doc": 100 * (id_acc - (id_conf - conf.mean().item())),
                  "atc_val": 100 * (conf > t_val).float().mean().item(), "atc_test": 100 * (conf > t_test).float().mean().item(),
                  "align_ms_acc": ma["acc"], "align_ms_ece": ma["ece"], "align_ms_signed_ece": ma["signed_ece"],
                  "align_m_acc": mm["acc"], "align_m_ece": mm["ece"],
                  "flip": 100 * (pred != pred_a).float().mean().item(),
                  "atc_al": 100 * (conf_a > t_val_al).float().mean().item(),
                  "mean_shift": (zs.mean(0) ** 2).mean().item(), "std_shift": ((zs.std(0) - 1) ** 2).mean().item(),
                  "mean_shift_c": (zc.mean(0) ** 2).mean().item(), "std_shift_c": ((zc.std(0) - 1) ** 2).mean().item(),
                  "frac_mean": ((d.mean(0) - 0) ** 2).sum().item() / (d ** 2).sum(1).mean().item(),
                  "pr_ratio": fm.pr(zs) / pr_id})
        return r

    all_x = torch.cat([tg[sp][0] for sp in ("train", "val", "test")])
    all_y = torch.cat([tg[sp][1] for sp in ("train", "val", "test")])
    res = {"model": model, "source": src, "target": tgt, "lambda": lam, "val_nll": {str(a): v for a, v in val_nll.items()},
           "atc_threshold_val": t_val, "atc_threshold_test": t_test, "k_flip": k,
           "streams": {"id": stream(zte, yte), "ood": stream(z(tg["test"][0]), tg["test"][1].to(dev)),
                       "ood_all": stream(z(all_x), all_y.to(dev))}}
    fid = res["streams"]["id"]["flip"]
    for v in res["streams"].values():
        v["flip_est"] = v["atc_al"] - k * max(0.0, v["flip"] - fid)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    k, n = stage1_k()
    print(f"k_flip = {k:.3f} from {n} stage-1 streams", flush=True)
    rows = []
    for m in args.models:
        res = {"model": m, "k_flip": k, "directions": {}}
        for dname, (src, tgt) in DIRS.items():
            res["directions"][dname] = run(m, src, tgt, k)
        json.dump(res, open(OUT / f"{m}.json", "w"), indent=1)
        rows.append(res)
        print(m, {d: (round(r["streams"]["id"]["acc"], 1), round(r["streams"]["ood"]["acc"], 1))
                  for d, r in res["directions"].items()}, flush=True)
    keys = ["acc", "ece", "signed_ece", "auroc", "ac", "doc", "atc_val", "align_ms_acc", "align_m_acc", "flip", "atc_al",
            "flip_est", "mean_shift_c", "std_shift_c", "frac_mean", "pr_ratio"]
    L = [f"k_flip = {k:.3f} (stage-1 synthetic streams: {n})"]
    for dname in DIRS:
        for sname in ("id", "ood", "ood_all"):
            L += [f"\n### {dname} / {sname}", f"{'model':15s} " + " ".join(f"{q[:11]:>11s}" for q in keys)]
            for res in rows:
                v = res["directions"][dname]["streams"][sname]
                L.append(f"{res['model']:15s} " + " ".join(f"{v[q]:11.3f}" if abs(v[q]) < 10 else f"{v[q]:11.1f}" for q in keys))
    open(OUT / "summary.txt", "w").write("\n".join(L) + "\n")
    print("DONE")
