"""Stage 1 metrics from the saved features (scripts/analysis/extract_feats.py), per backbone x dataset (AID, RESISC45).

Linear head (Beyond Accuracy style, frozen backbone):
  features standardized with clean train mean/std (a fixed affine, so the linear model class is unchanged; it puts all
  backbones on one scale for L-BFGS), multinomial logistic regression, full-batch L-BFGS, L2 weight lambda chosen by
  validation NLL (AID: stratified 10% of train held out; RESISC45: torchgeo val), final head refit on train.
  NLL rather than accuracy (Beyond Accuracy's choice): validation accuracy is flat across lambda (AID differences of a
  few images) and the accuracy pick landed on the strongest lambda, giving an underconfident clean head (SatMAE AID
  clean signed ECE -0.08); with NLL the clean head is calibrated, so ECE changes under shift come from the shift.
Per stream (clean test, each corruption x severity):
  acc, ECE (15 equal-width bins), signed ECE (mean confidence - accuracy; > 0 = overconfident), NLL,
  AUROC of max-softmax confidence for correct vs wrong
  drift (standardized space, z_c = corrupted, z = its clean image):
    rel_drift   = mean ||z_c - z||^2 / mean ||z - mean z||^2      (drift energy relative to the clean spread)
    frac_common = ||mean(z_c - z)||^2 / mean ||z_c - z||^2        (share of the drift that every sample shares)
    cka         = linear CKA(clean, corrupted) on the raw features (Kornblith 2019, feature-centred)
    cos         = mean per-sample cosine(raw clean, raw corrupted)
    pr_ratio    = participation ratio corrupted / clean (effective dimension)
    hdhc_wrong  = share of samples with drift above the stream median, confidence > 0.9 and a wrong prediction
  head-input alignment, mild protocol (stream shuffled with seed 0, batches of 64, no state across batches):
    align_ms  : z' = (z - mu_batch) / sd_batch * sd_src + mu_src (src = clean train; per channel)
    align_m   : z' = z - mu_batch + mu_src (mean only; FOA's activation shift with gamma = 1, no EMA)
    -> acc and ECE after alignment (also on the clean stream: cost of aligning when nothing is shifted)

  ~/REOBench/.venv/bin/python scripts/analysis/feat_metrics.py [--models ...]     # from repro/, GPU, ~1 min per model
Output: experiments/2026-11-16_backbone_feats/metrics/<model>.json and summary.txt (tables of the means)
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths  # noqa: E402
from extract_feats import MODELS  # noqa: E402

FEATS = "/data/rs/feats"
OUT = paths.EXP / "2026-11-16_backbone_feats"
GROUPS = {"geometry": ["rotate_new", "scale_image", "translate_image_new"], "occlusion": ["clouds", "data_gaps"],
          "radiometric": ["brightness_contrast", "haze_update_new"],
          "degradation": ["gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur",
                          "compression_artifacts"]}
LAMBDAS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
dev = torch.device("cuda")


def load(model, s):
    d = torch.load(f"{FEATS}/{model}/{s}.pt")
    return d["feats"].float(), d


def fit_head(x, y, c, lam):
    w = torch.zeros(x.shape[1], c, device=dev, requires_grad=True)
    b = torch.zeros(c, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w + b, y) + lam * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def stream_metrics(logits, y):
    p = logits.softmax(1)
    conf, pred = p.max(1)
    ok = (pred == y).float()
    bins = torch.clamp((conf * 15).long(), max=14)
    ece = sum((bins == k).float().mean() * (ok[bins == k].mean() - conf[bins == k].mean()).abs()
              for k in range(15) if (bins == k).any()).item()
    nll = F.cross_entropy(logits, y).item()
    npos, nneg = ok.sum(), (1 - ok).sum()
    if npos > 0 and nneg > 0:  # Mann-Whitney AUROC from ranks (ties get the average rank)
        _, inv, cnt = torch.unique(conf, return_inverse=True, return_counts=True)
        avg = (torch.cumsum(cnt, 0) - (cnt - 1) / 2.0).float()
        auroc = (((avg[inv] * ok).sum() - npos * (npos + 1) / 2) / (npos * nneg)).item()
    else:
        auroc = float("nan")
    return {"acc": 100 * ok.mean().item(), "ece": ece, "signed_ece": (conf.mean() - ok.mean()).item(), "nll": nll,
            "auroc": auroc, "conf": conf, "ok": ok}


def cka(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    return ((y.T @ x).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm())).item()


def pr(x):
    x = x - x.mean(0)
    ev = torch.linalg.eigvalsh(x.T @ x / len(x)).clamp(min=0)
    return (ev.sum() ** 2 / (ev ** 2).sum()).item()


def align(z, mode, bs=64):
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(z), generator=g).to(dev)
    out = torch.empty_like(z)
    for i in range(0, len(z), bs):
        idx = order[i:i + bs]
        zb = z[idx]
        mu = zb.mean(0)
        if mode == "m":
            out[idx] = zb - mu  # src mean = 0 in the standardized space
        else:
            out[idx] = (zb - mu) / (zb.std(0, unbiased=False) + 1e-5)  # src sd = 1
    return out


def run(model, ds):
    tr_x, tr = load(model, f"{ds}_train")
    te_x, te = load(model, f"{ds}_test")
    c_x, cd = load(model, f"{ds}_c")
    mu, sd = tr_x.mean(0), tr_x.std(0) + 1e-6
    z = lambda a: ((a - mu) / sd).to(dev)
    ytr = tr["y"].to(dev)
    nc = int(ytr.max()) + 1
    if ds == "aid":  # stratified 10% of train as validation
        g = torch.Generator().manual_seed(0)
        val = torch.zeros(len(ytr), dtype=torch.bool)
        for k in range(nc):
            ix = (tr["y"] == k).nonzero().flatten()
            val[ix[torch.randperm(len(ix), generator=g)[: max(1, len(ix) // 10)]]] = True
        val = val.to(dev)
        fx, fy, vx, vy = z(tr_x)[~val], ytr[~val], z(tr_x)[val], ytr[val]
    else:
        vx_raw, vd = load(model, "r45_val")
        fx, fy, vx, vy = z(tr_x), ytr, z(vx_raw), vd["y"].to(dev)
    val_acc, val_nll = {}, {}
    for lam in LAMBDAS:
        w, b = fit_head(fx, fy, nc, lam)
        val_acc[lam] = ((vx @ w + b).argmax(1) == vy).float().mean().item()
        val_nll[lam] = F.cross_entropy(vx @ w + b, vy).item()
    lam = min(LAMBDAS, key=lambda l: val_nll[l])
    w, b = fit_head(z(tr_x), ytr, nc, lam)
    head = lambda a: a @ w + b
    torch.save({"w": w.cpu(), "b": b.cpu(), "mu": mu, "sd": sd, "lambda": lam},  # reused by later analyses
               f"{FEATS}/{model}/head_{ds}.pt")

    res = {"model": model, "dataset": ds, "lambda": lam, "val_acc": {str(k): v for k, v in val_acc.items()},
           "val_nll": {str(k): v for k, v in val_nll.items()}, "streams": {}}
    zte, yte = z(te_x), te["y"].to(dev)
    clean_raw = te_x.to(dev)

    def one(name, zs, raw, ys, src):
        m = stream_metrics(head(zs), ys)
        conf, ok = m.pop("conf"), m.pop("ok")
        r = dict(m)
        if src is not None:
            zc = zte[src]
            d = zs - zc
            dn = (d ** 2).sum(1)
            r["rel_drift"] = (dn.mean() / ((zc - zc.mean(0)) ** 2).sum(1).mean()).item()
            r["frac_common"] = ((d.mean(0) ** 2).sum() / dn.mean()).item() if dn.mean() > 0 else 0.0  # identical images
            r["cka"] = cka(clean_raw[src], raw)
            r["cos"] = F.cosine_similarity(clean_raw[src], raw, dim=1).mean().item()
            r["pr_ratio"] = pr(raw) / pr(clean_raw[src])
            r["hdhc_wrong"] = ((dn > dn.median()) & (conf > 0.9) & (ok == 0)).float().mean().item()
        for mode in ("ms", "m"):
            ma = stream_metrics(head(align(zs, mode)), ys)
            r[f"align_{mode}_acc"], r[f"align_{mode}_ece"] = ma["acc"], ma["ece"]
        return r

    res["streams"]["clean"] = one("clean", zte, clean_raw, yte, None)
    res["clean_pr"] = pr(clean_raw)
    for n, cname in enumerate(common.NOISES):
        for s in range(1, 6):
            sel = (cd["corr"] == n) & (cd["sev"] == s)
            raw = c_x[sel].to(dev)
            res["streams"][f"{cname}_s{s}"] = one(cname, z(c_x[sel]), raw, cd["y"][sel].to(dev), cd["src"][sel].to(dev))
    return res


def summarize(all_res):
    keys = ["acc", "ece", "signed_ece", "auroc", "rel_drift", "frac_common", "cka", "cos", "pr_ratio", "hdhc_wrong",
            "align_ms_acc", "align_m_acc", "align_ms_ece"]
    lines = []
    for ds in ["aid", "r45"]:
        lines.append(f"\n### {ds}: clean | mean over 12 corruptions x 5 severities | group means of acc (corrupted)")
        hdr = f"{'model':15s} {'clean':>6s} {'cl_ece':>6s} {'cl_al':>6s} " + " ".join(f"{k[:11]:>11s}" for k in keys) + \
              " " + " ".join(f"{g[:5]:>6s}" for g in GROUPS)
        lines.append(hdr)
        for r in all_res:
            if r["dataset"] != ds:
                continue
            st = r["streams"]
            cor = [v for k, v in st.items() if k != "clean"]
            mean = {k: sum(v[k] for v in cor if not math.isnan(v[k])) / max(1, sum(not math.isnan(v[k]) for v in cor))
                    for k in keys}
            grp = {g: sum(st[f"{c}_s{s}"]["acc"] for c in cs for s in range(1, 6)) / (5 * len(cs)) for g, cs in GROUPS.items()}
            lines.append(f"{r['model']:15s} {st['clean']['acc']:6.1f} {st['clean']['ece']:6.3f} {st['clean']['align_ms_acc']:6.1f} "
                         + " ".join(f"{mean[k]:11.3f}" if abs(mean[k]) < 10 else f"{mean[k]:11.1f}" for k in keys) + " "
                         + " ".join(f"{grp[g]:6.1f}" for g in GROUPS))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--datasets", nargs="*", default=["aid", "r45"])
    args = ap.parse_args()
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    all_res = []
    for m in args.models:
        for ds in args.datasets:
            f = OUT / "metrics" / f"{m}_{ds}.json"
            if f.exists():
                all_res.append(json.load(open(f)))
                continue
            need = [f"{FEATS}/{m}/{ds}_{s}.pt" for s in ("train", "test", "c")] + ([f"{FEATS}/{m}/r45_val.pt"] if ds == "r45" else [])
            if not all(os.path.exists(p) for p in need):
                print(f"skip {m} {ds}: features missing", flush=True)
                continue
            r = run(m, ds)
            json.dump(r, open(f, "w"), indent=1)
            all_res.append(r)
            cl = r["streams"]["clean"]
            print(f"{m} {ds}: lambda {r['lambda']:g} clean {cl['acc']:.1f}", flush=True)
            torch.cuda.empty_cache()
    txt = summarize(all_res)
    open(OUT / "summary.txt", "w").write(txt)
    print(txt)
