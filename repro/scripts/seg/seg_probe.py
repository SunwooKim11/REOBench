"""Potsdam segmentation with all 17 frozen backbones and a linear head: the segmentation counterpart of the stage-1
classification study (scripts/analysis/extract_feats.py + feat_metrics.py). REOBench's own segmentors (fine-tuned ViT-L +
UPerNet, scripts/seg/seg_signals.py) exist for SatMAE and Scale-MAE only.

Model: the 512 px patch is resized to 448 and cut into 2 x 2 tiles of 224 (the pretraining size of every backbone, so no
position-embedding interpolation); patch features (reobench_tta.dense, grid 28 x 28 per image, 56 x 56 for the /8 ViTs) are
standardized with the train mean / std per channel and go through one linear layer (1 x 1 conv); logits are upsampled
bilinearly to 512. Head training: the 3,456 train patches, features cached, cross-entropy on the full-resolution
labels (clutter ignored), AdamW lr 1e-3, batch 32, 20 epochs, cosine schedule, seed 0. One fixed recipe for all backbones.

Streams and metrics = seg_signals.py (clean + 12 corruptions x 5 severities, the 2,016 test patches, batch 8, seed-0
shuffle; mIoU over the 5 classes, pixel accuracy, confidence, entropy filter, flip), with these four forwards:
  none   : as is
  fa     : head-input alignment, the feature statistics of the test batch replace the train statistics (per channel over
           the batch and all positions; the analogue of feat_metrics.align "ms" and of "bn" in seg_signals.py)
  isa    : input statistics aligned to the train pixel statistics (as seg_signals.py)
  isa_fa : both

  ~/REOBench/.venv/bin/python -u scripts/seg/seg_probe.py [--models ...] [--reverse]   # from repro/, GPU, resumes per stream;
      a second process with --reverse shares the list (lock file per model; remove stale signals/*.lock after a crash)
  ~/REOBench/.venv/bin/python scripts/seg/seg_probe.py --summary
Output: /data/rs/feats/<model>/seg_probe_head.pt, experiments/2026-11-25_seg_probe/signals/<model>.json, RESULTS.md
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import backbones, dense, paths  # noqa: E402

P = "/data/rs/reobench/Potsdam_extracted"
T = f"{P}/test"
FEATS = "/data/rs/feats"
OUT = paths.EXP / "2026-11-25_seg_probe"
MODELS = ["satmae", "scalemae", "mae_in", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16",
          "dino_in_s8", "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16",
          "sup_in_r50", "sup_in_vitb16", "sup_in_vitl16"]                   # = extract_feats.MODELS
CORR = ["brightness_contrast", "clouds", "compression_artifacts", "gaussian_blur", "gaussian_noise", "haze",
        "motion_blur", "salt_and_pepper_noise", "rotate", "scale_image", "translate_image", "data_gaps"]
SHORT = ["BC", "Cloud", "JPEG", "GBlur", "GNoise", "Haze", "MBlur", "S&P", "Rotate", "Scale", "Transl", "Gaps"]
C, IGNORE = 5, 5
VARS = ["none", "fa", "isa", "isa_fa"]
dev = torch.device("cuda")


class Files(Dataset):
    def __init__(self, img_dir, ann_dir, names):
        self.img_dir, self.ann_dir, self.names = img_dir, ann_dir, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = np.asarray(Image.open(f"{self.img_dir}/{n}").convert("RGB")).copy()
        lab = np.asarray(Image.open(f"{self.ann_dir}/{n}").convert("L")).copy()
        lab[lab == 255] = IGNORE
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(lab)


def test_dirs(cond, sev):
    if cond == "clean":
        return f"{T}/clean", f"{T}/clean_ann"
    ann = f"{T}/{cond}_ann/severity_{sev}"   # geometric corruptions have their own labels
    return f"{T}/{cond}/severity_{sev}", ann if os.path.isdir(ann) else f"{T}/clean_ann"


class Probe:
    def __init__(self, name):
        self.name = name
        model, norm, _ = backbones.build(name)
        self.model = dense.prepare(name, model).to(dev)
        self.mean, self.std = (torch.tensor(a, device=dev).view(1, 3, 1, 1) for a in norm)

    @torch.no_grad()
    def fmap(self, x):
        """x: (B, 3, 512, 512) float in 0-255 on GPU -> (B, D, 2g, 2g)."""
        B = len(x)
        x = F.interpolate(x, size=448, mode="bilinear", antialias=True, align_corners=False)
        x = (x / 255 - self.mean) / self.std
        tiles = x.view(B, 3, 2, 224, 2, 224).permute(0, 2, 4, 1, 3, 5).reshape(B * 4, 3, 224, 224)
        f = dense.feature_map(self.name, self.model, tiles)
        D, g = f.shape[1], f.shape[2]
        return f.view(B, 2, 2, D, g, g).permute(0, 3, 1, 4, 2, 5).reshape(B, D, 2 * g, 2 * g)


def train_head(pr, args):
    f = f"{FEATS}/{pr.name}/seg_probe_head{args.tag}.pt"
    if os.path.exists(f):
        return torch.load(f)
    t0 = time.time()
    names = sorted(os.listdir(f"{P}/images/train"))[:args.max_train]
    X, Y = None, torch.empty(len(names), 512, 512, dtype=torch.uint8, device=dev)
    i = 0
    for x, y in DataLoader(Files(f"{P}/images/train", f"{P}/anns/train", names), batch_size=16, num_workers=args.workers):
        fm = pr.fmap(x.to(dev).float()).half()
        if X is None:
            X = torch.empty(len(names), *fm.shape[1:], dtype=torch.float16, device=dev)
        X[i:i + len(x)], Y[i:i + len(x)] = fm, y.to(dev)
        i += len(x)
    D = X.shape[1]
    s1 = torch.zeros(D, dtype=torch.float64, device=dev); s2 = torch.zeros(D, dtype=torch.float64, device=dev)
    for j in range(0, len(X), 64):
        xb = X[j:j + 64].double()
        s1 += xb.sum((0, 2, 3)); s2 += (xb ** 2).sum((0, 2, 3))
    n = len(X) * X.shape[2] * X.shape[3]
    mu = (s1 / n).float(); sd = (s2 / n - (s1 / n) ** 2).sqrt().float() + 1e-6
    torch.manual_seed(0)
    head = torch.nn.Conv2d(D, C, 1).to(dev)
    steps = args.epochs * math.ceil(len(X) / 32)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device="cpu").manual_seed(0)
    for ep in range(args.epochs):
        perm = torch.randperm(len(X), generator=g).to(dev)
        tot = 0.0
        for j in range(0, len(X), 32):
            ix = perm[j:j + 32]
            z = (X[ix].float() - mu.view(1, -1, 1, 1)) / sd.view(1, -1, 1, 1)
            loss = F.cross_entropy(F.interpolate(head(z), size=512, mode="bilinear", align_corners=False), Y[ix].long(), ignore_index=IGNORE)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(ix)
    h = {"w": head.weight.detach().clone(), "b": head.bias.detach().clone(), "mu": mu, "sd": sd, "train_loss": tot / len(X),
         "n_train": len(X), "epochs": args.epochs}
    torch.save(h, f + ".part"); os.replace(f + ".part", f)
    print(f"[{time.strftime('%F %T')}] {pr.name} head: {len(X)} train patches, D {D}, grid {X.shape[2]}, final train loss {h['train_loss']:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    del X, Y
    torch.cuda.empty_cache()
    return h


@torch.no_grad()
def evaluate(pr, h, args):
    file = OUT / "signals" / f"{pr.name}{args.tag}.json"
    res = json.load(open(file)) if file.exists() else {"model": pr.name, "config": vars(args), "streams": {}}
    w, b = h["w"].to(dev), h["b"].to(dev)
    mu, sd = h["mu"].to(dev).view(1, -1, 1, 1), h["sd"].to(dev).view(1, -1, 1, 1)
    logits = lambda z: F.interpolate(F.conv2d(z, w, b), size=512, mode="bilinear", align_corners=False)
    # train pixel statistics for ISA (the same 500 patches as seg_signals.py / seg_tta.py)
    sub = random.Random(0).sample(sorted(os.listdir(f"{P}/images/train")), 500)
    s1 = torch.zeros(3, dtype=torch.float64); s2 = torch.zeros(3, dtype=torch.float64); npx = 0
    for n in sub:
        x = torch.from_numpy(np.asarray(Image.open(f"{P}/images/train/{n}").convert("RGB"), dtype=np.float64))
        s1 += x.sum((0, 1)); s2 += (x ** 2).sum((0, 1)); npx += x.shape[0] * x.shape[1]
    src_mean = (s1 / npx).float().view(1, 3, 1, 1).to(dev)
    src_std = (s2 / npx - (s1 / npx) ** 2).sqrt().float().view(1, 3, 1, 1).to(dev)

    def isa(x):
        m = x.mean((0, 2, 3), keepdim=True); s = x.std((0, 2, 3), keepdim=True).clamp_min(1e-6)
        return (x - m) / s * src_std + src_mean

    NAMES = sorted(os.listdir(f"{T}/clean"))
    streams = [("clean", 0)] + [(c, s) for c in args.corruptions for s in range(1, 6)]
    E0 = 0.4 * math.log(C)
    t0 = time.time()
    for si, (cond, sev) in enumerate(streams):
        key = "clean" if cond == "clean" else f"{cond}_s{sev}"
        if key in res["streams"]:
            continue
        names = NAMES if args.max_images is None else random.Random(si).sample(NAMES, args.max_images)
        g = torch.Generator().manual_seed(si)
        dl = DataLoader(Files(*test_dirs(cond, sev), names), batch_size=args.bs, shuffle=True, generator=g, num_workers=args.workers, pin_memory=True)
        conf = {v: torch.zeros(C * C, dtype=torch.long, device=dev) for v in VARS}
        hist = {v: torch.zeros(C, dtype=torch.long, device=dev) for v in VARS}
        acc = {k: 0.0 for k in ["n", "conf_sum", "ent_keep", "ent_keep_ok"] + [f"{k}_{v}" for v in VARS[1:] for k in ("flip", "flip_ok", "noflip_ok")]}
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True).float(), y.to(dev, non_blocking=True).long()
            v = y != IGNORE
            pred = {}
            for inp in ("", "isa"):
                fm = pr.fmap(isa(x) if inp else x)
                for al in (False, True):
                    name = (inp + "_fa" if inp else "fa") if al else (inp or "none")
                    z = (fm - fm.mean((0, 2, 3), keepdim=True)) / (fm.std((0, 2, 3), keepdim=True) + 1e-5) if al else (fm - mu) / sd
                    lg = logits(z)
                    pred[name] = lg.argmax(1)
                    if name == "none":
                        p = lg.softmax(1)
                        ent = -(p * lg.log_softmax(1)).sum(1)
                        ok = pred["none"] == y
                        acc["conf_sum"] += p.max(1)[0][v].sum().item()
                        k = (ent < E0) & v
                        acc["ent_keep"] += k.sum().item(); acc["ent_keep_ok"] += (k & ok).sum().item()
                    conf[name] += torch.bincount(C * y[v] + pred[name][v], minlength=C * C)
                    hist[name] += torch.bincount(pred[name][v], minlength=C)
            acc["n"] += v.sum().item()
            for name in VARS[1:]:
                fl = pred[name] != pred["none"]
                acc[f"flip_{name}"] += (fl & v).sum().item()
                acc[f"flip_ok_{name}"] += (fl & v & ok).sum().item(); acc[f"noflip_ok_{name}"] += (~fl & v & ok).sum().item()
        r = {"n": acc["n"], "mean_conf": acc["conf_sum"] / acc["n"], "ent_keep": acc["ent_keep"] / acc["n"],
             "ent_keep_acc": acc["ent_keep_ok"] / max(1, acc["ent_keep"])}
        for name in VARS:
            cm = conf[name].view(C, C).cpu().numpy().astype(np.float64)
            iou = np.diag(cm) / (cm.sum(0) + cm.sum(1) - np.diag(cm) + 1e-12)
            r[name] = {"miou": float(100 * iou[cm.sum(1) > 0].mean()), "pix_acc": float(100 * np.diag(cm).sum() / cm.sum()),
                       "iou": iou.tolist(), "pred_share": (hist[name].float() / hist[name].sum()).tolist()}
        r["label_share"] = (conf["none"].view(C, C).sum(1).float() / acc["n"]).tolist()
        for name in VARS[1:]:
            f = acc[f"flip_{name}"]
            r[f"flip_{name}"] = f / acc["n"]
            r[f"flip_acc_{name}"] = acc[f"flip_ok_{name}"] / max(1, f)
            r[f"noflip_acc_{name}"] = acc[f"noflip_ok_{name}"] / max(1, acc["n"] - f)
        res["streams"][key] = r
        json.dump(res, open(file, "w"), indent=1)
        if cond == "clean" or sev == 5:
            print(f"{pr.name} [{si + 1}/{len(streams)}] {key}: mIoU none {r['none']['miou']:.1f} fa {r['fa']['miou']:.1f} isa {r['isa']['miou']:.1f} "
                  f"isa_fa {r['isa_fa']['miou']:.1f} | flip fa {100 * r['flip_fa']:.1f}% acc {r['none']['pix_acc']:.1f} conf {100 * r['mean_conf']:.1f} "
                  f"({time.time() - t0:.0f}s)", flush=True)


def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(v)).astype(float) for v in (np.asarray(a), np.asarray(b)))
    return float(np.corrcoef(ra, rb)[0, 1])


def summary():
    R = {m: json.load(open(OUT / "signals" / f"{m}.json"))["streams"] for m in MODELS if (OUT / "signals" / f"{m}.json").exists()}
    R = {m: s for m, s in R.items() if len(s) == 1 + 5 * len(CORR)}
    keys = [f"{c}_s{s}" for c in CORR for s in range(1, 6)]
    mean = lambda a: float(np.mean(a))
    L = ["# Potsdam 분할, 17개 백본 (frozen backbone + 선형 head): 수치만 (2026-11-25~)", "",
         f"`scripts/seg/seg_probe.py`. 완료된 백본 {len(R)}개. 512px 패치를 448로 줄여 224 타일 4개로 인코딩, patch feature 위에 선형 층 하나(1x1 conv), logits를 512로 확대.",
         "head는 train 3,456 패치로 학습(AdamW 1e-3, 20 epoch, 모든 백본에 같은 설정). stream = clean + 손상 12종 × 5단계, 각 2,016장, 배치 8.",
         "none = 그대로, fa = head 입력 feature를 test 배치 통계로 표준화(분류의 head 입력 정렬과 같은 계산), isa = 입력 픽셀 통계 정렬, isa_fa = 둘 다. 모두 학습 없음.",
         "flip = none과 fa의 예측이 다른 픽셀 비율. 상관은 손상 stream 60개에서 none의 픽셀 정확도와의 Spearman.", "",
         "## 백본별 (손상 12종 × 5단계 평균 mIoU)", "",
         "| model | clean | none | fa | isa | isa_fa | fa − none | confidence − 정확도 (pt) | flip 상관 | confidence 상관 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    tab = []
    for m, S in R.items():
        g = lambda v: mean([S[k][v]["miou"] for k in keys])
        accs = [S[k]["none"]["pix_acc"] for k in keys]
        row = [S["clean"]["none"]["miou"], g("none"), g("fa"), g("isa"), g("isa_fa"), g("fa") - g("none"),
               mean([100 * S[k]["mean_conf"] - S[k]["none"]["pix_acc"] for k in keys]),
               spearman([S[k]["flip_fa"] for k in keys], accs), spearman([S[k]["mean_conf"] for k in keys], accs)]
        tab.append(row)
        L.append(f"| {m} | " + " | ".join(f"{x:.1f}" for x in row[:5]) + f" | {row[5]:+.1f} | {row[6]:+.1f} | {row[7]:+.2f} | {row[8]:+.2f} |")
    if tab:
        a = np.mean(tab, 0)
        L.append("| **평균** | " + " | ".join(f"{x:.1f}" for x in a[:5]) + f" | {a[5]:+.1f} | {a[6]:+.1f} | {a[7]:+.2f} | {a[8]:+.2f} |")
    L += ["", f"## 손상별 (백본 {len(R)}개 평균, 5단계 평균 mIoU)", "", "| 손상 | none | clean − none | fa | isa | isa_fa | fa − none | isa − none | confidence − 정확도 (pt) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    clean = mean([S["clean"]["none"]["miou"] for S in R.values()]) if R else 0.0
    for c, sh in zip(CORR, SHORT):
        g = lambda v: mean([S[f"{c}_s{s}"][v]["miou"] for S in R.values() for s in range(1, 6)])
        gap = mean([100 * S[f"{c}_s{s}"]["mean_conf"] - S[f"{c}_s{s}"]["none"]["pix_acc"] for S in R.values() for s in range(1, 6)])
        L.append(f"| {sh} | {g('none'):.1f} | {clean - g('none'):.1f} | {g('fa'):.1f} | {g('isa'):.1f} | {g('isa_fa'):.1f} | {g('fa') - g('none'):+.1f} | {g('isa') - g('none'):+.1f} | {gap:+.1f} |")
    L += ["", f"clean 평균 mIoU {clean:.1f}. clean stream에서 fa의 비용: {mean([S['clean']['fa']['miou'] - S['clean']['none']['miou'] for S in R.values()]) if R else 0:+.1f}pt.", "",
          "## 쌍 비교 (위성 − 일반, mIoU pt)", "", "| 쌍 | clean | 손상 none | 손상 fa |", "|---|---:|---:|---:|"]
    for a, b in backbones.PAIRS + [("mae_in", "scalemae")]:
        if a in R and b in R:
            d = lambda v, k=None: (mean([R[b][q][v]["miou"] for q in (k or keys)]) - mean([R[a][q][v]["miou"] for q in (k or keys)]))
            L.append(f"| {a} → {b} | {d('none', ['clean']):+.1f} | {d('none'):+.1f} | {d('fa'):+.1f} |")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--corruptions", nargs="*", default=CORR)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reverse", action="store_true", help="go through the models from the end (second process)")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(); sys.exit()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    (OUT / "signals").mkdir(parents=True, exist_ok=True)
    for m in (args.models[::-1] if args.reverse else args.models):
        f = OUT / "signals" / f"{m}{args.tag}.json"
        if f.exists() and len(json.load(open(f))["streams"]) == 1 + 5 * len(args.corruptions):
            continue
        lock = OUT / "signals" / f"{m}{args.tag}.lock"      # two processes can share the model list (one with --reverse)
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        try:
            pr = Probe(m)
            evaluate(pr, train_head(pr, args), args)
            del pr
        finally:
            lock.unlink()
        torch.cuda.empty_cache()
    print(f"SEG PROBE ALL DONE {time.strftime('%F %T')}", flush=True)
