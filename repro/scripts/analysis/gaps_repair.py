"""Data gaps: how much accuracy comes back when the invalid stripes are excluded or filled, given the invalid-pixel mask?

Method candidate 1 ("undo the acquisition condition"), upper bound for data gaps. The mask is treated as known (real
products ship a no-data mask). Here it is recovered from the image: the REOBench stripes are a one-parameter family
(make_corrupted_cls.data_gaps: mask = (x cos t + y sin t) mod (diag / count) < width, count 2..6 and width 5..9 by
severity), so only the angle t is fitted per image (stage "theta", GPU): the angle whose stripes are darkest inside and
least dark in a 2 px ring around them (dark = max channel <= 24). RESISC45-C was generated with a seed per image, so
the fitted angle is checked against the exact one there.

Variants (frozen backbone, stage-1 linear head; the mask is widened by 2 px on each side for JPEG ringing):
  raw        : the corrupted image (must reproduce the cached stage-1 features)
  meanfill   : masked pixels <- mean colour of the valid pixels
  interp     : masked pixels <- linear interpolation across the stripe between the nearest valid pixels on both sides
  drop_any   : ViT only, raw pixels; patch tokens that contain any masked pixel are excluded (reobench_tta.token_mask)
  drop_10    : ViT only; patch tokens with more than 10% masked area are excluded
Per stream (data_gaps severity 1..5) and variant: accuracy, accuracy after head-input alignment (feat_metrics.align
"ms"), mean confidence, share of the most predicted class.

  ~/REOBench/.venv/bin/python -u scripts/analysis/gaps_repair.py [--models ...]       # from repro/, GPU, resumes
  ~/REOBench/.venv/bin/python scripts/analysis/gaps_repair.py --summary
Output: /data/rs/feats/gaps_theta_<ds>.pt, /data/rs/feats/<model>/gaps/<ds>.pt = {feats fp16 (N, 3 or 5, D), y, sev, src,
dropped}; experiments/2026-11-25_undo_acquisition/gaps/<model>.json, RESULTS_gaps.md
"""
import argparse
import json
import math
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths, token_mask  # noqa: E402
import extract_feats as ef  # noqa: E402
import feat_metrics as fm  # noqa: E402

OUT = paths.EXP / "2026-11-25_undo_acquisition"
NUM, WID = [2, 3, 4, 5, 6], [5, 6, 7, 8, 9]      # stripes per image diagonal, stripe width (px at 256), by severity
MARGIN = 2
GAPS = common.NOISES.index("data_gaps")
PIX = ["raw", "meanfill", "interp"]
TOK = ["drop_any", "drop_10"]


def items(ds):
    """[(path, y, severity, clean index)] of the data_gaps streams, ordered by (severity, clean index)."""
    it = [r for r in (ef.aid_c_items() if ds == "aid" else ef.r45_c_items()) if r[2] == GAPS]
    return [(p, y, sev, src) for p, y, _, sev, src in sorted(it, key=lambda r: (r[3], r[4]))]


# ---------------------------------------------------------------- stage "theta": stripe angle per image (GPU)
class Raw(Dataset):
    def __init__(self, it):
        self.it = it

    def __len__(self):
        return len(self.it)

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(Image.open(self.it[i][0]).convert("RGB")).copy()), i


@torch.no_grad()
def fit_theta(dark, sev):
    """dark: (B, H, W) bool on GPU, one severity. Returns angle in degrees (B,), share of the stripe that is dark (B,),
    share of the 2 px ring around it that is dark (B,)."""
    B, H, W = dark.shape
    P = int(math.sqrt(W * W + H * H)) / NUM[sev - 1]
    wid = WID[sev - 1]
    yg, xg = torch.meshgrid(torch.arange(H, device=dark.device, dtype=torch.float32), torch.arange(W, device=dark.device, dtype=torch.float32), indexing="ij")

    def masks(th):                                     # degrees (...,) -> stripe (..., H, W), ring (..., H, W)
        t = torch.deg2rad(th)[..., None, None]
        c = xg * torch.cos(t) + yg * torch.sin(t)
        m = (c % P) < wid
        return m, (((c + MARGIN) % P) < wid + 2 * MARGIN) & ~m

    d = dark.flatten(1).float()
    coarse = torch.arange(0, 180, 0.5, device=dark.device)
    m, r = (a.flatten(1).float() for a in masks(coarse))
    score = d @ m.T / m.sum(1) - d @ r.T / r.sum(1)                                   # (B, 360)
    fine = coarse[score.argmax(1)][:, None] + torch.arange(-0.5, 0.5001, 0.025, device=dark.device)[None]   # (B, 41)
    best, cov, ring = [], [], []
    for i in range(0, B, 32):
        m, r = masks(fine[i:i + 32])
        dd = dark[i:i + 32, None]
        ci = (m & dd).flatten(2).sum(2) / m.flatten(2).sum(2)
        ri = (r & dd).flatten(2).sum(2) / r.flatten(2).sum(2)
        j = (ci - ri).argmax(1, keepdim=True)
        best.append(fine[i:i + 32].gather(1, j)); cov.append(ci.gather(1, j)); ring.append(ri.gather(1, j))
    return torch.cat(best).squeeze(1).cpu(), torch.cat(cov).squeeze(1).cpu(), torch.cat(ring).squeeze(1).cpu()


def thetas(ds, workers):
    f = f"{fm.FEATS}/gaps_theta_{ds}.pt"
    if os.path.exists(f):
        return torch.load(f)
    it = items(ds)
    th, cov, ring = (torch.zeros(len(it)) for _ in range(3))
    sev = torch.tensor([r[2] for r in it])
    t0 = time.time()
    for s in range(1, 6):
        ix = (sev == s).nonzero().flatten().tolist()
        for x, idx in DataLoader(torch.utils.data.Subset(Raw(it), ix), batch_size=256, num_workers=workers):
            dark = x.cuda().amax(3) <= 24
            a, c, r = fit_theta(dark, s)
            th[idx], cov[idx], ring[idx] = a, c, r             # Raw returns the index in `it`
    res = {"theta": th, "cover": cov, "ring": ring, "sev": sev}
    if ds == "r45":   # exact angle of our own generation: seed = crc32("<stem>|data_gaps|<severity>"), first torch.rand
        ex = torch.zeros(len(it))
        for i, (p, _, s, _) in enumerate(it):
            torch.manual_seed(zlib.crc32(f"{os.path.splitext(os.path.basename(p))[0]}|data_gaps|{s}".encode()))
            ex[i] = torch.rand(1).item() * 180
        res["theta_exact"] = ex
        err = (th - ex).abs()
        print(f"r45 angle check: |fit - exact| mean {err.mean():.3f} deg, 99% {err.quantile(0.99):.3f}, max {err.max():.3f}, > 0.5 deg: {(err > 0.5).sum().item()} of {len(it)}", flush=True)
    torch.save(res, f)
    print(f"[{time.strftime('%F %T')}] theta {ds}: {len(it)} images in {time.time() - t0:.0f}s | stripe dark share mean {cov.mean():.3f} "
          f"min {cov.min():.3f}, < 0.9: {(cov < 0.9).sum().item()} | ring dark share mean {ring.mean():.3f}", flush=True)
    return res


# ---------------------------------------------------------------- pixel repair (CPU workers)
def repair(x, theta, sev):
    """x: (H, W, 3) uint8. Returns meanfill, interp (uint8) and the widened mask (bool)."""
    h, w = x.shape[:2]
    P = int(math.sqrt(w * w + h * h)) / NUM[sev - 1]
    span = WID[sev - 1] + 2 * MARGIN
    xg, yg = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    cs, sn = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    c = (xg * cs + yg * sn + MARGIN) % P
    mask = c < span
    if not mask.any() or mask.all():
        return x, x, mask

    def sample(d):                                     # pixel at p + d * normal; valid if inside the image and unmasked
        xx, yy = np.rint(xg + d * cs).astype(np.int64), np.rint(yg + d * sn).astype(np.int64)
        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        xx, yy = xx.clip(0, w - 1), yy.clip(0, h - 1)
        return x[yy, xx].astype(np.float32), ok & ~mask[yy, xx]

    a, b = c + 1, span - c                             # distance back / forward to the first pixel outside the stripe
    A, okA = sample(-a)
    B, okB = sample(b)
    wa, wb = (b * okA)[..., None], (a * okB)[..., None]
    mean = x[~mask].astype(np.float32).mean(0)
    fill = np.where(wa + wb > 0, (wa * A + wb * B) / np.maximum(wa + wb, 1e-6), mean)
    m3 = mask[..., None]
    return (np.where(m3, mean, x).round().clip(0, 255).astype(np.uint8), np.where(m3, fill, x).round().clip(0, 255).astype(np.uint8), mask)


class Repaired(Dataset):
    def __init__(self, it, theta):
        self.it, self.theta = it, theta

    def __len__(self):
        return len(self.it)

    def __getitem__(self, i):
        x = np.asarray(Image.open(self.it[i][0]).convert("RGB"))
        mf, ip, mask = repair(x, float(self.theta[i]), self.it[i][2])
        r224 = lambda a: np.asarray(Image.fromarray(a).resize((224, 224), Image.BILINEAR)).copy()
        v = torch.stack([torch.from_numpy(r224(a)) for a in (x, mf, ip)]).permute(0, 3, 1, 2)
        return v, torch.from_numpy(r224(mask.astype(np.uint8) * 255)), i


@torch.no_grad()
def extract(name, ds, th, bs, workers):
    from reobench_tta import backbones
    f = f"{fm.FEATS}/{name}/gaps/{ds}.pt"
    if os.path.exists(f):
        return
    model, norm, dim = backbones.build(name)
    model = model.cuda()
    mean, std = (torch.tensor(a, device="cuda").view(1, 3, 1, 1) for a in norm)
    vit = token_mask.supported(name)
    if vit:
        chk = token_mask.check(name, model)
        assert chk[0] < 1e-3 and chk[1] < 1e-3 and chk[2] > 1e-2, (name, chk)
    it = items(ds)
    V = len(PIX) + (len(TOK) if vit else 0)
    feats = torch.empty(len(it), V, dim, dtype=torch.float16)
    dropped = torch.zeros(len(it), 2)
    t0 = time.time()
    os.makedirs(os.path.dirname(f), exist_ok=True)
    for x, inv, idx in DataLoader(Repaired(it, th), batch_size=bs, num_workers=workers, pin_memory=True):
        B = len(x)
        x = ((x.cuda(non_blocking=True).float() / 255) - mean) / std                   # (B, 3, 3, 224, 224)
        out = [model(x.flatten(0, 1)).view(B, len(PIX), dim)]
        if vit:
            frac = F.avg_pool2d(inv.cuda().float()[:, None] / 255, token_mask.patch(name))[:, 0]   # masked area per patch
            for j, thr in enumerate((1e-3, 0.10)):
                keep = frac <= thr
                keep[~keep.flatten(1).any(1)] = True                                   # nothing left: keep all
                out.append(token_mask.forward(name, model, x[:, 0], keep)[:, None])
                dropped[idx, j] = 1 - keep.flatten(1).float().mean(1).cpu()
        feats[idx] = torch.cat(out, 1).half().cpu()
    col = lambda k: torch.tensor([r[k] for r in it])
    torch.save({"feats": feats, "y": col(1), "sev": col(2), "src": col(3), "dropped": dropped, "variants": PIX + (TOK if vit else []),
                "model": name, "set": ds}, f + ".part")
    os.replace(f + ".part", f)
    print(f"[{time.strftime('%F %T')}] {name} {ds}: {len(it)} x {V} x {dim} in {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()


def analyse(name):
    res = {"model": name, "sets": {}}
    for ds in ("aid", "r45"):
        h = torch.load(f"{fm.FEATS}/{name}/head_{ds}.pt")
        w, b, mu, sd = h["w"].to(fm.dev), h["b"].to(fm.dev), h["mu"], h["sd"]
        z = lambda a: ((a.float() - mu) / sd).to(fm.dev)
        d = torch.load(f"{fm.FEATS}/{name}/gaps/{ds}.pt")
        te_x, te = fm.load(name, f"{ds}_test")
        c_x, cd = fm.load(name, f"{ds}_c")
        acc = lambda Z, y: 100 * ((Z @ w + b).argmax(1) == y).float().mean().item()
        yte = te["y"].to(fm.dev)
        r = {"variants": d["variants"], "clean": acc(z(te_x), yte), "clean_al": acc(fm.align(z(te_x), "ms"), yte), "streams": {}}
        diff = 0.0
        for s in range(1, 6):
            sel = d["sev"] == s
            y = d["y"][sel].to(fm.dev)
            csel = (cd["corr"] == GAPS) & (cd["sev"] == s)
            o = cd["src"][csel].argsort()
            assert (cd["src"][csel][o] == d["src"][sel]).all()
            diff = max(diff, (d["feats"][sel][:, 0].float() - c_x[csel][o]).abs().max().item())
            st = {"dropped": d["dropped"][sel].mean(0).tolist()}
            for v, vname in enumerate(d["variants"]):
                Z = z(d["feats"][sel][:, v])
                p = (Z @ w + b).softmax(1)
                st[vname] = {"acc": acc(Z, y), "acc_al": acc(fm.align(Z, "ms"), y), "conf": 100 * p.max(1)[0].mean().item(),
                             "top": 100 * torch.bincount(p.argmax(1), minlength=p.shape[1]).max().item() / len(y)}
            r["streams"][f"s{s}"] = st
        r["repro_maxdiff"], r["repro_scale"] = diff, c_x.abs().mean().item()
        res["sets"][ds] = r
    return res


def summary():
    R = [json.load(open(f)) for f in sorted((OUT / "gaps").glob("*.json"))]
    order = {m: i for i, m in enumerate(ef.MODELS)}
    R.sort(key=lambda r: order.get(r["model"], 99))
    V = [r for r in R if token_mask.supported(r["model"])]
    mean = lambda a: float(np.mean(a))
    L = ["# data gaps: 무효 영역을 빼거나 채우면 얼마나 회복되나 (2026-11-25, 수치만)", "",
         f"`scripts/analysis/gaps_repair.py`. 백본 {len(R)}개(그중 ViT {len(V)}개), frozen backbone + stage-1 선형 head. 값은 정확도(%).",
         "무효 영역 mask는 주어진 것으로 본다(줄무늬 각도만 이미지에서 맞춤). raw = 손상 이미지 그대로, meanfill = 무효 픽셀을 유효 픽셀 평균색으로, interp = 줄무늬 양쪽의 유효 픽셀로 선형 보간,",
         "drop_any = 무효 픽셀이 조금이라도 있는 patch token을 제외(ViT만, 픽셀은 raw), drop_10 = 무효 면적이 10%를 넘는 token만 제외. \"+정렬\" = head 입력 정렬을 더한 값.", ""]
    for ds, title in (("aid", "AID-C"), ("r45", "RESISC45-C")):
        g = lambda r, s, v, k="acc": r["sets"][ds]["streams"][f"s{s}"][v][k]
        L += [f"## {title} data gaps (백본 {len(R)}개 평균)", "", "| 단계 | raw | raw+정렬 | meanfill | interp | interp+정렬 | clean | clean+정렬 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        rows = []
        for s in range(1, 6):
            row = [mean([g(r, s, "raw") for r in R]), mean([g(r, s, "raw", "acc_al") for r in R]), mean([g(r, s, "meanfill") for r in R]),
                   mean([g(r, s, "interp") for r in R]), mean([g(r, s, "interp", "acc_al") for r in R]),
                   mean([r["sets"][ds]["clean"] for r in R]), mean([r["sets"][ds]["clean_al"] for r in R])]
            rows.append(row)
            L.append(f"| {s} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
        L.append("| **평균** | " + " | ".join(f"{v:.1f}" for v in np.mean(rows, 0)) + " |")
        L += ["", f"### {title}: ViT {len(V)}개 평균 (token 제외 포함)", "",
              "| 단계 | raw | raw+정렬 | drop_any | drop_any+정렬 | drop_10 | meanfill | interp | interp+정렬 | 제외된 token 비율 (any / 10%, patch 16) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        rows = []
        p16 = [r for r in V if token_mask.patch(r["model"]) == 16]
        for s in range(1, 6):
            row = [mean([g(r, s, "raw") for r in V]), mean([g(r, s, "raw", "acc_al") for r in V]), mean([g(r, s, "drop_any") for r in V]),
                   mean([g(r, s, "drop_any", "acc_al") for r in V]), mean([g(r, s, "drop_10") for r in V]), mean([g(r, s, "meanfill") for r in V]),
                   mean([g(r, s, "interp") for r in V]), mean([g(r, s, "interp", "acc_al") for r in V])]
            rows.append(row)
            dr = np.mean([r["sets"][ds]["streams"][f"s{s}"]["dropped"] for r in p16], 0)
            L.append(f"| {s} | " + " | ".join(f"{v:.1f}" for v in row) + f" | {100 * dr[0]:.0f}% / {100 * dr[1]:.0f}% |")
        L.append("| **평균** | " + " | ".join(f"{v:.1f}" for v in np.mean(rows, 0)) + " | |")
        L += ["", f"### {title}: 과신과 예측 쏠림 (백본 {len(R)}개 평균, 5단계 평균)", "", "| | confidence − 정확도 (pt) | 가장 많이 예측된 클래스의 비율 (%) |", "|---|---:|---:|"]
        for v in PIX:
            L.append(f"| {v} | {mean([g(r, s, v, 'conf') - g(r, s, v) for r in R for s in range(1, 6)]):+.1f} | {mean([g(r, s, v, 'top') for r in R for s in range(1, 6)]):.1f} |")
        L += ["", f"### {title}: 백본별 (5단계 평균)", "", "| model | clean | raw | raw+정렬 | meanfill | interp | interp+정렬 | drop_any | drop_10 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in R:
            m5 = lambda v, k="acc": mean([g(r, s, v, k) for s in range(1, 6)])
            tok = [f"{m5(v):.1f}" if v in r["sets"][ds]["variants"] else "–" for v in TOK]
            L.append(f"| {r['model']} | {r['sets'][ds]['clean']:.1f} | {m5('raw'):.1f} | {m5('raw', 'acc_al'):.1f} | {m5('meanfill'):.1f} | {m5('interp'):.1f} | {m5('interp', 'acc_al'):.1f} | " + " | ".join(tok) + " |")
        L.append("")
    L += ["## 점검", ""]
    for ds in ("aid", "r45"):
        f = f"{fm.FEATS}/gaps_theta_{ds}.pt"
        if os.path.exists(f):
            t = torch.load(f)
            s = f"- {ds} 각도 맞춤: 줄무늬 안이 어두운 비율 평균 {t['cover'].mean():.3f}, 0.9 미만 {int((t['cover'] < 0.9).sum())}장 / {len(t['cover'])}장, 줄무늬 바깥 2px 띠가 어두운 비율 평균 {t['ring'].mean():.3f}"
            if "theta_exact" in t:
                e = (t["theta"] - t["theta_exact"]).abs()
                s += f". 생성 때의 정확한 각도와의 차이: 평균 {e.mean():.3f}°, 최대 {e.max():.3f}°, 0.5° 초과 {int((e > 0.5).sum())}장"
            L.append(s)
    L.append("- raw feature와 stage-1 캐시 feature의 최대 절대 차이 (feature 절대값 평균 대비): " +
             ", ".join(f"{r['model']} {max(r['sets'][ds]['repro_maxdiff'] / r['sets'][ds]['repro_scale'] for ds in ('aid', 'r45')):.1e}" for r in R))
    (OUT / "RESULTS_gaps.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ef.MODELS)
    ap.add_argument("--datasets", nargs="*", default=["aid", "r45"])
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(); sys.exit()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    fm.dev = torch.device("cuda")
    (OUT / "gaps").mkdir(parents=True, exist_ok=True)
    TH = {ds: thetas(ds, args.workers)["theta"] for ds in args.datasets}
    for m in args.models:
        for ds in args.datasets:
            extract(m, ds, TH[ds], args.bs, args.workers)
        if set(args.datasets) == {"aid", "r45"} and not (OUT / "gaps" / f"{m}.json").exists():
            r = analyse(m)
            json.dump(r, open(OUT / "gaps" / f"{m}.json", "w"))
            a = r["sets"]["aid"]
            s3 = a["streams"]["s3"]
            print(f"{m:16s} AID clean {a['clean']:.1f} | gaps s3: raw {s3['raw']['acc']:.1f} meanfill {s3['meanfill']['acc']:.1f} interp {s3['interp']['acc']:.1f} "
                  + (f"drop_any {s3['drop_any']['acc']:.1f} drop_10 {s3['drop_10']['acc']:.1f} " if "drop_any" in s3 else "")
                  + f"| repro diff {a['repro_maxdiff']:.2e} (scale {a['repro_scale']:.2f})", flush=True)
    print(f"GAPS ALL DONE {time.strftime('%F %T')}", flush=True)
