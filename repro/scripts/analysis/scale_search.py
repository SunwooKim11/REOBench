"""Scale search: how much accuracy does undoing the apparent scale of a stream recover, and can the scale be picked without labels?

Method candidate 1 ("undo the acquisition condition"), upper bound for scale. Every test image is re-encoded at 11 apparent
magnifications plus a 2x2 tiling (frozen backbone, the stage-1 linear head):
  z0.5 .. z0.9 : zoom out = shrink the image to that fraction, reflect-pad to the original size (the REOBench scale_image
                 corruption; make_corrupted_cls.scale_reflect with PIL bilinear instead of cv2)
  z1.0         : the image as it is (must reproduce the cached stage-1 features)
  z1.11 .. z2.0: zoom in = centre crop of the fraction 0.9 .. 0.5; z(1/f) is the exact inverse of scale_image severity
                 (f = 0.9, 0.8, 0.7, 0.6, 0.5 for severity 1..5)
  tile2        : the four quadrants, each resized to 224, features averaged (zoom in x2 without losing the borders)
Streams: AID / RESISC45 clean test and scale_image severity 1..5; natural shift RESISC45 -> UC Merced (all 1,900 UCM
images) and UC Merced -> RESISC45 (2,659 test images), EarthShift 19 classes, head fitted on the source as natural_shift.py.

Per stream and op (analysis, from the saved features): accuracy, accuracy after head-input alignment (feat_metrics.align
"ms", batches of 64), and label-free scores of the op: stat (distance of the standardized feature mean / std to the
source, clamped to [-10, 10]), conf (mean max-softmax), ent (mean entropy), flip (share of predictions changed by
alignment). "oracle" = best op by accuracy (labels, upper bound); "ens" = mean of the softmax over the 11 single ops.

  ~/REOBench/.venv/bin/python -u scripts/analysis/scale_search.py [--models ...]      # from repro/, GPU, resumes
  ~/REOBench/.venv/bin/python scripts/analysis/scale_search.py --summary
Output: /data/rs/feats/<model>/scale_search/<set>.pt = {feats fp16 (N, 12, D), y, sev}; experiments/2026-11-25_undo_acquisition/
scale/<model>.json, RESULTS_scale.md
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths  # noqa: E402
import extract_feats as ef  # noqa: E402
import feat_metrics as fm  # noqa: E402

OUT = paths.EXP / "2026-11-25_undo_acquisition"
FR = [0.5, 0.6, 0.7, 0.8, 0.9]
OPS = [f"z{f}" for f in FR] + ["z1.0"] + [f"z{1 / f:.2f}" for f in FR[::-1]] + ["tile2"]   # 12 saved ops
MAG = FR + [1.0] + [1 / f for f in FR[::-1]] + [2.0]
ID = OPS.index("z1.0")
INV = {s: OPS.index(f"z{1 / f:.2f}") for s, f in zip(range(1, 6), FR[::-1])}                # severity -> inverse op
SCALE = common.NOISES.index("scale_image")
SETS = ["aid", "r45", "es_r45_test", "es_ucm_all"]


def items(s):
    """[(path, y, severity)]; severity 0 = clean."""
    if s in ("aid", "r45"):
        clean = ef.aid_items("test") if s == "aid" else ef.r45_items("test")
        cor = [r for r in (ef.aid_c_items() if s == "aid" else ef.r45_c_items()) if r[2] == SCALE]
        return [(p, y, 0) for p, y, *_ in clean] + [(p, y, sev) for p, y, _, sev, _ in sorted(cor, key=lambda r: (r[3], r[4]))]
    if s == "es_r45_test":
        return [(p, y, 0) for p, y, *_ in ef.es_items("r45", "test")]
    return [(p, y, 0) for sp in ("train", "val", "test") for p, y, *_ in ef.es_items("ucm", sp)]


def zoom_out(im, f):
    w, h = im.size
    nw, nh = round(w * f), round(h * f)
    y = np.asarray(im.resize((nw, nh), Image.BILINEAR))
    t, l = (h - nh) // 2, (w - nw) // 2
    return Image.fromarray(np.pad(y, ((t, h - nh - t), (l, w - nw - l), (0, 0)), mode="symmetric"))


def zoom_in(im, f):
    w, h = im.size
    nw, nh = round(w * f), round(h * f)
    t, l = (h - nh) // 2, (w - nw) // 2
    return im.crop((l, t, l + nw, t + nh))


def views(im):
    w, h = im.size
    v = [zoom_out(im, f) for f in FR] + [im] + [zoom_in(im, f) for f in FR[::-1]]
    v += [im.crop((x, y, x + w // 2, y + h // 2)) for y in (0, h - h // 2) for x in (0, w - w // 2)]
    return torch.stack([torch.from_numpy(np.asarray(a.resize((224, 224), Image.BILINEAR)).copy()) for a in v]).permute(0, 3, 1, 2)


class Views(Dataset):
    def __init__(self, it):
        self.it = it

    def __len__(self):
        return len(self.it)

    def __getitem__(self, i):
        return views(Image.open(self.it[i][0]).convert("RGB")), i


@torch.no_grad()
def extract(name, sets, bs, workers):
    from reobench_tta import backbones
    todo = [s for s in sets if not os.path.exists(f"{fm.FEATS}/{name}/scale_search/{s}.pt")]
    if not todo:
        return
    model, norm, dim = backbones.build(name)
    model = model.cuda()
    mean, std = (torch.tensor(a, device="cuda").view(1, 3, 1, 1) for a in norm)
    os.makedirs(f"{fm.FEATS}/{name}/scale_search", exist_ok=True)
    for s in todo:
        t0 = time.time()
        it = items(s)
        feats = torch.empty(len(it), len(OPS), dim, dtype=torch.float16)
        for x, idx in DataLoader(Views(it), batch_size=bs, num_workers=workers, pin_memory=True):
            B, V = x.shape[:2]
            x = ((x.cuda(non_blocking=True).flatten(0, 1).float() / 255) - mean) / std
            f = torch.cat([model(x[i:i + 480]) for i in range(0, len(x), 480)]).view(B, V, dim)
            feats[idx] = torch.cat([f[:, :len(OPS) - 1], f[:, len(OPS) - 1:].mean(1, keepdim=True)], 1).half().cpu()
        col = lambda k: torch.tensor([r[k] for r in it])
        f = f"{fm.FEATS}/{name}/scale_search/{s}.pt"
        torch.save({"feats": feats, "y": col(1), "sev": col(2), "ops": OPS, "model": name, "set": s}, f + ".part")
        os.replace(f + ".part", f)
        print(f"[{time.strftime('%F %T')}] {name} {s}: {len(it)} x {len(OPS)} x {dim} in {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()


def stream(Z, y, w, b):
    """Z: (n, ops, D) standardized, on GPU. Per-op lists + the ensemble."""
    r = {k: [] for k in ("acc", "acc_al", "stat", "conf", "ent", "flip", "top")}
    P = 0
    for o in range(Z.shape[1]):
        z = Z[:, o]
        p = (z @ w + b).softmax(1)
        pa = (fm.align(z, "ms") @ w + b).argmax(1)
        zc = z.clamp(-10, 10)
        r["acc"].append(100 * (p.argmax(1) == y).float().mean().item())
        r["acc_al"].append(100 * (pa == y).float().mean().item())
        r["stat"].append((zc.mean(0) ** 2).mean().item() + ((zc.std(0) - 1) ** 2).mean().item())
        r["conf"].append(100 * p.max(1)[0].mean().item())
        r["ent"].append(-(p * p.clamp_min(1e-12).log()).sum(1).mean().item())
        r["flip"].append(100 * (p.argmax(1) != pa).float().mean().item())
        r["top"].append(100 * torch.bincount(p.argmax(1), minlength=p.shape[1]).max().item() / len(y))
        if o < len(OPS) - 1:
            P = P + p
    r["ens"] = 100 * (P.argmax(1) == y).float().mean().item()
    r["n"] = len(y)
    return r


def analyse(name):
    load = lambda s: torch.load(f"{fm.FEATS}/{name}/scale_search/{s}.pt")
    res = {"model": name, "ops": OPS, "sets": {}}
    for ds in ("aid", "r45"):
        h = torch.load(f"{fm.FEATS}/{name}/head_{ds}.pt")
        w, b, mu, sd = h["w"].to(fm.dev), h["b"].to(fm.dev), h["mu"], h["sd"]
        d = load(ds)
        cached, _ = fm.load(name, f"{ds}_test")
        new = d["feats"][d["sev"] == 0][:, ID].float()
        res["sets"][ds] = {"repro_maxdiff": (new - cached).abs().max().item(), "repro_scale": cached.abs().mean().item(), "streams": {}}
        for s in range(6):
            sel = d["sev"] == s
            Z = ((d["feats"][sel].float() - mu) / sd).to(fm.dev)
            res["sets"][ds]["streams"]["clean" if s == 0 else f"scale_s{s}"] = stream(Z, d["y"][sel].to(fm.dev), w, b)
    for src, tgt, tset in (("r45", "ucm", "es_ucm_all"), ("ucm", "r45", "es_r45_test")):   # head as natural_shift.run
        tr_x, trd = fm.load(name, f"es_{src}_train")
        va_x, vad = fm.load(name, f"es_{src}_val")
        te_x, ted = fm.load(name, f"es_{src}_test")
        mu, sd = tr_x.mean(0), tr_x.std(0) + 1e-6
        z = lambda a: ((a - mu) / sd).to(fm.dev)
        nll = {lam: F.cross_entropy(z(va_x) @ wb[0] + wb[1], vad["y"].to(fm.dev)).item()
               for lam in fm.LAMBDAS for wb in [fm.fit_head(z(tr_x), trd["y"].to(fm.dev), 19, lam)]}
        w, b = fm.fit_head(z(tr_x), trd["y"].to(fm.dev), 19, min(nll, key=nll.get))
        d = load(tset)
        res["sets"][f"{src}_to_{tgt}"] = {"id_acc": 100 * ((z(te_x) @ w + b).argmax(1) == ted["y"].to(fm.dev)).float().mean().item(),
                                         "streams": {"ood": stream(z(d["feats"].float()), d["y"].to(fm.dev), w, b)}}
    return res


PICK = {"stat": min, "conf": max, "ent": min, "flip": min}     # label-free choice of the op, per stream


def pick(st, key):
    v = st[key]
    return v.index(PICK[key](v))


def summary():
    R = [json.load(open(f)) for f in sorted((OUT / "scale").glob("*.json"))]
    order = {m: i for i, m in enumerate(ef.MODELS)}
    R.sort(key=lambda r: order.get(r["model"], 99))
    mean = lambda a: float(np.mean(a))
    L = ["# 배율 탐색: 겉보기 배율을 되돌리면 얼마나 회복되나 (2026-11-25, 수치만)", "",
         f"`scripts/analysis/scale_search.py`. 백본 {len(R)}개, frozen backbone + stage-1 선형 head. 값은 정확도(%).",
         "op: z0.5–z0.9 = 축소 후 reflect 패딩(scale 손상과 같은 변환), z1.0 = 그대로, z1.11–z2.0 = 가운데를 잘라 확대(z(1/f)가 scale 손상 f의 정확한 역변환), tile2 = 4등분해 각각 인코딩 후 feature 평균.",
         "그대로 = z1.0, 정렬 = z1.0 + head 입력 정렬, 역변환 = 손상 단계에 맞는 확대(정답 배율을 아는 경우), oracle = 정확도가 가장 높은 op(라벨 사용, 상한),",
         "stat / conf / flip = 라벨 없이 고른 op(source 통계와의 거리 최소 / 평균 confidence 최대 / flip 최소), ens = 11개 op의 softmax 평균. \"+정렬\"은 그 op에 head 입력 정렬을 더한 값.", ""]
    for ds, title in (("aid", "AID"), ("r45", "RESISC45")):
        S = [r["sets"][ds]["streams"] for r in R]
        L += [f"## {title}: scale 손상 (백본 {len(R)}개 평균)", "",
              "| 단계 (축소 비율) | 그대로 | 정렬 | 역변환 | 역변환+정렬 | oracle | stat | stat+정렬 | conf | flip | ens | tile2 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        rows = []
        for s in range(1, 6):
            k = f"scale_s{s}"
            row = [mean([st[k]["acc"][ID] for st in S]), mean([st[k]["acc_al"][ID] for st in S]), mean([st[k]["acc"][INV[s]] for st in S]),
                   mean([st[k]["acc_al"][INV[s]] for st in S]), mean([max(st[k]["acc"]) for st in S]),
                   mean([st[k]["acc"][pick(st[k], "stat")] for st in S]), mean([st[k]["acc_al"][pick(st[k], "stat")] for st in S]),
                   mean([st[k]["acc"][pick(st[k], "conf")] for st in S]), mean([st[k]["acc"][pick(st[k], "flip")] for st in S]),
                   mean([st[k]["ens"] for st in S]), mean([st[k]["acc"][-1] for st in S])]
            rows.append(row)
            L.append(f"| {s} ({FR[::-1][s - 1]}) | " + " | ".join(f"{v:.1f}" for v in row) + " |")
        L.append("| **평균** | " + " | ".join(f"{v:.1f}" for v in np.mean(rows, 0)) + " |")
        c = "clean"
        L += ["", f"clean stream: 그대로 {mean([st[c]['acc'][ID] for st in S]):.1f}, stat으로 고른 op {mean([st[c]['acc'][pick(st[c], 'stat')] for st in S]):.1f}, "
              f"conf {mean([st[c]['acc'][pick(st[c], 'conf')] for st in S]):.1f}, flip {mean([st[c]['acc'][pick(st[c], 'flip')] for st in S]):.1f}, ens {mean([st[c]['ens'] for st in S]):.1f}", "",
              f"### {title}: 라벨 없이 고른 op가 정답 역변환과 같은 백본 수 ({len(R)}개 중)", "", "| 단계 | stat | conf | ent | flip | oracle |", "|---|---:|---:|---:|---:|---:|"]
        for s in range(6):
            k = "clean" if s == 0 else f"scale_s{s}"
            tgt = ID if s == 0 else INV[s]
            L.append(f"| {k} | " + " | ".join(str(sum(pick(st[k], q) == tgt for st in S)) for q in PICK) +
                     f" | {sum(int(np.argmax(st[k]['acc'])) == tgt for st in S)} |")
        L += ["", f"### {title}: op별 정확도 (백본 {len(R)}개 평균)", "", "| stream | " + " | ".join(OPS) + " |", "|---|" + "---:|" * len(OPS)]
        for k in ["clean"] + [f"scale_s{s}" for s in range(1, 6)]:
            L.append(f"| {k} | " + " | ".join(f"{mean([st[k]['acc'][o] for st in S]):.1f}" for o in range(len(OPS))) + " |")
        L += ["", f"### {title}: 백본별 (scale 5단계 평균)", "", "| model | clean | 그대로 | 정렬 | 역변환 | oracle | stat | stat+정렬 | ens |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r, st in zip(R, S):
            ks = [f"scale_s{s}" for s in range(1, 6)]
            row = [st["clean"]["acc"][ID], mean([st[k]["acc"][ID] for k in ks]), mean([st[k]["acc_al"][ID] for k in ks]),
                   mean([st[f"scale_s{s}"]["acc"][INV[s]] for s in range(1, 6)]), mean([max(st[k]["acc"]) for k in ks]),
                   mean([st[k]["acc"][pick(st[k], "stat")] for k in ks]), mean([st[k]["acc_al"][pick(st[k], "stat")] for k in ks]), mean([st[k]["ens"] for k in ks])]
            L.append(f"| {r['model']} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
        L.append("")
    L += [f"## 자연 shift (EarthShift 19클래스, 백본 {len(R)}개 평균)", "",
          "| 방향 | source test | 그대로 | 정렬 | oracle | oracle+정렬 | stat | stat+정렬 | conf | flip | ens | tile2 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in ("r45_to_ucm", "ucm_to_r45"):
        S = [r["sets"][d]["streams"]["ood"] for r in R]
        row = [mean([r["sets"][d]["id_acc"] for r in R]), mean([st["acc"][ID] for st in S]), mean([st["acc_al"][ID] for st in S]), mean([max(st["acc"]) for st in S]),
               mean([max(st["acc_al"]) for st in S]), mean([st["acc"][pick(st, "stat")] for st in S]), mean([st["acc_al"][pick(st, "stat")] for st in S]),
               mean([st["acc"][pick(st, "conf")] for st in S]), mean([st["acc"][pick(st, "flip")] for st in S]), mean([st["ens"] for st in S]), mean([st["acc"][-1] for st in S])]
        L.append(f"| {d} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
    for key, title in (("acc", "op별 정확도"), ("acc_al", "op별 정확도, head 입력 정렬 후")):
        L += ["", f"### 자연 shift: {title} (백본 {len(R)}개 평균)", "", "| 방향 | " + " | ".join(OPS) + " |", "|---|" + "---:|" * len(OPS)]
        for d in ("r45_to_ucm", "ucm_to_r45"):
            L.append(f"| {d} | " + " | ".join(f"{mean([r['sets'][d]['streams']['ood'][key][o] for r in R]):.1f}" for o in range(len(OPS))) + " |")
    L += ["", "### 자연 shift: 백본별 (그대로 → oracle op와 그 정확도, stat이 고른 op와 그 정확도)", "",
          "| model | R45→UCM 그대로 | oracle op | oracle | stat op | stat | UCM→R45 그대로 | oracle op | oracle | stat op | stat |", "|---|---:|---|---:|---|---:|---:|---|---:|---|---:|"]
    for r in R:
        cells = []
        for d in ("r45_to_ucm", "ucm_to_r45"):
            st = r["sets"][d]["streams"]["ood"]
            o, p = int(np.argmax(st["acc"])), pick(st, "stat")
            cells += [f"{st['acc'][ID]:.1f}", OPS[o], f"{st['acc'][o]:.1f}", OPS[p], f"{st['acc'][p]:.1f}"]
        L.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    L += ["", "## 점검", "", "z1.0 feature와 stage-1 캐시 feature의 최대 절대 차이 (feature 절대값 평균 대비):", ""]
    L.append(", ".join(f"{r['model']} {max(r['sets'][ds]['repro_maxdiff'] / r['sets'][ds]['repro_scale'] for ds in ('aid', 'r45')):.1e}" for r in R))
    (OUT / "RESULTS_scale.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ef.MODELS)
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary(); sys.exit()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    fm.dev = torch.device("cuda")
    (OUT / "scale").mkdir(parents=True, exist_ok=True)
    for m in args.models:
        extract(m, args.sets, args.bs, args.workers)
        if set(args.sets) == set(SETS) and not (OUT / "scale" / f"{m}.json").exists():
            r = analyse(m)
            json.dump(r, open(OUT / "scale" / f"{m}.json", "w"))
            a = r["sets"]["aid"]["streams"]
            print(f"{m:16s} AID clean {a['clean']['acc'][ID]:.1f} | scale s3: as is {a['scale_s3']['acc'][ID]:.1f} inverse {a['scale_s3']['acc'][INV[3]]:.1f} "
                  f"| repro diff {r['sets']['aid']['repro_maxdiff']:.2e} (scale {r['sets']['aid']['repro_scale']:.2f}) "
                  f"| r45->ucm {r['sets']['r45_to_ucm']['streams']['ood']['acc'][ID]:.1f} best {max(r['sets']['r45_to_ucm']['streams']['ood']['acc']):.1f}", flush=True)
    print(f"SCALE ALL DONE {time.strftime('%F %T')}", flush=True)
