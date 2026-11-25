"""Two RS properties LSCD-TTA names as reasons for TTA failure, checked under corruption on the saved features:
(A) class imbalance of the test stream, (B) information density / scene complexity of the images.

(A) Streams = clean + 12 corruptions x 5 severities of AID / RESISC45 (17 backbones, stage-1 linear head, batches of 64).
    The test stream is re-sampled / re-ordered, the same way for every corruption:
      balanced : all test images, shuffled (the setting of every other table)
      lt10, lt100 : long-tailed, class k keeps the share rho^(-rank_k / (C - 1)) of its images, rho = 10 / 100
                    (random class ranking, 3 seeds), shuffled
      sorted   : all test images ordered by class (every batch holds one or two classes)
    Methods (feat_tta.adapt, head-input affine, Adam lr 0.01, episodic): none, std = head-input standardization with the
    test batch (no learning), tent, dsbr (both on the raw head input).
    Reported: accuracy over the samples of the stream, mean over the corrupted streams / the clean stream.

(B) Per clean test image, complexity proxies of the 224 px image the backbones see:
      jpeg  : bits per pixel of the JPEG (quality 75) file            (compressibility = information density)
      edge  : mean gradient magnitude (central differences) of the grey image; cf. SI_mean of Yu & Winkler, QoMEX 2013 (edge density)
      hist  : entropy of the grey-level histogram, bits                (tonal variety)
    and, averaged over the 17 backbones: clean accuracy, accuracy over the 60 corrupted streams, drop, over-confidence
    (confidence - correct, corrupted), flip rate, gain of head-input standardization. Spearman over images and over
    class means; quartiles of jpeg. The same proxies on 5,000 ImageNet val images (natural_vs_rs.sets) for reference.

  /opt/conda/envs/segtta/bin/python scripts/analysis/rs_properties.py [--models ...] [--datasets aid r45]   # from repro/, GPU
Output: experiments/2026-11-26_rs_properties/{RESULTS.md, results.json}
"""
import argparse
import io
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reobench_tta import common, paths  # noqa: E402
from cluster_cot import FEATS, ORDER  # noqa: E402
from feat_tta import adapt, batch_align  # noqa: E402

OUT = paths.EXP / "2026-11-26_rs_properties"
dev = torch.device("cuda")
BS, LR = 64, 0.01
METHODS = ("none", "std", "tent", "dsbr")


def rank(a):
    a = np.asarray(a, float); o = a.argsort(kind="stable"); r = np.empty(len(a)); r[o] = np.arange(len(a))
    return r


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def load(m, ds):
    h = torch.load(f"{FEATS}/{m}/head_{ds}.pt"); te = torch.load(f"{FEATS}/{m}/{ds}_test.pt"); cd = torch.load(f"{FEATS}/{m}/{ds}_c.pt")
    z = lambda a: ((a.float() - h["mu"]) / h["sd"])
    n = len(te["y"])
    key = cd["corr"].long() * 10 + cd["sev"].long()
    streams = [z(te["feats"])]
    for k in key.unique().tolist():
        idx = (key == k).nonzero()[:, 0]
        idx = idx[cd["src"][idx].argsort()]
        assert len(idx) == n and (cd["src"][idx] == torch.arange(n)).all() and (cd["y"][idx] == te["y"]).all()
        streams.append(z(cd["feats"][idx]))
    return torch.stack(streams), te["y"].long(), h["w"].float().to(dev), h["b"].float().to(dev)   # (61, N, D): stream 0 = clean


def scenario_orders(y, nc):
    n = len(y)
    out = {"balanced": [torch.randperm(n, generator=torch.Generator().manual_seed(0))]}
    for rho in (10, 100):
        out[f"lt{rho}"] = []
        for seed in range(3):
            g = torch.Generator().manual_seed(seed)
            rk = torch.randperm(nc, generator=g)
            keep = []
            for k in range(nc):
                ix = (y == k).nonzero()[:, 0]
                m = max(1, round(len(ix) * rho ** (-rk[k].item() / (nc - 1))))
                keep.append(ix[torch.randperm(len(ix), generator=g)[:m]])
            keep = torch.cat(keep)
            out[f"lt{rho}"].append(keep[torch.randperm(len(keep), generator=g)])
    out["sorted"] = [y.argsort(stable=True)]
    return out


def part_a(Z, y, w, b, nc):
    res = {}
    for name, orders in scenario_orders(y, nc).items():
        acc = {m: [] for m in METHODS}
        for od in orders:
            Zs, ys = Z[:, od].to(dev), y[od].to(dev)
            with torch.no_grad():
                lg = {"none": Zs @ w + b,
                      "std": torch.cat([batch_align(Zs[:, i:i + BS]) for i in range(0, Zs.shape[1], BS)], 1) @ w + b}
            for loss in ("tent", "dsbr"):
                lg[loss] = adapt(Zs, w, b, "raw", loss, LR, BS)[0]
            for m in METHODS:
                acc[m].append(100 * (lg[m].argmax(-1) == ys).float().mean(1).cpu().numpy())    # (61,)
            del Zs, lg
        res[name] = {m: {"clean": float(np.mean([a[0] for a in acc[m]])), "corr": float(np.mean([a[1:].mean() for a in acc[m]]))} for m in METHODS}
        res[name]["n"] = int(np.mean([len(o) for o in orders]))
    return res


def per_image(Z, y, w, b):
    """per image, over the 60 corrupted streams (balanced order, seed 0 batches as cluster_cot.align)."""
    od = torch.randperm(Z.shape[1], generator=torch.Generator().manual_seed(0))
    inv = od.argsort()
    Zs = Z[:, od].to(dev)
    with torch.no_grad():
        lr = (Zs @ w + b)[:, inv]
        la = (torch.cat([batch_align(Zs[:, i:i + BS]) for i in range(0, Zs.shape[1], BS)], 1) @ w + b)[:, inv]
    yd = y.to(dev)
    ok, oka = (lr.argmax(-1) == yd).float(), (la.argmax(-1) == yd).float()
    conf = lr.softmax(-1).amax(-1)
    return {"clean": ok[0].cpu().numpy(), "corr": ok[1:].mean(0).cpu().numpy(), "over": (conf[1:] - ok[1:]).mean(0).cpu().numpy(),
            "flip": (lr[1:].argmax(-1) != la[1:].argmax(-1)).float().mean(0).cpu().numpy(), "gain": (oka[1:] - ok[1:]).mean(0).cpu().numpy()}


def proxies(job):
    path, mode = job
    im = Image.open(path).convert("RGB")
    if mode == "imagenet":   # Resize 256 + CenterCrop 224
        s = 256 / min(im.size); im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BICUBIC)
        l, t = (im.size[0] - 224) // 2, (im.size[1] - 224) // 2
        im = im.crop((l, t, l + 224, t + 224))
    else:
        im = im.resize((224, 224), Image.BICUBIC)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=75)
    g = np.asarray(im.convert("L"), np.float32) / 255
    gx = g[1:-1, 2:] - g[1:-1, :-2]; gy = g[2:, 1:-1] - g[:-2, 1:-1]
    hist = np.bincount((g * 255).astype(np.uint8).ravel(), minlength=256) / g.size
    hist = hist[hist > 0]
    return 8 * buf.tell() / (224 * 224), float(np.hypot(gx, gy).mean()), float(-(hist * np.log2(hist)).sum())


def image_paths(ds):
    if ds == "aid":
        return [p for p, _ in datasets.ImageFolder(f"{common.DATA_ROOT}/AID_clean_renamed/test").samples]
    from extract_feats import r45_items
    return [it[0] for it in r45_items("test")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ORDER)
    ap.add_argument("--datasets", nargs="+", default=["aid", "r45"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    res = {"A": {}, "B": {}}
    with Pool(8) as pool:
        from natural_vs_rs import IN_VAL, sets
        px = {"imagenet": np.array(pool.map(proxies, [(f"{IN_VAL}/{rel}", "imagenet") for rel, _ in sets()["imagenet"]], chunksize=64))}
        for ds in args.datasets:
            px[ds] = np.array(pool.map(proxies, [(p, ds) for p in image_paths(ds)], chunksize=64))
    print("proxies done", {k: v.mean(0).round(3).tolist() for k, v in px.items()}, flush=True)
    title = {"aid": "AID", "r45": "RESISC45"}
    out = ["# RS 성질 두 가지를 손상에서 확인: 클래스 불균형, 정보 밀도 (2026-11-26, 수치만)", "",
           "`scripts/analysis/rs_properties.py`. 저장된 feature와 stage-1 선형 head, 백본 " + str(len(args.models)) + "개, 배치 64. 정의는 스크립트 docstring.", "",
           "## A. test stream의 클래스 불균형", "",
           "balanced = test 전체를 섞음(다른 표의 설정), lt10 / lt100 = 클래스별로 남기는 비율이 최대 10배 / 100배 차이 나는 long-tail(무작위 클래스 순위 3번 평균),",
           "sorted = test 전체를 클래스 순서로 정렬(배치마다 클래스 한두 개). std = head 입력 표준화(학습 없음), tent / dsbr = head 입력 affine만 학습(Adam, lr 0.01).",
           "값 = stream 안 표본의 정확도(%), 손상 60개 stream 평균과 clean stream. 괄호 = none 대비 변화.", ""]
    per = {}
    for ds in args.datasets:
        A, P = {}, []
        for m in args.models:
            Z, y, w, b = load(m, ds)
            nc = w.shape[1]
            A[m] = part_a(Z, y, w, b, nc)
            P.append(per_image(Z, y, w, b))
            print(ds, m, {k: round(v["std"]["corr"] - v["none"]["corr"], 1) for k, v in A[m].items()}, flush=True)
            del Z
            torch.cuda.empty_cache()
        res["A"][ds] = A
        per[ds] = {k: np.mean([p[k] for p in P], 0) for k in P[0]}
        per[ds]["y"] = y.numpy()
        out += [f"### {title[ds]}", "", "| stream 구성 | 표본 수 | 손상: none | std | tent | dsbr | clean: none | std | tent | dsbr |", "|---|---:|" + "---:|" * 8]
        for sc in ("balanced", "lt10", "lt100", "sorted"):
            mean = lambda meth, part: np.mean([A[m][sc][meth][part] for m in args.models])
            cells = []
            for part in ("corr", "clean"):
                base = mean("none", part)
                cells.append(f"{base:.1f}")
                cells += [f"{mean(meth, part):.1f} ({mean(meth, part) - base:+.1f})" for meth in ("std", "tent", "dsbr")]
            out.append(f"| {sc} | {int(np.mean([A[m][sc]['n'] for m in args.models]))} | " + " | ".join(cells) + " |")
        out += ["", "std가 none보다 높은 백본 수 (손상 평균): " + ", ".join(
            f"{sc} {sum(A[m][sc]['std']['corr'] > A[m][sc]['none']['corr'] for m in args.models)} / {len(args.models)}" for sc in ("balanced", "lt10", "lt100", "sorted")), ""]
    PX = ("jpeg", "edge", "hist")
    PXN = {"jpeg": "JPEG 비트/픽셀", "edge": "edge 밀도", "hist": "밝기 히스토그램 엔트로피"}
    OC = (("clean", "clean 정확도"), ("corr", "손상 정확도"), ("drop", "하락 (clean − 손상)"), ("over", "과신 (confidence − 정답, 손상)"),
          ("flip", "flip 비율"), ("gain", "head 입력 표준화 이득"))
    out += ["## B. 정보 밀도 / 장면 복잡도", "",
            "이미지마다 224 px 입력의 복잡도를 세 가지로 잰다: JPEG(품질 75) 비트/픽셀, edge 밀도(밝기 gradient 크기의 평균, 중앙 차분), 밝기 히스토그램 엔트로피.",
            "결과 값은 17개 백본 평균(손상 = 60개 stream 평균).", "", "### 데이터셋별 복잡도 (평균, 괄호 = 사분위 25–75%)", "",
            "| 데이터 | 이미지 수 | " + " | ".join(PXN[p] for p in PX) + " |", "|---|---:|---:|---:|---:|"]
    for k, name in (("imagenet", "ImageNet val (클래스당 5장)"), ("aid", "AID test"), ("r45", "RESISC45 test")):
        if k in px:
            v = px[k]
            out.append(f"| {name} | {len(v)} | " + " | ".join(f"{v[:, i].mean():.3f} ({np.percentile(v[:, i], 25):.3f}–{np.percentile(v[:, i], 75):.3f})" for i in range(3)) + " |")
    out.append("")
    for ds in args.datasets:
        d = per[ds]; d["drop"] = d["clean"] - d["corr"]
        v, y = px[ds], d["y"]
        nc = int(y.max()) + 1
        cm = lambda a: np.array([a[y == k].mean() for k in range(nc)])
        out += [f"### {title[ds]}: 복잡도와 결과의 Spearman 순위상관 (이미지 {len(y)}장 / 클래스 평균 {nc}개)", "",
                "| 결과 | " + " | ".join(PXN[p] for p in PX) + " |", "|---|---:|---:|---:|"]
        B = {}
        for k, name in OC:
            B[k] = {p: (spear(v[:, i], d[k]), spear(cm(v[:, i]), cm(d[k]))) for i, p in enumerate(PX)}
            out.append(f"| {name} | " + " | ".join(f"{B[k][p][0]:+.2f} / {B[k][p][1]:+.2f}" for p in PX) + " |")
        q = np.digitize(rank(v[:, 0]), np.linspace(0, len(y), 5)[1:-1])
        out += ["", f"### {title[ds]}: JPEG 비트/픽셀 사분위별 (이미지 수 같게 나눔, 값은 % 또는 pt)", "",
                "| 사분위 | JPEG 비트/픽셀 | " + " | ".join(n for _, n in OC) + " |", "|---|---:|" + "---:|" * len(OC)]
        for i in range(4):
            out.append(f"| {i + 1}{' (낮음)' if i == 0 else ' (높음)' if i == 3 else ''} | {v[q == i, 0].mean():.2f} | " + " | ".join(f"{100 * d[k][q == i].mean():.1f}" for k, _ in OC) + " |")
        out.append("")
        res["B"][ds] = {"spearman": B, "quartile": [{k: float(d[k][q == i].mean()) for k, _ in OC} for i in range(4)]}
    res["B"]["proxy_mean"] = {k: v.mean(0).tolist() for k, v in px.items()}
    (OUT / "RESULTS.md").write_text("\n".join(out) + "\n")
    json.dump(res, open(OUT / "results.json", "w"))
    print("saved", OUT / "RESULTS.md")
