"""Result tables (numbers only) for the DSBR runs: feature level, full backbone, ImageNet-C check.

  ~/REOBench/.venv/bin/python scripts/tta/summarize_dsbr.py     # from repro/, CPU (a few seconds of GPU-free tensor work)
Output: experiments/2026-11-19_dsbr/RESULTS.md (only what is finished)
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # repro/src
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from reobench_tta import common, paths  # noqa: E402
import feat_metrics as fm  # noqa: E402

D = paths.EXP / "2026-11-19_dsbr"
FB = paths.EXP / "2026-11-17_fm_tta"
FT = paths.EXP / "2026-11-17_flip_tta" / "bs64"
ORDER = ["mae_in", "satmae", "scalemae", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16", "dino_in_s8",
         "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16", "sup_in_r50", "sup_in_vitb16",
         "sup_in_vitl16"]
TITLE = {"aid": "AID / AID-C", "r45": "RESISC45 / RESISC45-C"}
PAPER = {"none": 39.8, "tent": 51.9, "dsbr_a0.9": 64.4, "dsbr_a0.99": 64.9}  # Table 3, ViT-B/16, Default


def table(out, head, rows):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out.append("")


def summary_rows(ms, cols, get, fmt, base=("none",)):
    rows = [[m] + [fmt.format(get(m, k)) if get(m, k) is not None else "–" for k in cols] for m in ms]
    full = [m for m in ms if all(get(m, k) is not None for k in cols)]
    rows.append([f"**평균 ({len(full)}개)**"] + [fmt.format(sum(get(m, k) for m in full) / max(1, len(full))) for k in cols])
    return rows, full


out = ["# DSBR 결과표 (2026-11-19, 수치만)", "",
       "DSBR = 엔트로피를 1 / (C · P[예측 클래스])로 가중, P = 예측 클래스 빈도의 EMA(α 0.9, `99`/`a99` = 0.99), stream 시작 때 1/C.",
       "논문(arXiv 2606.02339) Algorithm 1을 보고 구현(공개 코드 없음). 손상 = 12종 × 5단계 60개 stream 평균, 배치 64, stream마다 초기화.", ""]

# ---- 1. feature level
out += ["## 1. feature 단계 (head 입력의 채널별 affine만 학습, Adam)", "",
        "`scripts/analysis/feat_tta.py`. raw = head 입력 그대로, al = 배치 정렬한 head 입력. flipf는 2026-11-17 실행 값.", ""]
for ds in ("aid", "r45"):
    R = {m: json.load(open(D / "feat_bs64" / f"{m}_{ds}.json")) for m in ORDER if (D / "feat_bs64" / f"{m}_{ds}.json").exists()}
    O = {m: json.load(open(FT / f"{m}_{ds}.json")) for m in R if (FT / f"{m}_{ds}.json").exists()}
    if not R:
        continue
    cols = ["none", "align_ms"] + [f"{i}_{l}_lr{lr}" for i in ("raw", "al") for lr in ("0.001", "0.01") for l in ("tent", "dsbr", "dsbr99")]
    cols.insert(2, "raw_flipf_lr0.001")

    def get(m, k, metric="acc"):
        src = O.get(m, {}) if "flipf" in k else R[m]
        x = src.get("methods", {}).get(k)
        if x is None:
            return None
        return x["m"][0]["acc"] if metric == "clean" else sum(s[metric] for s in x["m"][1:]) / 60

    for metric, name, fmt in (("acc", "손상 정확도 (%)", "{:.1f}"), ("clean", "clean 정확도 (%)", "{:.1f}"),
                              ("signed_ece", "손상 signed ECE (+ = 과신)", "{:+.3f}")):
        out += [f"### {TITLE[ds]}: {name} (백본 {len(R)}개)", ""]
        rows, full = summary_rows(list(R), cols, lambda m, k: get(m, k, metric), fmt)
        if metric == "acc":
            rows.append(["none보다 높은 백본 수"] + [str(sum(get(m, k) > get(m, "none") + 0.05 for m in full)) for k in cols])
            rows.append(["align_ms보다 높은 백본 수"] + [str(sum(get(m, k) > get(m, "align_ms") + 0.05 for m in full)) for k in cols])
        table(out, ["model"] + [c.replace("_lr", " ") for c in cols], rows)

# ---- 2. full backbone
out += ["## 2. 백본 전체 통과 (norm 층 affine 학습, SGD momentum 0.9, lr 0.001 LN / 0.00025 BN)", "",
        "`scripts/tta/cls_tta_fm.py`. dsbr는 tent와 파라미터 집합·optimizer·lr이 같고 손실만 다르다. dsbr_a0 = EMA 없이 배치 빈도만.",
        "`_al` = 정렬한 head 입력으로 예측·손실. `_x10` = lr 10배.", ""]
BASE = ["none", "none_al", "tent", "flipf", "dsbr", "dsbr_a99", "dsbr_a0", "tent_al", "dsbr_al"]
X10 = ["none", "none_al", "tent_x10", "flipf_x10", "dsbr_x10", "dsbr_a99_x10", "tent_al_x10", "dsbr_al_x10"]
for ds in ("aid", "r45"):
    R = {(m, k): json.load(open(FB / ds / m / f"{k}.json")) for m in ORDER for k in set(BASE + X10) if (FB / ds / m / f"{k}.json").exists()}
    for keys, tag in ((BASE, "프로토콜 lr"), (X10, "lr × 10")):
        dk = [k for k in keys if k.startswith("dsbr")]
        ms = [m for m in ORDER if any((m, k) in R for k in dk)]
        cols = [k for k in keys if any((m, k) in R for m in ms)]
        if not ms:
            continue
        for metric, name, fmt in (("avg", "손상 정확도 (%)", "{:.1f}"), ("clean", "clean 정확도 (%)", "{:.1f}"),
                                  ("avg_signed_ece", "손상 signed ECE (+ = 과신)", "{:+.3f}")):
            get = lambda m, k: R[(m, k)][metric] if (m, k) in R else None
            out += [f"### {TITLE[ds]}: {name}, {tag} (백본 {len(ms)}개)", ""]
            rows, full = summary_rows(ms, cols, get, fmt)
            if metric == "avg":
                rows.append(["none보다 높은 백본 수"] + [str(sum(get(m, k) > get(m, "none") + 0.05 for m in full)) for k in cols])
                rows.append(["none_al보다 높은 백본 수"] + [str(sum(get(m, k) > get(m, "none_al") + 0.05 for m in full)) for k in cols])
                tk = "tent" if tag == "프로토콜 lr" else "tent_x10"
                if tk in cols:
                    rows.append([f"{tk}보다 높은 백본 수"] + [str(sum(get(m, k) > get(m, tk) + 0.05 for m in full)) for k in cols])
            table(out, ["model"] + cols, rows)
        # per corruption: accuracy and the share of the most frequent predicted class (none from the saved features)
        d0 = dk[0]
        full = [m for m in ms if all((m, k) in R for k in cols)]
        if not full:
            continue
        top0 = {}
        for m in full:
            h = torch.load(f"{fm.FEATS}/{m}/head_{ds}.pt", map_location="cpu")
            x, cd = fm.load(m, f"{ds}_c")
            pred = (((x - h["mu"]) / h["sd"]) @ h["w"] + h["b"]).argmax(1)
            for n, c in enumerate(common.NOISES):
                for s in range(1, 6):
                    p = pred[(cd["corr"] == n) & (cd["sev"] == s)]
                    top0[(m, f"{c}_s{s}")] = torch.bincount(p).max().item() / len(p)
        out += [f"### {TITLE[ds]}: 손상별 (백본 {len(full)}개 × 5단계 평균), {tag}", "",
                "top = 가장 많이 예측된 클래스의 비율(%), 예측 쏠림의 크기. none의 top은 저장된 feature로 계산.", ""]
        acc_cols = [k for k in cols if k in ("none", "none_al", "tent", "tent_x10", "flipf", "flipf_x10", d0)]
        rows = []
        for c in common.NOISES:
            ss = [f"{c}_s{s}" for s in range(1, 6)]
            mean = lambda f: sum(f(m, s) for m in full for s in ss) / (len(full) * 5)
            rows.append([c] + [f"{mean(lambda m, s: R[(m, k)]['streams'][s]['acc']):.1f}" for k in acc_cols] +
                        [f"{100 * mean(lambda m, s: top0[(m, s)]):.1f}", f"{100 * mean(lambda m, s: R[(m, d0)]['streams'][s]['top_share']):.1f}"])
        table(out, ["손상"] + [f"{k} acc" for k in acc_cols] + ["none top", f"{d0} top"], rows)

# ---- 3. ImageNet-C check
I = D / "imagenet_c"
if I.exists():
    F = sorted(I.glob("*.json"))
    out += ["## 3. 구현 확인: ImageNet-C 강도 5, ViT-B/16, 배치 64 (논문 Table 3 Default 열)", "",
            "`scripts/verify/dsbr_imagenet_c.py`. 논문 값: 무적응 39.8, Tent 51.9, DSBR α=0.9 64.4, α=0.99 64.9.",
            "논문이 밝히지 않은 것(체크포인트, 학습 파라미터 집합, lr)은 가정이다. docstring 참고. 15종 평균 정확도 (%).",
            "`_inorm` = ImageNet mean/std 입력(체크포인트는 0.5/0.5로 학습). augreg2 + inorm의 무적응 값이 논문의 39.8과 같다.", ""]
    rows = []
    for f in F:
        r = json.load(open(f))
        if "mean" not in r:
            rows.append([f.stem, f"진행 중 ({len(r['acc'])}/15)", "–", "–"]); continue
        key = "none" if "_none" in f.stem else "tent" if "_tent" in f.stem else "dsbr_a0.99" if "_a0.99" in f.stem else "dsbr_a0.9"
        rows.append([f.stem, f"{r['mean']:.1f}", f"{PAPER[key]:.1f}", f"{100 * sum(r['top_share'].values()) / 15:.1f}"])
    table(out, ["실행", "우리", "논문", "최다 예측 클래스 비율 (%)"], rows)
    done = [json.load(open(f)) for f in F]
    done = [(f.stem, r) for f, r in zip(F, done) if "mean" in r]
    if done:
        cs = list(done[0][1]["acc"])
        table(out, ["손상"] + [n for n, _ in done], [[c] + [f"{r['acc'][c]:.1f}" for _, r in done] for c in cs])
open(D / "RESULTS.md", "w").write("\n".join(out) + "\n")
print("wrote", D / "RESULTS.md", len(out))
