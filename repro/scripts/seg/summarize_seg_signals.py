"""Result tables (numbers only) for the Potsdam segmentation signals and the November TTA methods.

  ~/REOBench/.venv/bin/python scripts/seg/summarize_seg_signals.py     # from repro/, CPU
Output: experiments/2026-11-23_seg_signals/RESULTS.md (only what is finished)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402

D = paths.EXP / "2026-11-23_seg_signals"
OLD = paths.SEG_TTA / "outputs"
CORR = ["brightness_contrast", "clouds", "compression_artifacts", "gaussian_blur", "gaussian_noise", "haze", "motion_blur",
        "salt_and_pepper_noise"]
SHORT = ["BC", "Cloud", "JPEG", "GBlur", "GNoise", "Haze", "MBlur", "S&P"]
VARS = ["none", "bn", "isa", "isa_bn"]


def rank(a):
    a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(len(a)); return r


def sp(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def table(out, head, rows):
    out += ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows] + [""]


out = ["# Potsdam 분할: 라벨 없는 신호와 11월 방법 (2026-11-23~, 수치만)", "",
       "모델 = REOBench 분할 B 설정(SatMAE, ScaleMAE; ViT-L + UPerNet). stream = clean + 손상 8종 × 5단계, 각 2,016장, 배치 8, stream마다 초기화.",
       "none = 그대로, bn = decoder BN을 test 배치 통계로(= bnadapt), isa = 입력 통계 정렬, isa_bn = 둘 다. 모두 학습 없음.",
       "flip = none과 정렬한 예측이 다른 픽셀의 비율(유효 픽셀 기준). 정의는 `scripts/seg/seg_signals.py`, `seg_tta.py` docstring.", ""]
for model in ("satmae", "scalemae"):
    f = D / "signals" / f"{model}_B.json"
    if not f.exists():
        continue
    S = json.load(open(f))["streams"]
    keys = [k for k in S if k != "clean"]
    out += [f"## {model}_B: 신호 (학습 없음)", "", "### 손상별 (5단계 평균). mIoU / 픽셀 정확도는 %, flip·confidence도 %", ""]
    rows = []
    for name, ks in [("clean", ["clean"])] + [(sh, [f"{c}_s{s}" for s in range(1, 6)]) for c, sh in zip(CORR, SHORT)] + [("**손상 평균**", keys)]:
        if not all(k in S for k in ks):
            continue
        m = lambda fn: float(np.mean([fn(S[k]) for k in ks]))
        rows.append([name] + [f"{m(lambda r: r[v]['miou']):.1f}" for v in VARS] + [f"{m(lambda r: r['none']['pix_acc']):.1f}",
                    f"{100 * m(lambda r: r['mean_conf']):.1f}", f"{100 * m(lambda r: r['mean_conf']) - m(lambda r: r['none']['pix_acc']):+.1f}"] +
                    [f"{100 * m(lambda r: r[f'flip_{v}']):.1f}" for v in VARS[1:]])
    table(out, ["", "mIoU none", "mIoU bn", "mIoU isa", "mIoU isa_bn", "픽셀 acc", "confidence", "conf − acc", "flip bn", "flip isa", "flip isa_bn"], rows)
    acc = [S[k]["none"]["pix_acc"] for k in keys]
    miou = [S[k]["none"]["miou"] for k in keys]
    out += [f"### 손상 stream {len(keys)}개에서 none의 정확도와의 Spearman 순위 상관", ""]
    rows = [[n, f"{sp(v, acc):+.2f}", f"{sp(v, miou):+.2f}"] for n, v in
            [("flip bn", [S[k]["flip_bn"] for k in keys]), ("flip isa", [S[k]["flip_isa"] for k in keys]),
             ("flip isa_bn", [S[k]["flip_isa_bn"] for k in keys]), ("평균 confidence", [S[k]["mean_conf"] for k in keys]),
             ("엔트로피 필터 통과 비율", [S[k]["ent_keep"] for k in keys])]]
    table(out, ["신호", "픽셀 정확도와", "mIoU와"], rows)
    out += ["### 픽셀 단위 필터: 남긴 픽셀의 none 예측 정확도 (%), 손상별 5단계 평균", "",
            "엔트로피 필터 = 엔트로피 < 0.4 ln 5. flip 필터 = none과 bn 예측이 같은 픽셀.", ""]
    rows = []
    for name, ks in [("clean", ["clean"])] + [(sh, [f"{c}_s{s}" for s in range(1, 6)]) for c, sh in zip(CORR, SHORT)] + [("**손상 평균**", keys)]:
        m = lambda fn: float(np.mean([fn(S[k]) for k in ks]))
        rows.append([name, f"{m(lambda r: r['none']['pix_acc']):.1f}", f"{100 * m(lambda r: r['ent_keep']):.1f}", f"{100 * m(lambda r: r['ent_keep_acc']):.1f}",
                     f"{100 * (1 - m(lambda r: r['flip_bn'])):.1f}", f"{100 * m(lambda r: r['noflip_acc_bn']):.1f}", f"{100 * m(lambda r: r['flip_acc_bn']):.1f}",
                     f"{100 * (1 - m(lambda r: r['flip_isa_bn'])):.1f}", f"{100 * m(lambda r: r['noflip_acc_isa_bn']):.1f}"])
    table(out, ["", "전체 acc", "엔트로피: 남김", "엔트로피: acc", "flip(bn): 남김", "flip(bn): acc", "flip한 픽셀 acc", "flip(isa_bn): 남김", "flip(isa_bn): acc"], rows)
    out += ["### 예측 쏠림: tv = 0.5 Σ|예측 클래스 비율 − 실제 클래스 비율| (%), 손상별 5단계 평균", ""]
    tv = lambda r, v: 50 * float(np.abs(np.array(r[v]["pred_share"]) - np.array(r["label_share"])).sum())
    rows = []
    for name, ks in [("clean", ["clean"])] + [(sh, [f"{c}_s{s}" for s in range(1, 6)]) for c, sh in zip(CORR, SHORT)] + [("**손상 평균**", keys)]:
        rows.append([name] + [f"{np.mean([tv(S[k], v) for k in ks]):.1f}" for v in VARS])
    table(out, ["", "tv none", "tv bn", "tv isa", "tv isa_bn"], rows)
    out += [f"none의 tv와 픽셀 정확도의 Spearman(손상 stream {len(keys)}개): {sp([tv(S[k], 'none') for k in keys], acc):+.2f}", ""]

    # TTA methods
    M = [("none", OLD), ("bnadapt", OLD), ("isa", OLD), ("tent", OLD), ("sar", OLD), ("tent_bnbatch", OLD), ("flipf", D / "outputs"), ("dsbr", D / "outputs"),
         ("dsbr_src", D / "outputs"), ("flipf_bnbatch", D / "outputs"), ("dsbr_bnbatch", D / "outputs"), ("dsbr_src_bnbatch", D / "outputs")]
    R = {n: json.load(open(p / f"{model}_B" / f"{n}.json")) for n, p in M if (p / f"{model}_B" / f"{n}.json").exists()}
    out += [f"## {model}_B: 학습하는 방법 (mIoU; none 행은 절대값, 나머지는 none 대비 변화)", "",
            "tent / sar / bnadapt / isa / tent_bnbatch는 2026-10-20 실행. `_bnbatch` = decoder BN을 test 배치 통계로 둔 채 학습.", ""]
    base = R["none"]
    rows = []
    for n, r in R.items():
        if not all(f"{c}_s{s}" in r["streams"] for c in CORR for s in range(1, 6)):
            continue
        per = [np.mean([r["streams"][f"{c}_s{s}"] for s in range(1, 6)]) for c in CORR]
        bper = [np.mean([base["streams"][f"{c}_s{s}"] for s in range(1, 6)]) for c in CORR]
        vals = [r["streams"]["clean"], float(np.mean(per))] + per
        bvals = [base["streams"]["clean"], float(np.mean(bper))] + bper
        rows.append([n] + ([f"{v:.2f}" for v in vals] if n == "none" else [f"{v - b:+.2f}" for v, b in zip(vals, bvals)]))
    table(out, ["method", "clean", "손상 평균"] + SHORT, rows)
(D / "RESULTS.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
