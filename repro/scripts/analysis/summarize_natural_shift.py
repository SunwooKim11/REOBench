"""Result tables (numbers only) for scripts/analysis/natural_shift.py.

  ~/REOBench/.venv/bin/python scripts/analysis/summarize_natural_shift.py     # from repro/, CPU
Output: experiments/2026-11-17_natural_shift/RESULTS.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402

D = paths.EXP / "2026-11-17_natural_shift"
ORDER = ["mae_in", "satmae", "scalemae", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16", "dino_in_s8",
         "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16", "sup_in_r50", "sup_in_vitb16",
         "sup_in_vitl16"]
R = {m: json.load(open(D / f"{m}.json")) for m in ORDER}
out = ["# 자연 shift 결과표: RESISC45 ↔ UC Merced (EarthShift 19개 공통 클래스, 2026-11-17, 수치만)", "",
       "frozen backbone + 선형 head(source train으로 학습, source val NLL로 λ 선택), 224 px, seed 0.",
       f"flip 추정의 기울기 k = {R[ORDER[0]]['k_flip']:.3f} (stage-1 합성 손상 stream 2,040개에서 추정, 이 데이터는 쓰지 않음).",
       "정의는 `scripts/analysis/natural_shift.py` docstring.", ""]


def table(head, rows, fmt):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    for r in rows:
        out.append("| " + " | ".join([str(r[0])] + [f.format(v) for v, f in zip(r[1:], fmt)]) + " |")
    out.append("")


NAMES = {"r45_to_ucm": "RESISC45 → UC Merced", "ucm_to_r45": "UC Merced → RESISC45"}
for d, title in NAMES.items():
    S = lambda m, s: R[m]["directions"][d]["streams"][s]
    n = {s: S(ORDER[0], s)["n"] for s in ("id", "ood", "ood_all")}
    out += [f"## {title}", "", f"ID = source test {n['id']:,}장, OOD = target test split {n['ood']:,}장, OOD 전체 = target 전체 {n['ood_all']:,}장.", ""]
    for s, name in (("ood", "OOD (target test split)"), ("ood_all", "OOD 전체 (target 모든 이미지)")):
        out += [f"### {name}: 정확도와 confidence", ""]
        table(["model", "ID acc", "OOD acc", "하락", "ID ECE", "OOD ECE", "signed ECE", "AUROC", "NLL"],
              [[m, S(m, "id")["acc"], S(m, s)["acc"], S(m, "id")["acc"] - S(m, s)["acc"], S(m, "id")["ece"], S(m, s)["ece"],
                S(m, s)["signed_ece"], S(m, s)["auroc"], S(m, s)["nll"]] for m in ORDER],
              ["{:.1f}", "{:.1f}", "{:.1f}", "{:.3f}", "{:.3f}", "{:+.3f}", "{:.3f}", "{:.2f}"])
        out += [f"### {name}: 라벨 없는 정확도 추정 (추정값 − 실제, pt)", "",
                "AC = 평균 confidence, DoC, ATC(임계값 = source val). 정렬 후 ATC는 정렬 후 정확도와 비교. flip 추정은 원래 정확도와 비교.", ""]
        table(["model", "OOD acc", "AC 오차", "DoC 오차", "ATC 오차", "정렬 후 acc", "정렬 후 ATC 오차", "flip (%)", "ID flip (%)",
               "flip 추정 오차"],
              [[m, S(m, s)["acc"], S(m, s)["ac"] - S(m, s)["acc"], S(m, s)["doc"] - S(m, s)["acc"], S(m, s)["atc_val"] - S(m, s)["acc"],
                S(m, s)["align_ms_acc"], S(m, s)["atc_al"] - S(m, s)["align_ms_acc"], S(m, s)["flip"], S(m, "id")["flip"],
                S(m, s)["flip_est"] - S(m, s)["acc"]] for m in ORDER],
              ["{:.1f}", "{:+.1f}", "{:+.1f}", "{:+.1f}", "{:.1f}", "{:+.1f}", "{:.1f}", "{:.1f}", "{:+.1f}"])
        out += [f"### {name}: 정렬 효과와 feature 통계", ""]
        table(["model", "OOD acc", "align ms acc", "align m acc", "변화 (ms)", "align ms ECE", "ID align ms acc − ID acc", "mean_shift",
               "std_shift", "frac_mean", "PR ratio"],
              [[m, S(m, s)["acc"], S(m, s)["align_ms_acc"], S(m, s)["align_m_acc"], S(m, s)["align_ms_acc"] - S(m, s)["acc"],
                S(m, s)["align_ms_ece"], S(m, "id")["align_ms_acc"] - S(m, "id")["acc"], S(m, s)["mean_shift_c"], S(m, s)["std_shift_c"],
                S(m, s)["frac_mean"], S(m, s)["pr_ratio"]] for m in ORDER],
              ["{:.1f}", "{:.1f}", "{:.1f}", "{:+.1f}", "{:.3f}", "{:+.1f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.2f}"])
open(D / "RESULTS.md", "w").write("\n".join(out) + "\n")
print(f"wrote {D / 'RESULTS.md'}: {len(out)} lines")
