"""Tables for weighted_flips_repro.py: our numbers next to the paper's (Press et al., ICML 2024, Table 1, row IN-C {75}).

  ~/REOBench/.venv/bin/python scripts/verify/summarize_wf_repro.py     # from repro/, CPU
Output: experiments/2026-11-19_weighted_flips_repro/RESULTS.md
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402

D = paths.EXP / "2026-11-19_weighted_flips_repro"
MAIN = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost",
        "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]
HOLD = ["speckle_noise", "gaussian_blur", "spatter", "saturate"]
PAPER = {"AC": 10.06, "DoC": 6.61, "ATC": 7.44, "COT": 2.23, "WF": 4.79}  # Table 1, IN-C {75}, ResNet-50
L = lambda s: json.load(open(D / f"{s}.json")) if (D / f"{s}.json").exists() else None
main = [L(f"{c}_{s}") for c in MAIN for s in range(1, 6)]
hold = [L(f"{c}_{s}") for c in HOLD for s in range(1, 6)]
val = L("val")
main, hold = [r for r in main if r], [r for r in hold if r]
EST = {"WF (가중 flip, 2차식)": lambda r: r["est"]["wf_quad"], "flip (가중 없음, 1차식)": lambda r: r["est"]["uf_lin"],
       "flip (가중 없음, 2차식)": lambda r: r["est"]["uf_quad"], "AC": lambda r: r["ac"], "DoC": lambda r: r["doc"],
       "ATC (max-softmax)": lambda r: r["atc_mc"], "ATC (negative entropy)": lambda r: r["atc_ne"]}
mae = lambda rs, f: float(np.mean([abs(f(r) - r["acc"]) for r in rs]))
out = ["# Weighted Flips 재현 (ImageNet-C, ResNet-50, 2026-11-19, 수치만)", "",
       "공식 코드(oripress/EntropyEnigma @ 04ac397)의 함수와 계수를 그대로 사용. seed 42, 배치 64, RDumb 990 iteration, 처음 1,000장의 flip.",
       "정의는 `scripts/verify/weighted_flips_repro.py` docstring. 논문 값 = Press et al. ICML 2024 Table 1의 IN-C {75} 행.", "",
       f"## ImageNet-C 15종 × 5단계 ({len(main)}개 split): 정확도 추정 오차 MAE (pt)", "",
       "| 방법 | 우리 실행 | 논문 |", "|---|---:|---:|"]
pk = {"WF (가중 flip, 2차식)": "WF", "AC": "AC", "DoC": "DoC", "ATC (max-softmax)": "ATC", "ATC (negative entropy)": "ATC"}
for k, f in EST.items():
    out.append(f"| {k} | {mae(main, f):.2f} | {PAPER[pk[k]]:.2f} |" if k in pk else f"| {k} | {mae(main, f):.2f} | – |")
out += ["| COT | (구현 안 함) | 2.23 |", ""]
if val:
    out += [f"ImageNet val: 정확도 {val['acc']:.2f}, WF {val['wf']:.1f}, 가중 없는 flip {val['uf']:.0f}.", ""]
if val and len(hold) == 20:
    pts = [val] + hold
    co = np.polyfit([r["wf"] for r in pts], [r["acc"] for r in pts], 2)
    f = np.poly1d(co)
    out += ["## f 다시 맞추기 (논문과 같이 ImageNet val + holdout 4종 × 5단계 = 21점, 2차식)", "",
            "| | x² | x | 상수 |", "|---|---:|---:|---:|",
            f"| 우리 실행 | {co[0]:.6f} | {co[1]:.4f} | {co[2]:.2f} |", "| 논문 (main.py) | 0.000356 | -0.3214 | 75.66 |", "",
            f"다시 맞춘 f로 추정한 IN-C {len(main)}개 split의 MAE: {float(np.mean([abs(f(r['wf']) - r['acc']) for r in main])):.2f} "
            f"(논문 계수 그대로: {mae(main, EST['WF (가중 flip, 2차식)']):.2f}).", ""]
out += ["## 손상별 (5단계 평균): 실제 정확도, WF 추정, 오차", "", "| corruption | acc | WF | WF 추정 | |오차| 평균 | ATC(MC) |오차| 평균 |",
        "|---|---:|---:|---:|---:|---:|"]
for c in MAIN + HOLD:
    rs = [r for r in (L(f"{c}_{s}") for s in range(1, 6)) if r]
    if rs:
        m = lambda f: float(np.mean([f(r) for r in rs]))
        out.append(f"| {c}{' (holdout)' if c in HOLD else ''} | {m(lambda r: r['acc']):.1f} | {m(lambda r: r['wf']):.1f} | {m(lambda r: r['est']['wf_quad']):.1f} | "
                   f"{mae(rs, EST['WF (가중 flip, 2차식)']):.2f} | {mae(rs, EST['ATC (max-softmax)']):.2f} |")
open(D / "RESULTS.md", "w").write("\n".join(out) + "\n")
print("\n".join(out[5:20]))
