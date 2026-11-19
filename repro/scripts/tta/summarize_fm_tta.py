"""Result tables (numbers only) for scripts/tta/cls_tta_fm.py.

  ~/REOBench/.venv/bin/python scripts/tta/summarize_fm_tta.py     # from repro/, CPU
Output: experiments/2026-11-17_fm_tta/RESULTS.md (only the datasets / backbones that are finished)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402

D = paths.EXP / "2026-11-17_fm_tta"
ORDER = ["mae_in", "satmae", "scalemae", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16", "dino_in_s8",
         "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16", "sup_in_r50", "sup_in_vitb16",
         "sup_in_vitl16"]
BASE = ["none", "none_al", "tent", "entf", "flipf", "bothf", "tent_al", "flipf_al", "sar", "eata", "eata_allnorm", "tent_bnstat"]
X10 = ["tent_x10", "entf_x10", "flipf_x10", "bothf_x10", "tent_al_x10", "flipf_al_x10"]


def table(out, head, rows):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out.append("")


out = ["# 17개 백본 전체 통과 TTA 결과표 (2026-11-17~18, 수치만)", "",
       "이미지를 백본에 통과시키고 norm 층 affine을 학습(상위 25% 블록과 마지막 norm 제외; EATA는 모든 norm). head는 stage-1 선형 head.",
       "stream마다 초기화, 배치 64, SGD momentum 0.9, lr 0.001(LayerNorm) / 0.00025(BatchNorm). `_x10` = 그 10배.",
       "none_al = 학습 없이 head 입력만 정렬. entf = 엔트로피 < 0.4 ln C인 샘플만, flipf = 정렬 전후 예측이 같은 샘플만, bothf = 둘 다,",
       "`_al` = 정렬한 head 입력으로 예측·손실. tent_bnstat = 모든 BN을 배치 통계로(교과서 TENT, BN 백본만).",
       "손상 = 12종 × 5단계 60개 stream 평균. 정의는 `scripts/tta/cls_tta_fm.py` docstring.",
       "eata = TENT·SAR과 같은 파라미터 집합(마지막 norm과 상위 블록 제외). eata_allnorm = 공식 코드처럼 모든 norm 층: 5개 백본에서 발산",
       "(마지막 norm의 일부 원소에서 Fisher가 12.6~5,050으로, SGD에서 2차 penalty가 안정한 한계 0.95를 넘음. docstring 참고).", ""]
for ds, title in (("aid", "AID / AID-C"), ("r45", "RESISC45 / RESISC45-C")):
    R = {(m, k): json.load(open(D / ds / m / f"{k}.json")) for m in ORDER for k in BASE + X10 if (D / ds / m / f"{k}.json").exists()}
    ms = [m for m in ORDER if (m, "none") in R]
    if not ms:
        continue
    for keys, tag in ((BASE, "프로토콜 lr"), (X10, "lr × 10")):
        keys = [k for k in keys if any((m, k) in R for m in ms)]
        if not keys:
            continue
        ref = ["none", "none_al"] if tag != "프로토콜 lr" else []
        for metric, name, fmt in (("avg", "손상 정확도 (%)", "{:.1f}"), ("clean", "clean 정확도 (%)", "{:.1f}"),
                                  ("avg_signed_ece", "손상 signed ECE (+ = 과신)", "{:+.3f}")):
            cols = ref + keys
            out += [f"## {title}: {name}, {tag} (백본 {len(ms)}개)", ""]
            rows = [[m] + [fmt.format(R[(m, k)][metric]) if (m, k) in R else "–" for k in cols] for m in ms]
            full = [m for m in ms if all((m, k) in R for k in cols if k not in ("tent_bnstat", "eata"))]
            rows.append([f"**평균 ({len(full)}개)**"] + [fmt.format(sum(R[(m, k)][metric] for m in full if (m, k) in R) /
                                                                 max(1, sum((m, k) in R for m in full))) for k in cols])
            if metric == "avg":
                rows.append(["none보다 높은 백본 수"] + [str(sum(R[(m, k)]["avg"] > R[(m, "none")]["avg"] + 0.05 for m in full if (m, k) in R)) for k in cols])
                rows.append(["none_al보다 높은 백본 수"] + [str(sum(R[(m, k)]["avg"] > R[(m, "none_al")]["avg"] + 0.05 for m in full if (m, k) in R)) for k in cols])
            table(out, ["model"] + cols, rows)
open(D / "RESULTS.md", "w").write("\n".join(out) + "\n")
print("wrote", D / "RESULTS.md", len(out))
