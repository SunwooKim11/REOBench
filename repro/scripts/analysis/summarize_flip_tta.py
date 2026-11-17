"""Result tables (numbers only) for flip_filter_tta.py.

  ~/REOBench/.venv/bin/python scripts/analysis/summarize_flip_tta.py [--bs 64]     # from repro/, CPU
Output: experiments/2026-11-17_flip_tta/bs<bs>/RESULTS.md
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402

ORDER = ["mae_in", "satmae", "scalemae", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16", "dino_in_s8",
         "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16", "sup_in_r50", "sup_in_vitb16",
         "sup_in_vitl16"]
LOSSES = ["tent", "entf", "flipf", "bothf"]

ap = argparse.ArgumentParser()
ap.add_argument("--bs", type=int, default=64)
args = ap.parse_args()
D = paths.EXP / "2026-11-17_flip_tta" / f"bs{args.bs}"
R = {(m, ds): json.load(open(D / f"{m}_{ds}.json")) for m in ORDER for ds in ("aid", "r45") if (D / f"{m}_{ds}.json").exists()}
lrs = sorted({float(k.split("_lr")[1]) for r in R.values() for k in r["methods"] if "_lr" in k})


def val(m, ds, key, metric):  # mean over the 60 corrupted streams, or the clean stream
    x = R[(m, ds)]["methods"][key]
    if metric == "clean":
        return x["m"][0]["acc"]
    if metric == "used":
        return 100 * sum(x["used"][1:]) / 60 if "used" in x else float("nan")
    return sum(s[metric] for s in x["m"][1:]) / 60


def table(out, head, rows):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out.append("")


out = [f"# flip 필터 TTA 결과표 (배치 {args.bs}, 2026-11-17, 수치만)", "",
       "저장된 feature 위에서 head 입력의 채널별 affine(gamma, beta)만 엔트로피 최소화로 학습. head는 고정. stream마다 초기화.",
       "입력: raw = 그대로, al = 배치마다 source 통계로 정렬한 것. 손실에 쓰는 샘플: tent = 전부, entf = 엔트로피 < 0.4 ln C,",
       "flipf = 정렬 전후 예측이 같은 샘플, bothf = 둘 다. Adam, 학습률은 전부 보고(라벨로 고르지 않음).",
       "손상 = 12종 × 5단계 60개 stream 평균. 정의는 `scripts/analysis/flip_filter_tta.py` docstring.", ""]
for ds, title in (("aid", "AID / AID-C"), ("r45", "RESISC45 / RESISC45-C")):
    ms = [m for m in ORDER if (m, ds) in R]
    if not ms:
        continue
    avg = lambda key, metric: sum(val(m, ds, key, metric) for m in ms) / len(ms)
    out += [f"## {title}: 백본 {len(ms)}개 평균", "",
            "이긴 수 = 손상 정확도가 none보다 높은 백본 수 / align_ms보다 높은 백본 수.", ""]
    rows = []
    for key in ["none", "align_ms"] + [f"{i}_{l}_lr{lr:g}" for i in ("raw", "al") for l in LOSSES for lr in lrs]:
        w0 = sum(val(m, ds, key, "acc") > val(m, ds, "none", "acc") + 1e-9 for m in ms)
        w1 = sum(val(m, ds, key, "acc") > val(m, ds, "align_ms", "acc") + 1e-9 for m in ms)
        u = avg(key, "used")
        rows.append([key, f"{avg(key, 'clean'):.1f}", f"{avg(key, 'acc'):.1f}", f"{avg(key, 'acc') - avg('none', 'acc'):+.1f}",
                     f"{avg(key, 'signed_ece'):+.3f}", f"{avg(key, 'ece'):.3f}", "–" if u != u else f"{u:.1f}", f"{w0} / {w1}"])
    table(out, ["method", "clean acc", "손상 acc", "none 대비", "signed ECE", "ECE", "쓴 샘플 (%)", "이긴 수 (none / align)"], rows)
    for metric, name, fmt in (("acc", "손상 정확도 (%)", "{:.1f}"), ("clean", "clean 정확도 (%)", "{:.1f}"),
                              ("signed_ece", "손상 signed ECE (+ = 과신)", "{:+.3f}")):
        for lr in lrs:
            keys = ["none", "align_ms"] + [f"{i}_{l}_lr{lr:g}" for i in ("raw", "al") for l in LOSSES]
            out += [f"### {title}: {name}, lr {lr:g}", ""]
            table(out, ["model", "none", "align_ms"] + [f"{i} {l}" for i in ("raw", "al") for l in LOSSES],
                  [[m] + [fmt.format(val(m, ds, k, metric)) for k in keys] for m in ms])
open(D / "RESULTS.md", "w").write("\n".join(out) + "\n")
print("wrote", D / "RESULTS.md", len(out), "models", {ds: sum((m, ds) in R for m in ORDER) for ds in ("aid", "r45")})
