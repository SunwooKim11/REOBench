"""Stage-1 result tables (numbers only) from the JSONs of feat_metrics.py, seg_drift.py, rrshid_analysis.py, natural_vs_rs.py.

  ~/REOBench/.venv/bin/python scripts/analysis/summarize_stage1.py     # from repro/, CPU, seconds
Output: experiments/2026-11-16_backbone_feats/RESULTS.md
"""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import paths  # noqa: E402

D = paths.EXP / "2026-11-16_backbone_feats"
ORDER = ["mae_in", "satmae", "scalemae", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16", "dino_in_s8",
         "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16", "sup_in_r50", "sup_in_vitb16",
         "sup_in_vitl16"]
CLS_C = ["rotate_new", "scale_image", "translate_image_new", "clouds", "data_gaps", "brightness_contrast", "haze_update_new",
         "gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur", "compression_artifacts"]
SHORT = {"rotate_new": "rotate", "scale_image": "scale", "translate_image_new": "translate", "clouds": "clouds",
         "data_gaps": "gaps", "brightness_contrast": "bright/contr", "haze_update_new": "haze", "gaussian_noise_new": "gauss noise",
         "salt_and_pepper_noise": "S&P", "gaussian_blur": "gauss blur", "motion_blur": "motion blur",
         "compression_artifacts": "JPEG", "gaussian_noise": "gauss noise", "haze": "haze", "impulse_noise": "impulse",
         "jpeg_compression": "JPEG", "brightness": "brightness", "contrast": "contrast"}
out = []


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def table(head, rows, fmt):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    for r in rows:
        out.append("| " + " | ".join([str(r[0])] + [("–" if (v is None or (isinstance(v, float) and math.isnan(v))) else f.format(v))
                                                    for v, f in zip(r[1:], fmt)]) + " |")
    out.append("")


# ---------------------------------------------------------------- A. classification
out += ["# Stage 1 결과표 (2026-11-16 실행, 수치만)", "",
        "백본 17개(대응 쌍은 ImageNet → RS 순서로 붙여 둠). 모든 값은 seed 0, 224 px, frozen backbone + 선형 head 1개.",
        "정의는 `scripts/analysis/{feat_metrics,seg_drift,rrshid_analysis,natural_vs_rs}.py` docstring.", ""]
for ds, title in (("aid", "AID / AID-C (test 2,000장)"), ("r45", "RESISC45 / RESISC45-C (test 6,300장)")):
    M = {m: json.load(open(D / "metrics" / f"{m}_{ds}.json"))["streams"] for m in ORDER}
    cm = lambda m, k, cs=CLS_C: mean([M[m][f"{c}_s{s}"].get(k) for c in cs for s in range(1, 6)])
    out += [f"## A. 분류 — {title}", "", "### 전체 요약 (손상 열 = 12종 × 5단계 평균)", "",
            "align ms = head 입력의 평균·표준편차를 source 통계로 맞춤(배치 64), align m = 평균만.", ""]
    table(["model", "clean acc", "clean ECE", "acc", "ECE", "signed ECE", "AUROC", "rel_drift", "frac_common", "CKA", "cos",
           "PR ratio", "hdhc_wrong", "align ms acc", "align m acc", "align ms ECE"],
          [[m, M[m]["clean"]["acc"], M[m]["clean"]["ece"]] + [cm(m, k) for k in
           ("acc", "ece", "signed_ece", "auroc", "rel_drift", "frac_common", "cka", "cos", "pr_ratio", "hdhc_wrong",
            "align_ms_acc", "align_m_acc", "align_ms_ece")] for m in ORDER],
          ["{:.1f}", "{:.3f}", "{:.1f}", "{:.3f}", "{:+.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.3f}",
           "{:.3f}", "{:.1f}", "{:.1f}", "{:.3f}"])
    for key, name, f in (("acc", "정확도 (%)", "{:.1f}"), ("align_gain", "align ms 후 정확도 변화 (pt)", "{:+.1f}"),
                         ("signed_ece", "signed ECE (+ = 과신)", "{:+.3f}"), ("frac_common", "frac_common", "{:.3f}"),
                         ("rel_drift", "rel_drift", "{:.2f}"), ("cka", "CKA", "{:.3f}")):
        out += [f"### 손상별 {name} — 5단계 평균", ""]
        rows = []
        for m in ORDER:
            r = [m]
            for c in CLS_C:
                if key == "align_gain":
                    r.append(mean([M[m][f"{c}_s{s}"]["align_ms_acc"] - M[m][f"{c}_s{s}"]["acc"] for s in range(1, 6)]))
                else:
                    r.append(cm(m, key, [c]))
            rows.append(r)
        table(["model"] + [SHORT[c] for c in CLS_C], rows, [f] * 12)
    out += ["### 단계별 정확도 (%) — 12종 평균", ""]
    table(["model", "clean", "s1", "s2", "s3", "s4", "s5"],
          [[m, M[m]["clean"]["acc"]] + [mean([M[m][f"{c}_s{s}"]["acc"] for c in CLS_C]) for s in range(1, 6)] for m in ORDER],
          ["{:.1f}"] * 6)

# ---------------------------------------------------------------- B. segmentation
out += ["## B. 분할 — Potsdam / Potsdam-C (UperNet, 학습된 B 체크포인트)", ""]
LAYERS = ["blocks.7", "blocks.11", "blocks.15", "blocks.23", "cls_in"]
for m in ("scalemae", "satmae"):
    S = json.load(open(D / "seg" / f"{m}_B.json"))["streams"]
    cs = sorted({re.sub(r"_s\d$", "", k) for k in S if k != "clean"})
    c0 = S["clean"]
    out += [f"### {m} — clean mIoU {c0['miou']:.2f}, pixel ECE {c0['ece']:.3f}, AUROC {c0['auroc']:.3f}", "",
            "mIoU (단계별), 나머지는 5단계 평균. fc = frac_common. cls_in = 분류 conv 입력.", ""]
    rows = []
    for c in cs:
        g = lambda k: mean([S[f"{c}_s{s}"].get(k) for s in range(1, 6)])
        rows.append([SHORT.get(c, c)] + [S[f"{c}_s{s}"]["miou"] for s in range(1, 6)] + [g("ece"), g("signed_ece"), g("auroc")]
                    + [g(f"{L}/frac_common") for L in LAYERS] + [g(f"{L}/cka") for L in LAYERS])
    table(["corruption", "s1", "s2", "s3", "s4", "s5", "ECE", "signed ECE", "AUROC"] + [f"fc {L.replace('blocks.', 'b')}" for L in LAYERS]
          + [f"CKA {L.replace('blocks.', 'b')}" for L in LAYERS], rows,
          ["{:.1f}"] * 5 + ["{:.3f}", "{:+.3f}", "{:.3f}"] + ["{:.3f}"] * 10)

# ---------------------------------------------------------------- C. RRSHID
R = {m: json.load(open(D / "rrshid" / f"{m}.json")) for m in ORDER}
out += ["## C. 실제 haze — RRSHID (clear/hazy 쌍, 라벨 없음)", "", "### 픽셀 통계", "",
        "affine R² = hazy에 채널별 a·x+b를 맞췄을 때 clear를 설명하는 비율(이미지별 평균). syn = 같은 clear 이미지에 x(1−i)+i.", ""]
table(["set", "PSNR (dB)", "affine R²", "mean ratio", "std ratio"],
      [[k, v["psnr"], v["affine_r2"], v["mean_ratio"], v["std_ratio"]] for k, v in R[ORDER[0]]["pixel"].items()],
      ["{:.2f}", "{:.3f}", "{:.2f}", "{:.2f}"])
for L in ("thin_fog", "moderate_fog", "thick_fog"):
    out += [f"### 실제 {L}", "",
            "agree = hazy와 clear의 예측이 같은 비율(%). 앞 6열은 전체 쌍, 뒤 6열은 val+test 쌍(보정 통계는 train clear에서).", ""]
    table(["model", "rel_drift", "frac_common", "CKA", "cos", "agree AID", "agree R45", "AID head", "+pixel ISA", "+feat align",
           "R45 head", "+pixel ISA", "+feat align"],
          [[m] + [R[m]["real"][L][k] for k in ("rel_drift", "frac_common", "cka", "cos", "agree_aid", "agree_r45", "ev_agree_aid",
                                                 "ev_agree_isa_aid", "ev_agree_align_aid", "ev_agree_r45", "ev_agree_isa_r45",
                                                 "ev_agree_align_r45")] for m in ORDER],
          ["{:.2f}", "{:.3f}", "{:.3f}", "{:.3f}"] + ["{:.1f}"] * 8)
out += ["### 합성 haze (같은 clear 이미지, i = 0.2 / 0.4 / 0.6)", ""]
rows = []
for m in ORDER:
    r = [m]
    for k in ("rel_drift", "frac_common", "cka", "agree_aid"):
        r += [R[m]["syn"][f"syn_i{i}"][k] for i in ("0.2", "0.4", "0.6")]
    rows.append(r)
table(["model"] + [f"{k} {i}" for k in ("rel_drift", "frac_common", "CKA", "agree AID") for i in ("0.2", "0.4", "0.6")], rows,
      ["{:.2f}"] * 3 + ["{:.3f}"] * 6 + ["{:.1f}"] * 3)

# ---------------------------------------------------------------- D. natural vs RS
NC = ["gaussian_noise", "impulse_noise", "gaussian_blur", "motion_blur", "jpeg_compression", "brightness", "contrast"]
DS = ["imagenet", "aid", "r45"]
P = json.load(open(D / "natural_vs_rs" / "pixel.json"))
N = {m: json.load(open(D / "natural_vs_rs" / f"{m}.json")) for m in ORDER}
out += ["## D. 같은 손상(ImageNet-C 함수·파라미터)을 자연 이미지와 RS 이미지에 — natural_vs_rs", "",
        "ImageNet val 5,000장(클래스당 5), AID test 2,000장, RESISC45 test 1,996장. 모두 224 px.", "",
        "### 픽셀: PSNR (dB) / MS-SSIM", ""]
table(["corruption", "sev"] + [f"PSNR {d}" for d in DS] + [f"MS-SSIM {d}" for d in DS],
      [[SHORT[c], s] + [P[f"{d}/{c}_{s}"]["psnr"] for d in DS] + [P[f"{d}/{c}_{s}"]["ms_ssim"] for d in DS]
       for c in NC for s in range(1, 6)], ["{}"] + ["{:.2f}"] * 3 + ["{:.3f}"] * 3)
out += ["### 백본별 feature 변화 — 7종 × 5단계 평균", ""]
nm = lambda m, d, k, cs=NC: mean([N[m][f"{d}/{c}_{s}"].get(k) for c in cs for s in range(1, 6)])
table(["model"] + [f"{k} {d}" for k in ("rel_drift", "frac_common", "CKA") for d in DS],
      [[m] + [nm(m, d, k) for k in ("rel_drift", "frac_common", "cka") for d in DS] for m in ORDER],
      ["{:.2f}"] * 3 + ["{:.3f}"] * 6)
out += ["### 손상별 feature 변화 — 백본 17개 × 5단계 평균", ""]
table(["corruption"] + [f"{k} {d}" for k in ("rel_drift", "frac_common", "CKA") for d in DS],
      [[SHORT[c]] + [mean([nm(m, d, k, [c]) for m in ORDER]) for k in ("rel_drift", "frac_common", "cka") for d in DS] for c in NC],
      ["{:.2f}"] * 3 + ["{:.3f}"] * 6)
out += ["### 정확도 (%) — clean / 7종 × 5단계 평균", "", "ImageNet 정확도는 자체 ImageNet head가 있는 supervised 3개만.", ""]
table(["model", "ImageNet clean", "ImageNet corrupted", "AID clean", "AID corrupted", "R45 clean", "R45 corrupted"],
      [[m] + [v for d in DS for v in (N[m][f"{d}/clean"]["acc"], nm(m, d, "acc"))] for m in ORDER], ["{:.1f}"] * 6)
out += ["### 손상별 정확도 유지율 (손상 정확도 / clean 정확도, %) — 5단계 평균", "",
        "ImageNet 열은 supervised 3개 평균, AID·R45 열은 백본 17개 평균과 supervised 3개 평균.", ""]
SUP = ["sup_in_r50", "sup_in_vitb16", "sup_in_vitl16"]
ret = lambda ms, d, c: mean([100 * nm(m, d, "acc", [c]) / N[m][f"{d}/clean"]["acc"] for m in ms])
table(["corruption", "ImageNet (sup 3)", "AID (sup 3)", "R45 (sup 3)", "AID (17)", "R45 (17)"],
      [[SHORT[c], ret(SUP, "imagenet", c), ret(SUP, "aid", c), ret(SUP, "r45", c), ret(ORDER, "aid", c), ret(ORDER, "r45", c)]
       for c in NC], ["{:.1f}"] * 5)

open(D / "RESULTS.md", "w").write("\n".join(out) + "\n")
print(f"wrote {D / 'RESULTS.md'}: {len(out)} lines")
