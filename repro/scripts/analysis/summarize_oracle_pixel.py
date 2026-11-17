"""Result tables (numbers only) for shift_direction_oracle.py and pixel_metrics.py.

  ~/REOBench/.venv/bin/python scripts/analysis/summarize_oracle_pixel.py     # from repro/, CPU
Output: experiments/2026-11-17_shift_oracle/RESULTS.md, experiments/2026-11-16_backbone_feats/pixel_metrics/RESULTS.md
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))  # repro/src
from reobench_tta import common, paths  # noqa: E402

ORDER = ["mae_in", "satmae", "scalemae", "moco_in", "moco_fmow", "gassl_fmow", "dino_in_s16", "satdino_s16", "dino_in_s8",
         "satdino_s8", "dino_in_b16", "satdino_b16", "dinov3_web_l16", "dinov3_sat_l16", "sup_in_r50", "sup_in_vitb16",
         "sup_in_vitl16"]
SHORT = {"rotate_new": "rotate", "scale_image": "scale", "translate_image_new": "transl", "clouds": "clouds", "data_gaps": "gaps",
         "brightness_contrast": "br/con", "haze_update_new": "haze", "gaussian_noise_new": "gnoise",
         "salt_and_pepper_noise": "S&P", "gaussian_blur": "gblur", "motion_blur": "mblur", "compression_artifacts": "JPEG",
         "gaussian_noise": "gnoise", "impulse_noise": "impulse", "jpeg_compression": "JPEG", "brightness": "bright",
         "contrast": "contrast"}
C = ["rotate_new", "scale_image", "translate_image_new", "clouds", "data_gaps", "brightness_contrast", "haze_update_new",
     "gaussian_noise_new", "salt_and_pepper_noise", "gaussian_blur", "motion_blur", "compression_artifacts"]
KS = [1, 4, 16, 64]


def table(out, head, rows, fmt):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    for r in rows:
        out.append("| " + " | ".join([str(r[0])] + [f.format(v) for v, f in zip(r[1:], fmt)]) + " |")
    out.append("")


def rank(a):
    a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(len(a))
    for v in np.unique(a):
        r[a == v] = r[a == v].mean()
    return r


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


# ------------------------------------------------------------------ oracle
D = paths.EXP / "2026-11-17_shift_oracle"
out = ["# Shift 방향 제거 oracle 결과표 (2026-11-17, 수치만)", "",
       "고정된 stage-1 선형 head, 평가 = 홀수 번째 test 이미지(AID 1,000 / RESISC45 3,150), 각 손상의 5단계 평균 정확도(%).",
       "pca_loco = 나머지 11종 손상의 clean−손상 차이로 구한 상위 k 방향 제거(평가 손상은 방향 추정에 안 씀).",
       "pca_same = 평가 손상 자체(짝수 번째 이미지)로 구한 방향(상한). refit = 방향을 뺀 clean train으로 head 재학습.",
       "mean_same = 같은 손상·단계의 평균 이동(짝수 번째 이미지)을 뺌(평균 보정의 상한). 정의는 `scripts/analysis/shift_direction_oracle.py`.", ""]
R = {(m, ds): json.load(open(D / f"{m}_{ds}.json")) for m in ORDER for ds in ("aid", "r45") if (D / f"{m}_{ds}.json").exists()}
METHODS = ["raw", "align_ms", "align_m", "clip3", "mean_same"] + [f"{p}_k{k}" for p in ("pca_loco", "pca_loco_refit", "pca_loco_align",
                                                                                         "pca_same", "dim_loco", "random") for k in KS
                                                                       if not (p == "pca_loco_refit" and k == 1)]
for ds, title in (("aid", "AID-C"), ("r45", "RESISC45-C")):
    ms = [m for m in ORDER if (m, ds) in R]
    mean = lambda m, k: sum(R[(m, ds)]["held_out"][c][k] for c in C) / len(C)
    out += [f"## {title}: 12종 평균 정확도", ""]
    cols = ["raw", "align_ms", "mean_same", "clip3", "pca_loco_k4", "pca_loco_k16", "pca_loco_k64", "pca_loco_refit_k16",
            "pca_loco_refit_k64", "pca_loco_align_k16", "pca_same_k4", "pca_same_k16", "pca_same_k64", "dim_loco_k16",
            "random_k16", "random_k64"]
    table(out, ["model", "clean"] + cols, [[m, R[(m, ds)]["clean"]["raw"]] + [mean(m, k) for k in cols] for m in ms], ["{:.1f}"] * (len(cols) + 1))
    out += [f"## {title}: 방법 적용 후 clean 정확도 (손상 없을 때의 비용)", ""]
    cc = ["raw", "align_ms", "clip3", "pca_loco_k4", "pca_loco_k16", "pca_loco_k64", "pca_loco_refit_k16", "pca_loco_refit_k64",
          "dim_loco_k16", "random_k16", "random_k64"]
    table(out, ["model"] + cc, [[m] + [R[(m, ds)]["clean"][k] for k in cc] for m in ms], ["{:.1f}"] * len(cc))
    out += [f"## {title}: 17개 백본 평균, 손상별 (raw 대비 변화, pt)", ""]
    rows = []
    for k in ["align_ms", "mean_same", "clip3"] + [f"pca_loco_k{x}" for x in KS] + [f"pca_loco_refit_k{x}" for x in (4, 16, 64)] + \
             [f"pca_loco_align_k{x}" for x in (4, 16, 64)] + [f"pca_same_k{x}" for x in KS] + [f"dim_loco_k{x}" for x in KS] + [f"random_k{x}" for x in KS]:
        rows.append([k] + [np.mean([R[(m, ds)]["held_out"][c][k] - R[(m, ds)]["held_out"][c]["raw"] for m in ms]) for c in C]
                    + [np.mean([mean(m, k) - mean(m, "raw") for m in ms])])
    raw_row = ["raw 정확도"] + [f"{np.mean([R[(m, ds)]['held_out'][c]['raw'] for m in ms]):.1f}" for c in C] + \
              [f"{np.mean([mean(m, 'raw') for m in ms]):.1f}"]
    rows = [raw_row] + [[r[0]] + [f"{v:+.1f}" for v in r[1:]] for r in rows]
    table(out, ["method"] + [SHORT[c] for c in C] + ["평균"], rows, ["{}"] * 13)
open(D / "RESULTS.md", "w").write("\n".join(out) + "\n")
print("wrote", D / "RESULTS.md", len(out))

# ------------------------------------------------------------------ pixel metrics
P = paths.EXP / "2026-11-16_backbone_feats" / "pixel_metrics"
F = paths.EXP / "2026-11-16_backbone_feats"
out = ["# 픽셀·지각 거리와 정확도 하락 (2026-11-17, 수치만)", "",
       "각 손상 이미지와 그 clean 원본 사이의 거리(224 px). LPIPS는 AlexNet·VGG 버전(Zhang et al. CVPR 2018, weights v0.1).",
       "정확도 하락 = clean 정확도 − 손상 정확도(pt). 정의는 `scripts/analysis/pixel_metrics.py`.", ""]
KEYS = ["psnr", "ms_ssim", "lpips_alex", "lpips_vgg"]
SIGN = {"psnr": -1, "ms_ssim": -1, "lpips_alex": 1, "lpips_vgg": 1}  # + = larger distance
if (P / "reobench.json").exists():
    X = json.load(open(P / "reobench.json"))
    for ds, title in (("aid", "AID-C"), ("r45", "RESISC45-C")):
        M = {m: json.load(open(F / "metrics" / f"{m}_{ds}.json"))["streams"] for m in ORDER}
        drop = {(c, s): np.mean([M[m]["clean"]["acc"] - M[m][f"{c}_s{s}"]["acc"] for m in ORDER]) for c in C for s in range(1, 6)}
        out += [f"## {title} (REOBench 12종): 5단계 평균", ""]
        table(out, ["corruption", "PSNR", "MS-SSIM", "LPIPS alex", "LPIPS vgg", "정확도 하락 (17개 평균)"],
              [[SHORT[c]] + [np.mean([X[f"{ds}/{c}_{s}"][k] for s in range(1, 6)]) for k in KEYS] + [np.mean([drop[(c, s)] for s in range(1, 6)])]
               for c in C], ["{:.2f}", "{:.3f}", "{:.3f}", "{:.3f}", "{:.1f}"])
        out += [f"### {title}: 거리와 정확도 하락의 순위 상관 (Spearman, + = 멀수록 많이 떨어짐)", "",
                "전체 60개 조건 / 기하 손상(rotate, scale, translate) 제외 45개 조건 / 손상 안에서 단계 순서만(12종 평균).", ""]
        rows = []
        for k in KEYS:
            v = {cs: SIGN[k] * X[f"{ds}/{cs[0]}_{cs[1]}"][k] for cs in drop}
            al = list(drop); ng = [cs for cs in al if cs[0] not in ("rotate_new", "scale_image", "translate_image_new")]
            within = np.mean([spear([v[(c, s)] for s in range(1, 6)], [drop[(c, s)] for s in range(1, 6)]) for c in C])
            rows.append([k, spear([v[x] for x in al], [drop[x] for x in al]), spear([v[x] for x in ng], [drop[x] for x in ng]), within])
        table(out, ["거리", "60개 조건", "45개 (기하 제외)", "단계 순서 (손상 안)"], rows, ["{:.2f}"] * 3)
        out += [f"### {title}: 백본별, 45개 조건(기하 제외)에서 거리와 정확도 하락의 Spearman", ""]
        ng = [(c, s) for c in C if c not in ("rotate_new", "scale_image", "translate_image_new") for s in range(1, 6)]
        table(out, ["model"] + KEYS, [[m] + [spear([SIGN[k] * X[f"{ds}/{c}_{s}"][k] for c, s in ng],
                                                   [M[m]["clean"]["acc"] - M[m][f"{c}_s{s}"]["acc"] for c, s in ng]) for k in KEYS] for m in ORDER],
              ["{:.2f}"] * 4)
if (P / "natural_vs_rs.json").exists():
    X = json.load(open(P / "natural_vs_rs.json"))
    NC = ["gaussian_noise", "impulse_noise", "gaussian_blur", "motion_blur", "jpeg_compression", "brightness", "contrast"]
    out += ["## 같은 손상(ImageNet-C 파라미터): ImageNet / AID / RESISC45, 5단계 평균", ""]
    table(out, ["corruption"] + [f"{k} {d}" for k in KEYS for d in ("IN", "AID", "R45")],
          [[SHORT[c]] + [np.mean([X[f"{d}/{c}_{s}"][k] for s in range(1, 6)]) for k in KEYS for d in ("imagenet", "aid", "r45")] for c in NC],
          ["{:.2f}"] * 3 + ["{:.3f}"] * 9)
    N = {m: json.load(open(F / "natural_vs_rs" / f"{m}.json")) for m in ORDER}
    SUP = ["sup_in_r50", "sup_in_vitb16", "sup_in_vitl16"]
    out += ["### 35개 조건에서 거리와 정확도 하락의 Spearman (supervised 3개 평균 하락; ImageNet은 자체 head)", ""]
    rows = []
    for k in KEYS:
        r = [k]
        for d in ("imagenet", "aid", "r45"):
            cs = [(c, s) for c in NC for s in range(1, 6)]
            dr = [np.mean([N[m][f"{d}/clean"]["acc"] - N[m][f"{d}/{c}_{s}"]["acc"] for m in SUP]) for c, s in cs]
            r.append(spear([SIGN[k] * X[f"{d}/{c}_{s}"][k] for c, s in cs], dr))
        rows.append(r)
    table(out, ["거리", "ImageNet", "AID", "RESISC45"], rows, ["{:.2f}"] * 3)
open(P / "RESULTS.md", "w").write("\n".join(out) + "\n")
print("wrote", P / "RESULTS.md", len(out))
