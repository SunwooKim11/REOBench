"""Human-rated scene complexity of AID classes (Peng et al., Remote Sens. 2021, 13, 742, Fig. 4: AID-22, class scores 1-10 =
memorability + visual search difficulty, three levels) against our per-class results under corruption.

Per AID class (22 of the 30 classes are rated), mean over the 17 backbones (stage-1 linear head, saved features):
clean accuracy, accuracy over the 60 corrupted streams, drop, over-confidence (confidence - correct), flip rate, gain of
head-input standardization (rs_properties.per_image), and the pixel proxy JPEG bits per pixel (rs_properties.proxies).
Spearman over the 22 classes, and means per complexity level.

  /opt/conda/envs/segtta/bin/python scripts/analysis/scene_complexity_check.py     # from repro/, GPU, a few minutes
Output: experiments/2026-11-26_rs_properties/RESULTS_scene_complexity.md
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import rs_properties as rp  # noqa: E402
from reobench_tta import common  # noqa: E402

SCORE = {"Viaduct": 7.9, "Airport": 7.4, "RailwayStation": 8.1, "Park": 8.2, "DenseResidential": 8.9, "Commercial": 9.3, "Center": 8.6,
         "Stadium": 5.2, "Bridge": 5.8, "River": 6.1, "Square": 6.5, "Playground": 5.0, "BaseballField": 6.1, "Port": 6.4, "Parking": 6.4,
         "Mountain": 4.6, "BareLand": 3.2, "Beach": 3.7, "Desert": 1.9, "Farmland": 3.4, "Meadow": 1.9, "Forest": 2.7}   # read off Fig. 4
LEVEL = lambda v: "high" if v >= 7 else "middle" if v >= 4.5 else "low"

if __name__ == "__main__":
    torch.set_num_threads(4)
    classes = datasets.ImageFolder(f"{common.DATA_ROOT}/AID_clean_renamed/test").classes
    assert set(SCORE) <= set(classes), set(SCORE) - set(classes)
    with Pool(8) as pool:
        px = np.array(pool.map(rp.proxies, [(p, "aid") for p in rp.image_paths("aid")], chunksize=64))
    P = []
    for m in rp.ORDER:
        Z, y, w, b = rp.load(m, "aid")
        P.append(rp.per_image(Z, y, w, b)); del Z
        torch.cuda.empty_cache(); print(m, flush=True)
    d = {k: np.mean([p[k] for p in P], 0) for k in P[0]}
    d["drop"] = d["clean"] - d["corr"]; d["jpeg"] = px[:, 0]
    y = y.numpy()
    cm = {k: np.array([v[y == classes.index(c)].mean() for c in SCORE]) for k, v in d.items()}
    sc = np.array(list(SCORE.values()))
    OC = (("clean", "clean 정확도 (%)", 100), ("corr", "손상 정확도 (%)", 100), ("drop", "하락 (pt)", 100), ("over", "과신 (pt)", 100),
          ("flip", "flip 비율 (%)", 100), ("gain", "head 입력 표준화 이득 (pt)", 100), ("jpeg", "JPEG 비트/픽셀", 1))
    out = ["# 사람이 매긴 장면 복잡도(AID-22)와 손상 결과 (2026-11-26, 수치만)", "",
           "`scripts/analysis/scene_complexity_check.py`. 복잡도 = Peng 외(Remote Sens. 2021, 13, 742) Fig. 4의 클래스 점수(1–10, 기억 난이도 + 시각 탐색 난이도, 사람 평가).",
           "AID 30개 클래스 중 22개에만 점수가 있다. 점수는 그림에서 읽은 값이다. 결과는 17개 백본 평균, 손상 = 60개 stream 평균.", "",
           "## 복잡도 수준별 평균", "", "| 수준 | 클래스 수 | 복잡도 점수 | " + " | ".join(n for _, n, _ in OC) + " |", "|---|---:|---:|" + "---:|" * len(OC)]
    lv = np.array([LEVEL(v) for v in sc])
    for L in ("low", "middle", "high"):
        out.append(f"| {L} | {(lv == L).sum()} | {sc[lv == L].mean():.2f} | " + " | ".join(f"{s * cm[k][lv == L].mean():.{1 if s == 100 else 2}f}" for k, _, s in OC) + " |")
    out += ["", "## 복잡도 점수와의 Spearman 순위상관 (클래스 22개; 5% 수준에서 유의한 크기는 약 0.42)", "", "| 결과 | 사람 복잡도 점수와 | JPEG 비트/픽셀과 |", "|---|---:|---:|"]
    for k, n, _ in OC:
        out.append(f"| {n} | {rp.spear(sc, cm[k]):+.2f} | " + ("–" if k == "jpeg" else f"{rp.spear(cm['jpeg'], cm[k]):+.2f}") + " |")
    out += ["", "## 클래스별", "", "| 클래스 | 수준 | 복잡도 | " + " | ".join(n for _, n, _ in OC) + " |", "|---|---|---:|" + "---:|" * len(OC)]
    for i in np.argsort(sc):
        c = list(SCORE)[i]
        out.append(f"| {c} | {lv[i]} | {sc[i]:.1f} | " + " | ".join(f"{s * cm[k][i]:.{1 if s == 100 else 2}f}" for k, _, s in OC) + " |")
    f = rp.OUT / "RESULTS_scene_complexity.md"
    f.write_text("\n".join(out) + "\n")
    print("saved", f)
