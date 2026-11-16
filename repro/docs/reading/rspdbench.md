# RSPDBench (arXiv 2609.23427) — 읽기 메모와 판단

PDF: `papers/benchmark/RSPDBench_ …pdf` = **v3 (2026-10-31)**. arXiv Comments: "Accepted to WACV 2027 (Round 1)".
코드: github.com/Tanjim131/RSPDBench 공개(MIT, `rspdbench/` 연산자 P1–P4·C1·C2, `results/` seed별 CSV, `pip install rspdbench`).
저장소는 README와 파일 목록만 확인했고 소스는 읽지 않음(2026-11-16).

## 1. 다른 세션 요약(v2 기준)에서 v3로 바뀐 점
| 항목 | v2 요약 | v3 본문 |
|---|---|---|
| 심사·코드 | venue 없음, 코드 미공개 | WACV 2027 채택, 코드 공개 |
| probe 재학습 회복 | "본문에서 확인 안 됨, 인용 금지" | §4.7에 있음. m-BigEarthNet S3 평균 drop: P4 15.6→3.7, C2 31.4→6.9, P1 52.1→19.7, C1 58.0→25.6. DINOv3는 P1 drop이 40% 근처로 남음 |
| TTA | 언급 없음 | §4.7 끝: 예상 못 한 결함은 label-free TTA(TENT 인용)가 필요, 해상도 손실로 사라진 정보를 TTA가 되살릴 수 있는지는 open question |
| S3 평균(최대) drop | P1 35.2(91.2), P4 11.0(41.9), P2 3.9(17.5), P3 2.4(10.6) | P1 32.8(89.3), P4 10.3(39.2), P2 4.6(29.5), P3 2.1(10.5) (Table 2) |
| seed | clean만 3 seed | "All results are means over three seeds" (§4.3) |

그대로인 것: 결함 정의와 Table 1 파라미터, 데이터 5종, 모델 7종 + scratch 2종, frozen encoder + clean 학습 head.

## 2. 설계에서 가져올 것
- 결함을 모델 정규화·리사이즈 **전에** dataset-native 단위로 적용. severity는 사양에 고정:
  P1 σ = N/2 (Sentinel-2 MTF 15–30%에 맞춤), P2 ±3/8/15% (S2 절대 방사 불확도 5%, Landsat ±3%),
  P3 0.3/1/2 px (S2 밴드 간 정합 요구 0.3 px), P4 SNR 100/33/20 (S2 요구 72–174).
- P1·P3 연산자는 Sentinel-2의 B08(10 m)·B8A(20 m) 쌍으로 검증(최적 σ 0.75–1.5, 잔차 변위 중앙값 0.02–0.05 px).
- 결과: RGB DINOv3는 gain에 거의 불변(≤1.2%)이지만 P1 drop이 5개 중 4개 데이터셋에서 최대
  (m-EuroSAT DINOv3-Sat P1-S3 81%). 해상도 조건화 모델 RAMEN은 반대. P1 순위와 P2 순위가 거의 뒤집힘(Kendall τ 중앙값 −0.52).
- §4.7의 해석(저자 추측): 노이즈·harmonization 잔차는 readout이 무시할 수 있는 방향으로 feature를 움직이고,
  해상도 손실은 쓸 정보 자체를 줄인다.

## 3. 우리 설정(RGB, AID / RESISC45 / Potsdam)에서의 판단
- 벤치마크 자체(데이터 5종, 다중분광)는 채택하지 않음. **연산자만** RGB 세트에 적용 = "RSPDBench 방식 연산자".
  물리 근거는 Sentinel-2/Landsat 사양이라 Google Earth 8-bit RGB(AID, RESISC45)에는 그대로 옮겨지지 않음 → 논문에서 "RSPDBench"라고 부르지 말 것.
- RGB에서 정보가 있는 축은 P1과 C1.
  - P2(전 채널 같은 gain): 채널별 평균·표준편차 정렬(ISA)이 정확히 되돌리는 변환. 통제 조건.
  - P4: σ = 0.01–0.05 × 채널 평균. REOBench·ImageNet-C 노이즈보다 훨씬 약함. 통제 조건.
  - P3·C2: 밴드 그룹이 필요해 RGB에는 정의되지 않음.
- P1 주의: native 해상도에서 적용하면 AID(600 px → 224 리사이즈, 2.7배 축소)에서는 N=2가 리사이즈에 거의 묻힘.
  RESISC45(256 px)와 Potsdam(512 px 그대로 입력)에서는 온전히 작용. 모델 입력 해상도(224) 기준 적용을 함께 둘지 정해야 함.
- 우리 연구에 주는 것:
  1. §4.7이 우리 질문을 open question으로 적어 둠 → 동기 인용.
  2. "정렬로 고쳐지는 shift vs 정보가 사라진 shift" 가설을 frac_common·align 지표로 직접 잴 수 있음(P2·P4 vs P1).
  3. REOBench OpenReview의 "RS 고유성" 지적에 대한 보완 세트.
