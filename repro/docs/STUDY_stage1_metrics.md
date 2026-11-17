# Stage 1 지표 공부 노트 — 무엇을, 어떻게 계산하고, 무엇을 뜻하는가

> 목적: stage-1 분석(`experiments/2026-11-16_backbone_feats/ANALYSIS.md`)과 자연 shift 결과를 직접 읽고 판단하기 위한 문서(2026-11-17).
> 코드 정의: `scripts/analysis/feat_metrics.py`, `labelfree_signals.py`, `natural_vs_rs.py` docstring.
> 각 절 끝의 **[직접 확인]** 은 손으로 확인해 볼 항목.

---

## 0. 한 장 요약

```
무엇을 재는가                 지표                              라벨 필요   clean 쌍 필요
이미지가 얼마나 망가졌나       PSNR, MS-SSIM                     X          O
모델이 얼마나 틀리나           accuracy                          O          X
confidence를 믿을 수 있나      ECE, signed ECE, NLL, AUROC       O          X
feature가 어떻게 움직였나      rel_drift, frac_common, cos, CKA, PR ratio   X   O (PR 제외)
위험한 샘플이 얼마나 되나      hdhc_wrong                        O          O
라벨 없이 정확도 추정          AC, DoC, ATC, flip                X          X
라벨 없이 고치기               정렬 (align ms / m)               X          X
```

배포 상황에서 실제로 쓸 수 있는 것은 **라벨도 clean 쌍도 필요 없는** 마지막 두 줄이다.
나머지는 "왜 그런가"를 설명하려는 진단용이다.

---

## 1. 기본 용어

- **stream**: 한 조건의 test 이미지 묶음. "AID test 2,000장에 haze 3단계를 건 것"이 stream 하나.
- **source / target**: source = head를 학습한 clean 데이터, target = 평가하는 (손상되거나 다른 데이터셋의) stream.
- **head**: frozen backbone feature 위에 얹은 선형 분류기 1개. 입력은 source train 통계로 표준화한 feature z = (f − μ_src) / σ_src.
  그래서 source에서는 z의 평균이 0, 표준편차가 1이다.
- **confidence**: softmax 확률의 최댓값.
- **pt**: 퍼센트포인트. 정확도 60%를 85%로 추정하면 오차 25pt.

---

## 2. 픽셀 지표: 이미지가 얼마나 망가졌나

### 2.1 PSNR (Peak Signal-to-Noise Ratio)
- 계산: MSE = 원본과 손상 이미지의 픽셀 차이 제곱 평균(값 범위 0–1). PSNR = 10 · log10(1 / MSE), 단위 dB.
- 읽는 법: 높을수록 원본에 가깝다. 대략 30 dB 이상이면 차이가 거의 안 보이고, 20 dB 아래면 뚜렷하다.
- 약점: 픽셀 차이만 본다. 밝기를 통째로 올리면 구조는 그대로인데 PSNR은 크게 떨어진다(brightness가 13–14 dB).

### 2.2 MS-SSIM (Multi-Scale Structural Similarity)
- 출처: SSIM은 Wang, Bovik, Sheikh, Simoncelli, "Image Quality Assessment: From Error Visibility to Structural Similarity",
  IEEE TIP 2004. MS-SSIM은 Wang, Simoncelli, Bovik, "Multiscale Structural Similarity for Image Quality Assessment",
  Asilomar Conference 2003.
- 표준인가: 원본이 있을 때 쓰는 화질 평가(full-reference IQA)에서는 PSNR과 함께 가장 널리 쓰는 지표다(영상 압축 평가의 기본).
  강건성 벤치마크에서 손상 세기를 재는 표준은 아니다. ImageNet-C 저자가 severity를 정할 때 MS-SSIM ≈ 1 − s/10을
  목표로 삼았다고 밝혀서(GitHub issue #12) 같은 잣대로 고른 것이다.
- SSIM 계산(한 위치, 11×11 Gaussian 창, σ = 1.5):
  - 밝기 항 l = (2 μx μy + C1) / (μx² + μy² + C1) — 두 창의 평균이 같은가
  - 대비·구조 항 cs = (2 σxy + C2) / (σx² + σy² + C2) — 두 창의 변동이 같은 모양·크기인가
  - C1 = 0.01², C2 = 0.03² (0으로 나누는 것 방지), SSIM = l · cs를 모든 위치에서 평균
- MS-SSIM 계산: 이미지를 2배씩 줄여 가며 5개 해상도에서 cs를 구하고, 가장 작은 해상도에서만 l을 곱한다.
  MS-SSIM = l₅^0.1333 · cs₁^0.0448 · cs₂^0.2856 · cs₃^0.3001 · cs₄^0.2363 · cs₅^0.1333 (지수는 논문의 고정 가중치).
- 읽는 법: 0–1, 1이면 동일. 사람 눈에 보이는 구조 손상에 PSNR보다 잘 맞는다.
- 우리 값: gauss noise 5단계 평균이 ImageNet 0.755, AID 0.710, RESISC45 0.716.
  "같은 손상이 세 데이터셋 이미지를 비슷한 정도로 망가뜨린다"는 근거.
- 우리 구현: 채널별 계산, 축소는 2×2 평균, 224 px (`natural_vs_rs.py`의 `ms_ssim`).

**[직접 확인]** `experiments/2026-11-16_backbone_feats/RESULTS.md` D절 픽셀 표에서 ImageNet 열의 MS-SSIM이 severity 1→5로
0.9 → 0.5 근처로 내려가는지(ImageNet-C의 설계 목표와 맞는지).

---

## 3. head 지표: 얼마나 틀리고, confidence를 믿을 수 있나

| 지표 | 계산 | 읽는 법 |
|---|---|---|
| accuracy | 맞은 비율 | |
| ECE | confidence를 15개 구간으로 나누고, 구간마다 \|평균 confidence − 정확도\|를 샘플 수로 가중 평균 | 0이면 confidence = 정확도. 0.1이면 평균 10pt 어긋남 |
| signed ECE | 평균 confidence − 정확도 | + 면 과신, − 면 과소신. 방향을 알려 줌 |
| NLL | 정답 클래스 확률의 −log 평균 | 틀리면서 확신할수록 급격히 커짐 |
| AUROC | 맞은 샘플과 틀린 샘플을 하나씩 뽑았을 때 맞은 쪽 confidence가 더 높을 확률 | 1 = 완벽히 구분, 0.5 = 무작위, 0.5 미만 = 틀린 쪽이 더 확신 |

- **ECE와 AUROC는 다른 것을 잰다.** ECE는 confidence의 **수준**이 맞는지, AUROC는 샘플 간 **순서**가 맞는지.
  stage 1에서는 AUROC 평균이 0.78–0.94로 유지되는데 signed ECE가 +0.13–0.19까지 가는 백본이 있다.
  → 순서는 남아 있고 수준이 통째로 올라간 것. 그래서 "수준만 다시 맞추면 된다"는 생각이 가능하다.
- signed ECE가 +0.164라는 것은 평균 confidence가 정확도보다 16.4pt 높다는 뜻(mae_in, AID 손상 평균).

**[직접 확인]** `RESULTS.md` A절 "손상별 signed ECE" 표에서 motion blur, data gaps, scale 열만 유독 큰지.

---

## 4. feature 지표: feature가 어떻게 움직였나

기호: z = clean 이미지의 feature(표준화), z′ = 같은 이미지에 손상을 건 feature, δ = z′ − z (그 샘플의 이동).
쌍(같은 이미지의 clean과 손상)이 있어야 계산할 수 있다. 실제 배포에서는 쌍이 없으므로 **진단용**이다.

### 4.1 rel_drift — 얼마나 멀리 움직였나
- 계산: 평균 ‖δ‖² / (clean feature가 자기 평균 주위로 퍼진 정도 = 평균 ‖z − z̄‖²)
- 읽는 법: 0 = 안 움직임. 1 = 이동 거리가 clean 데이터의 퍼짐과 같은 크기. 1을 넘으면 클래스 구조가 유지되기 어렵다.
- 예: mae_in AID 0.90, dinov3_web 0.33.
- 약점: MoCo ResNet은 blur에서 몇 개 채널이 수백 배 튀어 평균이 부풀려진다(중앙값이나 clamp가 필요).

### 4.2 frac_common — 그 이동이 모두에게 공통인가
- 계산: ‖평균 δ‖² / 평균 ‖δ‖². 이동을 "모든 샘플이 같이 밀린 벡터(평균 δ)"와 "샘플마다 다른 나머지"로 나눴을 때 앞쪽의 비중.
- 읽는 법: 1 = 모든 샘플이 똑같은 방향·거리로 평행이동. 0 = 제각각 흩어짐.
- 왜 중요한가: 공통 이동은 평균 하나만 빼면 되돌릴 수 있다(라벨 불필요). 제각각인 부분은 통계로는 못 고친다.
- 예: mae_in 0.49, satmae 0.12 (AID). mae_in은 정렬로 72.7 → 86.1, satmae는 74.6 → 79.7.
- 주의: stage-1 결과로는 손상 단위에서 "얼마나 고쳐지는가"를 잘 예측하지 못했다(motion blur는 0.42인데 회복 38%).
  공통 이동이 head의 결정 경계와 **무관한 방향**이면 커도 정확도에 영향이 없고, 나머지 부분이 크면 평균을 맞춰도 남는다.
- 파생: 공통 이동 크기 = rel_drift × frac_common = ‖평균 δ‖² / clean 퍼짐. 이것은 "stream 평균 − source 평균"이라
  **쌍 없이도** 잴 수 있다(문서의 mean_shift).

### 4.3 cos — 방향이 얼마나 유지됐나
- 계산: 샘플마다 clean feature와 손상 feature의 cosine을 구해 평균(표준화 전 raw feature).
- 읽는 법: 1 = 방향 그대로.
- 약점: feature에 큰 공통 성분(모든 이미지가 공유하는 평균 벡터)이 있으면 항상 1 근처로 나온다(mae_in 0.996, scalemae 0.991).

### 4.4 linear CKA — 샘플 사이의 관계가 유지됐나
- 출처: Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019.
- 계산: 두 feature 행렬 X(clean), Y(손상)를 각각 평균을 빼고, CKA = ‖YᵀX‖²_F / (‖XᵀX‖_F · ‖YᵀY‖_F).
- 뜻: "clean에서 서로 비슷했던 샘플들이 손상 후에도 서로 비슷한가". 1 = 샘플 간 관계가 그대로.
- 불변성: 평행이동(평균을 빼므로), 회전, 전체 크기 변화에 값이 안 변한다.
- 그래서 못 보는 것: 모든 샘플이 같이 밀리는 이동. mae_in은 CKA 0.941(17개 중 최고)인데 정확도가 22–25pt 떨어진다.
  고정된 head는 feature의 **절대 위치**로 판단하므로, 관계가 그대로여도 통째로 밀리면 틀린다.
- 또 다른 약점: 분산이 큰 몇 개 방향에 값이 좌우된다(알려진 비판).

### 4.5 PR ratio — 유효 차원이 줄었나
- 계산: participation ratio PR = (Σλᵢ)² / Σλᵢ² (λᵢ = feature 공분산의 고유값). PR ratio = 손상 PR / clean PR.
- 뜻: PR은 "분산이 고르게 퍼진 방향의 수". 1 미만이면 손상 때문에 feature가 더 적은 방향으로 찌그러졌다(정보 손실의 신호).
- 예: satmae 0.74, mae_in 1.05. RS 사전학습 백본은 손상 시 차원이 줄고, ImageNet 사전학습 DINO·MAE는 거의 안 준다.
- 쌍이 필요 없다(두 집합의 통계만 비교).

### 4.6 hdhc_wrong — 위험한 샘플의 비율
- 계산: (이동 ‖δ‖²이 stream 중앙값보다 큼) 그리고 (confidence > 0.9) 그리고 (틀림)인 샘플의 비율.
- 뜻: "많이 움직였는데 확신하면서 틀리는" 샘플. confidence로 걸러낼 수 없는 실패다.
- 예: moco_in 8.0%, mae_in 6.3%, DINO 계열 0.4–2.6% (AID).

**[직접 확인]** `RESULTS.md` A절 전체 요약 표에서 mae_in과 satmae 두 행을 나란히 놓고
rel_drift, frac_common, CKA, cos, PR ratio, 정렬 후 acc를 비교해 4.1–4.5의 설명과 맞는지.

---

## 5. 라벨 없이 쓰는 것: 정렬, flip, 정확도 추정

### 5.1 정렬 (align)
- align ms: 배치(64장)마다 z′의 채널별 평균·표준편차를 구해 (z′ − 평균) / 표준편차로 바꾼다. source에서는 평균 0·표준편차 1이었으므로
  이것이 "source 통계로 되돌리기"다. align m은 평균만 뺀다.
- 학습도 라벨도 없다. BatchNorm의 통계를 test 배치로 바꾸는 것(AdaBN)을 head 입력 한 곳에 적용한 셈.
- 4.2의 공통 이동을 제거하는 연산이다.

### 5.2 flip
- 계산: 정렬 전 예측과 정렬 후 예측이 다른 샘플의 비율(%).
- 뜻: "source 통계 관점에서 보면 판단이 달라지는 샘플이 얼마나 되나". shift가 없으면 0.8–5.7%(배치 통계의 잡음), 클수록 원래 예측을 믿기 어렵다.
- stage 1: 정확도와의 순위 상관 0.90–1.00(모든 백본). 정렬로 오르는 정확도 ≈ 0.47 × (flip − clean flip).

### 5.3 confidence 기반 정확도 추정
- AC (Average Confidence): 평균 confidence를 그대로 정확도 추정값으로.
- DoC (Guillory et al., ICCV 2021): source 정확도 − (source 평균 confidence − target 평균 confidence).
- ATC (Garg et al., ICLR 2022): source에서 "confidence > t인 비율 = 정확도"가 되는 t를 정하고, target에서 confidence > t인 비율을 추정값으로.
- 셋 다 confidence가 정확도와 함께 내려간다고 가정한다. 과신하면 추정이 높게 나온다.

### 5.4 우리 조합
- 정렬 후 ATC: 정렬한 head의 confidence에 ATC를 적용 → 정렬한 head의 정확도를 추정.
- flip 추정: 정렬 후 ATC − 0.47 × (flip − clean flip) → 원래 head의 정확도를 추정.

**[직접 확인]** `experiments/2026-11-17_natural_shift/RESULTS.md`의 "라벨 없는 정확도 추정" 표에서
ATC 오차와 정렬 후 ATC 오차, flip 추정 오차를 백본별로 비교(자연 shift에서도 줄어드는지).

---

## 6. 지표를 방법으로 연결하는 생각

| 관찰 | 쓸 수 있는 곳 |
|---|---|
| 순서(AUROC)는 남고 수준(signed ECE)이 올라감 | 수준만 다시 맞추면 됨 → 정렬, 또는 temperature를 shift 크기에 따라 조절 |
| 공통 이동(frac_common)이 큰 백본에서 정렬 이득이 큼 | 평균 이동 크기를 "정렬할지 말지 / 얼마나 세게"의 스위치로 |
| CKA가 높아도 정확도가 떨어짐 | 관계 보존은 "고칠 수 있다"는 뜻. CKA 높고 정확도 낮음 = 정렬로 회복 가능한 유형 |
| PR ratio가 줄어드는 손상·백본 | 정보가 사라진 유형. 통계 정렬로는 한계 → 다른 수단(입력 복원, 적응 학습)이 필요하다는 표시 |
| flip이 정확도를 따라감 | 라벨 없는 경보. flip이 크면 예측을 보류하거나 사람에게 넘기기 |
| hdhc_wrong이 높은 백본 | confidence 기반 필터(예: 엔트로피 TTA의 샘플 선택)가 틀린 샘플을 통과시킴 → 엔트로피 TTA가 실패하는 이유 후보 |

이 표는 가설이다. 어떤 것이 이미 연구됐는지는 `PRIOR_ART_stat_alignment.md`와 문헌 조사 결과로 확인한다.

## 7. 헷갈리기 쉬운 점 (2026-11-17 노트 검토)

### 7.1 rel_drift = 1의 뜻
- 분자 = 이미지 한 장이 손상으로 움직인 거리(제곱)의 평균. 분모 = clean 이미지가 전체 중심에서 떨어진 거리(제곱)의 평균, 즉 clean
  feature 구름의 반지름².
- 1이면 "이미지가 평균적으로 구름의 반지름만큼 움직였다"는 뜻이다. 클래스들은 그 구름 안에서 갈라져 있으므로, 반지름만큼 움직이면
  다른 클래스 영역으로 넘어갈 수 있는 크기다. 제곱 비율이라 0.5면 거리로는 반지름의 약 0.7배다.
- 한계: 방향을 보지 않는다. head가 읽지 않는 방향으로 움직인 것은 예측에 영향이 없다(7.4).

### 7.2 frac_common 0.2는 "20%의 샘플만 움직였다"가 아니다
- 모든 샘플이 움직인다. frac_common은 그 이동 에너지 중 **모든 샘플이 똑같이 공유하는 벡터 하나**가 차지하는 몫이다.
- 0.2는 작아 보이지만 크다. 표본별 이동은 수백~수천 개 방향에 흩어져 있고, 공통 이동은 방향 하나에 몰려 있다. 방향 하나가 전체
  이동의 20%를 차지한다는 뜻이다. 그래서 평균만 빼도 정확도가 5pt 넘게 돌아온다.
- 백본 평균은 0.11–0.49로 다르다(mae_in 0.49, scalemae 0.35, satmae·sup ViT 0.12–0.13). 손상별로도 다르다(sup_in_r50: data gaps
  0.38, motion blur 0.34, scale 0.32, 나머지 0.02–0.20).

### 7.3 cos는 백본끼리 비교하면 안 된다
- 범위는 0.75–0.996이다. mae_in이 0.996으로 가장 높은데 정확도 하락은 가장 큰 쪽이다.
- cos는 표준화하지 않은 feature로 계산한다. feature에 큰 상수 성분(모든 이미지에 공통인 평균 벡터)이 있으면 무엇이 변해도 1에
  가깝게 나온다. 같은 백본 안에서 손상끼리 비교하는 용도로만 쓴다.

### 7.4 "frac_common은 낮은데 CKA는 높다"가 뜻하는 것
- CKA가 못 보는 변화는 세 가지다: 평행이동(중심을 빼고 계산), 전체 확대·축소, 회전.
- 여기에 네 번째가 있다. linear CKA는 **분산이 큰 소수 방향**이 지배한다. 표본별 이동이 분산이 작은 많은 방향에 흩어져 있으면
  이동 에너지가 커도 CKA는 거의 안 변한다. 그래서 "CKA 높음 + frac_common 낮음"에서 회전이나 확대를 추론할 필요는 없다.
  흩어진 표본별 잡음으로 충분히 설명된다. (PR ratio가 1 근처라는 관찰과도 맞는다.)
- 측정(AID-C, 17개 백본): 이동 에너지 중 head가 읽는 부분공간(클래스 차이를 만드는 29개 방향)에 들어가는 몫은 공통 이동
  9–16%, 표본별 이동 3–15%다(무작위 기준 1.4–7.6%). **이동의 85–90%는 head가 읽지 않는 방향에서 일어난다.**
  rel_drift·cos·CKA는 이 부분까지 다 세기 때문에 정확도·ECE와 느슨하게만 연결된다.

### 7.5 feature geometry와 ECE의 관계
- head는 feature를 logit으로 바꾼다. 이동을 logit에서 보면 둘로 나뉜다.
  - 공통 이동 → 모든 이미지의 logit에 **같은 벡터**가 더해진다. head의 bias가 바뀐 것과 같다. 특정 클래스 logit이 일제히 올라가
    예측이 그 클래스로 쏠리고, 틀린 예측에도 confidence가 높다.
  - 표본별 이동 → logit에 이미지마다 다른 잡음이 더해진다. 오류가 여러 클래스로 흩어진다.
- 크기(AID-C 60조건 평균, logit 단위. clean에서 1등과 2등 logit의 차이는 5.2–8.5):
  공통 logit 이동의 클래스 간 표준편차 0.40–2.62, 표본별 logit 잡음의 표준편차 0.82–2.18.
- 백본 17개 사이에서 과신(signed ECE)과의 순위 상관: 공통 logit 이동 0.91, 표본별 logit 잡음 0.92, CKA −0.14, frac_common 0.21.
  즉 과신은 "feature 공간 전체의 모양"이 아니라 **logit이 얼마나 흔들렸는가**를 따른다.
- 두 성분은 손상이 세질수록 같이 커져서 상관만으로는 어느 쪽이 원인인지 못 가른다. 개입 실험이 가른다: 평균만 맞추면(공통 성분
  제거) 과신이 sup_in_r50 +13.2 → +5.3, mae_in +16.4 → +3.7, moco_in +18.3 → +7.9로 줄어든다. 과신의 큰 몫은 공통 성분이고, 남는
  몫이 표본별 잡음이다.
- CKA와 NEO: 샘플 사이 관계가 그대로여도 전체가 밀리면 고정된 head는 틀린다. 평균 재중심화(NEO, 우리 align m)가 듣는 이유가 이것이다.
