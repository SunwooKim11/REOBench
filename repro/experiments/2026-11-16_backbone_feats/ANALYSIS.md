# Stage 1 분석 (2026-11-17)

수치 출처: `RESULTS.md`(전체 표), `signals/summary.txt`(label-free 신호 평가).
범위: 백본 17개 × AID-C / RESISC45-C(12종 × 5단계 = 60개 stream), frozen backbone + 선형 head, seed 0.

## 한 줄 요약

confidence는 무작위로 틀리는 게 아니라 **특정 손상 3종과 특정 백본 계열에서 한 방향(과신)으로** 틀린다.
head 입력을 source 통계로 맞추면(라벨 불필요) 정확도 일부가 돌아올 뿐 아니라 **confidence가 다시 정확도를 따라간다.**

## 1. confidence는 어디서 틀리는가

confidence로 정확도를 추정하는 표준 방법 ATC(임계값 이상 confidence 비율)를 60개 stream에 적용한 오차.

| | 평균 오차 (pt) | 10pt 넘게 틀린 stream | 최대 오차 |
|---|---:|---:|---:|
| ATC (원래 head) | 5.2 | 12.8% | 84.4 |
| ATC (source 통계로 정렬한 head) | 1.6 | 1.4% | 18.6 |

(백본 17 × 데이터셋 2 × 60 = 2,040개 stream 전체. 중앙값은 1.1 → 0.7.)

- **중앙값은 1.1pt로 작다.** 대부분의 stream에서 confidence는 쓸 만하다. 문제는 꼬리다.
- **틀리는 손상은 3종에 몰려 있다.** 손상별 ATC 평균 오차(AID / RESISC45):
  motion blur 18.1 / 19.0, data gaps 12.2 / 16.2, scale 11.7 / 9.0. 나머지 9종은 0.6–4.6.
  세 손상 모두 방향이 같다: 정확도를 실제보다 높게 추정(과신).
- **틀리는 백본도 몰려 있다.** ATC 평균 오차(AID / RESISC45):
  mae_in 12.5 / 12.6, moco_in 13.6 / 16.0, moco_fmow 11.0 / 10.8, sup_in_r50 8.8 / 10.6, scalemae 4.0 / 10.7.
  DINO·SatDINO·supervised ViT는 0.8–3.9, dinov3_web은 0.5 / 0.9.
- **ResNet은 순위마저 틀린다.** RESISC45에서 confidence와 정확도의 순위 상관이 moco_in 0.54, sup_in_r50 0.63, moco_fmow 0.64
  (ViT 계열은 0.91–1.00).

## 2. source 통계 신호는 그 자리를 메우는가

- **정렬 후 ATC는 모든 백본에서 평균 오차 0.6–3.7pt.** 가장 나빴던 moco_in이 13.6 → 0.8(AID), 16.0 → 1.6(RESISC45).
  세 손상의 오차도 motion blur 18.1 → 4.8, gaps 12.2 → 3.1, scale 11.7 → 2.1(AID).
- **flip 비율이 정확도를 따라간다.** flip = 정렬 전후로 예측이 바뀐 샘플 비율(라벨 불필요).
  정확도와의 순위 상관이 모든 백본·두 데이터셋에서 0.90–1.00. confidence가 0.54였던 moco_in에서도 0.99.
- **flip은 정확도 단위로 바꿀 수 있다.** 정렬로 오른 정확도 ≈ 0.47 × (flip − clean에서의 flip), 상관 0.94.
  이 기울기를 다른 백본에서만 추정해 "정렬 후 ATC − 0.47 × flip"으로 원래 head의 정확도를 추정하면
  평균 오차 2.4pt, 10pt 초과 4.6%(원래 ATC는 5.2pt, 12.8%).
- **평균 이동 거리(‖stream 평균 − source 평균‖²)도 정확도 하락 순위를 따라간다**(0.73–0.97). 다만 단위가 백본마다 달라
  그대로는 정확도 추정에 못 쓴다. flip이 더 낫다.

## 3. 정렬은 정확도를 얼마나 되돌리는가

- 손상으로 잃은 정확도의 **35–58%**를 되돌린다(손상별, 17개 백본 평균). rotate만 0%, JPEG가 21–35%로 낮다.
- 평균만 맞추는 것과 평균·표준편차를 맞추는 것의 차이는 ViT에서 0.4pt 이내, ResNet에서 1.1–5.2pt(MoCo-ImageNet이 최대).
- clean에서 정렬하는 비용: 예측이 0.8–5.7% 바뀌고 정확도 변화는 +0.2 ~ −1.0pt.
- **frac_common은 "얼마나 고쳐지는가"의 예측 변수로는 약하다.** 손상별 회복 비율이 frac_common과 잘 안 맞는다
  (motion blur는 frac_common 0.42로 가장 높지만 회복 38%, haze는 0.31인데 58%).
  "공통 이동의 크기"(rel_drift × frac_common)는 회복 pt와 순위 상관 0.82–0.98이지만, 큰 하락에서 큰 회복이 나오는 효과가 섞여 있다.

## 4. 사전학습 데이터 쌍 (ImageNet → RS)

RS 사전학습의 이득은 작고 일관되지 않으며, 정렬하면 대부분 사라진다. (값 = RS − ImageNet, pt)

| 쌍 | AID clean | AID 손상 | AID 정렬 후 | R45 clean | R45 손상 | R45 정렬 후 |
|---|---:|---:|---:|---:|---:|---:|
| MAE → SatMAE | −1.6 | +1.9 | −6.4 | −5.2 | −0.4 | −8.0 |
| MoCo → MoCo-fMoW | −1.3 | +3.1 | −1.8 | −0.7 | +3.7 | −0.9 |
| DINO-S16 → SatDINO | +1.1 | +4.2 | +2.4 | +1.7 | +0.8 | +0.1 |
| DINO-S8 → SatDINO | +1.8 | +2.8 | +1.2 | +1.7 | −1.1 | −0.2 |
| DINO-B16 → SatDINO | +0.2 | +3.9 | +1.2 | +1.4 | +2.7 | +1.3 |
| DINOv3 web → sat | −2.3 | −14.3 | −8.6 | −4.3 | −12.7 | −9.6 |

- 손상 정확도에서 RS 쪽이 앞서는 쌍(MAE, MoCo)은 정렬 후 뒤집힌다. ImageNet 쪽의 손해가 "고쳐지는 공통 이동"이었다는 뜻
  (mae_in의 frac_common 0.49 vs satmae 0.12).
- DINOv3는 web이 sat보다 일관되게 강하다. RSPDBench도 DINOv3-Sat이 해상도 손실에 가장 약하다고 보고했다.
  단, dinov3_sat은 kNN 점검에서도 낮았으므로(52.7) feature 선택의 영향 가능성이 남아 있다.
- 과신은 사전학습 데이터보다 **방법·구조**를 따른다: MAE-ImageNet, MoCo, supervised ResNet이 크고 DINO·supervised ViT가 작다.

## 5. 같은 손상을 자연 이미지와 RS 이미지에 (b)

- **픽셀 수준에서 거의 같다.** MS-SSIM(5단계 평균, ImageNet / AID / RESISC45):
  gauss noise 0.755 / 0.710 / 0.716, gauss blur 0.849 / 0.834 / 0.831, JPEG 0.932 / 0.932 / 0.931,
  contrast 0.511 / 0.570 / 0.558.
- **정확도 유지율은 blur·JPEG·brightness가 같고, noise와 contrast만 RS가 낮다**(supervised 3개: noise 62 vs 48, contrast 69 vs 61).
  다만 ImageNet은 end-to-end로 학습한 자체 head, RS는 같은 feature 위의 선형 head라 head 조건이 다르다.
- 결론: 이 7종에서는 "RS 이미지라서 다르게 작용한다"는 증거가 약하다. RS 고유성은 이미지 도메인이 아니라
  **손상 종류**에서 찾아야 한다. 실제로 confidence가 무너지는 3종 중 2종(data gaps, scale)이 RS 고유 유형이다.

## 6. 실제 haze (RRSHID)

- 실제 haze는 채널별 선형 변환이 아니다(affine R² 0.35–0.43, 합성은 1.0).
- 평균·표준편차 비율이 비슷한 합성 haze보다 훨씬 파괴적이다. clear와 예측이 일치하는 비율(AID head, 백본 17개 범위)이
  실제 moderate 1.6–28.2%, 합성 i=0.4는 39–84%.
- feature 정렬이 픽셀 정렬보다 대체로 많이 되돌린다(moderate, mae_in: 6.9 → 픽셀 19.6 → feature 49.3).
  픽셀 정렬이 더 나은 백본은 17개 중 thin 3개, moderate 1개, thick 2개.
- 라벨이 없어 일치율만 잰 것이고, AID·RESISC45 head를 그 클래스가 아닌 이미지에 적용한 값이다.

## 7. 분할 (Potsdam)

- 두 모델 모두 JPEG에서 과신(signed ECE +0.11). gauss noise는 ScaleMAE만 과신(+0.09), SatMAE는 아님(−0.01).
- 분류 conv 입력에서 frac_common이 0.02–0.21로 분류(0.11–0.49)보다 낮다. 픽셀 단위 feature에서는 공통 이동의 비중이 작다.

## 8. 한계와 확인할 것

1. **정렬 조건이 쉽다.** 한 종류의 손상만 있는 stream을 섞어서 64장씩, 클래스가 고르게 들어온다. 손상이 섞이거나, 배치가 작거나,
   클래스가 치우치면 batch 통계가 흔들린다. 아직 재지 않았다.
2. **합성 손상뿐이다.** 자연 shift(RESISC45↔UCM)에서 같은 현상이 나오는지가 결정적 검증이다.
3. **ATC 임계값은 라벨 있는 clean test로 정했다**(표준 방식은 source validation). seed 1개, head 1개.
4. **선행 연구 확인이 필요하다.** 예측 불일치로 TTA 모델의 정확도를 추정하는 연구(AETTA 등), ATC·DoC·agreement 계열이 이미 있다.
   우리가 새로 말할 수 있는 것은 "RS foundation model에서, 어떤 손상·백본에서 confidence가 무너지고, source 통계 정렬이 그것을
   되돌린다"는 관찰과 그 분해 설명이다. 방법 자체의 새로움은 문헌 확인 전에는 주장하지 말 것.
5. **이득은 3종 손상에 몰려 있다.** 나머지 9종은 원래 ATC도 1–5pt로 맞는다. supervised ViT에서는 정렬 후 ATC가 약 2pt 과소추정한다.
6. **MoCo ResNet의 rel_drift는 꼬리가 무겁다.** blur에서 일부 채널이 표준화 값 수백까지 튄다(버그 아님, clamp하면 정확도 +1–2pt).
   이 백본의 rel_drift·frac_common 평균은 그 채널들이 좌우한다.

## 9. 다음 실험 제안

1. RESISC45↔UCM 자연 shift에서 §1–2 재현(정렬 후 ATC 오차, flip과 정확도의 관계). 기존 feature 파이프라인 그대로.
2. 정렬 조건 흔들기: 배치 크기 8 / 16 / 32 / 64, 손상 혼합 stream, 클래스 치우침.
3. 선행 연구 조사: label-free accuracy estimation × TTA.
4. 분할에서 같은 분석(픽셀 confidence, 정렬 위치).
