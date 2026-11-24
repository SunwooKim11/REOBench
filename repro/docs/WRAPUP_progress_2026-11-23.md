# 진행 정리 (2026-11-23 기준): 10월까지 한 것, 그 뒤에 한 것

수치의 출처는 각 절 끝에 적었다. 10월은 `docs/NOTES_tta.md`, 11월은 `docs/feedback/2026-11-16_direction.md`(경과 로그)가 원본이다.
한 줄 요약: 10월에는 "기존 TTA가 RS에서 왜 안 듣는가"를 확인했고, 11월에는 "confidence 대신 쓸 라벨 없는 신호"로 방향을 바꿔
17개 백본에서 검증했다.

## 1. 10월: 재현 → 기존 TTA 적용 → 통계 정렬

### 1.1 REOBench 재현 (10월 12~16일)
- AID 분류: 공개 코드로 재현하면 SatMAE 90.4%, ScaleMAE 91.3%가 나온다(논문 72.1 / 75.8). 코드의 SatMAE 체크포인트가 Sentinel용이라는
  것 등을 찾아 저자에게 issue로 알렸고, 저자는 "당신 결과를 baseline으로 쓰라"고 답했다.
- 입력 정규화를 추가하면(P → PN) 손상 평균이 약 +7pt 오른다. 뒤에 본 어떤 TTA보다 큰 효과다.
- Potsdam 분할 baseline: SatMAE 61.9, ScaleMAE 65.0 mIoU.

### 1.2 TTA 구현과 검증
- Tent, SAR, DeYO, EATA, CoTTA, ViDA, LSCD를 공식 코드에서 옮겼다.
- ImageNet-C에서 논문 수치와 대조했다(예: DeYO Table 1은 0.35pt 이내). 분할은 Cityscapes → ACDC에서 대조했다.

### 1.3 AID에서 기존 TTA 결과 (SatMAE, ScaleMAE)
- 기존 TTA의 효과는 ±1~2pt이고, 설정에 따라 붕괴한다. CoTTA는 모든 설정에서, ScaleMAE P에서는 SAR·DeYO·Tent가 붕괴했다.
- 꾸준히 이득인 것은 학습 없는 입력 통계 정렬(ISA)뿐이었다. radiometric 손상(brightness, haze)에서 +10~29pt.

### 1.4 왜 안 듣는가
- 손상에서는 엔트로피를 믿을 수 없다. 정확도 15~60%인 손상에서도 91~99%의 이미지가 "확신"으로 분류되어 필터가 거르지 못한다.
- 예측이 한 클래스로 쏠린다. clean에서는 최다 예측 클래스가 4%인데, SatMAE motion blur에서는 60%.
- DeYO의 패치 섞기 신호(PLPD)는 질감형 RS 클래스에서 약하다.

### 1.5 정렬을 여러 깊이로 확장 (10월 20~28일)
- 같은 식을 입력, 블록, head 입력에 넣어 봤다(depth sweep). radiometric은 입력에서, degradation(blur, noise)은 중간 블록(7~11)에서
  맞추는 것이 가장 좋았다. 분류와 분할, 두 백본 모두 같았다.
- 분할에서는 Tent + test 배치 BN이 +3.8 mIoU인데, BN 통계만 바꿔도 +3.1이다. 이득은 엔트로피가 아니라 통계에서 온다.
- src 통계는 라벨 없는 clean 32장이면 충분했다.

## 2. 11월 초: 발표 피드백과 방향 전환

LSCD-TTA 발표 뒤 교수님 피드백(`docs/feedback/2026-11-W1_LSCD-TTA.md`):
- RS corruption의 고유성을 문제 제기로 쌓을 것 (F2)
- 벤치마크를 2개 이상으로, REOBench 계열 논문을 더 읽을 것 (F3, 완료)
- fMoW로 사전학습한 ResNet·ViT를 찾고 head 학습 방식을 확인할 것 (F4, 표로 정리할 것이 남음)
- 백본 선택의 근거를 댈 것 (F5, 남음)

연구 질문:
> shift 아래에서 confidence를 믿을 수 없다면, confidence 대신 쓸 라벨 없는 신호를 얻을 수 있는가?

후보는 source 통계와의 거리(정렬)와, 정렬 전후 예측이 바뀌는지(flip)다.

## 3. 11월 16일 이후: 17개 백본에서 신호와 TTA 검증

### 3.1 실험 기반
- 백본 17개: 같은 구조·같은 사전학습 방식의 ImageNet 모델과 위성 모델을 쌍으로 구성(MAE ↔ SatMAE·ScaleMAE, MoCo ↔ MoCo-fMoW·GASSL,
  DINO ↔ SatDINO, DINOv3 web ↔ sat). 지도학습 ImageNet 3개는 기준점. 상세는 `docs/SETTINGS.md`.
- 데이터: AID-C, RESISC45-C(우리가 생성), 자연 shift RESISC45 ↔ UC Merced.
- feature를 저장해 두고 선형 head를 학습했다(stage 1).

### 3.2 신호 분석 (합성 손상)
- confidence로 정확도를 추정하면(ATC) motion blur, data gaps, scale에서 크게 과신한다. 10pt 넘게 틀리는 경우가 12.8%인데,
  head 입력을 정렬하면 1.4%로 줄어든다.
- flip 비율은 17개 백본 모두에서 정확도를 따라간다(순위 상관 0.90~1.00).
- 과신의 원인은 구조가 아니라 "특정 손상에서 feature가 집단으로 한 클래스 쪽으로 이동하는 것"이었다. "feature 크기가 커져서"라는 가설은 기각.
- 과신 백본에서 SAR의 엔트로피 필터는 거의 못 거른다(남긴 표본 오류율 23~27%). flip 필터는 7~13%.
- shift 방향을 지우는 pruning은 근거가 없었다(oracle 실험). SAE 아이디어는 접었다.

### 3.3 TTA 결과 (17개 백본 평균 손상 정확도 %, 백본 전체 통과, 기본 lr)

| 방법 | AID-C | RESISC45-C |
|---|---:|---:|
| none | 81.8 | 75.2 |
| tent | 81.4 | 71.2 |
| 엔트로피 필터 | 81.5 | 71.8 |
| sar | 80.6 | 71.8 |
| eata | 81.9 | 71.1 |
| dsbr | 83.6 | 77.7 |
| flip 필터 | 84.3 | 79.1 |
| 정렬만 (학습 없음) | 87.3 | 81.1 |
| flip 필터 + 정렬 | 87.9 | 82.4 |
| tent + 정렬 | 88.1 | 83.0 |
| dsbr + 정렬 | 88.1 | 83.1 |

- 엔트로피 기반 방법(Tent, 엔트로피 필터, SAR, EATA)은 평균적으로 손해이거나 제자리이고, 과신 백본에서 붕괴한다.
- confidence를 쓰지 않는 flip 필터와 DSBR은 이득이다.
- 학습 없는 정렬이 그보다 크고, 정렬 위에 Tent를 얹은 것이 가장 높다.
- lr을 10배로 올리면(AID-C) tent 76.1, flip 필터 83.8, dsbr 80.4, tent + 정렬 88.6, dsbr + 정렬 88.8. DSBR도 moco_in에서는 붕괴한다(23.1).
- EATA는 처음에 5개 백본에서 붕괴했다. 원인(분산이 거의 0인 채널의 Fisher penalty 발산)을 찾아 고쳤다.

출처: `experiments/2026-11-17_fm_tta/RESULTS.md`, `experiments/2026-11-19_dsbr/RESULTS.md`.

### 3.4 검증과 통제
- **Weighted Flips 재현** (ImageNet-C): 비교 기준 3개가 논문과 소수 둘째 자리까지 일치. WF 자체는 0.7pt 높다.
- **DSBR 구현 확인** (ImageNet-C, ViT-B/16): 무적응 평균이 논문과 같은 39.8인 설정(timm augreg2 + ImageNet mean/std)에서
  Tent 49.7, DSBR α 0.9 61.2, α 0.99 61.8 (논문 51.9 / 64.4 / 64.9). 이득 크기는 비슷하고(DSBR − Tent +11.5, 논문 +12.5) 절대값은 2~3pt 낮다.
  평균이 같은 다른 모델(EATA-C의 source)이 있어 논문의 체크포인트는 확정하지 못했다.
- **예측 쏠림 측정**: 쏠림의 약 2/3가 정렬로 사라진다. 쏠림이 클수록 Tent는 더 잃고 DSBR은 더 얻는다. stream의 절반은 쏠림이 작아 어떤 방법도 차이가 없다.
- **head 학습 방식 통제** (11-23): 표준화 없이 학습한 head에서도 정렬 이득이 같다(AID-C +5.4 대 +5.5, RESISC45-C +5.8 대 +5.9).
  단, 쓰는 식은 head에 따라 다르다. 규칙은 "head 입력의 채널별 통계를 head가 학습 때 본 입력의 통계로 맞춘다"이고,
  표준화한 입력으로 학습한 head는 (h − μ_test) / σ_test, 그대로 학습한 head는 여기에 · σ_src + μ_src를 붙인다.
- **선행 연구의 linear probe 코드** (11-23, 공식 repo 확인): MAE, Scale-MAE는 head 앞에 affine 없는 BatchNorm(= 우리 표준화와 같은 계산)을 두고,
  DINO, DINOv2, DINOv3, MoCo v3, SatDINO, CLIP 예시는 두지 않는다. 구조가 아니라 사전학습 방식에 따른 관행 차이다.

출처: `experiments/2026-11-19_weighted_flips_repro/`, `2026-11-19_dsbr/`, `2026-11-23_head_nostd/`, `docs/STUDY_alignment.md` §2.4.

### 3.5 자연 shift 해석 (RESISC45 ↔ UC Merced, 11-23)

| 17개 백본 평균 (target 전체 이미지) | R45 → UCM | UCM → R45 | 합성 손상 (AID-C / R45-C) |
|---|---:|---:|---:|
| 정확도 하락 (pt) | 15.3 | 30.9 | 13.0 / 15.4 |
| signed ECE (평균 confidence − 정확도, 0~1 단위) | +0.072 | +0.145 | |
| 정확도 추정의 평균 절대 오차: 평균 confidence (pt) | 7.5 | 14.5 | |
| 정확도 추정의 평균 절대 오차: ATC (pt) | 6.0 | 7.6 | |
| 정확도 추정의 평균 절대 오차: flip 추정 (pt) | 1.7 | 5.2 | |
| flip과 정확도의 순위 상관 (백본 17개) | −0.99 | −0.91 | |
| confidence와 정확도의 순위 상관 (백본 17개) | +0.89 | +0.81 | |
| 정렬 이득 (pt) | +2.2 | +2.1 | +5.5 / +5.9 |
| 정렬 뒤 남는 하락 (pt) | 13.1 | 28.8 | |

- 신호 쪽 결론(confidence는 과신, flip은 정확도를 따름)은 자연 shift에서도 선다. 과신은 일부 백본에 몰려 있다(ScaleMAE +0.295, SatMAE +0.188, DINOv3 sat +0.164).
- 정렬 이득은 합성의 절반 이하다. 자연 shift에서는 이동 중 "공통 평균 이동"의 몫이 3~27%로 작아서, 통계 정렬이 되돌릴 부분이 적다(해석).
  → 정렬은 공통 이동이 큰 shift용 처방으로 범위를 좁혀 말해야 하고, 자연 shift에서는 신호를 만드는 도구로서의 근거가 더 강하다.
- 위성 사전학습의 효과는 방식마다 반대다(원인 미확인):

| 쌍 (위성 − 일반, 정확도 차이 pt) | R45 → UCM | UCM → R45 |
|---|---:|---:|
| SatDINO − DINO (S/16, S/8, B/16) | +3.8, +1.1, +3.8 | +8.0, +6.2, +9.4 |
| GASSL − MoCo-ImageNet | +8.0 | +13.5 |
| MoCo-fMoW − MoCo-ImageNet | +2.9 | −1.7 |
| SatMAE − MAE-ImageNet | −14.1 | −0.9 |
| ScaleMAE − MAE-ImageNet | −18.4 | −3.4 |
| DINOv3 sat − DINOv3 web | −29.8 | −28.2 |

  데이터 양으로는 설명되지 않는다(DINOv3 sat의 SAT-493M은 4.9억 장으로 fMoW 36만 장보다 훨씬 많다). MoCo 쪽 이득은 위성 데이터가 아니라
  GASSL의 방법(같은 장소의 다른 시점을 positive로, 위치 예측)에서 온다(같은 방법의 MoCo-fMoW는 MoCo-ImageNet과 비슷).
- 한계: 데이터 쌍 1개, 19클래스, seed 1개. 상관은 백본 17개를 표본으로 한 순위 상관이다. 두 데이터셋의 클래스 정의와 GSD 차이가 섞여 있다.

출처: `experiments/2026-11-17_natural_shift/RESULTS.md`.

## 4. 공부한 것
- 논문 노트(`docs/reading/`): ImageNet-C, Beyond Accuracy, EarthShift, RSPDBench, REOBench 리뷰, matched-pair 연구(arXiv 2611.13849),
  Entropy Enigma(Weighted Flips), GOLD, DSBR. 이 중 Entropy Enigma, GOLD, matched-pair 연구는 Claude 요약만 있고 직접 읽지 않았다.
- 선행연구 조사(`docs/PRIOR_ART_*`): 통계 정렬, feature 신호, CLIP 분할 TTA, SAE pruning.
- 개념 노트(`docs/STUDY_*`): stage 1 지표(drift, frac_common, CKA, ECE, flip), 정렬의 원리(LayerNorm과의 축 차이, 1·2단계, AdaBN, raw / std head별 식).
- 읽을 것: Entropy Enigma(발표 후보, DSBR과 한 묶음), 백본 방식 3개(MoCo, DINO, MAE)의 방법 절, matched-pair 연구(우리 쌍 설계의 근거).

## 5. 지금 서 있는 결론과 빈 곳

결론 (합성 손상, 분류, 배치 64, stream마다 초기화 기준):
1. RS 기초모델은 shift 아래에서 confidence가 틀리고, 합성 손상에서 그 원인은 feature의 집단 이동과 예측 쏠림이다.
2. 합성 손상에서는 그 이동의 큰 부분을 학습 없는 통계 정렬로 되돌릴 수 있다. 자연 shift에서는 일부만 된다.
3. 정렬 전후 flip은 confidence와 다른 정보를 주는 라벨 없는 신호다. 정확도 추정과 표본 필터에 쓸 수 있고, 자연 shift에서도 정확도를 따른다.
4. 엔트로피 최소화는 정렬 위에서만 안전하다.

빈 곳:
- 우리 flip을 Weighted Flips와 같은 조건에서 비교하지 않았다.
- 혼합·연속 stream과 작은 배치 조건이 약하다. 배치 16에서는 정렬 이득이 줄고 clean 비용이 생긴다.
- 11월 방법을 분할에 적용하지 않았다(Potsdam-C, Potsdam → Vaihingen). → 11-23부터 진행.
- 자연 shift는 데이터 쌍 하나뿐이고, 정렬로 되돌리지 못하는 부분에 대한 처방이 없다.
- 남은 하락이 큰 손상(occlusion, scale)에 대한 처방이 없다.
- 교수님 요구 F4, F5(백본별 사전학습·head 학습 방식·선택 근거 표)가 남았다.
- seed는 대부분 1개다.
