# The Entropy Enigma (ICML 2024) — flip으로 정확도를 추정한 선행 연구

Press, Shwartz-Ziv, LeCun, Bethge. "The Entropy Enigma: Success and Failure of Entropy Minimization", ICML 2024,
arXiv 2405.05012. 코드: github.com/oripress/EntropyEnigma.
출처: Claude가 PDF(`papers/TTA/`) 본문과 부록 A–J의 **텍스트**를 직접 읽음(2026-11-17). 그림은 보지 못했고 캡션만 읽음.

## 1. 논문이 말하는 것

**질문**: 엔트로피 최소화(EM, TENT의 손실)는 왜 처음엔 정확도를 올리고 오래 돌리면 무너지는가.

**(1) "confidence 높은 샘플이 맞으니까 더 확신시키면 된다"는 직관은 일부만 맞다 (§2, 부록 G)**
- ImageNet-C Gaussian noise 3에서, 처음에 맞게 분류된 이미지를 전부 빼고(전체의 45%, 평균 엔트로피 1.85 vs 전체 2.84) TENT를
  돌려도 holdout 정확도 이득이 +10.50pt로, 아무것도 안 뺀 경우 +12.38pt와 비슷하다. 정답이 top-10 안에 있는 이미지를 다 빼도 +7.88pt.
- 엔트로피가 낮은 샘플을 빼도 결과가 비슷하다.
- 결론: EM의 성공은 개별 샘플의 정확도·엔트로피에 크게 의존하지 않는다.

**(2) EM은 embedding을 군집화하는 과정이고, 두 단계가 있다 (§3, 부록 A)**
- 측정 두 가지: Silhouette score(k-means 군집이 얼마나 뚜렷한가), Shift distance(군집 중심과 **학습 데이터 클래스 평균 embedding**
  사이의 평균 거리, Hungarian 매칭).
- 1단계(성공): 군집이 뚜렷해지고, 학습 데이터 embedding 근처에 머문다. Silhouette 변화와 정확도 변화의 상관 ρ = 0.70.
- 2단계(실패): 군집은 계속 뚜렷해지지만 학습 데이터 embedding에서 멀어진다. Shift distance 변화와 정확도 변화의 상관 ρ = −0.79.
- 부록 A는 EM을 Expectation-Maximization 군집화로 해석한다. 초기값(사전학습 모델)이 좋으면 맞는 군집으로 가고, 오래 돌리면 test
  데이터에 과적합해 원래 위치에서 떠난다.
- 실험: ResNet-50, TENT 50,000 iteration, 16-class ImageNet, IN-C / IN-C̄ / IN-3DCC.

**(3) Weighted Flips (WF): 라벨 없이 정확도 추정 (§4)**
- 근거: 초기 Silhouette score와 정확도가 강하게 상관 → 이미 잘 군집된 데이터는 EM이 별로 바꿀 것이 없어 flip이 적다.
- 방법: RDumb(ETA 기반, BatchNorm 파라미터만 학습, SGD lr 2.5e-4, 배치 64, 1,000 iteration마다 초기화)으로 적응시키면서, 처음
  1,000장의 예측이 iteration 0과 1,000 사이에 바뀌었는지 센다.
  `WF = Σ_i 1[flip_i] · c_i` (c_i = 초기 confidence의 percentile. 확신하던 예측이 뒤집히면 더 크게 센다.)
- 변환식: `acc = f(WF)`, f는 2차식 `0.00036x² − 0.32x + 75.66`. ImageNet val과 IN-C holdout 손상의 (WF, 정확도) 쌍에 맞춤(라벨 필요).
- 추가 비용: 1,000 step당 forward 20회.

**결과 (ResNet-50, 23개 데이터셋 326 split, 추정 오차 MAE %)**

| | AC | DoC | ATC | COT | WF |
|---|---:|---:|---:|---:|---:|
| 평균 | 26.02 | 22.29 | 11.81 | 8.17 | 5.75 |
| 최악 | 99.90 | 96.34 | 99.90 | 30.38 | 21.61 |
| IN-C (75 split) | 10.06 | 6.61 | 7.44 | 2.23 | 4.79 |

- 손상(noise) 계열에서는 COT가 WF보다 낫다(IN-C 2.23 vs 4.79). WF가 앞서는 것은 domain shift·adversarial 쪽을 합친 평균이다.
- 다른 구조: ResNet-50으로 맞춘 f를 그대로 쓰면 ViT-B/16의 IN-C 오차 8.34, f를 ViT로 다시 맞추면 1.64 (부록 F).
- 측정 구간을 500 iteration으로 줄이면 오차가 상대 26.89% 줄어든다 (부록 D.1).
- TENT, RPL, CPL로 바꿔도 작동(IN-C 6.75 / 6.85 / 5.14) (부록 E).
- **CIFAR-10에서는 실패**: flip이 데이터셋당 0–5개뿐이라 MAE 16.64. CIFAR-100은 9.10 (부록 I).
- 한계 절은 따로 없다. agreement 기반 추정(Baek 2022, Kim 2023 등)은 "여러 모델과 ID 데이터가 필요해 TTA 설정 밖"이라며 비교하지 않았다.

## 2. 우리 flip과의 차이

| | Weighted Flips | 우리 flip |
|---|---|---|
| 무엇과 무엇 사이의 flip | 적응 전 모델 vs EM으로 1,000 step 학습한 모델 | 같은 frozen 모델에서 head 입력 그대로 vs source 통계로 정렬 |
| 필요한 연산 | gradient 학습(BN 파라미터) | 배치 평균·표준편차 한 번, 학습 없음 |
| 필요한 source 정보 | 없음(단, f를 맞출 라벨 있는 shift 데이터 필요) | source feature의 평균·표준편차(채널당 2개 값) |
| flip→정확도 변환 | 2차식(계수 3개), 구조가 바뀌면 다시 맞춰야 최선 | 정렬 후 ATC − 0.472 × (flip − clean flip), 계수 1개를 17개 백본에 공통 |
| 가중 | 초기 confidence percentile | 없음 |
| flip이 정확도를 따르는 이유(설명) | 정확도가 낮으면 군집이 흐릿 → EM이 많이 바꿈 | 손상이 logit에 공통 이동을 만들고, 정렬이 그것을 되돌리며 예측이 바뀜 |
| 모델 | end-to-end 학습한 ImageNet 분류기(BN ResNet 중심, ViT-B/16) | frozen EO·일반 foundation model + 선형 head, 17개 |
| 실패 조건 | 클래스가 적으면 flip이 거의 없음(CIFAR-10) | 공통 이동이 없는 shift, 작은 배치(미검증), UCM→R45에서 오차 5.5pt |
| 측정 단위 | 데이터셋 하나당 추정 1개(처음 1,000장) | 배치마다 계산 가능 |

우리 수치(2,040개 합성 stream MAE 2.4pt, R45→UCM 1.7pt)와 그쪽 5.75%는 데이터·모델이 달라 직접 비교할 수 없다.

## 3. 우리에게 주는 것

**주장할 수 없는 것**
- "예측 flip이 정확도를 따라간다"는 관찰 자체. 2024년에 이 논문이 했다.

**남는 것**
- 학습 없이, 통계 정렬 한 번으로 얻는 flip. WF는 적응 학습이 끝나야 값이 나온다.
- frozen EO foundation model에서의 검증.
- 표본 단위 필터로 쓰는 것. WF는 데이터셋 수준 추정만 한다.

**꼭 해야 하는 것**
1. **WF를 baseline으로 직접 비교.** 우리 설정에 옮기면: head 입력의 affine(마지막 norm에 해당)을 EM으로 학습하고 전후 flip을 센 뒤 같은
   방식으로 정확도를 추정. `scripts/analysis/feat_tta.py`의 틀을 쓸 수 있다. 리뷰어가 가장 먼저 물을 비교다.
2. **confidence 가중을 빌려 시험.** 우리도 "확신하던 예측이 뒤집힘"에 가중치를 줄 수 있다. 오늘 확인한 "confidence × 정렬 후 확률"의
   오류 탐지 AUROC 0.92–0.97과 같은 방향이다.
3. **COT도 baseline에 추가.** 손상 계열에서는 COT가 WF보다 좋았다.

**방법 아이디어로 이어지는 것**
- 이 논문의 실패 설명은 "embedding이 학습 데이터 위치에서 멀어진다"이고, 그 거리를 재려면 학습 데이터 클래스 평균이 필요해서
  실용적이지 않다고 스스로 적었다(§4). 우리는 source 통계를 쓴다는 가정이라 **그 거리를 실제로 잴 수 있다.**
  → "source 통계에서 멀어지면 적응을 멈추거나 초기화"하는 규칙. RDumb의 "1,000 step마다 무조건 초기화"를 대체하는 형태.
- §2의 결과(확신 샘플을 빼도 EM이 작동)는 "엔트로피로 샘플을 고르는 것이 핵심이 아니다"라는 뜻이다. confidence 필터의 근거를
  약하게 하지만, 동시에 "어떤 필터든 큰 차이가 없을 수 있다"는 뜻이기도 하다. flip 필터 TTA 결과를 읽을 때 같이 봐야 한다.
  다만 그쪽은 BN이 있는 end-to-end 모델이고 과신으로 예측이 쏠리는 상황(우리 mae_in, ResNet)을 다루지 않았다.

## 4. 사용자 노트와의 연결
- "frac_common이 이 논문 그림과 비슷한가": 같은 것은 아니다. Figure 1은 **EM을 오래 돌렸을 때** embedding이 학습 데이터 위치에서
  떠나는 모습이고, frac_common은 **손상이** feature를 한 방향으로 미는 정도다. 공통점은 "source embedding 위치에서의 거리가
  정확도를 좌우한다"는 관점이다.
