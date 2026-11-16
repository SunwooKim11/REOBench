# Beyond Accuracy: Assessing Calibration of Geospatial Foundation Models and Their Sensitivity to Distribution Shifts

- arXiv 2608 (Xiao Xiang Zhu 그룹; "No one knows the SOTA of EO FM" 저자들). PDF: `~/REOBench/papers/Beyond Accuracy_ ….pdf`
- 메모 작성: 사용자 (2026-11-16 전달). 아래 "사용자 메모"는 원문 그대로, "Claude 보충"은 메모 속 질문에 대한 원문 근거.

---

## 사용자 메모 (원문)

# 내 요약

- under Covariate shift: physically motivated EO distribution shifts and across training-data budgets
    - GeoFM calibration이 얼마나 sensitive 한지 → trustworthy behavior를 측정한다→ confidence으 초점.
- centered kernel alignment (CKA) representational analysis
- 보통의 calibration methods이 shift하에서 calibration문제를 해결 못함
    - an evaluation of post-hoc calibration, ensembles, and a Gaussian-process probe.
    - selective-prediction이 확신 기반 기피가 확신에 찬 잘못된 예측을 회피할 수 없음을 보임. 특히 구름에서
- → 그리고 표준화 된거를 지향? GeoBench

# Introduction

- EO FM 이 주목받은 이유: motivated by the scale of available geospatial data and the potential for improved label efficiency and downstream performance at lower fine-tuning cost [6]
- Research Question: mainly calibration, the agreement between a model’s confidence and its correctness → GeoFM calibration이 얼마나 sensitive 한지 → trustworthy behavior를 측정한다.
    - Accuracy on clean benchmark conditions is also blind to what deployment conditions a model encounters, since safety-critical decision making [8] and natural distribution
    shifts of geography, sensors, and atmospheric conditions all change the input distribution at test time
    - Distribution shift - semantic, covariate, concept shift → 이중 covariate shift 타겟
    - shift도 far(다른 센서 및 지역), near(같은 장면과 센서이나 습기 또는 센서 저하) 로 나뉨
        - → near shift 타겟
        - two orthogonal axes; under physically motivated EO distribution shifts and across training-data budgets,
    - Needs: Calibration metrics are well established in the general machine learning (ML) literature but have barely been applied in the geospatial domain
- Motivation(Practice)
    - we aim to establish a more holistic view of GeoFM performance that includes how well these models are calibrated and how this perspective can  further support real-world application cases, where predictive uncertainty and model prediction confidence matter.
- 분석방법 및 결과
    - use a centered kernel alignment (CKA) representational analysis to assess feature sensitivity,
        - a representation-drift analysis that ties calibration failure to a domain-conditioned pattern
        - clean vs corrupted의 head-layer CKA가 EO 0.82, 일반 0.61입니다.
        - 즉 EO embedding은 corruption에 덜 움직이는데 task 정보는 똑같이 잃습니다. 저자들은 이를 “representational rigidity”라고 부릅니다
    - and furthermore show that common calibration methods do not fully resolve these issues.
        - an evaluation of post-hoc calibration, ensembles, and a Gaussian-process probe, none of which fully restores calibration under shift.
    - Furthermore, a selective-prediction evaluation shows that confidence-based abstention
    cannot defer around confidently wrong predictions: at the high- est cloud severity, coverage is near total and mean accuracy falls to 0.15, yet models remain overconfident.

# Related work

## Geo FM의 평가 현실

- 해당 논문은 ‘No one knows the SOTA of EO FM’의 저자. → evaluation이 재현 가능하지도, 표준화 되어있지도 않음.
    - The gap we identify, however, extends further than reproducibility. We systematically
    audited this same corpus, as well as available benchmark frameworks, for evidence of calibration, predictive uncertainty, and robustness evaluation.
    - every “calibration” mention refers to radiometric sensor calibration

## Reliable Geo FM

1. **Calibration:** **전체적으로(empirical)** confidence와 prediction accruacy가 잘 맞는지→**marginal**
    1. 정답일 확률 80%”라고 했다면, 그 예측들 가운데 실제로 약 80%가 맞아야 잘 보정된 모델.
    2. Calibration(보정)은 확률 구간별 평균을 맞추는 **주변적 성질**입니다. 그래서 모든 입력에 같은 기본 비율(uniformative)을 내는, 정보가 거의 없는 모델도 전체적으로는 보정되어 보일 수 있습니다.
2. **Predictive uncertainty**:  **어떤 개별 입력**에서 틀릴 가능성이 높은지 → **Conditional**
    1. Calibration과 구분하기→ 평균적으로 확률이 잘 맞더라도, 모델이 **바로 어떤 사례에서 틀릴지** 구별하지 못할 수 있다.
    2. Predictive uncertainty는 입력별로 “이 사례는 어렵다”를 식별하는 **조건부 성질**입니다. 평균적인 확률이 맞는지만으로는 이를 보장할 수 없다.
3. **Robustness(강건성):** 입력 조건이 달라져도 앞의 능력들이 유지되는가

# Method

## A. problem Setting

- linear probing → 분류, 분할 둘다. 분할은 왜? calibration 때문에?
    - FPN- style decoder head [46] gϕ takes the backbone’s feature maps
    at L = 4 coarse-to-fine depths and produces per-pixel class logits gϕ({zl}L
    l=1) ∈ RK×H×W over the H × W spatial grid of the input image
    - ~~→ FPN-style decoder 를 linear probe 라고 하네 신기하다.    →~~ but replace the linear
    probe with a lightweight FPN decoder 라고 하네

## B. dataset

- REObench(AID, Potsdam)와 다른 데이터셋 → GeoFM eval의 standard를 고려하기 위함이였나? GEO-bench 1, 2, PANGEA에서만 기반한?
    - Classification: ADVANCE, m-EuroSAT, RESISC45, and So2Sat spanning diverse sensors, spatial resolutions, and label spaces
    - Segmentation: CloudSEN12 [47], DynamicEarthNet [48], FLAIR#2 [49], Fields of the World [50], and SpaceNet2 [51] covering cloud/shadow masking, land cover, agricultural parcels, and building footprints across Sentinel-2, aerial, and high-resolution commercial sensors.

## C. Encoders

- GeoFMs pretrained on 위성 항공 이미지 and general-purpose enocers pretained on ImageNet → in-domain과 non-domain  enocder의 calibration 능력을 비교하기 위해
- reconstruction(MIM), self-distillation(DINO),hybrid 로 pretraining-objectives로도 encoder 갈랐고
- 입력 설정도 비교.
    - 모든 영상은 $224 \times 224$ 크기로 맞추고 채널별 정규화를 적용합니다.
    - RGB 데이터셋에서는 모든 인코더가 같은 RGB 입력을 받습니다. 다중분광 데이터셋에서는 RGB와 센서의 원래 분광 밴드 설정을 모두 시험한 뒤, 정확도가 더 높은 설정을 후속 보정 평가에 사용합니다.
    - 따라서 이 비교는 완전히 동일한 입력 조건만을 보는 것은 아닙니다. 저자들은 RGB만 쓰는 데이터셋에서도 GeoFM의 분포 변화 아래 보정 격차가 나타나므로, 그 결과가 밴드 선택만으로 설명되지는 않는다고 덧붙입니다.

## F. distribituion shift conditions(중요, 여기 있는 주장을 가져가는게 현명할듯)

- **stress-tested:**
    - 분류: all three corruption types below (cloud/shadow, Poisson–Gaussian sensor noise, motion blur)
    - 분할: Poisson–Gaussian sensor noise and motion blur only.  CloudSEN12 already contains real cloud as its labeled target,
- 왜 해당 3개의 Corruption을 골랐는지, 최대한 unrelated 한 corruption을 골라서, calibration effect가 하나의 corruption에 specific 한지 아니면 broad 한지 보기 위함.
    - The three corruptions are chosen to span distinct physical degradation mechanisms rather than variations of one: scene-level occlusion (cloud and shadow), optical smearing from platform motion (motion blur), and detector shot and read noise (Poisson–Gaussian). Real EO degradation is of course broader, but sampling three mechanistically unrelated corruptions lets us ask whether a calibrationeffect is specific to one corruption or common to all of them; an effect that appears under every one cannot be an artifact of any single degradation type.
- implementation detail - test, tta 상황에 맞게
    - All corruptions are applied in raw pixel space before normalization and seeded deterministically by (base seed, image index) for reproducibility across UQ methods and severity levels. Corrupting before normalization is likewise deliberate: at deployment only the training-set statistics are available, so the normalization mismatch a shifted input induces is itself part of the shift a model sees in practice.
- 방어하는 문장. 물리적으로 motivated 되었지만 합성이다.
    - These corruptions are physically motivated but synthetic; they approximate rather than replace real EO shift (Section II). Fig. 13 in the appendix shows example images at each severity level for all three corruption types.
- 각 세가지 corrpution을 어떤 식으로 구현했고, 어디 실제 degradation에서 motivated 받은건지?
    - a) Cloud and shadow: Cloud masks are synthesized using the satellite-cloud-generator library [68],  producing Perlin-noise-based patterns with soft edges and coregistered shadow offset
    - b) Poisson–Gaussian sensor noise: We model CCD/CMOS detector degradation as [69]
    - c) Motion blur: Satellite platform motion during acquisition causes spatial smearing in the flight direction [70], [71].

## J. Representational Drift Analysis

The three metrics are therefore complementary rather than redundant: CKA is a  structural, rotation-invariant comparison; cosine similarity is a metric that directly reflects a given sample’s own movement; and participation ratio catches collapse onto fewer effective dimension

- **추가로 확인하는 관계**
    - 마지막 층에서 샘플별 표현 변화량과 예측 confidence, 그리고 정답 여부의 관계를 계산합니다.
    - 특히 표현은 크게 달라졌는데도 confidence가 높고 예측이 틀린 샘플의 비율을 셉니다. 이는 모델이 입력 변화에 흔들리면서도 자기 예측을 확신하는 사례를 직접 포착하려는 분석입니다.

# Results

!image.png

!image.png

- **Figure 11. SVGP는 깨끗한 데이터에서의 보정 성능을 희생하는 대신, 심한 구름 오염에서 과신과 NLL 증가를 완화**하는 모습입니다. 따라서 “항상 가장 좋은 방법”이라기보다는, 깨끗한 조건과 심한 이동 조건 사이의 **절충**으로 이해하는 것이 적절합니다.

# Questions

- Selective predictions:  모델이 모든 입력에 반드시 답하지 않고, 자신이 불확실하다고 판단한 예측은 보류해 사람의 검토나 추가 분석으로 넘기는 방식
    - metrics
        - **90% coverage에서의 selective accuracy**
        - **eAURC (excess Area Under the Risk–Coverage Curve)**
- Centered Kernel Alignment (CKA): 샘플 $i$와 $j$ 사이의 표현상 유사성이  activation-level 에서 corruption 전후에 얼마나 보존되는가  → 모든 샘플 쌍의 관계를 모아 본 **집단적·전역적 구조 비교 →** 괜찮은 테크닉인듯? corruption이 activation에 대한 영향을  분석하는 툴로
    - 샘플별 cosine similarity: 같은 샘플의 깨끗한 activation $X_i$와 손상된 activation $Y_i$가 얼마나 비슷한지 직접 비교
- 모든 $\rho$ 값이 양수입니다. 즉, 대체로 CKA가 높은 모델일수록 손상 후 ECE가 더 크게 증가하는 경향이 있다.
    - 여기서 CKA가 높다는 것은 손상된 입력을 넣어도 모델의 내부 표현 구조가 깨끗한 입력 때와 비슷하게 유지된다는 의미다. 저자들은 이를 표현이 더 경직되어 있다고 해석합니다. → 근데 비슷해야 맞지 않나? 비슷하게 feature가 나와야지 clean때 있던 성능을 유지할수 있잖아.

---

## Claude 보충 (메모 속 질문에 대한 원문 근거)

### Q1. 분할은 왜 linear probe가 아니라 FPN decoder?
- 원문 §III-A: 분류는 linear probe, 분할은 "keep the same frozen-backbone approach but replace the linear probe with a
  lightweight FPN decoder" (4개 깊이의 feature를 1×1 lateral → top-down 합 → 3×3 → concat → 1×1 분류).
- calibration 때문이 아니라 **dense prediction에는 공간 해상도가 다른 여러 층이 필요해서**. 핵심 원칙은 "백본 고정 + 가벼운 head"
  (GeoFM 배포 패턴 = 공유 백본 + 가벼운 head). 한계로 스스로 "UperNet 같은 풍부한 decoder에서는 다를 수 있음"이라고 씀
  → 우리 분할(UperNet)과 설정이 다르다는 점은 비교할 때 명시 필요.

### Q2. 왜 REOBench(AID, Potsdam)와 다른 데이터셋?
- 분류 4개(ADVANCE, m-EuroSAT, RESISC45, So2Sat)는 다중분광·SAR 포함 "diverse sensors, resolutions, label spaces"를 목표로 고름.
  GEO-Bench·PANGAEA 등 기존 GeoFM 평가 묶음에서 가져온 것(표준 평가에 calibration을 얹는 게 논문 목적).
- 우리와 겹치는 건 **RESISC45** (RGB만이라 모든 인코더가 같은 입력) → 우리가 만든 RESISC45-C가 이 논문과 바로 비교 가능.

### Q3. "CKA가 높으면 경직(rigid)"인데, 비슷해야 성능이 유지되는 것 아닌가? (핵심 질문)
- 맞는 직관이고, 저자들도 "비슷하다 = 나쁘다"고 말하는 게 아님. 핵심은 **비슷한데도 정확도는 똑같이 떨어진다**는 점:
  > "EO-pretrained embeddings move less under corruption **while losing just as much task information** and remaining overconfident."
- 즉 세 가지가 동시에 성립:
  1. (B1) EO 인코더 CKA ≈0.82 > 일반 ≈0.61 (덜 움직임). participation ratio ≈13 vs ≈31 → 표현이 소수 방향에 몰려 있음.
  2. 그런데 정확도 하락은 비슷함 → 덜 움직였지만 **task에 중요한 방향**에서는 정보를 잃음 (CKA는 전체 구조를 보므로, 소수의 큰
     방향이 유지되면 높게 나옴; 작은 task 방향의 변화는 가려짐).
  3. (B2) 표현이 덜 움직이니 logit·confidence도 거의 안 변함(drift–confidence 상관 ρ ≈ −0.07 vs −0.17) → (B3) "많이 변했는데 확신하고
     틀린" 샘플 비율 2.4배(≈0.14 vs 0.06).
- 해석: "성능을 지키는 불변성"이 아니라 "**틀려도 모르는 둔감함**". 좋은 불변성이면 CKA가 높고 정확도도 유지돼야 하는데, 정확도는
  유지되지 않음. 그래서 rigidity가 calibration 실패(과신)와 연결됨.
- 우리 실험과의 연결:
  - massive activations(상위 1% 채널이 에너지 60~95%) = participation ratio가 낮다는 것과 같은 현상 → CKA·코사인은 큰 채널이
    지배해서 "안 변했다"고 나오기 쉬움.
  - 우리 §4.3 "SatMAE는 91~99%를 확신하는데 정확도는 전체와 같음", entropy AUROC 0.93→0.66~0.82 = 이 논문의 과신 결과와 같은
    현상을 TTA 관점에서 본 것 (엔트로피 기반 TTA가 신호를 잃는 이유).
  - 단, 이 논문의 결과는 상관 분석(intervention 아님, single seed)이라고 저자 스스로 한계로 씀.

### Q4. §F "정규화 전에 corruption"이 우리에게 주는 근거
> "Corrupting before normalization is likewise deliberate: at deployment only the training-set statistics are available,
> so the normalization mismatch a shifted input induces is itself part of the shift a model sees in practice."
- 우리 설정(REOBench도 픽셀 공간에서 손상 → 고정 Normalize)과 같고, **ISA가 고치는 대상(정규화 불일치)이 실제 shift의 일부**라는 근거로
  인용 가능. 또한 "학습 셋 통계만 쓸 수 있다"는 가정이 우리 소스 통계 사용(§5.8)과 같은 가정.

### Q5. corruption 구현 차이 (REOBench와 비교)
| | Beyond Accuracy | REOBench |
|---|---|---|
| 구름 | satellite-cloud-generator (Perlin + 그림자 offset) | Perlin threshold + 회색 혼합 (그림자 없음) |
| 노이즈 | Poisson–Gaussian (CCD/CMOS shot + read noise) | Gaussian, salt & pepper |
| motion blur | 비행 방향 smearing | 수평 커널 (AID-C 실제 길이 10~30) |
| 선정 원칙 | 물리 메커니즘이 서로 다른 3종 → 효과가 하나에 특이한지 확인 | 12종 (기하·환경·센서) |
| seed | (base seed, image index)로 고정 | 없음 (우리 RESISC45-C는 고정) |

---

## Claude 분석: Results 이후 읽는 법과 가져갈 재료 (2026-11-16, 사용자 요청)

### 0. 지표 입문 (이 논문을 읽는 데 필요한 만큼)
- **ECE**: 예측을 confidence 구간(15개)으로 나누고, 구간마다 |평균 confidence − 실제 정확도|를 재서 가중 평균. 0이면 "90% 확신한
  예측이 실제로 90% 맞음". **signed ECE** = 평균 confidence − 평균 정확도. 양수 = 과신.
- **NLL, Brier**: 정답 클래스에 준 확률을 직접 채점. ECE는 구간 나누기에 의존하므로 보조 확인용으로 같이 봄.
- **temperature scaling(TS)**: logit을 상수 T로 나눠 confidence만 낮춤(예측 클래스는 안 바뀜). T는 clean 검증 400장으로 한 번 맞춤.
- **selective prediction**: confidence 낮은 예측을 버리면 남은 것의 정확도가 오르는가. "하위 10% 버렸을 때 정확도 이득"과 eAURC(0에
  가까울수록 confidence가 맞고 틀림을 잘 가름).
- **Kendall τ**: 두 조건에서 인코더 16개의 순위가 얼마나 같은가 (1 = 같음, 0 = 무관).
- **linear CKA**: 같은 샘플 N개의 clean feature X와 corrupted feature Y에 대해 "샘플들 사이의 상대 배치"가 보존됐는지. 계산 전에
  **feature마다 샘플 평균을 뺌(열 중심화)**. 회전·전체 스케일에 불변.
- **샘플별 cosine**: 같은 샘플의 clean/corrupted feature가 얼마나 같은 방향인지 (개별 이동).
- **participation ratio(PR)**: 분산이 몇 개 방향에 퍼져 있는지 (유효 차원 수).

### 1. Results 한 줄씩
| 절 | 결과 | 핵심 숫자 |
|---|---|---|
| A | clean에서는 GeoFM과 일반 인코더가 구분 안 됨 | acc 0.847 vs 0.839, ECE 0.034 vs 0.038 |
| B | shift에서 정확도와 calibration이 같이 무너지고, 방향은 **과신** | 평균 acc 0.84 → cloud s3 0.34 / s5 0.15, Poisson–Gaussian s3 0.15, motion blur s3 0.66. signed ECE 거의 전부 양수 |
| B | clean 순위로는 shift 아래 calibration 순위를 못 맞힘 | acc 순위 τ 0.39 (clean→cloud s3), ECE 순위 τ 0.07. shift끼리는 일관(s3→s5 τ 0.72) |
| B | 분할도 같음. mIoU 하락은 비슷한데 ECE 악화는 GeoFM이 큼 | s5/clean pixel-ECE 비: motion blur 4.75× vs 3.47×, P–G 8.30× vs 7.34× |
| B | RGB만 쓰는 데이터(ADVANCE, RESISC45)에서도 EO가 더 과신 | cloud s3 ECE 0.44 vs 0.25, clean은 0.023 vs 0.024 |
| C | EO 인코더는 덜 움직이고(B1), 움직여도 confidence가 안 변하고(B2), "많이 움직였는데 확신하고 틀린" 샘플이 많음(B3) | CKA 0.82 vs 0.61, PR 13 vs 31, drift–confidence ρ −0.07 vs −0.17, 0.14 vs 0.06 |
| C | 강도별 CKA 하락 속도가 다름 | EO 0.93→0.71, 일반 0.83→0.40 (s1→s5) |
| C | CKA가 높은 인코더일수록 ECE가 더 늘어남 | 15개 조건 모두 ρ > 0 (중앙값 0.39), 6개만 p < 0.05 |
| D | 학습 데이터를 1%로 줄여도 같은 패턴 | acc 순위 τ 0.75, ECE 순위 τ 0.28 |
| E | TS는 shift에서 효과 0, ensemble은 일부, GP head는 clean을 희생 | EO cloud s3 ECE 0.400 → 0.403 (TS). SVGP: 심한 구름 ECE 절반, clean ECE 3배(0.108 vs 0.036) |
| F | confidence로 틀린 예측을 걸러낼 수 없음 | 하위 10% 기권 이득 → 0 (일부 음수), eAURC 0.038 → 0.187 (cloud s3) |

논문이 스스로 인정한 한계: C절은 상관 분석·seed 1개·n = 8 vs 7. Panopticon은 rigid한데 가장 덜 과신(반례). EO와 일반 그룹은
아키텍처·크기·augmentation도 달라 "도메인 효과"로 단정 못 함. 기하 손상 미실험. fine-tuning·UperNet 미실험.

### 2. 가져갈 재료

**M1. "엔트로피 기반 TTA가 GeoFM에서 안 듣는 이유"의 외부 증거 (가장 직접적)**
- F절: confidence 하위 10%를 버려도 정확도 이득이 0 → confidence가 맞고 틀림을 못 가름. EATA·SAR·DeYO의 "엔트로피 낮은 샘플만 신뢰"는
  selective prediction과 같은 가정이므로 같은 이유로 실패.
- B절: 실패 방향이 과신 → 엔트로피 최소화는 과신을 더 키우는 방향.
- 우리 §4.3(SatMAE 91~99% 확신인데 정확도는 전체와 같음, entropy AUROC 0.93 → 0.66~0.82)과 같은 현상을 16개 인코더·4개 데이터셋에서
  독립적으로 본 것. → "SatMAE/AID만의 현상이 아니다"의 근거 (교수님 F3 reward hacking 우려 대응).

**M2. "신호를 confidence가 아니라 표현에서 가져와야 한다"는 저자들의 제안 (우리 방향의 인용 근거)**
> "A more promising direction may be to base correction and abstention on signals other than the model's own confidence. …
> the representation itself carries information about shift: the CKA and participation-ratio drift measures … operate on the
> embedding rather than the softmax." (§V-e)
- 통계 정렬(ISA, 블록 정렬)은 정확히 이 방향. 이 논문은 진단까지만 하고 적응은 안 함 → 우리가 채우는 자리.
- TS가 실패하는 이유 = "clean 데이터로 한 번 맞추고 shift를 본 적이 없어서". TTA는 라벨 없는 테스트 배치를 본다는 점이 차이.

**M3. rigidity 퍼즐에 대한 해석 (Claude 추론, 논문 주장 아님 → 검증 필요)**
- CKA는 feature마다 샘플 평균을 빼고 계산함. 그래서 **모든 샘플에 공통인 이동(Y = X + c)은 CKA에 전혀 안 잡힘** (CKA = 1).
- 그런데 linear head는 절대 좌표에서 동작: logit = Wz + b이므로 공통 이동 c는 모든 샘플의 logit에 Wc를 더함 → 특정 클래스 쪽으로
  예측이 쏠리고 confidence는 그대로 높음.
- 이 그림은 세 가지를 동시에 설명: (B1) CKA 높음, (B2) drift가 커도 confidence 안 떨어짐, 우리 §4.3의 예측 쏠림(motion blur → Beach 60%).
- 그리고 우리 §5.7(손상이 채널 평균 이동으로 나타남)과 같은 말. 공통 이동은 통계 정렬이 정확히 되돌리는 성분.
- 예측: "rigid한 인코더(CKA 높음)일수록 통계 정렬로 회복되는 비율이 크다". 맞으면 "RS FM은 rigid해서 과신하지만, 같은 이유로 싸게
  고쳐진다"는 문제 제기 → 해법 연결이 됨.
- 주의: 논문 본문에는 cosine 결과 숫자가 없어(정의만 있음) 이 해석을 논문만으로는 확인 못 함.

**M4. latent 분석 도구 (교수님 F2 "latent feature 분포 보기")**
- CKA + 샘플별 cosine + PR + drift–confidence 상관 + "high-drift & confidence > 0.9 & 오답" 비율. 4개 층에 hook.
- 우리 `feature_shift.py`(채널 평균 이동)에 이 지표들을 추가하면 선행 논문과 같은 도구로 보고 가능.
- 추가로 M3 검증용 분해: drift ‖x_corr − x_clean‖²를 (공통 이동) + (샘플별 잔차)로 나눠 공통 이동이 차지하는 비율.

**M5. calibration을 TTA의 두 번째 평가축으로**
- none / Tent / EATA / ISA / 블록 정렬에 대해 정확도와 함께 ECE·NLL 보고. 가설: 엔트로피 최소화는 ECE를 악화, 통계 정렬은 개선.
- FOA도 ECE를 같이 보고함(3.2). logit만 저장하면 되므로 비용 작음.

**M6. 설정 방어 문장 (사용자 메모 §F에 이미 표시)**
- 정규화 전 손상("the normalization mismatch … is itself part of the shift") → ISA 대상의 정당성, 학습 셋 통계만 쓴다는 가정.
- 물리 메커니즘이 다른 손상 여러 개에서 같은 효과 → 한 손상의 artifact가 아님. 합성이지만 물리적 동기 + 실제 shift 벤치마크로 보완.

**M7. 이 논문이 비워 둔 곳 = 우리가 할 수 있는 곳**
- 기하 손상 미실험 (REOBench에는 있음). UperNet 등 풍부한 decoder 미실험 (우리 분할이 UperNet). 적응 방법 없음.
- "pretraining 도메인만 다른 통제 비교가 필요"(§IV-C, §V-c)라고 명시. **MAE ViT-L(ImageNet) vs SatMAE ViT-L(fMoW)** 는 아키텍처와
  objective가 같고 사전학습 데이터만 다름 → 이 통제 실험이 됨 (교수님 F4, F5와 연결. 체크포인트 세부는 확인 필요).
- 저자들의 대안 가설: 자연 영상 사전학습은 Gaussian blur·photometric jitter augmentation을 많이 쓰고, EO MIM은 crop·flip·masking 위주
  → 격차가 도메인이 아니라 augmentation 노출 때문일 수 있음.

**M8. 겹치는 데이터**
- RESISC45가 이 논문의 분류 4종에 포함(Ntest 6,300 = 우리 RESISC45-C와 같은 크기. 같은 torchgeo split인지는 미확인).
- motion blur는 "REOBench를 따라" 수평 커널이지만 폭이 3~21(224px). 이 논문에서 motion blur가 "약한 손상"인 것은 이 설정 때문일 수
  있음. REOBench(MIM 최대 하락)와 반대로 보이므로 "motion blur는 약하다"는 가져오면 안 됨.

### 3. 가져오면 안 되는 것
- "EO pretraining이 과신의 원인" → 저자도 가설로만 둠 (상관, 교란 변수, 반례).
- "CKA가 높으면 나쁘다" → 높은 CKA 자체가 아니라 "정확도는 떨어졌는데 confidence가 안 변함"이 문제.
- 우리 백본(SatMAE, ScaleMAE)은 이 논문의 16개에 없음. 수치를 그대로 옮겨 말할 수 없고, 같은 지표를 우리 모델에서 재야 함.
