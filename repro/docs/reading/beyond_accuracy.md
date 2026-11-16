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
