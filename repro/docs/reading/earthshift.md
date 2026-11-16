# EarthShift: a benchmark for measuring robustness to real-world distribution shifts in Earth observation

- Doerksen & Kerner (ASU), arXiv 2605.29330. PDF: `~/REOBench/papers/EarthShift_….pdf`. 코드: github.com/kerner-lab/earthshift
- 메모 작성: 사용자 (2026-11-16 전달). "사용자 메모"는 원문 그대로, "Claude 보충"은 메모 속 질문에 대한 원문 확인.

---

## 사용자 메모 (원문)

# Motivation

Despite this, most benchmarks evaluating ML models for satellite data (SatML) test generalization to new samples that are closely related to the in-distribution training set, and do not account for the unique distribution shifts common to remote sensing data.  When SatML models are deployed in real applications, differences in data distributions between data seen during training and inference time are common. In a changing climate with unpredictable shifts in ecosystems and the built environment, distributionally-robust models are vital for monitoring our changing planet [2].

natural distribution shifts characteristic of real-world deployment scenarios

distributional robustness tells a different story of model applicability in real-world contexts, where  sufficient progress has not been made.

- Task - 분류 및 분할
- frozen, full fine-tune, frozen backbone decoder fine-tune
- → 실험 결과 분석 후 clear한 trend
    - All models are robust to temporal shifts but not geographic, scale, sensor, and source shifts.
    - GFMs have similar (or sometimes worse) robustness compared to vision foundation models(e.g., CLIP or ImageNet-pretrained models), despite domain-specific pretraining.
    - Neither GFMs or VFMs outperform traditional supervised models (randomly initialized ResNets or ViTs) in terms of distributional robustness.
    - Model robustness does not improve, and sometimes worsens, when encoders are fine-tuned.

# 2. Related work

- Test-time adaptation [12 ] offers promise by adapting during deployment without labeled target data, though its effectiveness for EO distribution shifts remains unexplored
- 그리고 natural image에서 natural shfit를 다루는 데이터셋은 WILDS, ImageNetV2 등이 있는데, SatML계열엔 없다.
- synthetic dataset shift 보다 natural shift가 고려되어야하는 이유
    - Datasets capturing natural distribution shifts are particularly important because robustness to synthetic dataset shifts is not representative of robustness to natural distribution shifts [ 17], and out of distribution scenarios are constantly encountered in SatML contexts. For example, [18 ] showed that the accuracy of models trained to detect post-disaster building damage in remote sensing images dropped by up to 30% when
    evaluated on a new disaster site. Similarly, agriculture field delineation models fail in new countries with different farming and climate patterns than in training data [19].

# Experiment

- Metric: Effective robustness score
    - $\rho(f) = score_{OOD}(f) - \beta(score_{ID}(f))$
    - $\beta(\operatorname{score}_{\mathrm{ID}}(f))$: ID 성능을 바탕으로 추정한 기준 OOD 성능
        - 논문에서는 같은 shift 유형에서 여러 모델의 ID 점수와 OOD 점수 사이의 관계를 선형 회귀로 추
    - 이식은 ID 성능에 따라 일반적으로 기대되는 OOD 기준선 $\beta$와 실제 OOD 점수를 비교해 이 둘을 구분하려는 방법
    - Metircs 해석:

        예를 들어 ID 성능이 비슷한 두 모델을 생각해 볼 수 있습니다. 두 모델의 ID 성능으로 예상되는 기준 OOD 점수가 같더라도, 실제 OOD 점수가 더 높은 모델은 더 큰 \(\rho(f)\)를 얻습니다. 즉, 이 지표는 “얼마나 높은 점수를 받았나”보다 “기본 성능을 고려했을 때 분포 변화에 얼마나 잘 대응했나”에 초점을 둡니다.
        다만 이 지표는 shift 유형별로 모델들의 점수 관계를 바탕으로 한 상대적 측정값입니다. 절대적인 성능이나 모든 실제 배포 상황에서의 견고성을 단독으로 보장하지는 않으므로, 논문은 ID·OOD 점수와 절대 성능 차이도 함께 제시합

- Table 2
    - Surprisingly, in both the full fine-tuning and frozen backbone settings, the ImageNet
    ResNet50 (R50-imgnet) is the most robust model overall, and the randomly-initialized ResNet50
    (R50-rnd) – which uses random convolutions in the encoder – is second in the frozen setting

# 내 생각

- 아니 어이가 없는게, Resnet50(random, imgnet)이 그냥 대부분보다 좋은데 성능이?. 그리고 Full F.T가 아닌 Frozen이여도 ImageNet으로 학습된 모델은 1등인 것 → 근데 왜 그런거지? ResNet의 어떤이점?
    - Scaleshift(분류)에서 TerraFM이 그나마 높은성능을 보여주는데 UCMerced→RESISC45에서 성능 저하가 너무 크고(45정도)
    - Sensor shift(분류)에서도 Resnet50(random, imgnet)
- Datashift(분할,  DeepGlobe-DFC2022)에서는 DINOv3가 1등, 근데 압도적이진않음. 80점대가 self-supervised 계열(EOFM, ViT, CLIP), 70점대가 ResNet계열. → 근데 신기한게 segmentation
- Geographcial, Temporal shift 하에서는 비슷한느낌?

---

## Claude 보충 (2026-11-16)

### Q. 왜 ResNet50이 가장 강건한가?
- **논문의 설명은 추측 한 문단뿐** (§6.2): EO 사전학습 데이터는 특정 센서·지역에 치우쳐 있고, ImageNet은 다양해서 강건할 수 있다.
  ResNet과 ViT의 차이(아키텍처)는 설명하지 않음. 무작위 초기화 ResNet이 왜 강건한지도 설명 없음.
- **절대 점수를 보면 학습 프로토콜의 영향이 커 보임** (Table 10, scale shift, full fine-tuning, 5 seeds; 아래는 Claude 해석):

| 모델 | RESISC45 ID | → UCM OOD | UCM ID | → RESISC45 OOD |
|---|---|---|---|---|
| ResNet50-imgnet | 97.3 | 86.4 | 97.6 | 64.5 |
| TerraFM | 94.7 | 79.3 | 95.0 | 48.9 |
| CLIP | 94.3 | 73.4 | 92.7 | 42.5 |
| ResNet50-random | 90.4 | 59.6 | 80.8 | 31.5 |
| Clay | 81.9 | 34.5 | 69.7 | 25.0 |
| ViT-imgnet | 77.5 | 34.0 | 71.2 | 26.9 |
| DINOv3 | 77.3 | 35.9 | 69.9 | 27.9 |
| Prithvi-2.0 | 70.3 | 27.2 | 60.1 | 19.2 |
| Clay (frozen) | 34.8 | 18.2 | 23.6 | 9.6 |

  - ViT-imgnet의 RESISC45 ID가 77.5%, 무작위 초기화 ResNet50이 90.4%. 19클래스·학습 7,915장·50 epoch·단순 head 설정에서
    **ViT 계열이 ID부터 덜 학습됨** → CNN의 data 효율(귀납 편향)이 유리한 설정. "ResNet이 강건"의 일부는 "ViT가 이 프로토콜에서
    잘 안 맞춰짐"일 수 있음.
  - Clay, Prithvi, CROMA, Galileo, TerraMind, DOFA는 Sentinel 계열(중해상도·다중분광) 모델. 고해상도 RGB 장면 분류는 이들의 입력 영역
    밖 (frozen Clay 34.8%). 고해상도 RGB로 학습한 TerraFM·CLIP은 잘함.
  - RGB 데이터는 모든 모델에 ImageNet 정규화 상수를 씀 (§4 Evaluation). 모델별 사전학습 통계와 다를 수 있음
    (우리 §5.8: 맞지 않는 상수로 ISA를 하면 clean −2.6~−3.5).
- 따라서 "GFM은 강건하지 않다"는 결론은 **이 13개 모델·이 프로토콜**에 대한 것. 우리 백본(SatMAE fMoW-RGB, ScaleMAE, RemoteCLIP)은
  고해상도 RGB 모델인데 여기 없음 → 같은 scale shift에서 직접 재야 함.

### effective robustness 읽을 때 주의
- ρ는 "ID 점수로 예상한 OOD 점수 대비 초과분"(모델 13개로 맞춘 회귀선의 잔차). 절대 성능이 아님.
- R50-rnd(frozen 2위)는 OOD 점수가 높아서가 아니라 예상보다 덜 떨어져서. 회귀선 기울기: temporal 1.02, source 1.0, geographic 0.8,
  scale 0.65, sensor 0.39 (frozen). 기울기가 1보다 작을수록 어려운 shift.

### 우리에게 쓸 재료
- **TTA 공백을 직접 명시**: "Test-time adaptation offers promise … though its effectiveness for EO distribution shifts remains unexplored." (§2)
- **fine-tuning으로는 안 고쳐짐** (강건성이 안 오르거나 나빠짐) → 학습 단계가 아니라 배포 단계 적응의 필요성.
- **scale shift가 RGB 고해상도에서 가장 큰 실제 shift**: 평균적으로 ID 성능의 43%를 잃음 (full FT, 범위 32~63%).
  RESISC45 ↔ UCM은 이미 준비됨(`/data/rs/{resisc45,ucmerced}/earthshift19`). TTA가 회복할 여지가 큼.
- **합성 shift만으로는 부족** ("robustness to synthetic dataset shifts is not representative of robustness to natural distribution
  shifts") → 합성(AID-C, RESISC45-C) + 실제(RESISC45↔UCM, Potsdam→Vaihingen) 둘 다 필요. Beyond Accuracy도 EarthShift를 보완재로 인용.
- 보고 형식: ID / OOD / Δ를 같이 제시. TTA는 "ID–OOD 격차 중 회복한 비율"로 보고하면 이 논문과 바로 비교됨.
- 주의: 논문 표의 UCM split(420/420/1,260)은 torchgeo 필터 결과(1,142/379/379)와 안 맞음 (DATASETS_2026-11.md §3).
