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

## 5. 우리 백본에서 확인한 것 (2026-11-24)
수치: `experiments/2026-11-24_cluster_cot/RESULTS.md`, `RESULTS_subspace.md`, `RESULTS_em_phases.md`. 그림: `docs/figs/2026-11-24_feature_dist/`.

- **부록에서 꼭 볼 것**: B(가중·차수 비교: 가중 없음 1차 6.14, 가중 2차 5.75, 가중 1차 6.47 → 가중의 효과는 작고 일관되지 않다),
  F(ViT-B/16에서 COT 16.44, WF 6.61 / 재적합 4.15), I(CIFAR-10 실패), D.1(500 iteration이면 4.08 → 2.98).
- **Table 1을 묶음별로 보면** (Claude 계산): Noises 4행 COT 2.6 / WF 4.7, Domain shifts 11행 6.2 / 4.2, Adversarial 6행 13.0 / 7.9, OOD 2행 15.3 / 10.4.
  합성 손상에서는 네 행 모두 COT가 이긴다.
- **COT를 우리 설정에 붙이면** (17개 백본, 2,040개 stream): 평균 절대 오차 COT 3.0, COT + temperature 2.3, 우리 flip 추정 2.4, ATC 5.2.
  10pt 넘는 오차는 COT 0.7%, flip 추정 4.6%. → §3 "꼭 해야 하는 것" 3번 완료. flip 추정은 COT보다 낫다고 할 수 없다.
- **"정확도가 높으면 잘 군집돼 있다"(§4.1의 전제)는 우리 백본에서 일반적으로 성립하지 않는다.** k-means Silhouette과 정확도의 순위상관 평균
  AID +0.21, RESISC45 +0.28 (백본에 따라 −0.87 ~ +0.87). clean에서 선형 head 94.8%인데 k-means는 59.1%(AID).
- **두 단계(§3)**: feature 단계 tent에서 정확도가 오른 뒤 떨어지는 것은 재현(81.8 → 83.0 → 70.5). 그러나 DSBR도 Shift distance가 tent만큼 늘면서
  정확도는 오른다(88.0). 우리 설정에서 실패를 가르는 것은 예측 클래스 수(tent 23.5 / 30, DSBR 29.8 / 30).
- 남은 것: §3 "꼭 해야 하는 것" 1번(WF를 같은 조건에서 직접 비교)은 아직 하지 않았다.

## 6. 후속 연구: Entropy Enigma를 TTA 성능 개선에 쓴 곳 (2026-11-26 조사)

Claude가 서브에이전트로 조사했다. Semantic Scholar 기준 인용 약 36편이고, 각 행은 arXiv 초록이나 본문을 열어 확인했다(원문은 아직 정독하지 않음).
"source 가중치만" = 되돌아갈 사전학습 가중치는 필요하지만 source 데이터와 source 통계는 필요 없음.

**요약.** flip을 TTA 방법 안에서 쓴 것은 ABR(ICASSP 2026) 하나이고, 용도는 "언제 리셋할지"다.
Weighted Flips나 2차식을 조기 종료, step 수 선택, 하이퍼파라미터 선택에 쓴 논문은 찾지 못했다. 대부분은 "EM이 붕괴한다"의 근거로만 인용한다.
우리 flip 필터와 가장 가까운 것은 Entropy Enigma를 인용하지 않는 DeYO, AETTA, 그리고 동기로만 인용하는 SICL이다.

| # | 논문 | 저자, venue, 연도 | 링크 | 가져온 것 (또는 flip·불일치 아이디어) | TTA를 어떻게 개선하나 | source 데이터·통계 | 공식 코드 |
|---|---|---|---|---|---|---|---|
| 1 | Adaptive and Balanced Re-initialization for Long-timescale CTTA (ABR). 이전 판 "Maintain Plasticity…" (arXiv 2412.20034) | Y. Wang, J. Tong 외, H. Chen; ICASSP 2026 (arXiv 주석 기준) | arxiv.org/abs/2602.06328 | Entropy Enigma를 인용. 현재 모델과 이전 모델 사이의 예측 flip을 confidence와 그 증가로 가중하고 EMA로 평활 | flip 궤적의 기울기가 최저점 이후 β/√Δt를 넘으면 리셋. flip으로 정한 계수만큼 source 가중치 쪽으로 섞는다. CCC에서 RDumb 대비 평균 +2.3pt (CCC-Medium +4.8). EATA / RDumb 위에 구현 | source 가중치만 | 못 찾음 |
| 2 | AETTA: Label-Free Accuracy Estimation for TTA | T. Lee, S. Chottananurak, T. Gong, S.-J. Lee; CVPR 2024 | arxiv.org/abs/2404.01351 | 인용 안 함(더 먼저 나옴). 적응한 모델과 그 dropout 표본들 사이의 예측 불일치. 배치 예측이 낮은 엔트로피로 쏠리면 벌점 | 온라인 정확도 추정으로 리셋 결정(최근 5개 추정이 그 전 5개보다 낮거나 0.2 미만). TTA 방법 6개 평균 +11.7pt | 없음 (배치만 씀, 리셋 대상은 명시 안 됨) | 있음, github.com/taeckyung/AETTA |
| 3 | Entropy is not Enough for TTA (DeYO) | J. Lee 외; ICLR 2024 Spotlight | arxiv.org/abs/2403.07366 | 인용 안 함. PLPD: 입력과 patch를 섞은 입력 사이의 pseudo-label 확률 감소. 두 view 사이의 예측 변화를 필터와 가중으로 씀 | 엔트로피와 PLPD 임계값으로 표본을 고르고 PLPD로 가중. 치우친 shift와 "wild" 설정에서 개선. Multi-Hypothesis TTA(ECCV 2026)가 이 필터를 재사용 | 없음 | 있음, github.com/Jhyun17/DeYO |
| 4 | Towards Reliable TTA: Style Invariance as a Correctness Likelihood (SICL) | G. Nam, T. Kim, J. Joeong, E. Yang; WACV 2026 | arxiv.org/abs/2512.07390 | EM이 과신으로 간다는 근거로 인용. 원래 예측과 스타일을 흔든 20개 변형의 예측 일치율. 입력 자신의 첫 층 채널 평균·표준편차에 잡음을 넣고, 잡음 크기는 test 배치로 정함 | 주로 forward만 하는 calibration 부품(보정 오차 약 13pt 감소). 부록 실험 하나에서만 SoTTA의 표본 필터를 이것으로 바꿈 | 없음 (논문 Table 1) | 못 찾음 |
| 5 | When and Where to Reset Matters for Long-Term TTA | T. Lim, J.-W. Hwang, K. Lee; ICLR 2026 | arxiv.org/abs/2603.03796 | 인용 안 함. 같은 실패 단계를 다룸. 붕괴 신호 = 배치 평균 예측의 음의 엔트로피를 그 EMA와 비교 | 신호가 EMA를 넘을 때만, 뒤쪽 일부 층만 리셋. 온라인 Fisher 정규화로 잃은 지식 회복. ETA, EATA, ROID, CMF, PeTTA에 붙는 부품 | source 가중치만 (Fisher는 test 데이터로 누적) | 있음, github.com/YonseiML/asr |
| 6 | DSBR (Entropy Minimization without Model Collapse: Mitigating Prediction Bias in Medical Imaging) | T. Nielen, S. Ambekar, J. Kiechle, D. Lang, J. Schnabel; arXiv 2026, venue 미표기 | arxiv.org/abs/2606.02339 | Entropy Enigma의 군집 관점을 직접 이어받음(shift가 적응 전에 이미 군집을 합침). 20% 보류 평가 프로토콜도 같음 | 예측 클래스 빈도의 EMA 역수로 엔트로피를 가중 | 없음 (1/K에서 시작) | 못 찾음 |
| 7 | RDumb++: Drift-Aware Continual TTA | H. Mishra; arXiv 2026, venue 없음 | arxiv.org/abs/2601.15544 | 인용 안 함. 엔트로피의 z-score(EMA 평균·분산 기준)와 KL 점수(기준 분포가 논문에 정의되지 않음) | 고정 1,000 step 리셋 대신 drift가 보이면 전체 또는 절반 리셋. CCC-Medium 약 +3pt | source 가중치만 | 못 찾음 (CCC notebook만) |
| 8 | CAMA: Calibrated Anti-Collapse Marginal Alignment for TTA of VLMs | J. Kang, Y. Guo; arXiv 2026, venue 없음 | arxiv.org/abs/2611.28461 | 관련 연구에서 한 번(embedding drift). 자기 설명은 feature 압축이 클래스 비율을 치우치게 한다는 것 | 배치 상호정보량 + 배치 평균 예측을 기준 클래스 분포에 맞춤(기준은 처음 몇 배치에서 추정) | 없음 | 못 찾음 |
| 9 | EATA-C: Uncertainty-Calibrated Test-Time Model Adaptation without Forgetting | M. Tan, G. Chen 외, S. Niu; TPAMI 2025 (DOI는 검색 결과에서만 확인) | arxiv.org/abs/2403.11491 | 인용 안 함. 전체 망과 부분 망 사이의 차이(모델 불확실성), 예측 라벨 사이의 불일치(데이터 불확실성) | 순수 EM 대신 일관성 손실 + 표본별로 confidence를 올리거나 내리는 min-max 엔트로피 정규화 | 논문은 Fisher를 test 표본으로 낸다고 씀. 불확실: EATA 실제 코드에서는 clean 분포 내 데이터였음 | 확인 안 함 |
| 10 | TTA Induces Stronger Accuracy- and Agreement-on-the-Line | E. Kim, M. Sun, C. Baek, A. Raghunathan, J. Z. Kolter; NeurIPS 2024 | arxiv.org/abs/2310.04941 | 인용 안 함. 모델 쌍 사이의 불일치가 OOD 정확도를 예측 | 라벨 없이 TTA 하이퍼파라미터와 전략 선택 | 아마 필요 (여러 모델 + 라벨 있는 ID 정확도). 초록으로는 확인 안 됨 | 못 찾음 |
| 11 | Realistic Evaluation of TTA Algorithms: Unsupervised Hyperparameter Selection | S. Cygert, D. Sójka, T. Trzciński, B. Twardowski; arXiv 2024 (venue 미확인) | arxiv.org/abs/2407.14231 | 라벨 없는 대리 지표로 TTA 하이퍼파라미터 선택 | 방법이 아니라 벤치마크. 라벨 없는 대리 지표는 들쭉날쭉하고, 믿을 만한 것은 test 라벨 몇 개나 사전학습 데이터를 쓰는 전략뿐 | 일부 전략은 필요 | 공개라고 씀, URL 미확인 |
| 12 | Monitoring Risks in Test-Time Adaptation | M. Schirmer, M. Jazbec, C. Naesseth, E. Nalisnick; NeurIPS 2025 (Semantic Scholar 기준, 미확인) | arxiv.org/abs/2507.08721 | 붕괴와 정확도 추정의 근거로 인용. 최대 softmax 확률을 손실의 대리로 써서 confidence sequence로 추적 | 경보만 울리고 멈추거나 리셋하지는 않음 | 필요 (라벨 있는 source 보정 표본 1,000장) | 있음, github.com/monasch/tta-monitor |

배경으로만 인용해서 뺀 것: Multi-Hypothesis TTA (arXiv 2607.00259, ECCV 2026), COME (2410.10894), TTVD (2412.07980).
쓰임을 확인 못 한 것: LCoTTA (NeurIPS 2025), PEA (ICLR 2026, backprop 없는 공분산 정렬, source 통계 필요 여부 미확인). MGP (ICML 2026)는 인용 여부 미확인.

**우리에게 가까운 것.**
- **SICL (#4)**: 입력 자신의 채널 평균·표준편차를 test 배치 기준으로 흔들어 예측 일치율을 본다. 우리 flip(표준화 전후 예측 차이)을 source 통계 없이 만드는 본보기가 될 수 있다.
- **DeYO (#3)**: 두 view 사이의 예측 변화를 업데이트의 필터와 가중으로 바로 쓴다. 공식 코드가 있고 10월에 이미 구현해 둔 방법이다.
- **ABR (#1)**: Entropy Enigma의 flip을 TTA 안에서 쓴 유일한 예. 정확도 회귀가 아니라 "실패 단계에 들어섰으니 되돌려라"는 신호로 쓴다.
- **AETTA (#2)**: 불일치를 온라인 정확도 추정으로 바꿔 리셋을 정한다. source 통계 없이 되고 코드가 있다.
- 빈 곳: Entropy Enigma식 flip을 표본별 업데이트 필터로 쓰거나, 온라인으로 step 수를 고르는 데 쓴 논문은 찾지 못했다.
