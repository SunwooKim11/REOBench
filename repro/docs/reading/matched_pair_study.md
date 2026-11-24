# "Satellite Pretraining Is Not One Thing" (arXiv 2611.13849) — 우리 백본 쌍 설계와 겹치는 선행 연구

Agrawal, Sathish, Nambiar (SatSure Analytics). 2026-11-10 제출. GeoCV workshop @ WACV 2027 채택. 코드 링크 없음.
출처: 요약 agent가 arXiv 초록과 HTML 본문(1–6절)을 읽고 보고한 내용(2026-11-17). 2026-11-24에 Claude가 PDF(`papers/`, 11쪽) 본문 1–8쪽의
**텍스트**를 직접 읽고 아래 "원문 직접 확인" 절을 추가함. 그림은 캡션만, Table 2·3의 개별 수치는 다시 대조하지 않음. 부록(supplementary)은 PDF에 없음.

## 무엇을 했나
- 질문: 위성 사전학습 encoder가 강한 일반 encoder보다 나은가(라벨이 적은 장면 분류에서).
- frozen encoder 14개, 그중 대응 쌍 5개:
  DINO ViT-B/16 ↔ SatDINO, DINOv3 ViT-L/16 web ↔ SAT, MAE ViT-L/16 ↔ SatMAE, MAE ViT-L/16 ↔ Scale-MAE, CLIP ViT-L/14 ↔ RemoteCLIP.
- 데이터: PatternNet, RSICD, EuroSAT, UC Merced, RESISC45 (AID, Potsdam 없음). train 10% / test 90%, seed 6개.
- 평가: k-NN, 선형 probe, attentive probe, retrieval. fine-tuning 없음.

## 결론 (그쪽 수치, 위성 − 일반, k-NN / 선형 probe pt)
- SatDINO: RESISC45 +5.2 / +1.7, UC Merced +2.1 / +0.8 → 도움.
- DINOv3-SAT: RESISC45 −27.4 / −12.4, UC Merced −41.8 / −22.5 → 크게 해로움. CLS만 쓰면 UCM k-NN 84.9 vs 43.2.
- SatMAE: RESISC45 +0.0 / −6.5, UC Merced −5.9 / −8.8 → 해로움.
- Scale-MAE: RESISC45 +14.0 / +1.9 → 도움. 같은 fMoW-RGB MAE인 SatMAE와 반대.
- RemoteCLIP: RSICD·UC Merced는 사전학습에서 봤으므로 판정에서 제외. 나머지에서 +0.4~+1.7 선형 probe.
- 조언: "위성으로 사전학습했다"는 선택 기준이 못 된다.

## 다루지 않은 것 (agent가 절 단위로 확인)
손상 강건성, 데이터셋 간 shift, calibration / ECE, CKA, feature drift, test-time adaptation, 통계 정렬, 라벨 없는 정확도 추정: 전부 없음.

## 우리에게 주는 것
- **clean 정확도에서의 대응 쌍 비교는 더 이상 새롭지 않다.** 인용하고, 우리 기여는 shift·손상·confidence 쪽으로 한정할 것.
- **우리 clean 결과와 방향이 일치한다.** 우리(선형 head, RS − ImageNet): SatMAE AID −1.6 / RESISC45 −5.2, DINOv3 sat −2.3 / −4.3,
  SatDINO +0.2~+1.8.
- **DINOv3-SAT의 낮은 CLS k-NN은 우리 로딩 문제가 아니다.** 우리 AID k-NN 49.7(CLS)과 같은 현상을 그쪽도 보고.
- 그쪽 한계: 쌍마다 corpus·학습 길이·recipe가 달라 어떤 요인 때문인지는 못 가림. SatDINO·Scale-MAE를 GSD 입력 없이 돌림(우리도 같음).
- 그쪽에 없는 쌍: MoCo(ResNet50) ImageNet ↔ fMoW. 그쪽에 있고 우리에 없는 쌍: CLIP ↔ RemoteCLIP.

## 원문 직접 확인 (2026-11-24)

**설계의 이유 (F5에 그대로 인용할 수 있는 부분)**
- 기존 비교는 objective, 구조, 크기, 사전학습 데이터, 전처리가 한꺼번에 달라서 "왜 이 encoder가 이기는지, 위성 데이터 때문인지"를 말할 수 없다(§1).
  그래서 objective와 구조가 같은 쌍으로 비교한다. 우리 쌍 설계와 같은 논리다.
- RGB만 쓰는 이유: 14개 encoder가 모두 받을 수 있는 유일한 입력이고, 많은 EO encoder가 원래 RGB 모델이다(§3).
- frozen feature를 쓰는 이유: 표현의 질을 fine-tuning 선택과 분리하기 위해서(§1).
- 쌍이 통제하지 못하는 것도 적었다: DINOv3 쌍은 데이터 규모(16.9억 대 4.9억)도 다르고, Scale-MAE는 scale-aware 위치 인코딩이 더 있고,
  RemoteCLIP은 짝인 CLIP에서 출발한다(가중치 cosine 0.999).

**선형 probe가 우리 stage-1 head와 같은 방식이다 (F4)**
- "features standardised with training-split statistics, the scikit-learn analogue of the parameter-free BatchNorm that MAE places before its linear probe" (§3).
  L-BFGS 다항 로지스틱 회귀, C = 1. 우리 head(clean train 통계로 표준화 → L-BFGS, L2는 검증 NLL로 선택)와 같은 계열이다.
- 입력 정규화는 각 encoder의 사전학습 통계를 쓴다(Scale-MAE는 TorchGeo대로 ImageNet 통계). SatDINO는 GSD token을 빼고, Scale-MAE는 기본 해상도 1 m로 돌렸다. 우리와 같다.

**read-out에 따른 차이 (Table 5, 위성 − 일반, 데이터셋 평균 pt)**

| EO 모델 | k-NN (원래 feature) | k-NN (patch 평균) | 선형 probe | 선형 probe (patch 평균) | attentive probe |
|---|---:|---:|---:|---:|---:|
| SatDINO | +3.3 | +2.6 | +1.0 | +1.1 | +1.1 |
| DINOv3-SAT | −22.7 | −11.7 | −10.3 | −5.0 | −4.6 |
| SatMAE | −0.2 | +12.0 | −4.2 | −2.6 | −2.9 |
| Scale-MAE | +8.9 | +11.4 | +1.0 | +0.8 | +0.8 |
| RemoteCLIP | +4.4 | +2.8 | +1.2 | +0.8 | +0.9 |

- 다섯 쌍 중 넷은 read-out과 무관하게 판정이 같다. SatMAE만 k-NN에서 뒤집힌다(MAE ViT-L의 patch 평균 k-NN이 무너져서).
- 선형 probe는 k-NN보다 masked 모델에 훨씬 너그럽다: 선형 − k-NN이 MAE ViT-L 22.9, I-JEPA 19.8, SatMAE 19.0pt인데 RemoteCLIP 4.2, DINOv3 ViT-L 5.5pt.
  → 우리 결과(선형 head 94.8 대 k-means 59.1, MAE 계열이 가장 낮음)와 같은 방향이다.

**DINOv3-SAT이 왜 뒤지는가 (§5.1, Table 6)**
- CLS token이 장면 기술자로 약하다. patch 평균으로 바꾸면 UC Merced k-NN 43.2 → 60.5, 선형 72.7 → 84.9. web 모델은 feature 선택에 거의 무관(95.2 / 95.2).
  CLS ⊕ patch(우리가 쓰는 조합)는 선형 81.9로 그 사이.
- 가장 좋은 feature를 써도 web 모델에 UC Merced 선형 10.6pt, RSICD 6.2pt 뒤진다. 입력 정규화(SAT-493M 통계 대 ImageNet 통계)의 영향은 작다.
- DINOv3 저자들의 GEO-Bench 평가에서도 web 7B가 위성 7B를 분류 6개 중 5개에서 이긴다(위성 모델은 canopy height에서 앞섬).
- 저자 해석: 0.6 m Maxar 영상과 dense prediction에 맞춰진 patch token에 정보가 있는 모델.

**어디서 차이가 나는가 (§4, Table 4)**
- 효과는 작은 인공 구조물 클래스에 몰린다. DINOv3-SAT은 basketball court −34.7, palace −34.4, tennis court −33.0, SatMAE는 palace −19.9, airport −19.5, storage tank −17.4.
  자연 장면 13개 클래스의 변화는 쌍마다 평균 0.8–3.0pt뿐.
- 저자 해석: 겉보기 크기가 GSD에 달린 작은 구조물에서 사전학습 데이터와 recipe가 중요하다. 인과는 확인하지 못했고,
  한계 (2)에서 "controlled rescaling으로 시험할 수 있다"를 future work로 적었다. → 우리 방법 후보 1(배율 탐색)의 외부 근거.
- 라벨이 적을수록 차이가 크다: 1% 라벨에서 DINOv3-SAT은 UC Merced 선형 −46.1, RESISC45 −28.2pt, 50%에서 −8.0, −6.1.

**Silhouette (§5.2) — 우리 결과와 헷갈리지 말 것**
- 그쪽: **encoder 14개 사이에서**, 라벨로 잰 Silhouette(cosine)이 k-NN 정확도·retrieval mAP와 ρ 0.96–0.98 (EuroSAT만 0.68). 선형 probe와의 상관은 보고하지 않았다.
  "k-NN, retrieval, Silhouette이 모두 cosine 이웃에 기대므로 일치는 예상된 것"이라고 스스로 적었다.
- 우리(2026-11-24): **한 백본 안에서 손상 stream 60개 사이**, k-means로 잰 Silhouette과 선형 head 정확도의 상관이 평균 +0.21 / +0.28.
- 축(encoder 간 대 stream 간), Silhouette의 정의(라벨 대 k-means), 정확도의 종류(k-NN 대 선형)가 모두 달라 서로 모순이 아니다.
  공통점: 군집이 뚜렷한 encoder(SatDINO, DINOv3 web)는 k-NN이 높고 masked 모델은 낮다.
