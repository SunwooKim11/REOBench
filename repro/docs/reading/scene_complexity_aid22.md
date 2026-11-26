# "Scene Complexity" (Peng 외, Remote Sensing 2021, 13(4), 742) — RS 장면의 복잡도를 사람 평가로 정의한 논문

Peng, Mei, Li, Hong, Sun, Li (Central South University 등). 사용자가 찾아 `papers/`에 넣음(2026-11-26).
출처: Claude가 PDF 본문 텍스트(19쪽)를 읽고, 클래스 점수는 Fig. 4를 확대해 읽음. 코드 저장소(github.com/GeoX-Lab/Scene_Complexity)는 요약 agent가 확인.

## 무엇을 했나
- 질문: 모든 이미지를 같은 구조의 CNN으로 처리하는 것이 맞나. 쉬운 장면과 어려운 장면이 있다.
- **장면 복잡도의 정의**: 사람 실험 두 가지의 합.
  - 기억 난이도(memorability): 이미지 열을 1초씩 보여 주고 전에 본 이미지를 알아보게 한다. 알아보는 데 필요한 반복 횟수로 1–10점.
  - 탐색 난이도(search difficulty): "비행기가 있나?", "집이 어디 있나?" 같은 질문에 맞게 답하는 데 걸린 시간의 평균.
  - 두 점수의 합을 1–10으로 정규화한 것이 이미지의 복잡도 점수. 평가자 수는 본문에서 찾지 못했다.
- **AID-22**: AID 30개 클래스 중 "복잡도가 잘 구별되는" 22개, 클래스당 360장, 256 px. 클래스 평균 점수를 k-means(K = 3)로 세 수준으로 나눔.

| 수준 (군집 중심) | 클래스 (점수, Fig. 4에서 읽음) |
|---|---|
| 낮음 (2.80) | desert 1.9, meadow 1.9, forest 2.7, bareland 3.2, farmland 3.4, beach 3.7 |
| 중간 (5.79) | mountain 4.6, playground 5.0, stadium 5.2, bridge 5.8, river 6.1, baseball field 6.1, port 6.4, parking 6.4, square 6.5 |
| 높음 (8.34) | airport 7.4, viaduct 7.9, railway station 8.1, park 8.2, center 8.6, dense residential 8.9, commercial 9.3 |

  빠진 AID 클래스 8개: Church, Industrial, MediumResidential, Pond, Resort, School, SparseResidential, StorageTanks.
- 복잡도 예측기: ImageNet 사전학습 ResNet-18을 이 데이터로 학습하고 마지막 층 feature에 SVM. 세 수준 분류 정확도 92.85%.
- 결과: 복잡한 장면은 더 깊고 여러 배율의 feature가 필요하고(VGG-19가 고복잡도에서 +3%), 여러 물체의 조합으로 표현된다(CAM). 손상이나 shift, TTA는 다루지 않는다.

## 공개된 것
- 저장소에는 CAM 시각화 코드(Caffe, Matlab)만 있다. **이미지별 점수, 클래스 점수 파일, 복잡도 예측 모델은 공개되지 않았다.** 쓸 수 있는 것은 Fig. 4의 클래스 점수 22개뿐이다.

## 우리 실험에 적용 (2026-11-26, `experiments/2026-11-26_rs_properties/RESULTS_scene_complexity.md`)
클래스 점수 22개를 AID-C 결과(17개 백본 평균)와 견줬다. 그림에서 읽은 점수의 수준별 평균이 논문의 군집 중심(2.80, 5.79, 8.34)과 일치한다.

| 수준 | 클래스 수 | clean 정확도 | 손상 정확도 | 하락 | 과신 | flip | JPEG 비트/픽셀 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 낮음 | 6 | 98.5 | 87.9 | 10.5 | 3.6 | 9.2 | 1.18 |
| 중간 | 9 | 95.8 | 84.0 | 11.9 | 5.1 | 12.0 | 1.60 |
| 높음 | 7 | 92.5 | 79.2 | 13.3 | 7.4 | 14.2 | 1.88 |

| 사람 복잡도 점수와의 순위상관 (22개 클래스, 5% 유의 크기 약 0.42) | ρ |
|---|---:|
| clean 정확도 | −0.68 |
| 손상 정확도 | −0.47 |
| 하락 (clean − 손상) | +0.20 |
| 과신 (confidence − 정답) | +0.46 |
| flip 비율 | +0.54 |
| head 입력 표준화 이득 | −0.13 |
| 우리 픽셀 지표(JPEG 비트/픽셀) | +0.81 |

- 사람이 복잡하다고 본 클래스일수록 clean에서도 어렵고, 손상에서 과신과 flip이 크다. 하락 폭 자체와의 관계는 약하다(유의하지 않음).
- 우리가 임의로 고른 픽셀 지표(JPEG 비트/픽셀)가 사람 평가와 클래스 단위에서 0.81로 맞는다. 픽셀 지표를 쓴 근거가 된다.
- 한계: 클래스 22개의 평균 점수만 쓴 것이다(이미지별 점수는 없음). 점수는 그림에서 읽었다. 복잡한 클래스는 서로 비슷한 클래스이기도 해서 인과는 가를 수 없다.
  RESISC45에는 적용할 수 없다(점수가 AID 클래스에만 있음).
