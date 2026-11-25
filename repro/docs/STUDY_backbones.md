# 백본 17개: 사전학습 방식 요약 (2026-11-26 작성, F4·F5 질문 대비)

목적: 발표에서 "이 백본은 어떻게 학습된 것인가"에 한두 문장으로 답할 수 있게 한다. 표의 원본은 `docs/SETTINGS.md` §1이고,
이 문서는 그것을 방식별로 풀어 쓴 것이다. augmentation·epoch 표는 2026-11-24에 하위 agent가 공식 코드·README·논문을 읽고 Claude가 정리했다.
**방식 설명(2절)은 Claude가 아는 내용으로 쓴 것이고 이번에 원문과 다시 대조하지 않았다.** 발표 전에 3절의 논문에서 확인할 것.

## 0. 한 장 요약 (슬라이드용)

| 방식 | 무엇을 맞히도록 학습하나 (라벨 없음) | 구조 | 일반 이미지 | 위성 | 쌍 안에서 다른 점 |
|---|---|---|---|---|---|
| MAE | 패치의 75%를 가리고 가린 픽셀을 복원 | ViT-L/16 | mae_in (ImageNet) | satmae, scalemae (fMoW-RGB) | 데이터. Scale-MAE는 방식도 다름(GSD 위치 인코딩, 여러 배율 복원) |
| MoCo-v2 | 같은 이미지의 두 변형은 가깝게, 다른 이미지는 멀게 | ResNet-50 | moco_in (ImageNet) | moco_fmow, gassl_fmow (fMoW) | 데이터. GASSL은 방식도 다름(같은 장소의 다른 시점을 짝으로, 위치 분류 과제) |
| DINO | student가 teacher(student의 이동 평균)의 출력 분포를 따라가게 | ViT-S/16, S/8, B/16 | dino_in ×3 (ImageNet) | satdino ×3 (fMoW-RGB) | 데이터, epoch(800 / 800 / 400 대 200 / 100 / 미확인). SatDINO는 GSD 예측 head와 균등 view sampling 추가 |
| DINOv3 | DINO 계열 + 큰 모델(ViT-7B)을 ViT-L로 distill | ViT-L/16 | dinov3_web (LVD-1689M, 16.9억 장) | dinov3_sat (SAT-493M, 4.9억 장) | 데이터, 정규화 상수, 학습 길이 |
| 지도학습 | ImageNet 라벨 분류 | ResNet-50, ViT-B/16, ViT-L/16 | sup_in ×3 | 없음 | 쌍이 아니라 기준점 |

- 모두 frozen이고, 224 px 전체 이미지를 넣고, 각 방법의 공식 linear evaluation이 쓰는 feature를 쓴다(ViT는 CLS, ResNet은 global average pooling, DINOv3는 CLS + patch 평균).
- "데이터만 다른 쌍"으로 세는 것은 6개다: MAE ↔ SatMAE, MoCo ↔ MoCo-fMoW, DINO ↔ SatDINO 3개, DINOv3 web ↔ sat. Scale-MAE와 GASSL은 방식이 달라 참고로만 둔다.
- 쌍이 완전히 같지는 않다. 데이터 규모(ImageNet 128만 장 대 fMoW-RGB 약 36만 장), epoch, 위성 쪽에 더해진 요소(GSD)가 함께 다르다. 비교는 쌍 안에서만 한다.

## 1. 발표에서 쓸 수 있는 세 가지 관찰 (우리 결과와 이어지는 것)

| 사전학습의 성질 | 우리 결과 | 근거 |
|---|---|---|
| Gaussian blur augmentation이 있나 (MoCo, DINO, SatDINO, DINOv3는 있고 MAE와 지도학습은 없음) | 있는 11개는 Gaussian blur 손상 하락이 평균 1.3 / 2.6pt, 없는 6개는 14.1 / 16.3pt. 두 묶음의 범위가 겹치지 않는다 | T7, fig7 |
| 데이터가 위성인가 | scale 손상은 5쌍에서 위성 쪽이 강하다(AID: MAE 25.7 대 SatMAE 78.9). DINOv3만 반대(92.1 대 80.5) | T7 |
| 목적함수가 무엇인가 | feature가 클래스별로 얼마나 뭉치나(k-means, 5-NN)는 DINO 계열이 높고 MAE 계열이 낮다. 선형 head 정확도는 비슷하다 | T8, `RESULTS_shape.md` |

노이즈, motion blur, JPEG augmentation을 쓰는 백본은 없다. 17개 평균에서 가장 크게 떨어지는 손상이 motion blur인 것과 맞는다(T12: AID-C 40.0, RESISC45-C 33.1).

## 2. 방식별 설명

### 2.1 MAE 계열 (masked image modeling)
- **MAE** (He 외, CVPR 2022): 이미지를 패치로 나누고 75%를 가린 뒤, 보이는 25%만 encoder에 넣고 가벼운 decoder가 가린 패치의 픽셀을 복원한다.
  색 변형과 blur augmentation이 없다(crop과 flip만). 복원 과제라 feature가 픽셀 수준 정보를 많이 담고, 선형으로는 잘 갈라지지만 가까운 이웃이 같은 클래스인 정도(k-NN)는 낮다고 알려져 있다.
- **SatMAE** (Cong 외, NeurIPS 2022): MAE를 위성 영상에 맞춘 것. 논문의 기여는 시계열·다분광 입력용 인코딩이고, 우리가 쓰는 것은 fMoW-RGB로 학습한 ViT-L 가중치다. (사용자가 읽음)
- **Scale-MAE** (Reed 외, ICCV 2023): 위치 인코딩에 GSD를 넣고, 낮은 주파수와 높은 주파수를 서로 다른 배율로 복원한다. 추론 때 GSD 입력이 필요한데 우리는 기본값 1 m로 돌린다(matched-pair 연구도 같음). (사용자가 읽음)

### 2.2 MoCo 계열 (contrastive)
- **MoCo-v2** (Chen 외, 2020 기술 보고서; MoCo는 He 외, CVPR 2020): 한 이미지에서 augmentation 두 개를 만들어 하나는 query encoder, 하나는 key encoder에 넣는다.
  같은 이미지에서 온 쌍은 가깝게, queue에 쌓아 둔 다른 이미지의 key들과는 멀게 만든다(InfoNCE 손실).
  key encoder는 gradient로 학습하지 않고 query encoder의 이동 평균으로 갱신한다(momentum encoder). v2는 v1에 MLP head, 강한 색 변형, Gaussian blur를 더한 것이다.
  augmentation에 불변이 되도록 학습하므로 augmentation에 들어 있는 변형(색, blur, crop 배율)에 강해진다.
- **MoCo-v2 fMoW**: GASSL 저장소가 baseline으로 공개한 가중치. 같은 방식을 fMoW로 200 epoch 학습했다. 공개 스크립트에 이 baseline의 학습 명령이 없어 정확한 설정은 확인하지 못했다.
- **GASSL** (Ayush 외, ICCV 2021, "Geography-Aware Self-Supervised Learning"): MoCo-v2에 두 가지를 더했다.
  (1) positive를 같은 이미지의 augmentation이 아니라 **같은 장소를 다른 시점에 찍은 이미지**로 잡는다. (2) 이미지의 위치를 100개 군집 중 하나로 분류하는 과제를 더한다.
  목적함수가 다르므로 moco_in의 정식 짝이 아니다.

### 2.3 DINO 계열 (self-distillation)
- **DINO** (Caron 외, ICCV 2021): student와 teacher 두 network가 같은 구조다. 한 이미지에서 큰 crop 2개와 작은 crop 여러 개(multi-crop)를 만들고,
  student가 본 crop의 출력 분포가 teacher가 본 다른 crop의 출력 분포와 같아지도록 학습한다(cross-entropy). teacher는 student의 이동 평균이다.
  negative가 없으므로 모든 출력이 같아지는 붕괴를 teacher 출력의 centering과 sharpening으로 막는다.
  CLS feature가 k-NN으로도 잘 분류된다는 것이 논문의 대표 결과다. 우리 측정에서도 5-NN과 k-means는 SatDINO와 DINOv3 web이 가장 높고 DINO(ImageNet)가 그다음이다.
- **SatDINO** (Straka 외, 2025, github.com/strakaj/SatDINO): DINO를 fMoW-RGB로 학습하면서 (1) GSD를 예측하는 head(MSE, 가중 0.1)를 더하고
  (2) local crop의 배율 구간을 균등하게 나눠 뽑는다(uniform view sampling). 공개 가중치는 ViT-S/16, S/8, B/16 세 개이고 우리가 셋 다 쓴다.
  B/16의 epoch는 공식 자료에서 찾지 못했다.

### 2.4 DINOv3
- **DINOv3** (Siméoni 외, 2025): DINOv2 계열을 16.9억 장으로 키운 ViT-7B를 학습하고 ViT-L 등으로 distill했다. 같은 방법으로 Maxar 위성 영상 4.9억 장(SAT-493M, 0.6 m)에서
  학습한 위성 모델도 공개했다. 논문에 따르면 web과 sat의 차이는 데이터, 정규화 상수, 학습 길이다. (사용자가 읽음)
- 우리 결과에서 유일하게 web이 sat보다 크게 강한 쌍이다(AID-C −14.4pt, 자연 shift −29.8 / −28.2pt). 원인은 가르지 못했다.
  DINOv3-SAT은 CLS가 약하다는 보고가 있어(matched-pair 연구) CLS와 patch 평균을 이어 붙여 쓴다.

### 2.5 지도학습 (기준점)
- torchvision의 ImageNet 가중치(V1). ResNet-50은 색 변형 없이 90 epoch, ViT-B/16과 ViT-L/16은 RandAugment / TrivialAugment, mixup, cutmix를 쓴다.
- 넣은 이유: 자기지도 방식이 아닌 기준이 있어야 "자기지도라서 그런가"를 볼 수 있고, ResNet-50은 BatchNorm이 있어 기존 TTA의 원래 대상이다.

## 3. 누가 무엇을 읽었나, 읽을 순서

| 논문 | 읽은 사람 | 발표 전 확인할 곳 |
|---|---|---|
| SatMAE, Scale-MAE, DINOv3 | 사용자 | – |
| MAE (He 외, CVPR 2022) | 미확인 (유명 논문) | §3 방법, mask 비율과 augmentation |
| MoCo / MoCo-v2 (He 외 CVPR 2020, Chen 외 2020) | Claude·agent (코드 기준) | MoCo §3.1–3.2 (queue, momentum), v2의 Table 1 (MLP, aug+) |
| GASSL (Ayush 외, ICCV 2021) | Claude·agent (README, 코드) | §3 temporal positive와 geo-location classification 두 절 |
| DINO (Caron 외, ICCV 2021) | Claude·agent (코드) | §3.1 (손실, centering), Algorithm 1 |
| SatDINO (Straka 외, 2025) | Claude·agent (README, HuggingFace) | GSD 인코딩과 uniform view sampling 절, 공개 가중치 표 |

읽는 순서(시간이 없을 때): DINO → MoCo-v2 → GASSL → SatDINO. DINO와 MoCo는 방식의 뼈대이고 뒤의 둘은 그 위에 얹은 것이다.

## 4. 예상 질문과 답

| 질문 | 답 |
|---|---|
| 왜 DINO 쌍이 셋인가 | SatDINO가 공개한 가중치가 S/16, S/8, B/16 셋이고 DINO(v1)에 같은 구조의 가중치가 모두 있다. 세 쌍의 방향이 같다는 것이 크기와 무관한 결론이라는 근거다. ViT-L은 양쪽 다 가중치가 없다 |
| 쌍이라고 해도 epoch와 데이터 규모가 다르지 않나 | 맞다. 그래서 "데이터 종류 때문"이 아니라 "공개된 위성 가중치로 바꿨을 때"의 차이라고 말한다. matched-pair 연구도 같은 한계를 적었다 |
| GASSL과 Scale-MAE는 왜 쌍이 아닌가 | 목적함수가 다르다(시점 positive와 위치 분류 / GSD 인코딩과 다중 배율 복원). 데이터만 다른 비교가 아니므로 참고 행으로 둔다 |
| 왜 CLIP 계열은 없나 | 이번 구성은 RGB 이미지 encoder의 자기지도 방식 네 가지와 지도학습 기준점이다. REOBench의 CLIP 계열과 MLLM은 넣지 않았다(범위 밖으로 둔 것이고 이유를 따로 검증하지는 않았다) |
| feature를 왜 그렇게 뽑았나 | 각 방법의 공식 linear evaluation이 쓰는 것을 따랐다. head는 17개 모두 같은 절차(train 통계로 표준화 → L-BFGS 선형 head)다 |
