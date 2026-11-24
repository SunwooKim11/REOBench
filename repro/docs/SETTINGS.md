# 실험 설정: 백본·데이터셋·손상 (단일 관리 파일, 마지막 갱신 2026-11-24)

백본, 데이터셋, 손상, 실험 범위에 관한 설정은 이 파일 하나에서 관리합니다. 2026-11-24에 `WRAPUP_backbones_datasets_2026-11-17.md`와
`DATASETS_2026-11.md`를 합쳤습니다. 개념 설명은 `NOTES_concepts.md`, `STUDY_*.md`에 있습니다.

지금까지 실제로 돌린 것만 적었습니다. 준비만 하고 돌리지 않은 것은 §4에 따로 모았습니다.
수치 출처: `experiments/2026-11-16_backbone_feats/RESULTS.md`, `experiments/2026-11-17_natural_shift/RESULTS.md`.

## 1. 백본 17개

모두 frozen, 224 px(전체 이미지를 resize, crop 없음), 각 모델의 정규화 상수, seed 0. feature는 각 방법의 공식
linear evaluation이 쓰는 것입니다. 정의는 `src/reobench_tta/backbones.py`, 체크포인트는 `/data/rs/pretrained_weights`.

| model | 구조 | 사전학습 방법 | 사전학습 데이터 | feature (dim) | 역할 |
|---|---|---|---|---|---|
| mae_in | ViT-L/16 | MAE | ImageNet-1k | CLS (1024) | MIM 쌍 (ImageNet) |
| satmae | ViT-L/16 | SatMAE (MAE) | fMoW-RGB | CLS (1024) | MIM 쌍 (RS) |
| scalemae | ViT-L/16 | Scale-MAE (MAE + GSD PE) | fMoW-RGB | CLS (1024) | MIM 쌍 (RS, 변형) |
| moco_in | ResNet-50 | MoCo-v2 200ep | ImageNet-1k | GAP (2048) | contrastive 쌍 (ImageNet) |
| moco_fmow | ResNet-50 | MoCo-v2 200ep (GASSL baseline) | fMoW-RGB | GAP (2048) | contrastive 쌍 (RS) |
| gassl_fmow | ResNet-50 | MoCo-v2 + Geo + TP (GASSL) | fMoW-RGB | GAP (2048) | 추가 (목적함수가 달라 정식 쌍 아님) |
| dino_in_s16 | ViT-S/16 | DINO | ImageNet-1k | CLS (384) | DINO 쌍 (ImageNet) |
| satdino_s16 | ViT-S/16 | SatDINO | fMoW-RGB | CLS (384) | DINO 쌍 (RS) |
| dino_in_s8 | ViT-S/8 | DINO | ImageNet-1k | CLS (384) | DINO 쌍 (ImageNet) |
| satdino_s8 | ViT-S/8 | SatDINO | fMoW-RGB | CLS (384) | DINO 쌍 (RS) |
| dino_in_b16 | ViT-B/16 | DINO | ImageNet-1k | CLS (768) | DINO 쌍 (ImageNet) |
| satdino_b16 | ViT-B/16 | SatDINO | fMoW-RGB | CLS (768) | DINO 쌍 (RS) |
| dinov3_web_l16 | ViT-L/16 | DINOv3 (7B에서 distill) | LVD-1689M (web) | CLS + patch 평균 (2048) | DINOv3 쌍 (web) |
| dinov3_sat_l16 | ViT-L/16 | DINOv3 (7B에서 distill) | SAT-493M (Maxar) | CLS + patch 평균 (2048) | DINOv3 쌍 (위성) |
| sup_in_r50 | ResNet-50 | supervised (torchvision V1) | ImageNet-1k | GAP (2048) | 기준점 (쌍 없음) |
| sup_in_vitb16 | ViT-B/16 | supervised (torchvision V1) | ImageNet-1k | CLS (768) | 기준점 (쌍 없음) |
| sup_in_vitl16 | ViT-L/16 | supervised (torchvision V1) | ImageNet-1k | CLS (1024) | 기준점 (쌍 없음) |

정식 쌍은 6개입니다(`backbones.PAIRS`): mae_in↔satmae, moco_in↔moco_fmow, DINO↔SatDINO 3개(S/16, S/8, B/16),
DINOv3 web↔sat. 각 쌍은 구조와 방법이 같고 사전학습 데이터만 다릅니다. scalemae는 mae_in과 비교하되 GSD PE가 더해진
변형이고, gassl_fmow는 목적함수가 달라 쌍에서 뺐습니다.

### 백본별 핵심 수치 (정확도 %, 17개 전부)

손상 = REOBench 12종 × 5단계 평균. 정렬 = head 입력을 배치(64)마다 source 평균·표준편차로 다시 표준화(align ms).
자연 shift = EarthShift 19클래스, target test split.

| model | AID clean | AID-C | AID-C 정렬 | R45 clean | R45-C | R45-C 정렬 | R45→UCM | UCM→R45 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mae_in | 94.3 | 72.7 | 86.1 | 90.3 | 65.6 | 78.7 | 72.8 | 52.1 |
| satmae | 92.7 | 74.6 | 79.7 | 85.1 | 65.2 | 70.7 | 60.7 | 52.2 |
| scalemae | 95.5 | 76.8 | 83.4 | 89.8 | 64.4 | 73.7 | 53.8 | 49.8 |
| moco_in | 93.3 | 71.3 | 85.1 | 88.0 | 64.1 | 77.5 | 77.0 | 57.4 |
| moco_fmow | 92.0 | 74.4 | 83.3 | 87.3 | 67.8 | 76.6 | 81.8 | 55.6 |
| gassl_fmow | 94.8 | 81.6 | 86.2 | 90.8 | 74.8 | 80.0 | 87.1 | 71.6 |
| dino_in_s16 | 95.6 | 85.3 | 89.4 | 91.5 | 79.8 | 84.5 | 85.8 | 68.5 |
| satdino_s16 | 96.7 | 89.5 | 91.8 | 93.2 | 80.6 | 84.6 | 91.3 | 76.2 |
| dino_in_s8 | 95.4 | 86.5 | 90.2 | 92.0 | 81.4 | 85.5 | 88.1 | 71.5 |
| satdino_s8 | 97.2 | 89.3 | 91.4 | 93.7 | 80.3 | 85.3 | 91.6 | 76.5 |
| dino_in_b16 | 96.1 | 85.8 | 91.0 | 92.4 | 81.1 | 86.2 | 86.8 | 68.1 |
| satdino_b16 | 96.3 | 89.7 | 92.2 | 93.8 | 83.8 | 87.5 | 91.6 | 77.5 |
| dinov3_web_l16 | 97.8 | 94.8 | 95.3 | 95.5 | 90.6 | 91.8 | 95.3 | 88.2 |
| dinov3_sat_l16 | 95.5 | 80.5 | 86.7 | 91.2 | 77.9 | 82.2 | 66.0 | 58.5 |
| sup_in_r50 | 90.8 | 71.1 | 78.6 | 86.7 | 64.9 | 72.4 | 78.9 | 61.7 |
| sup_in_vitb16 | 92.7 | 81.7 | 85.0 | 88.7 | 76.1 | 79.0 | 79.4 | 63.5 |
| sup_in_vitl16 | 94.9 | 86.1 | 88.9 | 91.1 | 80.7 | 84.0 | 74.9 | 67.1 |

### 백본별 사전학습 데이터와 augmentation (2026-11-24 추가)

출처: 각 방법의 공식 코드·README·논문을 2026-11-24에 확인(하위 agent가 읽고 Claude가 정리). "미확인"은 공식 자료에서 찾지 못한 것.
공통: 모두 좌우 flip(p 0.5)을 쓴다. 단 DINOv3는 학습 config에서 flip이 꺼져 있다. 노이즈·motion blur·JPEG augmentation을 쓰는 백본은 없다.

| model | 데이터 (규모) | epoch | crop | 색 변형 | Gaussian blur | 그 밖 |
|---|---|---:|---|---|---|---|
| mae_in | ImageNet-1k (128만) | 800 | RandomResizedCrop 224, scale 0.2–1 | 없음 | 없음 | mask 75% |
| satmae | fMoW-RGB (train 약 36만) | 800 | 위와 같음 | 없음 | 없음 | mask 75%, fMoW 정규화 상수 |
| scalemae | fMoW-RGB | 800 | RandomCrop 448 뒤 정사각 RandomResizedCrop scale 0.2–1, encoder 입력 224 | 없음 | 없음 (코드에 정의는 있으나 호출 미확인) | GSD positional encoding, 복원 목표 448 |
| moco_in | ImageNet-1k | 200 | RandomResizedCrop 224, scale 0.2–1 | jitter (0.4, 0.4, 0.4, 0.1) p 0.8, grayscale p 0.2 | σ 0.1–2, p 0.5 | 공식 파일은 404로 읽지 못함. GASSL 저장소가 복사한 MoCo-v2 블록 기준 |
| moco_fmow | fMoW | 200 | 위와 같음 (추정) | 위와 같음 | 위와 같음 | 공개 스크립트에 이 baseline의 학습 명령이 없어 정확한 설정 미확인 |
| gassl_fmow | fMoW | 200 | Resize 448, CenterCrop 448, RandomResizedCrop 224 (scale 기본값) | 위와 같음 | 위와 같음 | positive = 같은 장소의 다른 시점 이미지, 위치 100개 군집 분류 과제 추가 |
| dino_in_s16 | ImageNet-1k | 800 | global 2 × 224 (0.25–1), local 10 × 96 (0.05–0.25) | jitter (0.4, 0.4, 0.2, 0.1) p 0.8, grayscale p 0.2, solarize p 0.2 (global 2) | global 1 p 1.0, global 2 p 0.1, local p 0.5 | |
| dino_in_s8 | ImageNet-1k | 800 | global 0.4–1, local 0.05–0.4, 10개 | 위와 같음 | 위와 같음 | |
| dino_in_b16 | ImageNet-1k | 400 | global 0.25–1, local 0.05–0.25, 10개 | 위와 같음 | 위와 같음 | |
| satdino_s16 | fMoW-RGB (train 360,105) | 200 | global 0.25–1, local 0.05–0.25, 10개. local scale 구간을 균등 분할(uniform view sampling) | DINO와 같음 | DINO와 같음 | GSD 예측 head(MSE, 가중 0.1), 시점 augmentation 없음, fMoW 정규화 상수 |
| satdino_s8 | fMoW-RGB | 100 | S/16과 같음 (README 명령은 S/16만 있음) | DINO와 같음 | DINO와 같음 | 위와 같음 |
| satdino_b16 | fMoW-RGB | 미확인 | 위와 같음 | DINO와 같음 | DINO와 같음 | 위와 같음 |
| dinov3_web_l16 | LVD-1689M: Instagram 공개 게시물 170억 장에서 고른 16.89억 장 + ImageNet-1k/22k, Mapillary | iteration 단위 | global 2 × 256 (0.32–1), local 8 × 112 (0.05–0.32) | DINO와 같음 | DINO와 같음 | ViT-7B를 distill. flip 꺼짐. distill에 쓴 데이터는 미확인 |
| dinov3_sat_l16 | SAT-493M: Maxar RGB 정사영상 0.6 m, 512 × 512, 4.93억 장 | iteration 단위 | web과 같음 (논문: 정규화 상수와 학습 길이만 다름) | DINO와 같음 | DINO와 같음 | 위성 ViT-7B를 distill |
| sup_in_r50 | ImageNet-1k (라벨 사용) | 90 | RandomResizedCrop 224 (scale 기본값) | 없음 | 없음 | |
| sup_in_vitb16 | ImageNet-1k (라벨 사용) | 300 | 위와 같음 | RandAugment (밝기·대비·색·solarize 등 포함) | 없음 | mixup 0.2, cutmix 1.0, label smoothing 0.11 |
| sup_in_vitl16 | ImageNet-1k (라벨 사용) | 600 | 위와 같음 | TrivialAugmentWide | 없음 | mixup 0.2, cutmix 1.0, random erase 0.1 |

**blur augmentation 유무와 Gaussian blur 손상에서의 하락** (clean − Gaussian blur 5단계 평균 정확도, pt)

| 묶음 | 백본 | AID 하락 | RESISC45 하락 |
|---|---|---:|---:|
| blur augmentation 있음 (11개) | MoCo 3, DINO 3, SatDINO 3, DINOv3 2 | 0.4–3.9 (평균 1.3) | 1.0–6.2 (평균 2.6) |
| 없음 (6개) | MAE 3, supervised 3 | 4.5–25.1 (평균 14.1) | 6.5–33.2 (평균 16.3) |

motion blur에서는 이 구분이 성립하지 않는다(AID 하락: MoCo 60–76, MAE 53–61, DINO 27–36, DINOv3 web 16, DINOv3 sat 60).

**데이터만 다른 쌍의 scale 손상 정확도** (AID, augmentation의 crop 범위는 쌍 안에서 같음)

| 쌍 | 일반 이미지 | fMoW / 위성 |
|---|---:|---:|
| mae_in / satmae | 25.7 | 78.9 |
| moco_in / moco_fmow | 46.3 | 77.7 |
| dino_in_s16 / satdino_s16 | 75.8 | 89.6 |
| dino_in_s8 / satdino_s8 | 70.8 | 91.3 |
| dino_in_b16 / satdino_b16 | 67.1 | 89.4 |
| dinov3_web_l16 / dinov3_sat_l16 | 92.1 | 80.5 |

## 2. 데이터셋

### 2.1 원본 데이터 (torchgeo 표 형식, License 제외)

열은 [torchgeo Benchmark Datasets 표](https://docs.torchgeo.org/en/stable/api/datasets.html)와 같습니다.
Task: C = 분류, S = 의미 분할. 값은 **원본 데이터 기준**이고, 우리가 실제로 쓴 크기·split은 §2.2에 있습니다.
torchgeo 표에 없는 데이터(AID, RRSHID, ImageNet)는 원 논문 값이며, 확인하지 못한 칸은 "미확인"으로 두었습니다.

| Dataset | Task | Source | # Samples | # Classes | Size (px) | Resolution (m) | Bands | torchgeo |
|---|---|---|---:|---:|---|---|---|---|
| AID | C | Google Earth | 10,000 | 30 | 600x600 | 0.5–8 | RGB | 없음 |
| RESISC45 | C | Google Earth | 31,500 | 45 | 256x256 | 0.2–30 | RGB | 있음 |
| UC Merced | C | USGS National Map | 2,100 | 21 | 256x256 | 0.3 | RGB | 있음 |
| Potsdam | S | Aerial | 38 (타일) | 6 | 6,000x6,000 | 0.05 | MSI (RGB + IR, 우리는 RGB만) | 있음 |
| Vaihingen | S | Aerial | 33 (타일) | 6 | 1,281–3,816 | 0.09 | IRRG (torchgeo 표기는 RGB) | 있음 |
| LoveDA | S | Google Earth | 5,987 | 7 | 1,024x1,024 | 0.3 | RGB | 있음 |
| RRSHID | haze 쌍 (라벨 없음) | 위성, 같은 지역 다시기 촬영 (중국) | 3,053 쌍 | – | 256x256 | 미확인 | RGB | 없음 |
| ImageNet-1k (val) | C | 웹 사진 | 50,000 | 1,000 | 제각각 | – | RGB | 없음 |

- **Potsdam·Vaihingen의 "타일"과 "패치"**
  - 타일 = 원본 파일 한 장. ISPRS가 도시 한 구역을 항공 정사사진으로 크게 찍어 배포한 단위입니다. Potsdam은 6,000x6,000 타일
    38장, Vaihingen은 크기가 제각각인 타일 33장입니다. torchgeo는 파일 단위로 세기 때문에 # Samples에 타일 수가 들어갑니다.
  - 패치 = 타일을 모델 입력 크기로 자른 조각. 6,000x6,000은 GPU에 통째로 넣을 수 없어서 잘라 씁니다.
  - 분할은 픽셀마다 라벨이 있어서, "샘플 수"가 자르는 방식(크기, 간격, 겹침)에 따라 바뀝니다. 같은 Potsdam이라도 512 간격이면
    타일당 144장, 256 간격이면 529장이 됩니다. 그래서 원본 단위인 타일 수로 세는 것이 데이터 크기를 비교하기에 안정적입니다.
    실제 학습 신호의 양은 픽셀 수입니다(Potsdam 38 × 6,000² ≈ 13.7억 픽셀).
  - REOBench Potsdam test: 타일 14장(2_13, 2_14, 3_13, 3_14, 4_13, 4_14, 4_15, 5_13, 5_14, 5_15, 6_13, 6_14, 6_15, 7_13) ×
    타일당 144장(512x512, 512 간격, 마지막 칸은 가장자리에 맞춤 → 가로 12 × 세로 12) = 2,016 패치. 파일 이름
    `3_13_4608_5488_5120_6000.png` = 타일 3_13의 x 4,608–5,120, y 5,488–6,000 구간입니다.
  - Vaihingen mmseg 패치: train 344 / val 398 (512x512, §2.8).
- torchgeo에 없는 것은 AID와 fMoW입니다. UC Merced는 torchgeo에 "UC Merced"라는 이름으로 있습니다.
- Vaihingen 파일은 근적외·적·녹(IRRG) 3채널입니다. torchgeo 표에는 RGB로 적혀 있지만 실제 채널 구성은 IRRG라 RGB 모델에는
  센서 shift가 같이 들어갑니다(§2.8).
- RRSHID: Zhu et al. 2025, "Real-World Remote Sensing Image Dehazing: Benchmark and Baseline"
  ([arXiv 2503.17966](https://arxiv.org/abs/2503.17966), [github.com/lwCVer/RRSHID](https://github.com/lwCVer/RRSHID)).
  가시거리로 thin(>10 km) 763 / moderate(5–10 km) 1,526 / thick(<5 km) 764쌍으로 나뉘며, 우리 서버 파일 수와 일치합니다.
  센서와 해상도는 확인하지 못했습니다.

### 2.2 우리가 쓴 형태

| 데이터 | Task | # Samples (우리 split) | # Classes | Size (px, 파일) | 만든 곳 | 쓴 실험 |
|---|---|---|---:|---|---|---|
| AID | C | train 7,999 / test 2,000 | 30 | 256x256 | REOBench 배포본 (원본 600을 256으로 줄여 배포) | stage 1, 정렬, 신호, oracle, 픽셀 지표 |
| AID-C | C | 120,000 (test 2,000 × 12종 × 5단계) | 30 | 256x256 | REOBench 배포본 | 위와 같음 |
| RESISC45 | C | train 18,900 / val 6,300 / test 6,300 | 45 | 256x256 | torchgeo split | 위와 같음 |
| RESISC45-C | C | 378,000 (test 6,300 × 12종 × 5단계) | 45 | 256x256 | **우리가 생성** (REOBench 코드, 파라미터 4개를 AID-C에 맞춤) | 위와 같음 |
| EarthShift R45 | C | train 7,915 / val 2,726 / test 2,659 | 19 | 256x256 | RESISC45에서 거름 (EarthShift 표 1과 일치) | 자연 shift |
| EarthShift UCM | C | train 1,142 / val 379 / test 379 (전체 1,900) | 19 | 256x256 | UC Merced에서 우리가 split | 자연 shift |
| Potsdam-C | S | test 2,016 패치 × 12종 × 5단계 (8종 배포 + 4종 우리가 생성) | 5 (+clutter 무시) | 512x512 | REOBench 배포본 | 분할 drift (2개 모델만) |
| RRSHID | 쌍 | thin 763 / moderate 1,526 / thick 764 쌍 | – | 256x256 | 공개 데이터 그대로 | 실제 vs 합성 haze |
| natural_vs_rs | C | ImageNet val 5,000 / AID test 2,000 / R45 test 1,996 × 7종 × 5단계 | 1,000 / 30 / 45 | 224x224 | ImageNet-C 함수·파라미터로 우리가 생성 (ImageNet은 공개 ImageNet-C 파일) | 자연 vs RS 손상 비교, 픽셀 지표 |

분류 백본에는 모두 224x224로 resize해서 넣었습니다(crop 없음).

REOBench 12종 손상: rotate, scale, translate, clouds, data gaps, brightness/contrast, haze, gaussian noise,
salt & pepper, gaussian blur, motion blur, JPEG. natural_vs_rs의 ImageNet-C 7종: gaussian noise, impulse noise(salt & pepper),
gaussian blur, motion blur, JPEG, brightness, contrast.

### 2.3 사전학습 데이터 (같은 열 형식)

백본이 사전학습에 쓴 데이터입니다. 우리 실험에서는 이 데이터를 직접 쓰지 않았고, 테스트에서도 뺐습니다.
Task는 데이터에 붙은 라벨 종류입니다. self-supervised 백본(MAE, MoCo, DINO 계열)은 라벨을 쓰지 않습니다.

| Dataset | Task | Source | # Samples | # Classes | Size (px) | Resolution (m) | Bands | 쓴 백본 |
|---|---|---|---:|---:|---|---|---|---|
| ImageNet-1k (train) | C | 웹 사진 (Flickr·검색 엔진에서 모아 사람이 라벨링) | 1,281,167 | 1,000 | 제각각 | – | RGB | mae_in, moco_in, dino_in × 3, sup_in × 3 (라벨은 sup_in만 사용) |
| fMoW-RGB | C | Maxar(DigitalGlobe) WorldView 위성 | 약 36만 (train) | 62 + false detection | 제각각 (물체 주변을 잘라냄) | 이미지별 메타데이터 (서브미터, 예 0.53) | RGB (원본 4/8 band를 JPEG로 변환) | satmae, scalemae, moco_fmow, gassl_fmow, satdino × 3 |
| LVD-1689M | – | 웹 (Instagram 공개 게시물 170억 장에서 큐레이션) | 16.89억 | – | 제각각 | – | RGB | dinov3_web_l16 |
| SAT-493M | – | Maxar RGB 정사 영상에서 무작위 추출 | 4.93억 | – | 512x512 | 0.6 | RGB | dinov3_sat_l16 |

- **fMoW** (Functional Map of the World, Christie et al. CVPR 2018)
  - 원본(fMoW-full)은 207개국에서 모은 100만 장 이상의 4/8 band 다중분광 TIFF입니다. fMoW-RGB는 이를 RGB JPEG로 바꾼 판입니다.
  - 클래스는 공항, 병원, 경기장 같은 **시설·건물 종류**입니다. 장면 전체를 분류하는 AID·RESISC45와 달리 물체 중심입니다.
  - 같은 장소를 여러 날짜에 찍은 시계열이 있습니다. GASSL이 "같은 장소 다른 시각"을 positive 쌍으로 쓰는 이유가 이것입니다.
  - 이미지마다 GSD가 메타데이터에 있지만 우리 코드(Scale-MAE 포함)는 쓰지 않습니다(`docs/NOTES_concepts.md`).
  - train 약 36만 장은 REOBench 저자 답변의 "fMoW 36만 장"과 SatMAE 논문 기재값입니다. 정확한 split 수는 fMoW 공식 저장소에서
    다시 확인하지 않았습니다.
  - fMoW-Sentinel(SatMAE, Sentinel-2 13 band)은 같은 장소의 다중분광판이지만 우리 백본은 쓰지 않습니다.
- **ImageNet-1k** (ILSVRC-2012): train 1,281,167 / val 50,000, 1,000클래스. 물체 중심 자연 사진입니다. natural_vs_rs에서 쓴
  ImageNet은 이 val에서 클래스당 5장(5,000장)입니다.
- **DINOv3 두 데이터**: 둘 다 각자 데이터로 ViT-7B를 먼저 학습하고, 우리가 쓴 ViT-L/16은 그 7B에서 distill한 모델입니다.
  SAT-493M은 Maxar 상용 영상이라 공개되지 않았습니다.
- **규모 차이가 쌍 비교에 섞입니다.** MAE↔SatMAE, DINO↔SatDINO, MoCo 쌍은 ImageNet 128만 장 대 fMoW-RGB 약 36만 장(약 3.5배)이라
  "도메인 차이"와 "데이터 양 차이"가 같이 들어갑니다. DINOv3 쌍은 둘 다 수억 장 규모라 이 혼동이 상대적으로 작습니다.

### 2.4 배경지식: Source, Bands, Resolution

**Source (영상을 어디서 얻었나)**
- **Google Earth**: Google이 여러 위성·항공 영상을 모자이크하고 색 보정해서 보여주는 8-bit RGB 지도 영상입니다. 이미지마다 원
  센서·촬영일·처리가 다르고 공개되지 않아서, 해상도가 한 값이 아니라 범위(RESISC45 0.2–30 m)로 적힙니다. 물리 단위(반사율)가
  아니라 "보기 좋게 만든 화면"이라 RSPDBench식 센서 물리 근거를 그대로 옮기기 어렵습니다. AID, RESISC45, LoveDA가 여기 속합니다.
- **USGS National Map**: 미국 지질조사국의 항공 정사사진(orthoimagery)입니다. UC Merced는 미국 도시 지역을 1 ft(약 0.3 m)로
  잘랐습니다. 한 기관의 일관된 항공 사진이라 Google Earth 데이터보다 색·해상도가 균일합니다.
- **Aerial**: 비행기에 실은 카메라로 직접 찍은 영상입니다. ISPRS Potsdam·Vaihingen이 여기 속하고 해상도가 수 cm로 매우 세밀합니다.
- **Sentinel-2 / Landsat**: 공공 위성입니다(ESA / NASA·USGS). 반사율 단위의 다중분광 영상을 줍니다. 해상도는 10–60 m(Sentinel-2)로
  거칩니다. EuroSAT, BigEarthNet이 여기 속하고 Prithvi·DOFA 같은 다중분광 모델이 주로 이쪽으로 학습됩니다.
- **fMoW (사전학습 데이터)**: Maxar(DigitalGlobe) 상용 위성 영상입니다. fMoW-RGB는 RGB JPEG이고, fMoW-Sentinel은 같은 장소의
  Sentinel-2 다중분광판입니다. 우리 RS 백본은 전부 fMoW-RGB로 학습됐습니다(DINOv3 sat은 Maxar SAT-493M).

**Bands (채널 구성)**
- **RGB**: 가시광선 빨강·초록·파랑 3채널. 자연 사진과 같은 형식이라 ImageNet 모델을 그대로 쓸 수 있습니다.
- **MSI (multispectral)**: 가시광 + 근적외(NIR) + 단파적외(SWIR) 등 수~십여 개 채널. Sentinel-2는 13개입니다. torchgeo는 RGB에
  IR 채널이 하나만 더 있어도(Potsdam의 RGBIR) MSI로 적습니다.
- **IRRG**: 근적외·빨강·초록을 RGB 자리에 넣은 false-colour 영상. 식물이 빨갛게 보입니다(Vaihingen).
- **HSI (hyperspectral)**: 좁은 파장 band가 수백 개.
- **SAR**: 레이더 영상. 스스로 전파를 쏘는 능동 센서라 구름과 밤에도 찍히지만, 광학 영상과 모양이 완전히 다릅니다.

**Resolution (m)** = GSD(ground sample distance), 픽셀 한 변이 지상에서 몇 m인지입니다. 작을수록 세밀합니다. 같은 256 px라도
UC Merced(0.3 m)는 약 77 m 범위를, RESISC45의 30 m 이미지는 약 7.7 km 범위를 담습니다. EarthShift가 R45↔UCM을 "scale shift"라고
부르는 이유가 이 GSD 차이입니다.

### 2.5 손상: 종류, 단계별 배치, 생성 방법

**REOBench 12종을 SenseBench의 단계 분류에 놓은 표** (2026-11-24). SenseBench는 열화를 촬영–전송–지상 처리 단계와 품질 표준(ISO 19157-1 등)에
맞춰 6개 대분류 22종으로 나눈다(`docs/reading/sensebench.md`). "RS 고유인가" 열은 Claude의 판단이다.

| 단계 | REOBench 손상 | SenseBench 대응 | RS 고유인가 |
|---|---|---|---|
| 센서 | gaussian noise | Gaussian noise | 일반 |
| 센서, 전송 | salt & pepper | impulse noise | 일반 |
| 센서 (검출기 고장) | data gaps | deadline, linear blindness | 고유 (push-broom 주사 방식) |
| 광학, 대기 | gaussian blur | Gaussian blur | 일반 |
| 플랫폼 운동 | motion blur | motion blur | 일반 (원인은 궤도 속도·자세 흔들림) |
| 대기 | clouds | cloud | 고유 (위에서 내려다보는 시점) |
| 대기 | haze | haze | 일부 고유 |
| 방사 보정 | brightness / contrast | 없음 (가까운 것: band attenuation) | 일반 |
| 전송, 저장 | JPEG | JPEG | 일반 (위성은 JPEG2000, SPIHT도 사용) |
| 기하 | scale | 없음 (가까운 것: 기하 압축·늘림) | 고유 (GSD 변화) |
| 기하 | rotate | 없음 | 고유이지만 열화가 아님 (정해진 방향이 없음) |
| 기하 | translate | 없음 | 일반 |

- 12종 중 7종은 ImageNet-C와 같은 일반 영상 열화다(REOBench OpenReview의 "RS 고유성" 지적과 같은 지점).
- 고유한 쪽(data gaps, clouds, scale, rotate)에서 백본 간 차이가 가장 크다: AID scale은 fMoW 쪽이 5쌍 모두 높고(mae_in 25.7, satmae 78.9),
  AID data gaps 정확도는 백본에 따라 33.0–96.9%로 퍼진다(`2026-11-16_backbone_feats/signals/`).
- SenseBench에는 있고 REOBench에는 없는 것: 전송·보정 단계(JPEG2000, WebP, SPIHT, band attenuation, band switch, tile missing), stripe noise.
  후보 실험(미실행, 차선): 복합 stream 6쌍, 통제 2종(band attenuation, band switch). 방법이 정해진 뒤에 판단하기로 함(2026-11-24).

**AID-C**: REOBench 배포본 그대로.

**RESISC45-C** (`scripts/tools/make_corrupted_cls.py`, 우리가 생성)
- REOBench 손상 코드(`~/REOBench/Corrupt Image Generation/`)를 그대로 실행하되, **AID-C 실제 파일과 맞춰 본 결과 공개 코드와 다른
  파라미터 4개를 AID-C 쪽으로 고침** (AID 테스트 이미지 8~10장, JPEG 재저장 후 MAE 비교):

| 손상 | 공개 코드 | AID-C에서 역추정 (= 사용값) | MAE (공개값 → 사용값) |
|---|---|---|---|
| motion_blur 커널 길이 | 2, 4, 6, 8, 10 | 10, 15, 20, 25, 30 | 7.4~8.4 → 1.4~1.9 |
| compression JPEG quality | 30, 25, 20, 15, 10 | 50, 40, 30, 20, 10 | 5.2 → 1.1 (s1) |
| data_gaps 줄무늬 폭 | 3~7 | 5~9 (줄 수 2~6은 같음) | 검은 면적 0.117 → 0.150 (AID-C 0.157, s5) |
| scale_image | 검은 캔버스 | 축소 후 reflect 패딩 (AID-C엔 검은 픽셀 없음) | 7.7 → 4.5 (s5) |

- 공개값과 일치: gaussian blur, haze, brightness/contrast, rotate, translate(4방향 중 최적), gaussian noise σ
  (AID-C noise std 0.028/0.056은 JPEG 저장 효과로 재현됨).
- **clouds는 미일치**: AID-C 구름 면적(추정 0.013~0.129)이 공개 density(추정 0.04~0.22)보다 작음. 원인 못 찾아 공개값 유지.
- 모든 출력은 AID-C처럼 PIL 기본 JPEG(q75). (이미지, 손상, 강도)마다 고정 seed → 재현 가능 (공개 코드는 seed 없음).
- 출력 구조는 AID-C와 같음: `<Class>/<corruption>/severity_<s>/<class idx>/<file>.jpg`.
- 환경: `~/REOBench/.venv-corrupt` (segtta + scikit-image 0.21 + conda-forge `noise`; 서버에 gcc 없음).

**Potsdam-C**
- REOBench 배포본에는 8종의 이미지만 있다. rotate, scale, translate, data gaps는 **라벨만 있고 이미지가 없다**(`anns/…`, 2,016장 × 5강도).
  배포된 기하 손상 라벨은 선형 보간돼 있어(0~4가 아닌 값이 섞인 3채널 PNG) 그대로 쓰면 평가가 오염된다.
- 2026-11-24에 4종을 직접 생성(`scripts/tools/make_potsdam_c_missing.py`): 각 10,080장(2,016 × 5단계).
  기존 8종 이미지가 공개 생성 코드와 정확히 같음(MAE 0.00)을 확인하고 공개 코드의 severity 표를 그대로 사용(data gaps 줄 2~6개, 폭 3~7).
  라벨은 nearest 보간·reflect 경계로 새로 만듦. translate는 배포 라벨에서 방향을 역추정해 전부 일치. data gaps는 clean 라벨 사용.
  위치: `/data/rs/reobench/Potsdam_extracted/test/<corr>/severity_<s>/`, 라벨 `<corr>_ann/severity_<s>/`.

**natural_vs_rs**: ImageNet-C 7종의 함수·파라미터로 AID / RESISC45 test에 생성(§2.2 표).

### 2.6 준비 상태와 위치

선정 기준: (i) 선행 RS 강건성 연구(REOBench, Beyond Accuracy, EarthShift)에서 쓴 데이터, (ii) 평가 백본과 같은 고해상도 RGB,
(iii) 합성 손상과 실제 shift를 둘 다 포함, (iv) 사전학습 데이터(fMoW)는 테스트에서 제외.

| 단계 | 데이터 | 용도 | 상태 | 위치 |
|---|---|---|---|---|
| 1 | AID-C | 분류 합성 손상 | 있음 (REOBench) | `/data/rs/reobench/AID` |
| 1 | RESISC45-C | 분류 합성 손상 | **생성** (test 6,300장 × 12종 × 5강도) | `/data/rs/resisc45/RESISC45-C` |
| 1 | Potsdam-C | 분할 합성 손상 | 8종 있음(REOBench) + 4종 **생성** (2026-11-24, §2.5) | `/data/rs/reobench/Potsdam_extracted` |
| 2 | RESISC45 ↔ UCM | 분류 scale shift | **split 생성** (EarthShift 19클래스) | `/data/rs/{resisc45,ucmerced}/earthshift19` |
| 2 | Potsdam → Vaihingen | 분할 센서·도시 shift | 있음 (mmseg 패치) | `/data/rs/seg/isprs/Vaihingen/derived/mmseg` |
| 3 | LoveDA Urban → Rural | 분할 UDA | 있음 | `/data/rs/loveda_uda`, `/data/rs/seg/loveda` |
| 3 | DeepGlobe → DFC2022 | 분할 data source shift | DeepGlobe만 (비공식 HF parquet, 803장) | `/data/rs/seg/deepglobe_landcover` |

다운로드: `scripts/tools/download_extra_rs.sh` (torchgeo HF 미러).

### 2.7 RESISC45 ↔ UCM split (`scripts/tools/make_rs_splits.py`)
- 클래스 매핑 = EarthShift 공식 코드 `configs/scale-shift-exp.json` (github.com/kerner-lab/earthshift). 19개 1:1, 병합 없음.
  논문은 "highway"라고 썼지만 코드는 양쪽 모두 freeway. 라벨 = 알파벳 순서(양쪽 같음).
- RESISC45 (torchgeo split을 19클래스로 거름): train 7,915 / val 2,726 / test 2,659 = **EarthShift 표 1과 정확히 일치**.
- UCM: train 1,142 / val 379 / test 379 / 전체 1,900. 논문 표의 UCM 숫자(420/420/1,260)는 어떤 필터 split과도 안 맞음
  → OOD 평가는 전체 1,900장(또는 test 379장)으로 두 가지 다 제공.
- 해상도: RESISC45 0.2~30m, UCM 0.3m. 둘 다 256px (UCM 일부 tif는 크기 확인 필요).

### 2.8 Potsdam → Vaihingen
- Vaihingen은 IRRG(근적외-적-녹)만 있음. 우리 Potsdam 모델은 RGB로 학습 → RGB→IRRG 센서 shift가 같이 들어감
  (RS UDA 문헌의 P(RGB)→V(IRRG) 설정). Potsdam 5cm vs Vaihingen 9cm.
- mmseg 패치: train 344 / val 398 (512px). 라벨 1~6 (0 = 경계 무시, reduce_zero_label 형식) → Potsdam REOBench 라벨
  (0~4, 5 = clutter 무시, 255 무시)과 매핑 필요.

## 3. 실험별 범위

| 실험 | 백본 | 데이터 | 측정한 것 | 결과 위치 |
|---|---|---|---|---|
| stage 1 분류 | 17 | AID(-C), RESISC45(-C) | 정확도, ECE, signed ECE, NLL, AUROC, rel_drift, frac_common, CKA, cos, PR ratio, hdhc_wrong, 정렬 후 정확도·ECE | `2026-11-16_backbone_feats/RESULTS.md` §A |
| 라벨 없는 신호 | 17 | AID-C, RESISC45-C (stream 2,040개) | AC, DoC, ATC, 정렬 후 ATC, flip, 평균·표준편차 이동, flip 추정 | `2026-11-16_backbone_feats/signals/` |
| 분할 drift | SatMAE B, ScaleMAE B (UperNet, 미세조정) | Potsdam-C 8종 | mIoU, pixel ECE, AUROC, 층별 frac_common·CKA | 같은 RESULTS.md §B |
| RRSHID | 17 | RRSHID 3단계 + 같은 이미지의 합성 haze | 픽셀 통계, feature 변화, 예측 일치율, ISA·정렬 후 일치율 | §C |
| natural_vs_rs | 17 (ImageNet 정확도는 supervised 3개만) | ImageNet / AID / R45 × ImageNet-C 7종 | PSNR, MS-SSIM, feature 변화, 정확도 유지율 | §D |
| 자연 shift | 17 | R45 ↔ UCM (양방향) | 정확도, ECE, AUROC, AC/DoC/ATC, 정렬 후 ATC, flip 추정, 정렬 효과 | `2026-11-17_natural_shift/RESULTS.md` |
| shift 방향 oracle | 17 | AID-C, RESISC45-C (홀수 번째 test 이미지) | PCA·차원·무작위 방향 제거, head 재학습, 평균 보정 상한 | `2026-11-17_shift_oracle/RESULTS.md` |
| 픽셀·지각 거리 | 백본과 무관 (하락은 17개 평균) | AID-C, RESISC45-C, natural_vs_rs | PSNR, MS-SSIM, LPIPS(alex, vgg)와 정확도 하락의 순위 상관 | `2026-11-16_backbone_feats/pixel_metrics/RESULTS.md` |

### 10월 단계 (REOBench 재현, 참고)

- 백본: SatMAE, ScaleMAE (ViT-L, 256 px, REOBench 설정). 분류는 AID-C에 선형 head(P / PN), 분할은 Potsdam-C에 UperNet.
- TTA: TENT, SAR, DeYO, EATA, CoTTA, LSCD, ISA(배치·이미지) 등. 결과는 `experiments/2026-10-13_cls_tta/`,
  `experiments/2026-10-20_seg_tta/`.
- TTA 구현 검증: ImageNet-C 등에서 원 논문 수치와 비교(R50, ViT-B). `experiments/2026-10-14_verify_imagenetc/`.
- 11월의 17개 백본은 224 px라 10월의 256 px 결과와 바로 비교되지 않습니다.

## 4. 빈 곳과 한계

**준비했지만 아직 안 돌린 것**
- Potsdam → Vaihingen (분할 자연 shift). 데이터는 있으나 Vaihingen은 IRRG뿐이고 라벨 매핑이 필요합니다.
- LoveDA Urban → Rural, DeepGlobe (DeepGlobe는 비공식 803장만 있음).
- Potsdam-C 나머지 4종(rotate, scale, translate, data gaps): 2026-11-24에 생성 완료(§2.5), 실행은 `experiments/2026-11-23_seg_signals/`에서 진행 중.

**설계상 빠진 것**
- contrastive ViT의 fMoW 쌍. 공개 체크포인트가 없어 contrastive 쌍은 ResNet-50(MoCo-v2)뿐입니다.
- CLIP↔RemoteCLIP, DOFA, Prithvi, Galileo, TerraMind는 넣지 않았습니다. 다중분광 모델은 한계로 둘 예정입니다.
- 분할은 REOBench의 2개 모델(SatMAE B, ScaleMAE B)만 했습니다. 17개 백본 비교는 분류에만 있습니다.

**데이터 쪽 주의점**
- RESISC45-C의 clouds는 AID-C와 파라미터를 맞추지 못해 공개 코드 값을 그대로 썼습니다(구름 면적이 AID-C보다 큼).
- EarthShift 논문의 UCM split 크기(420/420/1,260)는 재현되지 않았습니다. 우리 split과 전체 1,900장 결과를 둘 다 냈습니다.
- 모두 seed 0 한 번입니다. 정렬 결과는 배치 64, 단일 손상 stream 기준입니다.

**선행 연구와 겹치는 부분**
- 사전학습 데이터만 다른 쌍을 clean 정확도로 비교하는 설계는 arXiv 2611.13849가 먼저 했습니다
  (`docs/reading/matched_pair_study.md`). 우리 기여는 손상·shift·confidence 쪽입니다.

## 5. 데이터·백본 선택 이유 (질문과 답)

### RESISC45는 어디서 나왔고, RESISC45-C는 왜 만들었나
- RESISC45는 EarthShift가 만든 데이터가 아닙니다. 원래 NWPU-RESISC45(Cheng et al., Proc. IEEE 2017)라는 독립된 장면 분류
  데이터입니다. EarthShift는 이걸 UC Merced와 묶어 scale shift(R45↔UCM)에 썼고, Beyond Accuracy도 분류 4종 중 하나로 씁니다.
- REOBench 12종 손상을 입힌 RESISC45-C는 우리만 만들었습니다. REOBench의 분류 손상 세트는 AID-C뿐입니다. Beyond Accuracy도
  RESISC45에 손상을 입히지만 종류가 3종(cloud/shadow, Poisson–Gaussian 노이즈, motion blur)으로 다르고 데이터는 공개되지 않았습니다.
  공개된 RESISC45-C가 따로 있는지는 저장한 노트에 없고 웹 검색은 하지 않았습니다.
- 만든 이유는 지도교수 피드백 F3입니다(`docs/feedback/2026-11-W1_LSCD-TTA.md`). 벤치마크 하나에만 맞추면 reward hacking 우려가
  있으니 2개 이상으로 하라는 지적이었습니다. RESISC45를 고른 이유:
  - EarthShift의 R45↔UCM 자연 shift와 같은 데이터라 합성 손상과 실제 shift를 같은 데이터 위에서 비교할 수 있습니다.
  - Beyond Accuracy와 겹치는 RGB 데이터라 그 논문과 직접 비교할 수 있습니다.

### RRSHID는 왜 넣었고, REOBench의 haze는 무엇인가
- 실제 haze와 합성 haze를 비교하려고 넣었습니다. 출처는 REOBench OpenReview 토론입니다(`docs/reading/reobench_openreview.md`).
  리뷰어 h8bd가 "haze가 회색 덧칠 같다"고 지적했고, 저자들은 RRSHID 기준 FID 90.56이라는 숫자만 답했습니다. 같은 장면의
  clear/hazy 쌍이라 FID보다 직접적인 비교(feature 이동, 예측 일치율)가 가능해서 썼습니다.
- REOBench haze는 이미지 전체를 흰색 쪽으로 균일하게 섞는 연산입니다: `x·(1−i) + 1·i`, i = 0.2 / 0.3 / 0.4 / 0.5 / 0.6
  (단계 1→5, `~/REOBench/Corrupt Image Generation/image_generation_without_annotations.py` `add_haze_batch`).
  픽셀마다 같은 affine이라 공간 구조가 없습니다. ImageNet-C의 fog는 plasma fractal을 더해 공간적으로 불균일합니다.
- 그래서 결과가 크게 갈립니다. 실제 haze는 채널별 affine으로 설명되는 비율(R²)이 0.35–0.43이고 합성 haze는 정의상 1.0입니다.
  hazy와 clear의 예측 일치율도 실제 1.6–28.2%, 합성 39–84%였습니다(`RESULTS.md` §C).

### ImageNet-C 7종과 REOBench의 대응 (impulse, contrast)
두 종류가 REOBench에 없는 것이 아니라 이름이 다르거나 하나로 합쳐져 있습니다(`docs/reading/imagenet_c.md` §2).

| ImageNet-C | REOBench |
|---|---|
| gaussian_noise | gaussian_noise |
| impulse_noise | salt_and_pepper_noise (ImageNet-C 코드가 s&p를 impulse라고 부름) |
| gaussian_blur (추가 세트) | gaussian_blur |
| motion_blur | motion_blur |
| jpeg_compression | compression_artifacts |
| brightness, contrast (따로) | brightness_contrast (밝기↑ + 대비↓를 한 번에) |

- natural_vs_rs의 질문은 "같은 함수·같은 파라미터가 자연 이미지와 RS 이미지에 다르게 작용하는가"입니다. 그래서 REOBench가 아니라
  ImageNet-C 파라미터를 AID·RESISC45에 적용했고, ImageNet 쪽은 공개 ImageNet-C 파일을 그대로 썼습니다.
- REOBench의 clouds, haze, data gaps, rotate, scale, translate는 ImageNet-C에 같은 함수가 없어 이 비교에서 빠졌습니다
  (haze↔fog는 함수가 달라 짝으로 두지 않음).

### 선행 연구 저자들은 백본을 어떻게 골랐나 (F5, 2026-11-24)

출처: 각 논문 PDF의 모델 선택 절을 Claude가 직접 읽음(REOBench §3, Beyond Accuracy §III-C와 §V 논의, EarthShift §4.1과 표 4, matched-pair 연구 §1·§3).

| 논문 | 고른 모델 | 저자가 밝힌 기준 | 일반 영상 모델 | 스스로 적은 한계 |
|---|---|---|---|---|
| REOBench | MIM 4 (SatMAE, RVSA, ScaleMAE, SatMAE++), CLIP 계열 2 (RemoteCLIP, GeoRSCLIP, 각각 ViT-B/32와 ViT-L/14), MLLM 6 (GeoChat, LHRS-Bot, RS-LLaVA, VHM, SkySenseGPT, Falcon) | "RS의 현재 state-of-the-art"를 세 유형으로 분류. 코드와 가중치가 공개된 것. 두 크기는 크기에 따른 강건성을 보려고 | 없음 | 고해상도 광학만 다룸. 모델 구성에 대한 한계 언급은 없음 |
| Beyond Accuracy | frozen 16개: GeoFM 8 (Clay V1.5, Prithvi V2, DOFA, OLMo-Earth base·tiny, TerraMind, Panopticon, DINOv3-Sat ViT-L), 일반 7 (ConvNeXt-L DINOv3, ResNet-18·50, Swin-T, ViT-B·L, MobileNetV3), 무학습 RCF 1 | 사전학습 도메인으로 묶음. 일반 encoder는 "EO 모델이 더 잘 calibrate되는지 보려고 의도적으로" 포함 | 있음 (구조가 다름) | 두 묶음이 구조, 파라미터 수, recipe도 달라 도메인만 따로 변하지 않는다. "구조와 용량을 고정하고 사전학습 데이터만 바꾸는 통제 연구가 필요하다". augmentation 노출 차이를 배제하지 못함. self-distillation encoder가 2개뿐 |
| EarthShift | GFM 8 (DOFA, CROMA, Clay, Prithvi V2.0, TerraFM, DINOv3 위성, Galileo, TerraMind), 일반 3 (ImageNet 지도학습 ResNet50·ViT-B/16, CLIP), 무작위 초기화 2 (ResNet50, ViT-B/16) | "전 지구 RS archive로 사전학습한 open-source 모델". 일반 모델은 GFM과 강건성을 비교하려고, 무작위 초기화는 transfer의 이득을 보려고 | 있음 (구조가 다름) | 한계 절은 데이터와 shift 범위만 다룸. 모델 구성에 대한 언급은 없음 |
| matched-pair 연구 (arXiv 2611.13849) | 14개, 그중 쌍 5개 | objective와 구조가 같은 쌍. 기존 비교는 여러 요인이 한꺼번에 달라 원인을 말할 수 없기 때문 | 있음 (쌍) | 데이터 규모와 recipe 세부는 통제하지 못함 |

- 세 가지 방식으로 나뉜다. (1) RS 최신 모델 목록(REOBench): 일반 모델이 없어 "RS 모델이라서 약한가"를 말할 기준이 없다.
  (2) 일반 모델을 함께 넣기(Beyond Accuracy, EarthShift): 비교는 되지만 구조가 달라 원인을 가르지 못한다. Beyond Accuracy가 이를 스스로 적고 통제 연구를 요청했다.
  (3) 쌍(matched-pair 연구, 우리): 구조와 방법을 고정하고 데이터만 바꾼다.
- Beyond Accuracy가 배제하지 못한 설명("일반 영상 pipeline은 Gaussian blur와 색 변형 augmentation이 많고 EO masked 모델은 crop·flip·masking뿐")은
  §1의 augmentation 표가 직접 본다: blur augmentation이 있는 11개 백본은 Gaussian blur 손상 하락이 평균 1.3 / 2.6pt, 없는 6개는 14.1 / 16.3pt.
- Beyond Accuracy와 EarthShift의 GFM은 대부분 Sentinel 다중분광 모델이다(EarthShift는 1 × 1 conv 채널 어댑터로 입력을 맞춤). 우리는 RGB 모델만 쓴다.
  겹치는 모델: SatMAE·ScaleMAE(REOBench), DINOv3-Sat(Beyond Accuracy, EarthShift, matched-pair), SatDINO·MAE·DINO·DINOv3(matched-pair).
- EarthShift는 본문에서 DINOv3를 SAT-493M 사전학습이라고 쓰고 표 4에는 "LVD-142M + Aerial"이라고 적어 표기가 엇갈린다.

### contrastive ViT 쌍은 왜 없나
- fMoW-RGB로 처음부터 contrastive 학습한 ViT의 공개 체크포인트가 없습니다. SatMAE 논문의 MoCo-v3 baseline은 ViT-B, fMoW-Sentinel
  (10 band, 96 px)이고 공개되지 않았습니다. RemoteCLIP 등 CLIP 계열은 모두 OpenAI CLIP이나 LAION 가중치에서 이어 학습했습니다.
- 그래서 contrastive 쌍은 ResNet-50 MoCo-v2뿐이고, 이 점은 한계로 적습니다. ImageNet MoCo-v3(ResNet-50, ViT-B)로 "contrastive냐
  ResNet이냐"를 가르는 실험은 하지 않기로 했습니다(2026-11-17).
