# 백본·데이터셋 정리 (2026-11-17 기준)

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

- torchgeo에 없는 것은 AID와 fMoW입니다. UC Merced는 torchgeo에 "UC Merced"라는 이름으로 있습니다.
- Vaihingen 파일은 근적외·적·녹(IRRG) 3채널입니다. torchgeo 표에는 RGB로 적혀 있지만 실제 채널 구성은 IRRG라 RGB 모델에는
  센서 shift가 같이 들어갑니다(`docs/DATASETS_2026-11.md` §5).
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
| Potsdam-C | S | test 2,016 패치 × 8종 × 5단계 | 5 (+clutter 무시) | 512x512 | REOBench 배포본 | 분할 drift (2개 모델만) |
| RRSHID | 쌍 | thin 763 / moderate 1,526 / thick 764 쌍 | – | 256x256 | 공개 데이터 그대로 | 실제 vs 합성 haze |
| natural_vs_rs | C | ImageNet val 5,000 / AID test 2,000 / R45 test 1,996 × 7종 × 5단계 | 1,000 / 30 / 45 | 224x224 | ImageNet-C 함수·파라미터로 우리가 생성 (ImageNet은 공개 ImageNet-C 파일) | 자연 vs RS 손상 비교, 픽셀 지표 |

분류 백본에는 모두 224x224로 resize해서 넣었습니다(crop 없음).

REOBench 12종 손상: rotate, scale, translate, clouds, data gaps, brightness/contrast, haze, gaussian noise,
salt & pepper, gaussian blur, motion blur, JPEG. natural_vs_rs의 ImageNet-C 7종: gaussian noise, impulse noise(salt & pepper),
gaussian blur, motion blur, JPEG, brightness, contrast.

### 2.3 배경지식: Source, Bands, Resolution

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
- Potsdam-C 나머지 4종(rotate, scale, translate, data gaps). 배포 라벨이 보간돼 있어 직접 만들어야 합니다.

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

### contrastive ViT 쌍은 왜 없나
- fMoW-RGB로 처음부터 contrastive 학습한 ViT의 공개 체크포인트가 없습니다. SatMAE 논문의 MoCo-v3 baseline은 ViT-B, fMoW-Sentinel
  (10 band, 96 px)이고 공개되지 않았습니다. RemoteCLIP 등 CLIP 계열은 모두 OpenAI CLIP이나 LAION 가중치에서 이어 학습했습니다.
- 그래서 contrastive 쌍은 ResNet-50 MoCo-v2뿐이고, 이 점은 한계로 적습니다. ImageNet MoCo-v3(ResNet-50, ViT-B)로 "contrastive냐
  ResNet이냐"를 가르는 실험은 하지 않기로 했습니다(2026-11-17).
