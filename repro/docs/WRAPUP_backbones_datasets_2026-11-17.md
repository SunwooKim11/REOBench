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

| 데이터 | 작업 | 클래스 | 규모 | 출처 | 쓴 실험 |
|---|---|---:|---|---|---|
| AID | 분류 | 30 | train 7,999 / test 2,000 | REOBench 배포본 | stage 1, 정렬, 신호, oracle, 픽셀 지표 |
| AID-C | 분류, 합성 손상 | 30 | 120,000 (test 2,000 × 12종 × 5단계) | REOBench 배포본 | 위와 같음 |
| RESISC45 | 분류 | 45 | train 18,900 / val 6,300 / test 6,300 | torchgeo split | 위와 같음 |
| RESISC45-C | 분류, 합성 손상 | 45 | 378,000 (test 6,300 × 12종 × 5단계) | **우리가 생성** (REOBench 코드, 파라미터 4개를 AID-C에 맞춤) | 위와 같음 |
| EarthShift R45 | 분류, 자연 shift | 19 | train 7,915 / val 2,726 / test 2,659 | RESISC45에서 거름 (EarthShift 표 1과 일치) | 자연 shift |
| EarthShift UCM | 분류, 자연 shift | 19 | train 1,142 / val 379 / test 379 (전체 1,900) | UC Merced에서 우리가 split | 자연 shift |
| Potsdam-C | 분할, 합성 손상 | 5 (+clutter 무시) | test 2,016 패치 × 8종 × 5단계 | REOBench 배포본 | 분할 drift (2개 모델만) |
| RRSHID | 실제 haze 쌍 (라벨 없음) | – | 쌍 thin 763 / moderate 1,526 / thick 764 | 공개 데이터 | 실제 vs 합성 haze |
| natural_vs_rs | 같은 손상을 자연·RS에 | – | ImageNet val 5,000 / AID test 2,000 / R45 test 1,996 × 7종 × 5단계 | ImageNet-C 함수·파라미터로 우리가 생성 | 자연 vs RS 손상 비교, 픽셀 지표 |

REOBench 12종 손상: rotate, scale, translate, clouds, data gaps, brightness/contrast, haze, gaussian noise,
salt & pepper, gaussian blur, motion blur, JPEG. natural_vs_rs의 ImageNet-C 7종: gaussian noise, impulse noise,
gaussian blur, motion blur, JPEG, brightness, contrast.

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
