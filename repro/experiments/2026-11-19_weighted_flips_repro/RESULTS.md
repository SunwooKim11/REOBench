# Weighted Flips 재현 (ImageNet-C, ResNet-50, 2026-11-19, 수치만)

공식 코드(oripress/EntropyEnigma @ 04ac397)의 함수와 계수를 그대로 사용. seed 42, 배치 64, RDumb 990 iteration, 처음 1,000장의 flip.
정의는 `scripts/verify/weighted_flips_repro.py` docstring. 논문 값 = Press et al. ICML 2024 Table 1의 IN-C {75} 행.

## ImageNet-C 15종 × 5단계 (75개 split): 정확도 추정 오차 MAE (pt)

| 방법 | 우리 실행 | 논문 |
|---|---:|---:|
| WF (가중 flip, 2차식) | 5.49 | 4.79 |
| flip (가중 없음, 1차식) | 5.28 | – |
| flip (가중 없음, 2차식) | 5.51 | – |
| AC | 10.06 | 10.06 |
| DoC | 6.62 | 6.61 |
| ATC (max-softmax) | 3.68 | 7.44 |
| ATC (negative entropy) | 7.45 | 7.44 |
| COT | (구현 안 함) | 2.23 |

ImageNet val: 정확도 76.15, WF 12.9, 가중 없는 flip 99.

## f 다시 맞추기 (논문과 같이 ImageNet val + holdout 4종 × 5단계 = 21점, 2차식)

| | x² | x | 상수 |
|---|---:|---:|---:|
| 우리 실행 | 0.000501 | -0.3417 | 74.14 |
| 논문 (main.py) | 0.000356 | -0.3214 | 75.66 |

다시 맞춘 f로 추정한 IN-C 75개 split의 MAE: 5.31 (논문 계수 그대로: 5.49).

## 손상별 (5단계 평균): 실제 정확도, WF 추정, 오차

| corruption | acc | WF | WF 추정 | |오차| 평균 | ATC(MC) |오차| 평균 |
|---|---:|---:|---:|---:|---:|
| gaussian_noise | 31.3 | 177.4 | 34.9 | 3.60 | 2.14 |
| shot_noise | 28.9 | 191.7 | 32.5 | 3.59 | 3.19 |
| impulse_noise | 26.0 | 188.4 | 31.8 | 5.86 | 3.11 |
| defocus_blur | 38.6 | 207.6 | 29.1 | 9.49 | 8.81 |
| glass_blur | 26.6 | 220.4 | 27.3 | 3.67 | 2.72 |
| motion_blur | 38.6 | 140.1 | 40.6 | 2.07 | 5.51 |
| zoom_blur | 36.1 | 123.2 | 42.2 | 6.09 | 4.28 |
| snow | 32.2 | 151.0 | 36.2 | 3.98 | 2.10 |
| frost | 37.9 | 145.4 | 37.8 | 0.97 | 3.16 |
| fog | 45.4 | 69.0 | 55.5 | 10.12 | 3.55 |
| brightness | 68.0 | 22.6 | 68.6 | 2.18 | 2.43 |
| contrast | 38.8 | 122.5 | 47.4 | 8.66 | 4.62 |
| elastic_transform | 44.8 | 79.5 | 53.1 | 8.35 | 4.69 |
| pixelate | 44.8 | 66.4 | 56.4 | 11.60 | 1.68 |
| jpeg_compression | 53.7 | 87.1 | 51.5 | 2.16 | 3.26 |
| speckle_noise (holdout) | 37.0 | 132.4 | 41.5 | 4.44 | 1.96 |
| gaussian_blur (holdout) | 41.6 | 193.8 | 33.4 | 8.30 | 6.64 |
| spatter (holdout) | 49.1 | 85.2 | 52.1 | 3.35 | 3.72 |
| saturate (holdout) | 62.0 | 32.1 | 65.8 | 4.12 | 2.31 |
