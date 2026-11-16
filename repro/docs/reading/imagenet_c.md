# ImageNet-C: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations

- Hendrycks & Dietterich, ICLR 2019 (arXiv 1903.12261). PDF: `~/REOBench/papers/[ImageNet-C] ….pdf`.
  Code: github.com/hendrycks/robustness `ImageNet-C/create_c/make_imagenet_c.py` (parameters below are from it).
  Data here: `/data/tta/ImageNet_Family/ImageNet_C/<corruption>/<1-5>/<wnid>/*.JPEG` (50,000 per severity; 19 types incl. extras).
- 정리: Claude (2026-11-16, 사용자 요청: 방법론이 아니라 데이터셋 기준점이므로 요점만). 우리 연구 관점의 확인 4가지.

## 1. 구성
- ImageNet val 50,000장 → Resize 256 → CenterCrop 224 → 손상 → JPEG(quality 85) 저장. 손상 15종 × 강도 5 = 75 조건.
- 4그룹: noise (Gaussian, shot, impulse) / blur (defocus, glass, motion, zoom) / weather (snow, frost, fog) /
  digital (brightness, contrast, elastic, pixelate, JPEG). 검증용 추가 4종(speckle, Gaussian blur, spatter, saturate)은 따로.
- 이미지마다 무작위성 있음(예: fog 패턴은 이미지마다 다름).

## 2. REOBench 손상과 짝짓기 (같은 이름이라도 세기가 다름)

| REOBench (AID-C 실측 파라미터) | ImageNet-C 짝 | ImageNet-C 파라미터 (s1 → s5) | 세기 비교 |
|---|---|---|---|
| gaussian_noise: σ 0.04→0.08 | gaussian_noise | σ 0.08→0.38 | **REOBench s5 = IN-C s1**. REOBench가 훨씬 약함 |
| salt_and_pepper: amount 0.005→0.05 | impulse_noise (s&p) | amount 0.03→0.27 | REOBench s5 ≈ IN-C s1~2 |
| gaussian_blur: 커널 3→11 (σ 0.8→2.0) | gaussian_blur (추가 세트) | σ 1→6 | REOBench s5 ≈ IN-C s2 |
| motion_blur: 수평 균일 커널 10→30 px | motion_blur | Wand radius 10→20, σ 3→15, 각도 ±45° 무작위 | 커널 모양이 다름(박스 vs 가우시안 꼬리), 방향 고정 vs 무작위. REOBench s5(30px 박스)는 강함 |
| compression: JPEG quality 50→10 | jpeg_compression | quality 25→7 | REOBench가 약함 (REOBench s4 = IN-C s1 근처) |
| brightness_contrast: x·c + b, (b, c) = (0, 1)→(0.4, 0.2) | brightness / contrast (따로) | V(HSV)+0.1→0.5 / (x−평균)·0.4→0.05 + 평균 | REOBench는 두 개를 동시에 적용 (밝기↑ + 대비↓) |
| haze: x(1−i) + i, i 0.2→0.6 | fog | plasma fractal 더하고 재스케일 | IN-C fog는 공간적으로 불균일, REOBench haze는 균일한 affine |
| clouds: Perlin 마스크 + 회색 혼합 | (없음; snow·frost가 가림이지만 성격 다름) | – | RS 고유 |
| data_gaps, rotate, scale, translate | (없음; ImageNet-P에 아주 작은 rotate·scale·translate) | – | RS 고유 (기하·센서 결손) |
| (없음) | shot, defocus, glass, zoom, snow, frost, elastic, pixelate | – | 자연 영상 고유 |

- 저장 형식: ImageNet-C JPEG q85, AID-C JPEG q75(PIL 기본) → AID-C noise가 JPEG 때문에 더 약해짐 (σ 0.08 → 실측 0.056).
- 결론: **겹치는 손상은 5종(noise, s&p, blur, motion blur, JPEG)과 밝기/대비 1종**. REOBench는 noise·blur·JPEG가 ImageNet-C보다
  대체로 약하게 정의되어 있음. "같은 손상이 자연 영상과 RS 영상에 다르게 작용하는가"를 비교하려면 **강도를 맞춰야 함**
  (예: 같은 함수·같은 파라미터를 두 데이터에 적용, 또는 clean 대비 PSNR 같은 물리량으로 맞추기).

## 3. 평가 규칙 (TTA 프로토콜 문장에 인용)
> "Networks should be trained on datasets such as ImageNet and not be trained on IMAGENET-C corruptions." (§4.1)
> "Researchers should clearly state whether they trained on these corruptions or perturbations; however, this training
> strategy is discouraged." (§4.2) 표준 augmentation(crop, flip)과 다른 왜곡(예: uniform noise)으로 학습하는 것은 허용.
- 우리: head는 clean train으로만 학습, 손상은 평가에만 사용, TTA는 라벨 없는 테스트 배치만 사용 → 규칙 준수.

## 4. 지표
- **CE**(corruption error) = Σ_s 오류(모델) / Σ_s 오류(AlexNet), 15종 평균 = **mCE**. 손상마다 어려움이 다르니 AlexNet으로 정규화.
- **Relative mCE** = Σ_s(오류_s − clean 오류) / AlexNet의 같은 값 → clean 성능과 분리된 "하락 폭".
- RS에는 AlexNet 같은 공통 기준 모델이 없음 → 우리는 그대로 평균 정확도 + clean 대비 하락으로 보고. 기준 모델을 정한다면
  supervised ImageNet ResNet50을 "AlexNet 역할"로 쓸 수 있음 (선택).
- 논문 결과 중 참고: AlexNet → ResNet의 mCE 개선은 대부분 clean 정확도 개선으로 설명되고, relative mCE는 오히려 나빠짐
  (EarthShift의 effective robustness와 같은 생각: 강건성은 clean 성능과 분리해서 봐야 함).
- 강건성 향상 방법 중 **CLAHE(히스토그램 평활화) 전처리가 mCE 개선** (76.7 → 74.5) → 입력 통계를 표준화하는 전처리가 효과가 있다는
  오래된 근거. 음성 인식에서 입력 통계 표준화가 쓰인다는 언급도 있음 (§2). ISA의 선행 사례로 인용 가능.

## 5. 손상 함수 구현
- `make_imagenet_c.py`가 공개되어 있음. AID에 같은 함수를 적용하려면 224 crop을 가정한 부분(fog의 plasma_fractal 256 크기,
  motion_blur의 shape 검사, elastic의 상수)을 256 입력에 맞게 고쳐야 함. Wand(ImageMagick) 의존성 있음(motion blur, snow).
- 저장된 ImageNet-C 이미지를 쓰는 비교(예: 같은 백본의 ImageNet-C feature drift vs AID-C feature drift)는 코드 없이 가능.

## 6. 강도 맞추기 논의 (2026-11-16)
- 용어: **PSNR** = 10·log10(255² / MSE(clean, corrupted)) dB. 픽셀 손상량(높을수록 덜 손상). 의미 손상과는 다름(밝기 이동은 PSNR을
  크게 떨어뜨리지만 인식은 쉬움). **mCE** = 손상별 오류를 기준 모델(AlexNet) 오류로 나눈 값의 평균 (§4).
- Beyond Accuracy의 방식: 강도를 **데이터셋 안에서 "비슷한 정도의 손상"이 되도록** 정함. 구름 불투명도는 데이터셋별, Poisson–Gaussian은
  센서별로 조정, motion blur만 모든 데이터셋에 같은 커널(3~21). 그래서 "s3은 데이터셋 간 같은 물리 조건이 아니다"(GSD가 달라 같은 커널도
  지상 거리가 다름)를 스스로 한계로 적음. 이 논문은 자연 영상 vs RS 영상을 비교하지 않고, **같은 RS 데이터에서 사전학습만 다른
  인코더**를 비교 → 강도 맞추기 문제가 생기지 않음.
- 우리 F2 질문을 둘로 나누면:
  (a) 같은 RS 이미지·같은 손상에서 ImageNet 사전학습 vs fMoW 사전학습 = 백본 짝 실험. 강도 맞추기 불필요 (현재 진행 중).
  (b) 같은 손상이 자연 영상과 RS 영상에 다르게 작용하는가 = 강도 맞추기 필요.
- (b)의 선택지:
  1) 단계 맞추기 (REOBench s5 vs ImageNet-C s5): 정의가 달라 불공정. 쓰지 않음.
  2) **파라미터 맞추기**: 같은 함수·같은 값을 두 데이터에 적용 (예: ImageNet-C 함수를 AID에도). 재현 쉽고 "같은 손상"의 가장 직관적인 뜻.
     단 같은 값도 영상 통계(AID 픽셀 std 0.19 vs ImageNet 약 0.27)와 GSD 때문에 상대 세기가 다름.
  3) **손상량 맞추기**: PSNR(픽셀 손상) 또는 기준 모델의 clean 대비 정확도 하락(의미 손상)이 같아지도록 값을 고름.
  → 제안: 2)를 주 비교로 하고 PSNR을 함께 보고(차이가 손상량 차이로 설명되는지 확인). 값은 결과를 보기 전에 정하고 공개하면 조정 자체는 문제
  없음(ImageNet-C·REOBench의 값도 설계자가 고른 임의값). 우리 방법이 좋아 보이도록 값을 고르는 것만 피하면 됨.

## 7. 손상 세기 값의 근거 조사 (2026-11-16, 조사 에이전트 + 원문 확인)
결론: **세 벤치마크 모두 물리적 근거가 없음.** 모두 "현실 상황"이 아니라 스트레스 테스트로 봐야 함.

- **ImageNet-C**: 저자 답변(GitHub issue #12, Claude가 API로 원문 확인) — "The values were initially chosen so that the
  MS-SSIM(x, corruption_s(x)) = 1 - s/10 for severity s ∈ {1..5}. Since MS-SSIM is not a perfect measure I used some discretion."
  후속 답변: TF MS-SSIM 코드로 소수 이미지만 보고 정했고, "the current severity levels probably do not track the MS-SSIM well".
  즉 **지각 유사도 0.9 → 0.5 사다리를 출발점으로 손으로 조정**. ImageNet-C-bar(Mintun 2021)는 ImageNet-C 오류와 맞추도록 자기 값을 정했을
  뿐 원래 값의 근거는 아님.
- **REOBench**: README·issue(4건)에 값 근거나 코드/데이터 불일치 언급 없음. 논문은 현상의 존재만 인용.
- **Beyond Accuracy**: 공개 코드 없음. Poisson–Gaussian 값은 "센서별로 경험적으로" 정했다고만 하고 숫자 미공개. 인용한 노이즈 문헌(Foi 2008)은
  카메라 raw 노이즈 모델 추정법이고 위성 크기는 없음. motion blur 인용(Sieberth 2014/2015)은 UAV이고 "2 px 정도부터 문제"라는 내용.

### 물리량과 비교 (VHR 광학 위성; [추정]은 에이전트 계산)
| 손상 | ImageNet-C | REOBench (AID-C 데이터) | 실제 위성 수준 (출처) | 판단 |
|---|---|---|---|---|
| Gaussian noise σ (0~1) | 0.08~0.38 | 0.04~0.08 | 약 0.002~0.01 [추정]. SNR 150~250 (Pleiades, Sentinel-2 사양·실측), 어두운 수면에서 26~98 (PlanetScope) | 둘 다 과함. REOBench 약 5~20배, ImageNet-C 약 10~100배 |
| salt & pepper | 픽셀 3~27% | 0.5~5% | L1 제품에서 불량 픽셀은 보정됨 → ≪0.1% [추정] | 둘 다 비현실적 |
| Gaussian blur 추가 σ | 1~6 px | 약 0.8~2.0 px | 시스템 PSF σ 약 0.5~0.7 px (Pleiades MTF 0.16~0.3 at Nyquist), 추가 열화 ≲1~1.5 px [추정] | REOBench s1~3 그럴듯, s4~5 강함. ImageNet-C s3~5 비현실적 |
| motion blur 길이 | 반경 10~20 | 10~30 px (공개 코드는 2~10) | 위성 jitter ≤1 px (ZY-3, ALOS), UAV 약 2 px | 둘 다 위성에 비현실적. **여기서는 REOBench가 약하지 않음** |
| JPEG | q 25~7 | q 50~10 | 위성 온보드 웨이블릿 압축 2.9~3.3 bpp(12-bit) ≈ JPEG q 75 이상 [추정] | 둘 다 실제 운용보다 훨씬 가혹 |
| 밝기/대비 | +0.1~0.5 / ×0.4~0.05 | +0~0.4 / ×1~0.2 | 태양 고도에 따른 곱셈 변화 ×0.5~1.5 정도 [추정, 출처 없음] | REOBench s1~2 그럴듯, s4~5 극단 |
| haze | fog (불균일) | i = 0.2~0.6 | 경로 반사도 0.03~0.15 → i ≈ 0.05~0.3 [추정]. 전 지구 구름 피복 55~67% (MODIS) | REOBench s1~2 현실적, s4~5는 얇은 구름 수준 |

- 센서 사양으로 강도를 정한 RS 손상 벤치마크는 찾지 못함 (HyperTTA도 "통일된 벤치마크 부재"를 지적).
- 주의: AID·RESISC45 자체가 Google Earth에서 재표본·압축된 RGB라, "센서 기준 깨끗한 영상"이라는 출발점부터 근사임.
- 우리에게 주는 의미: (1) 강도 값 자체를 문제 삼는 리뷰에는 "ImageNet-C도 MS-SSIM 기반 임의값, REOBench는 대부분 그보다 물리량에 가까움"으로
  답할 수 있음. (2) 손상량을 맞출 지표로 PSNR보다 **MS-SSIM**이 ImageNet-C 설계 원리와 같음 → (b) 분석에 MS-SSIM도 함께 보고하는 것을 고려.
  (3) motion blur s4~5와 JPEG는 위성 현실보다 훨씬 가혹하므로 해석할 때 "극한 조건"으로 명시.
