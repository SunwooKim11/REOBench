# 방향 정리 (2026-11-16): 사용자 브레인스토밍 + 두 논문 + 교수님 피드백

입력: [2026-11-W1_LSCD-TTA.md](2026-11-W1_LSCD-TTA.md) (F1~F5), [reading/beyond_accuracy.md](../reading/beyond_accuracy.md) (M1~M8),
[reading/earthshift.md](../reading/earthshift.md), 아래 브레인스토밍. 상태: **제안 단계, 사용자 확정 전**.

## 1. 사용자 브레인스토밍 (원문)

### (1) RS 고유 특징
- Remote Senosry Image 의 고유 특징을 고려하여 Distribution shift 아래 성능 하락 완화에 도움을 줄 수 있는가?
    - Satellite Data is a Distinct Modality in ML (ICML 2024) ·  정리
        - **Overhead view**: 고정 고도, 위에서 내려다봄
        - **No canonical orientation**: rotation invariance
        - **Log-scale object size**: GSD, multi-scale
        - **Multispectral / hyperspectral**: 3채널 초과
        - **High radiometric resolution**: 8비트 초과
        - **Multi-sensor**: 광학, SAR, LiDAR
        - **Temporal revisit**: 같은 장소 반복 촬영
        - **Georeferenced**: 위치·시간 메타데이터
        - **Sparse, clustered labels**: 현장 측정 라벨
        - **Geographic bias**
        - **Atmospheric / meteorological variation**: 구름, 연무, 조명
        - **Background clutter**: 높은 정보 밀도
        - **Small objects**
        - **Dense prediction**: 지역 단위 지도
        - **Massive data volume**
    - LSCD-TTA: *“multiple cross-domain styles, high information density, variable meteorological environments, and large-scale data”*
    - CTAOD-RS: *“large variations in spectral characteristics, complex background clutter, and extreme object scale diversity”*
    - LSCD-TTA
        - weak-category sample - $L_{wsce}$ - 2nd
            - RS Image is complex information → so weak logit 고려하자
        - category-diversity - $L_{bsce}$ - 3rd
            - Class-imbalance 문제 해결하자
        - Low Saturation - $L_{lsd}$ 1st
            - Gradient Vanishing
            - RS scenario에선 inter-class similarites and environmental variations can make 특정 카테고리를 내재적으로 구분하기 힘들다
    - RS 에서 자주 쓰이는 테크닉; 이미지에 고-저주파 FFT로 구분해서 따로 학습하기 → complex background clutter이기 때문에?

### (2) niche 찾기
- [x]  EarthShift, Beyond Accuracy 성능 표 보면서 niche 찾기.
    - 작은양의 test-data가 주는 unlabeled img로 전체 test 분포를 파악? → positive aug, MEMO
    - 이미지 전체에 어떤 corruption이 가해진지 feature 파악? → Panda
        - 모델이 corruption을 보고 판단하는 경우를 지워버리자 → DeYO
    - 비슷한 지역의 임베딩? → Tesellating the Earth, ECCV 2026 → 지구좌표 모델링? → geographic shift 담당?
    - Scale Shift → scale-MAE approach?
    - feature CKA, topology, Neural Collapse -> corruption이 모델에 어떤영향을 미치는가? - Beyond Accuracy의 ECE, NLL 값도 보면서 파악 → 모델 및 activation 기반 접근법
    - ResNet↔ ViT: Locality의 특성
        - rs img 장면 is complex. patch안 자체도 정보가 많다.
        - log-scale object size. patch간 관계만 고려하는 ViT는 한계 - Swin이 약간 보완?
        - ConvNext도 유리하지 않나
    - supervised vs contrastive vs MIM → Loss, gradient, activation 수준까지의 이해를 해야 성능 향상의 빈틈을 찾을 수 있지 않을까
    - ImageNet의 특성  파악해야 하지 않나.
    - GEO-Bench의 Augmentation
        - Data Augmentation Data augmentation plays a crucial role in the training of deep learning models, especially with small training datasets. Hence, we consider it to be part of the fine-tuning process. As a guideline, we propose limiting the augmentations to 90◦ rotations and vertical and horizontal flips5. On the other hand, we also encourage users to study what are the best data augmentations for remote sensing as this could lead to useful findings for practitioners and the benchmark is well-suited for evaluating such findings.

## 2. 정리 (Claude 제안)

### 2.1 하나로 묶는 질문
> RS 기초모델은 shift 아래에서 confidence를 믿을 수 없다(Beyond Accuracy, 우리 §4.3). 그렇다면 **confidence 대신 쓸 수 있는,
> 라벨 없는 신호를 RS 영상의 고유 구조에서 얻을 수 있는가?**

- 신호 후보는 두 종류: (a) **소스 통계와의 거리** (표현 수준, FOA·Beyond Accuracy §V-e), (b) **RS에서 라벨을 안 바꾸는 변환**(방향, 스케일).
- 교수님 F1과 맞음: 샘플별 예측이 아니라 분포(집단) 수준 신호, 손실 항 없음(forward만).

### 2.2 RS 특징 → shift → 쓸 수 있는 신호
| RS 특징 (ICML 2024 목록) | 만드는 shift | 라벨 없는 신호 / 처방 | 상태 | 남은 하락 (SatMAE AID-C, clean 92.8) |
|---|---|---|---|---|
| Atmospheric variation (affine 물리 모델) | radiometric | 입력 통계 정렬 (ISA) | 완료 | 80.3 → 91.8 (거의 회복) |
| 센서 열화 (blur, noise) | degradation | 중간 블록 통계 정렬 | 완료 | 58.1 → 70.1 (**−22 남음**) |
| 구름, data gaps | occlusion | 가린 영역을 통계에서 제외, MIM 마스킹 | 미착수 | 79.7 → 80.2 (**−13 남음**) |
| Log-scale object size, GSD | scale (합성 + RESISC45↔UCM) | 입력 배율 탐색, ScaleMAE GSD 인코딩 | 미착수 | 81.1 (−12), 실제 shift는 ID의 43% 손실 |
| No canonical orientation | rotate, translate | D4(90° 회전·뒤집기) 예측 평균 | 미착수 | 88.4, 89.3 (−4, 여지 작음) |
| 한 장면의 타일 = 균질한 배치 | (shift 아님) | 배치 통계가 믿을 만함. 단 클래스 치우침에 주의 | 전제로 사용 중 | – |
| Multispectral, multi-sensor, temporal, georeferenced | sensor, geographic, temporal | 메타데이터·다중분광 필요 | **범위 밖** (RGB 백본, AID에 좌표 없음) | – |

### 2.3 브레인스토밍 항목별 판단
| 항목 | 판단 | 이유 |
|---|---|---|
| CKA, activation 분석, ECE·NLL | **채택 (1단계)** | F2의 latent 분석. Beyond Accuracy와 같은 도구 |
| ImageNet 특성 / sup vs contrastive vs MIM | **일부 채택** | MAE(ImageNet) vs SatMAE(fMoW) 통제 비교로 좁힘 (F4, F5). gradient 수준 분석은 보류 |
| Scale shift → Scale-MAE | **채택 (3단계 후보 1순위)** | RS 특징 중 남은 여지가 가장 큼. RESISC45↔UCM 준비됨 |
| positive aug, MEMO, GEO-Bench D4 | **조건부** | D4는 RS에서 근거 있는 변환. 단 MEMO의 엔트로피 최소화는 confidence 의존 → 예측 평균이나 통계 거리로 대체. 회전 여지는 작음 |
| Panda, DeYO | **비교 대상으로만** | Panda = 최종 층 공통 성분 제거. 1단계 분해로 언제 듣는지 설명 가능 |
| ResNet↔ViT, Swin, ConvNeXt | **보류** | EarthShift의 ResNet 우위는 학습 설정과 섞여 있음. BN 유무 관점(ResNet은 층마다 BN = 정렬 지점이 내장)만 논의에 사용 |
| Tessellating the Earth, 지리 임베딩 | **보류** | Claude가 내용을 모르는 논문. geographic shift는 좌표가 필요해 현재 데이터로 불가 |
| FFT 고·저주파 | **분석 축으로 가능** | radiometric = 저주파(평균·대비), degradation = 고주파 변화 (pixel_affine.py의 HF ratio). 방법으로는 보류 |
| topology, Neural Collapse | **보류** | 1단계 결과에서 필요해지면 |
| LSCD의 RS 성질(클래스 유사성, 불균형) | **문제 제기에 사용** | 손상 아래 예측이 질감 없는 클래스(Beach, Desert, Farmland)로 쏠림 = 클래스 유사성이 손상에서 드러난 형태 (§4.3) |

### 2.4 실험 순서 (제안)
1. **shift 분해 분석** (기존 코드 확장, GPU 수십 분): 손상별·층별로 feature 이동을 공통 이동과 샘플별 잔차로 분해 + CKA, cosine, PR, ECE.
   모델: SatMAE, ScaleMAE, + MAE ViT-L(ImageNet) 통제. 데이터: AID-C, RESISC45-C.
   답하는 것: 어떤 shift가 "집단 공통"이라 정렬로 고쳐지는가 (M3 검증), EO와 ImageNet 사전학습이 다른가 (F2, F4, F5).
2. **두 번째 벤치마크로 재현** (F3): none / Tent / EATA / ISA / 블록 정렬 / 조합을 RESISC45-C와 RESISC45→UCM에서. ECE 같이 보고.
   답하는 것: AID에서 본 결과가 다른 데이터, 실제 shift에서도 성립하는가.
3. **정렬로 안 고쳐지는 shift에 RS 구조 사용**: 스케일부터. "소스 통계 거리가 최소인 입력 배율"이 정확도 최대 배율과 일치하는지 먼저 확인.
4. 보류: CLIP 2종, 분할 누락 손상 4종, occlusion, D4.

필요한 준비: RESISC45 linear head (SatMAE, ScaleMAE, MAE-IN), MAE ViT-L ImageNet 체크포인트, logit 저장.
