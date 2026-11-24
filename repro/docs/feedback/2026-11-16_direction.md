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

## 3. 백본 구성 (2026-11-16 논의, 사용자 우선순위 = 백본 넓히기 먼저)

질문: fMoW/위성 사전학습 백본으로 통일할까, general 백본도 넣을까. **제안: 둘 중 하나가 아니라 "쌍"으로.**
(아키텍처, 사전학습 방식)이 같은 fMoW 모델과 ImageNet 모델을 짝지어, general 백본을 통제군으로 넣는다.

| 사전학습 방식 / 구조 | fMoW | ImageNet 짝 | 상태 |
|---|---|---|---|
| MIM, ViT-L/16 | SatMAE fMoW-RGB, ScaleMAE (보유) | MAE ViT-L (facebookresearch/mae) | 짝 다운로드 필요 |
| contrastive, ResNet50 | MoCo-v2 fMoW (GASSL repo의 baseline, 200ep), GASSL = MoCo-v2+Geo+TP | MoCo-v2 ImageNet | GASSL repo에 4종 체크포인트 표 확인(링크 미확인). torchgeo `ResNet50_Weights.FMOW_RGB_GASSL` 존재 |
| self-distillation, ViT-S/B | SatDINO (github.com/strakaj/SatDINO) | DINO ImageNet ViT-S/16, B/16 | 공개 가중치 종류 미확인 |

- 통일안의 문제: 결과가 RS 사전학습 때문인지 ViT/MAE 일반 성질인지 못 가림 (F2 미충족).
- general을 그냥 추가하는 안의 문제: 구조·크기가 섞임 (EarthShift, Beyond Accuracy가 스스로 한계로 적음).
- 쌍 구성: F4(fMoW ResNet·ViT), F5(선택 근거), F2(자연 vs RS), Beyond Accuracy가 "필요하다"고 한 통제 실험을 동시에 충족. 6~7개로 작음.
- ResNet을 넣는 추가 이유: BN이 있어 기존 TTA(BN adapt, Tent)의 원래 대상. ViT에는 그 자리가 없어서 "어디서 정렬할지"가 문제가 됨.
- 주의: 짝이 완전히 같지는 않음(epoch, 해상도, SatDINO·ScaleMAE의 GSD 인코딩, GASSL의 지리 정보). 비교는 쌍 안에서만. 입력 정규화 상수는 모델별 확인.

읽을 논문과 확인할 것:
- ImageNet-C: 손상 15종 중 REOBench와 겹치는 것, 강도 정의, "테스트 손상으로 학습 금지" 규칙. 같은 손상 함수를 ImageNet val과 AID에
  적용해 latent 지표(head 불필요)를 비교하는 기준점.
- GASSL: 체크포인트 4종, 입력 정규화, linear probe 프로토콜, ImageNet MoCo-v2와의 비교.
- SatDINO: 공개 가중치(ViT-S/B, patch 크기), GSD 인코딩이 추론 때 GSD 입력을 요구하는지(ScaleMAE와 같은 문제), ImageNet DINO와의 비교.

### 3.1 체크포인트 확인 결과 (2026-11-16) → `scripts/tools/download_backbones.sh`
- MAE ViT-L ImageNet, MoCo-v2 200ep ImageNet, DINO ViT-S/16·S/8·B/16 ImageNet: 공식 URL 200 확인.
- SatDINO: HF strakajk/satdino-{vit_small-16, vit_small-8, vit_base-16} (Apache-2.0). 키 'teacher'(백본), 'linear_head'(fMoW 63클래스).
  README에 정규화 값과 추론 시 GSD 입력 여부는 없음(코드 확인 필요). 학습에 GSD 예측 손실만 언급.
- GASSL: 공식 Zenodo가 이 서버에서 403(트래픽 차단). torchgeo HF 미러(cc-by-4.0)는 **MoCo-v2+Geo+TP만** 있음 → ImageNet MoCo-v2와의
  짝에 Geo+TP 차이가 섞임. 순수 MoCo-v2 fMoW는 다른 곳에서 받아 복사 필요. GASSL 정규화 = ImageNet 값, 224px. repo에 라이선스 파일 없음.

### 3.2 supervised ImageNet 백본을 어떻게 넣나 (2026-11-16 논의)
- 짝(주 비교)은 사전학습 방식을 맞춰야 하므로 SSL끼리: fMoW 쪽이 전부 SSL(SatMAE, GASSL, SatDINO)이라 ImageNet 쪽도 SSL(MAE, MoCo-v2, DINO).
  SatMAE vs supervised ImageNet ViT는 데이터와 학습 방식(라벨 유무)이 동시에 달라 원인 분리가 안 됨.
- supervised ImageNet ResNet50·ViT는 **참고 기준(anchor)** 으로 추가 제안: 선행 논문과 숫자를 잇는 연결점(EarthShift에서 가장 강건한 R50-imgnet,
  Beyond Accuracy의 Nat-supervised 그룹). fMoW supervised 짝은 없음.
- 확인: EarthShift는 "ResNet50, ViT-b-16 pre-trained on ImageNet"이라고만 쓰고 supervised인지 명시 안 함(코드 확인 필요). EarthShift의
  "fully-supervised"는 무작위 초기화 후 과제에 바로 학습한 모델을 뜻함. Beyond Accuracy는 Nat-supervised로 명시.

### 3.3 백본 구성 수정 (2026-11-16)
- DINO 짝은 DINOv1(facebookresearch/dino, ICCV 2021) 유지: SatDINO가 DINOv1 코드·목적함수 위에 만들어져 정확한 짝은 v1뿐.
- 최신 baseline 보강: **DINOv3 ViT-L/16 web(LVD-1689M) vs sat(SAT-493M)** 짝 추가. 둘 다 같은 구조, 각자 데이터의 ViT-7B에서 distill.
  Beyond Accuracy(DINOv3-Sat), EarthShift(DINOv3)와 직접 연결. sat는 보유, web은 HF gated(DINOv3 license) → 사용자 계정으로 승인·다운로드 필요.
- gassl_fmow(MoCo-v2+Geo+TP)는 주 짝에서 제외, 선택 항목으로 유지 (moco_fmow가 moco_in의 정확한 짝).
- 확인 필요: DINOv3-Sat의 CLS kNN(AID 49.7)이 patch 평균(72.7)보다 낮음. 로딩 누락 키 없음. DINOv2/v3 linear 평가는 CLS+patch 평균을
  같이 쓰는 관례라, 모델별 공식 linear 특징을 따를지 CLS로 통일할지 결정 필요.

## 진행 기록 2026-11-17

- stage 1 전부 완료: 분류 지표 34개(백본 17 × AID/RESISC45), 분할 drift 2개, RRSHID 17개, natural_vs_rs 17개 + 픽셀.
- 결과표: `experiments/2026-11-16_backbone_feats/RESULTS.md` (생성: `scripts/analysis/summarize_stage1.py`). 수치만, 해석 없음.
- RSPDBench 메모: `docs/reading/rspdbench.md`. 판단: 벤치마크째 채택하지 않고 연산자(P1·C1, 통제로 P2·P4)만 보조 표로.
- 우선순위: (1) stage-1 분석 → (2) 자연 shift(RESISC45↔UCM, Potsdam→Vaihingen) → (3) 선별 백본 8–9개로 신호·TTA 실험
  → (4) RSPDBench 방식 연산자, DOFA, CLIP↔RemoteCLIP → (5) 다중분광·Prithvi·Galileo·TerraMind는 한계로 기술.
- stage-1 분석: `experiments/2026-11-16_backbone_feats/ANALYSIS.md`. 추가 계산 `scripts/analysis/{labelfree_signals,eval_labelfree}.py`
  → `signals/`. 핵심: confidence 기반 정확도 추정(ATC)은 motion blur·data gaps·scale과 MAE-ImageNet·MoCo·ResNet에서 크게 과신
  (10pt 초과 12.8%), head 입력을 source 통계로 정렬하면 1.4%로 줄고, 정렬 전후 예측 flip 비율이 모든 백본에서 정확도를 따라감(0.90–1.00).
  다음: RESISC45↔UCM에서 재현, 정렬 조건(배치 크기·혼합 stream) 흔들기, 선행 연구(AETTA 등) 확인.
- 자연 shift RESISC45↔UCM(EarthShift 19클래스) 실행: `scripts/analysis/natural_shift.py`, 결과표 `experiments/2026-11-17_natural_shift/RESULTS.md`.
  지표 공부 노트 `docs/STUDY_stage1_metrics.md`. 문헌 조사(feature 통계를 쓰는 완화·정확도 추정 방법) 진행 중.
- 문헌 조사 완료: `docs/PRIOR_ART_feature_signals.md`. 평균 재중심화는 NEO(ICLR 2026)가 선점, flip 기반 추정은 Weighted Flips(ICML 2024)가
  가장 가까움(단, 학습 필요). 비어 있는 곳: frozen EO FM, 통계 정렬 후 ATC 회복 주장, 정렬 전후 flip, 공통 이동으로 보정 여부 결정.
- SAE + pruning 아이디어(Deletion Tuning에서 착안) 조사: `docs/PRIOR_ART_sae_pruning.md`. LLM 불필요, 아이디어 그대로의 선행 연구는 없음,
  그러나 단위 pruning이 손상 정확도를 올린다는 근거도 없음. SAE 전에 저장된 feature로 "shift 방향 사영 제거" oracle 실험을 먼저 할 것.
- shift 방향 제거 oracle(SAE 전 검증): `scripts/analysis/shift_direction_oracle.py`, 결과표 `experiments/2026-11-17_shift_oracle/RESULTS.md`.
  17개 백본 평균(AID-C): 다른 손상에서 구한 방향 제거는 k=4 +0.2pt, k=16 −2.6pt, k=64 −17.8pt(무작위 64개는 −0.7pt).
  같은 손상에서 구한 방향(상한)도 k=4 +5.8pt로 정렬(+5.2pt)과 같은 수준. 개별 차원 제거는 0. → head 입력에서의 pruning은 근거 없음.
- LPIPS 추가: `scripts/analysis/pixel_metrics.py`, 결과표 `experiments/2026-11-16_backbone_feats/pixel_metrics/RESULTS.md`.
  손상 종류 간 해로움 순위(기하 3종 제외 45개 조건, 17개 백본 평균 하락과의 Spearman): AID PSNR 0.34 / MS-SSIM 0.70 / LPIPS 0.88–0.90,
  RESISC45 0.52 / 0.83 / 0.82–0.88. 손상 안의 단계 순서는 네 지표가 같음(0.85–0.89). 기하 손상은 쌍 거리로 못 잼.
- fMoW contrastive ViT 조사(agent): **처음부터 fMoW-RGB로 contrastive 학습한 ViT의 공개 가중치는 없음.**
  SatMAE 논문의 MoCo-v3 baseline은 ViT-B, fMoW-Sentinel(10밴드, 96px)이고 공개된 적 없음(README·Zenodo 2건·HF 확인). Claude의 이전 기억
  ("ViT-L, fMoW-RGB")은 틀림. torchgeo의 fMoW 가중치는 GASSL ResNet-50과 Scale-MAE뿐. SatMIP(ECCV 2024)은 가중치 미공개.
  Cross-Scale MAE(NeurIPS 2023)는 InfoNCE+MAE 혼합이라 ImageNet 짝이 없음. MillionAID의 iBOT·MoCo-v3 ViT는 가중치 못 찾음.
  CLIP 계열(RemoteCLIP, GeoRSCLIP, SkyCLIP, Git-RSCLIP)은 전부 CLIP/SigLIP에서 이어 학습. ImageNet 쪽 MoCo-v3 ViT-S/B 300ep는 공식 공개.
- 겹치는 선행 연구 발견: arXiv 2611.13849 (GeoCV@WACV 2027), 같은 대응 쌍 설계로 clean 장면 분류 비교. 메모 `docs/reading/matched_pair_study.md`.

### 2026-11-17 백본·데이터셋 정리
- 지금까지 돌린 백본 17개, 데이터셋, 실험별 범위, 빈 곳을 `docs/SETTINGS.md`에 정리.
- MoCo-v3 ImageNet ResNet50/ViT-B 추가는 하지 않기로 함(사용자 판단: 지금 범위로 충분).

### 2026-11-17 과신의 원인 확인 (AID-C, 저장 feature, scratch 스크립트)
- 가설 "ResNet은 손상에서 feature 크기가 커져 logit이 부풀어 과신한다"는 **sup_in_r50에서 기각**. 표본별 feature norm을 clean 평균으로
  고정해도 과신(confidence − 정확도, 60조건 평균)은 +13.2 → +12.4. norm 비는 1.08, 최대 logit은 오히려 clean의 0.86배(ViT-B 0.85배와 같음).
  moco_in만 부분 효과(+18.3 → +13.0, norm 비 1.44).
- 과신은 3종 손상에 몰려 있음: sup_in_r50 data gaps +50.4 / scale +31.2 / motion blur +29.9, 나머지 9종 평균 +5.2.
  이 3종에서 예측이 한 클래스로 쏠림(가장 많이 예측된 클래스의 비율 52 / 32 / 31%, clean 약 5%).
  sup_in_vitb16은 같은 3종에서 +5.7 / +12.1 / +3.4, 쏠림 20 / 20 / 12%.
- 평균만 맞추는 정렬(align m)이 과신을 크게 줄임: sup_in_r50 +13.2 → +5.3, mae_in +16.4 → +3.7, moco_in +18.3 → +7.9
  (표준편차까지 맞추면 moco_in만 +4.2로 더 줄어듦).
- 정리: 과신은 구조 고유의 calibration 성질이라기보다 "특정 손상에서 feature가 집단으로 한 클래스 쪽으로 이동 → 예측 쏠림"의 증상.
  ResNet이 왜 그 3종에서 쏠리고 supervised ViT는 덜 쏠리는지는 미확인.

### 2026-11-17 flip을 표본 단위 필터로 쓸 수 있는가 (AID-C 120,000장, 17개 백본, scratch 스크립트)
- 필터 후 남긴 표본의 오류율(%): "flip 안 한 표본만" 3.6–14.7 (남김 75–98%), SAR 엔트로피 필터(E < 0.4 ln C) 3.9–27.4 (남김 80–98%).
  과신 백본에서 SAR 필터는 거의 못 거름: mae_in 26.0, moco_in 27.4, moco_fmow 23.3, sup_in_r50 24.7 (전체 오류율 27.3 / 28.7 / 25.6 / 28.9).
  같은 백본에서 flip 필터는 7.0 / 7.8 / 9.8 / 13.0. calibration이 좋은 백본(satdino_s16, sup_in_vitl16)에서는 SAR이 같거나 약간 나음.
- 오류 탐지 AUROC: flip(이진) 0.66–0.89, confidence 0.78–0.95, 둘의 곱(원래 예측 클래스의 정렬 전 확률 × 정렬 후 확률) 0.917–0.968.
  곱은 17개 전부에서 0.92 이상 → confidence와 flip은 서로 다른 정보를 가짐.
- sup_in_r50의 손상별 frac_common은 data gaps 0.381 / motion blur 0.339 / scale 0.320이 상위 3개(나머지 0.02–0.20) = 과신이 몰린 3종과 일치.
- 주의: 합성 손상, 단일 손상 stream, 배치 64, 선형 head 기준의 오류 탐지 결과이고 TTA에 넣어 본 것은 아님. SAR 임계값은 원 논문 기본값.

### 2026-11-17 feature 이동을 head 부분공간·logit에서 분해 (AID-C, 17개 백본, scratch 스크립트)
- 이동 에너지 중 head가 읽는 29차원 부분공간에 들어가는 몫: 공통 9–16%, 표본별 3–15% (무작위 기준 1.4–7.6%).
- logit에서: 공통 이동의 클래스 간 표준편차 0.40–2.62, 표본별 잡음 표준편차 0.82–2.18 (clean margin 5.2–8.5).
- 백본 간 과신(signed ECE)과의 Spearman: 공통 logit 이동 0.91, 표본별 logit 잡음 0.92, CKA −0.14, frac_common 0.21.
- 사용자 노트 검토 내용은 `docs/STUDY_stage1_metrics.md` §7에 정리.

### 2026-11-17 flip 필터 TTA 실행 시작, Entropy Enigma 독해
- `scripts/analysis/feat_tta.py`: 저장 feature 위에서 head 입력 affine만 EM으로 학습(ViT에서는 마지막 LayerNorm만 학습하는 TENT와 같음).
  입력 raw/정렬 × 필터 tent/entf/flipf/bothf × lr 1e-4…1e-1, 17개 백본 × AID-C/RESISC45-C, 배치 64 → 16. CPU, tmux `fliptta`.
  결과: `experiments/2026-11-17_flip_tta/bs{64,16}/`, 표는 `scripts/analysis/summarize_flip_tta.py`.
- Entropy Enigma(ICML 2024) PDF 텍스트 독해 → `docs/reading/entropy_enigma.md`. Weighted Flips = EM 적응 전후 flip(초기 confidence 가중) → 2차식으로
  정확도 추정. "flip이 정확도를 따른다"는 관찰은 선행. 우리 쪽에 남는 것: 학습 없는 flip, frozen EO FM, 표본 단위 필터.
  해야 할 것: WF와 COT를 baseline으로 직접 비교.

### 2026-11-17 밤샘 실행: 17개 백본 전체 통과 TTA (`scripts/tta/cls_tta_fm.py`, tmux `fmtta`)
- 이미지를 백본에 통과시키고 norm 층 affine을 학습(상위 25% 블록과 마지막 norm 제외 = SAR/DeYO 규칙). head는 stage-1 선형 head.
  방법: none, none_al, tent, entf, flipf, bothf, tent_al, flipf_al, sar, eata (+ BN 백본만 tent_bnstat).
  순서: AID-C 프로토콜 lr → AID-C lr×10 (tent·필터 변형) → RESISC45-C. 결과: `experiments/2026-11-17_fm_tta/<ds>/<model>/<method>.json` (이어 돌리기 가능).
- 축소 시험(3 stream)에서 확인: none이 stage-1 정확도와 일치(dino_in_s16 95.6, satdino_s16 96.7, sup_in_vitb16 92.7, satmae 92.7, dinov3_web 97.8, sup_in_r50 90.8).
- **BN 백본**: 모든 BN을 배치 통계로 바꾸는 교과서 TENT는 sup_in_r50의 AID clean을 90.8 → 62.1로 떨어뜨림(running 통계가 ImageNet 것이고 head가
  그 feature로 학습됐기 때문). 그래서 BN도 running 통계 유지 + affine만 학습을 기본으로 하고, 배치 통계판은 tent_bnstat으로 따로 남김.
- **미해결**: dinov3_web_l16에서 EATA가 clean 6.2%로 붕괴(3 stream 시험). 원인 미확인 → DINOv3의 EATA 수치는 확인 전까지 쓰지 말 것.
- 서버 CPU 제한이 12코어(cgroup cpu.max)라 CPU 작업이 DataLoader를 굶김 → feature 수준 실행(`feat_tta.py`)을 `--device cuda`로 옮김.

### 2026-11-17 flip 필터 TTA (feature 수준) 결과 — `experiments/2026-11-17_flip_tta/bs{64,16}/RESULTS.md`
- 17개 백본 평균 손상 정확도, 배치 64 (AID-C / RESISC45-C): none 81.8 / 75.2, align_ms 87.3 / 81.2,
  raw tent lr 0.01 78.7 / 63.7, raw entf lr 0.01 79.4 / 66.2, raw flipf lr 0.001 82.4 / 76.3 (17 / 17개 백본에서 none보다 높음),
  raw flipf lr 0.01 84.0 / 74.3 (16 / 7개), al tent lr 0.001 87.3 / 81.0. lr 0.1은 전부 붕괴.
- 배치 16: align_ms 85.8 / 79.3 (이득 +4.0 / +4.0, clean 비용 −1.4 / −1.9), raw tent lr 0.01 74.1 / 54.5, raw flipf lr 0.001 83.1 / 76.8.
- 정리: head 입력 affine만 학습하는 조건에서는 (1) EM과 엔트로피 필터는 해롭다, (2) flip 필터는 손해를 막고 소폭 이득, (3) 학습 없는
  정렬을 넘는 방법은 없다, (4) 배치가 작아지면 정렬 이득이 줄고 clean 비용이 생긴다.

### 2026-11-18 밤샘 TTA 결과(AID-C 완료), head 부분공간 분해를 저장소로 옮김
- `experiments/2026-11-17_fm_tta/RESULTS.md` (`scripts/tta/summarize_fm_tta.py`). AID-C 17개 백본 평균 손상 정확도:
  none 81.8, none_al(정렬만) 87.3, tent 81.4, entf 81.5, flipf 84.3, bothf 84.2, sar 80.6, tent_al 88.1, flipf_al 87.9, tent_bnstat 68.0(BN 4개).
  lr×10: tent 76.1, entf 76.4, flipf 83.8, bothf 83.8, tent_al 88.6, flipf_al 86.9.
  → 백본 안쪽을 학습해도 EM·엔트로피 필터·SAR은 평균적으로 손해, flip 필터는 +2.5pt, 정렬 위의 TENT가 정렬만보다 +0.8(lr×10: +1.3, 16/17 백본).
- **EATA 붕괴**: mae_in, dino_in_b16, dinov3_web/sat, sup_in_vitl16에서 약 6%. 원인 미확인(구현 또는 Fisher 스케일 의심). EATA 열은 사용 금지.
- RESISC45-C는 진행 중(작은 백본부터).
- `scripts/analysis/head_subspace.py` → `experiments/2026-11-18_head_subspace/` (어제의 scratch 측정을 정식화, AID + RESISC45).
  이동 에너지 4분할 평균: head 부분공간 안 = 공통 1.2–6.5% + 표본별 2.6–13.0%, 밖 = 83–94%. 과신·정확도 하락과의 백본 간 Spearman:
  logit 공통 이동 0.90 / 0.82–0.92, logit 표본별 잡음 0.91–0.94 / 0.91–0.93, CKA −0.12~−0.41, frac_common 0.11–0.39 (두 데이터셋).

### 2026-11-19 전체 통과 TTA 완료 (AID-C, AID-C lr×10, RESISC45-C; 11-19 11:19 종료)
- RESISC45-C 17개 백본 평균 손상 정확도: none 75.2, none_al 81.1, tent 71.2, entf 71.8, sar 71.8, flipf 79.1, bothf 79.0,
  tent_al 83.0, flipf_al 82.4, tent_bnstat 61.2(BN 4개). tent_al은 17개 전부에서 none_al보다 높음(+1.9pt).
- 두 데이터셋 공통: (1) EM·엔트로피 필터·SAR은 평균 손해(과신 백본에서 붕괴: moco_in RESISC45-C tent 27.4, sar 20.3),
  (2) flip 필터는 17/17 이득(+2.5 / +3.9pt), (3) 정렬 + TENT가 가장 높음(정렬만 대비 +0.8 / +1.9pt), (4) 정렬 뒤에는 flip 필터가 보탬이 안 됨.
- EATA 붕괴 지속(5개 백본에서 약 3%, 원인 미확인) → EATA 열 사용 금지. RESISC45-C는 lr×10 미실행.

### 2026-11-19 EATA 붕괴 원인과 수정
- 원인: 공식 EATA는 모든 norm 층을 학습하고 Fisher penalty(α = 2000)를 건다. 5개 백본(mae_in, dino_in_b16, dinov3_web/sat, sup_in_vitl16)에서
  마지막 norm의 원소 1~44개의 Fisher가 12.6~5,050 → SGD(momentum 0.9)에서 2차 penalty가 안정하려면 lr·2αF < 3.8, 즉 F < 0.95(lr 0.001)인데
  이를 크게 넘어 발산. 붕괴한 5개 = 한계를 넘는 5개와 정확히 일치. 그 원소들은 clean-train 표준편차가 거의 0인 feature 채널
  (최소 2e-6)로 이어지고, head 입력 표준화가 그 값으로 나누면서 gradient가 커진다.
- 수정: `eata` = TENT·SAR과 같은 파라미터 집합(마지막 norm·상위 25% 블록 제외), 이 집합의 Fisher 최대값은 5e-4 이하. 3-stream 시험에서
  clean 정확도 유지(mae_in 94.3, dino_in_b16 96.1, sup_in_vitl16 94.9, dinov3_web 97.8). 기존 결과는 `eata_allnorm.json`으로 보존.
  17개 백본 × AID-C/RESISC45-C 재실행 중(tmux `eata`, `run_eata.log`).
- 분산이 거의 0인 채널(중앙값의 1/100 미만): dino_in_b16 5개, dinov3_web 2개, dinov3_sat 5개, sup_in_vitl16 34개, 나머지 0개.
  이 채널을 0으로 두어도 stage-1 정확도 변화는 0.3pt 이하, 평균 이동에서 차지하는 몫 3.6% 이하 → 기존 분석에는 영향 없음.
- flip 계산의 교차 확인: 따로 짠 두 코드(labelfree_signals.py / feat_tta.py의 lr 1e-4 실행)의 AID-C 평균 flip이 17개 백본에서
  0.2pt 이내로 일치. cls_tta_fm.py의 flipf는 백본이 학습되면서 flip이 줄어들어(예: mae_in 24.4 → 17.7) 같은 값이 아니며 교차 확인으로 쓸 수 없음.
  외부 기준과의 검증은 아직 없음: Weighted Flips(Entropy Enigma)는 구현하지 않았고, 그 논문 수치 재현도 하지 않음.

### 2026-11-19 EATA 재실행 완료 (파라미터 집합 = TENT·SAR과 동일)
- 17개 백본 평균 손상 정확도 (AID-C / RESISC45-C): eata 81.9 / 71.1 (none 81.8 / 75.2, tent 81.4 / 71.2, sar 80.6 / 71.8, eata_allnorm 58.6 / 48.0).
  none보다 높은 백본 수 12 / 12. 수치 발산(약 3~6%로 붕괴)은 사라짐.
- 남은 하락은 발산이 아니라 엔트로피 최소화 자체의 실패: RESISC45에서 moco_in clean 88.0 -> 16.0, sup_in_r50, mae_in 등
  과신 백본에서 TENT·SAR과 같은 양상(RESISC45-C: moco_in 15.7, sup_in_r50 44.2, mae_in 50.8).
- 표: `experiments/2026-11-17_fm_tta/RESULTS.md` (eata, eata_allnorm 열 모두 수록).

### 2026-11-19 Weighted Flips 재현 완료 (ImageNet-C, ResNet-50, 공식 코드)
- 공식 코드(oripress/EntropyEnigma @ 04ac397, `third_party/entropy_enigma/`)의 함수·계수 그대로, seed 42 한 번. 96개 split
  (val + IN-C 15종×5 + holdout 4종×5). `scripts/verify/weighted_flips_repro.py`, 표 `experiments/2026-11-19_weighted_flips_repro/RESULTS.md`.
- IN-C 75개 split 추정 오차 MAE (우리 / 논문 Table 1): WF 5.49 / 4.79, AC 10.06 / 10.06, DoC 6.62 / 6.61,
  ATC(negative entropy) 7.45 / 7.44. 비교 기준 3개가 소수 둘째 자리까지 일치 → 데이터·모델·평가 파이프라인은 논문과 같음.
  WF는 0.70pt 높음(seed 1개, 처음 1,000장의 표본 차이일 수 있으나 확인하지 않음).
- 논문에 없는 값: ATC(max-softmax) 3.68, 가중 없는 flip 1차식 5.28 / 2차식 5.51 → 이 실행에서는 confidence 가중의 이득이 보이지 않음.
- f 다시 맞추기(val + holdout 21점): 0.000501x² − 0.3417x + 74.14 (논문 0.000356x² − 0.3214x + 75.66), IN-C MAE 5.31.
- 손상별 WF 오차가 큰 곳: pixelate 11.6, fog 10.1, defocus blur 9.5, contrast 8.7, elastic 8.4.
- 이것은 논문 방법의 재현이다. 우리 flip(정렬 전후 예측 변화)의 검증은 아니며, 두 flip을 같은 조건에서 비교하는 일은 남아 있음.

### 2026-11-19 DSBR 읽기·구현, feature 단계 결과
- 논문: Nielen et al., arXiv 2606.02339. 노트 `docs/reading/dsbr.md`. 공개 코드를 찾지 못해 부록 A Algorithm 1대로 구현
  (`cls_tta_fm.py`의 dsbr / dsbr_a99 / dsbr_a0 / dsbr_al, `feat_tta.py`의 dsbr / dsbr99). 논문 수치와의 대조는
  `scripts/verify/dsbr_imagenet_c.py`(ViT-B/16, ImageNet-C 강도 5)로 진행 중.
- feature 단계(head 입력 affine만 학습, Adam, 배치 64), 17개 백본 평균 손상 정확도 (AID-C / RESISC45-C):
  none 81.8 / 75.2, align_ms 87.3 / 81.2, flipf lr 0.001 82.4 / 76.3,
  tent lr 0.001 81.7 / 74.0, lr 0.01 78.7 / 63.7,
  dsbr(α 0.9) lr 0.001 82.2 / 76.3, lr 0.01 83.7 / 78.8 (두 lr 모두 17/17에서 none보다 높음),
  dsbr(α 0.99) lr 0.001 81.8 / 75.4, lr 0.01 80.5 / 73.7,
  정렬 입력: tent lr 0.01 86.3 / 74.8, dsbr lr 0.01 87.3 / 81.3, dsbr lr 0.001 87.4 / 81.3.
- 읽기: (1) TENT가 무너지는 lr 0.01에서 DSBR은 무너지지 않고 none보다 +1.9 / +3.6pt. (2) α 0.99는 우리 stream 길이(31~99 step)에서
  P가 1/C에서 거의 못 벗어나 TENT에 가까워짐. (3) 정렬(align_ms)에는 못 미침(−3.6 / −2.4pt). (4) 정렬 뒤에는 TENT의 lr 0.01 하락
  (74.8)을 막아 주지만 정렬만 한 것 대비 이득은 0.1pt.
- 표: `experiments/2026-11-19_dsbr/RESULTS.md`. 백본 전체 통과 실행은 tmux `dsbr`(`scripts/archive/tta/run_dsbr.sh`, `run.log`).

### 2026-11-19 DSBR: 사용자 메모 검토, 예측 쏠림 측정
- 사용자 메모(Problem / Solution / 부록 E·F / 실험 / future work)와 질문 5개를 논문과 대조해 `docs/reading/dsbr.md` 4~7절에 정리.
  고친 점 2개: (1) SAR이 엔트로피를 쓴 이유는 "label이 없어 gradient를 못 구해서"가 아니라 gradient norm 문턱값을 모델·데이터에
  무관하게 정하기 어려워서, (2) 경계 근처 샘플은 적응 전에 shift로 이미 넘어온 것. Fig. 5·6은 페이지를 이미지로 렌더링해 확인.
- 예측 쏠림 측정(`scripts/analysis/pred_bias.py`, `experiments/2026-11-19_dsbr/pred_bias.md`), tv = 예측·실제 클래스 비율의 차이:
  손상 평균 tv none 12.2 / 14.9% → 정렬 후 4.5 / 5.1% (clean 1.3 / 1.6%). 쏠림 상위 1/4 stream에서 none 대비
  tent −14.1 / −31.9, dsbr +4.5 / +9.8, align +17.9 / +18.7pt. 하위 절반에서는 모든 방법이 ±0.9pt 이내.
  tv와의 Spearman: tent 이득 −0.28 / −0.62, dsbr 이득 +0.63 / +0.87, align 이득 +0.93 / +0.94.
- 읽기: 쏠림이 클수록 TENT가 잃는다는 논문 주장은 우리 17개 백본에서도 같은 방향. 우리 쏠림의 약 2/3는 공통 이동(정렬로 제거)이고
  DSBR은 그 구간에서 정렬의 1/4~1/2만 회복. tv는 오류율과 기계적으로 묶여 있어 인과는 주장 불가.
- 백본 전체 통과(AID-C, 17개 백본, 2026-11-19 20:57 완료). 프로토콜 lr: none 81.8, none_al 87.3, tent 81.4, flipf 84.3,
  dsbr 83.6 (none보다 16/17), tent_al 88.1. lr×10: tent_x10 76.1, flipf_x10 83.8, dsbr_x10 80.4 (tent보다 17/17, none보다 13/17),
  tent_al_x10 88.6. moco_in은 dsbr_x10에서도 23.1로 붕괴(tent 15.9, flipf 54.0), sup_in_r50 68.6, mae_in 71.6.
  읽기: 백본 전체에서는 dsbr가 tent보다 낫지만 flip 필터보다 낮고, 큰 lr에서 붕괴를 막지 못함. 표는 `experiments/2026-11-19_dsbr/RESULTS.md` 2절.
- ImageNet-C 구현 확인(강도 5, ViT-B/16, 배치 64, 15종 평균, 21:25 완료): 무적응 51.6(augreg) / 55.5(augreg2), 논문 39.8 →
  체크포인트 불일치로 절대값 비교 불가. augreg에서 Tent(SGD) 59.3, DSBR α 0.9(Adam) 66.2 (논문 51.9 / 64.4). DSBR이 15종 전부에서
  Tent·무적응보다 높음. 남은 것: DSBR α 0.99, Tent(Adam)는 파이프라인 끝에서 실행.

### 2026-11-20 DSBR 파이프라인 종료 (06:22, 실패 없음)
- AID-C 변형: dsbr_a99 82.4 (lr×10 78.1), dsbr_al 88.1 (lr×10 88.8) vs tent_al 88.1 (88.6).
- RESISC45-C 프로토콜 lr: none 75.2, none_al 81.1, tent 71.2, flipf 79.1, dsbr 77.7 (none보다 16/17, moco_in 56.7 < 64.1),
  tent_al 83.0, dsbr_al 83.1.
- ImageNet-C(augreg): Tent(Adam) 59.9 (snow 27.0, frost 20.4로 붕괴, 나머지 13종 평균 65.5), DSBR α 0.9 66.2 (13종 65.9), α 0.99 66.4.
  DSBR − Tent(SGD) 차이의 대부분은 optimizer, DSBR 손실의 몫은 2종의 붕괴 방지.
- 읽기: DSBR은 TENT의 붕괴를 줄이는 안전장치로는 일관되게 작동(전 설정에서 tent 이상)하지만, 우리 설정에서 flip 필터보다 낮고
  정렬 뒤에는 추가 이득이 없음. 표는 `experiments/2026-11-19_dsbr/RESULTS.md`.
- timm augreg2 ViT-B/16 + ImageNet mean/std 입력(체크포인트는 0.5/0.5로 학습)의 무적응 평균이 논문과 같은 39.8. 단 EATA-C의
  source도 평균 39.84인데 손상별 값이 전혀 달라(gauss 12.9 vs 35.2) 논문 설정의 확정 근거는 아님. 이 설정에서
  Tent(SGD) 49.7, Tent(Adam) 47.0(snow 3.2, frost 7.2로 붕괴), DSBR α 0.9 61.2, α 0.99 61.8 (논문 51.9 / – / 64.4 / 64.9).
  이득의 크기는 논문과 비슷(DSBR − Tent +11.5 vs +12.5), 절대값은 2~3pt 낮음. seed 1개. 구현 확인으로 충분하다고 판단.
  `scripts/archive/verify/run_dsbr_norm.sh`, `run_dsbr_match.sh`.

### 2026-11-23 head 학습 방식 caveat 확인: 표준화 없이 학습한 head에서의 정렬
- 질문: 정렬(head 입력을 test 배치 통계로 표준화)의 이득이 "clean train 통계로 표준화한 입력으로 head를 학습"한 우리 방식 덕인가?
- 실험(`scripts/analysis/head_nostd.py`, `experiments/2026-11-23_head_nostd/RESULTS.md`): 같은 절차(로지스틱 회귀, L-BFGS, 검증 NLL로 λ)로
  표준화 없이 h로 head를 학습하고, 일반식 h′ = (h − μ_test) / σ_test · σ_src + μ_src로 정렬. 17개 백본, 배치 64, 학습 없음.
- 결과(손상 평균 정확도, none → 정렬): AID-C std head 81.8 → 87.3 (+5.5), raw head 81.6 → 86.9 (+5.4);
  RESISC45-C std 75.2 → 81.2 (+5.9), raw 74.9 → 80.7 (+5.8). raw head에서도 17/17 백본에서 이득. clean은 94.8 / 94.6, 90.6 / 90.4.
- 읽기: 정렬의 이득은 head 학습 방식에 거의 의존하지 않음(평균 차이 0.1pt). 백본별로는 moco_in, moco_fmow에서 raw head의 none이
  2.5 / 1.4pt 높아 이득이 작지만 정렬 후 정확도는 같은 수준.
- 주의: 쓰는 식은 head에 따라 다르다. 규칙은 "head 입력의 채널별 통계를 head가 학습 때 본 입력의 통계로 맞춘다"이고,
  raw head는 (μ_src, σ_src), std head는 (0, 1)을 넣는다. 따라서 필요한 정보는 h의 src 통계가 아니라 head 학습 입력의 통계이며,
  head와 함께 저장돼야 한다(BN running 통계와 같은 처지). 이 실험이 보인 것은 식을 맞게 쓰면 이득 크기가 head 학습 방식에 거의 무관하다는 것.

### 2026-11-23 자연 shift 해석 (RESISC45 ↔ UC Merced, 17개 백본, `experiments/2026-11-17_natural_shift/RESULTS.md`)
수치는 OOD 전체(target 모든 이미지: UCM 1,900장 / RESISC45 13,300장) 기준, R45→UCM / UCM→R45 순. 실행은 11-17, 해석만 새로 함.
- 하락: ID 대비 평균 −15.3 / −30.9pt (최소 4.0 / 11.6, 최대 40.3 / 45.5). 합성 손상 평균 하락(AID-C 13.0, RESISC45-C 15.4)보다 크다.
- confidence는 과신: signed ECE 평균 +0.072 / +0.145, 정확도 추정의 평균 절대 오차(17개 백본 평균, 모델 MAE와 무관) AC 7.5 / 14.5, DoC 8.0 / 14.8, ATC 6.0 / 7.6pt (거의 전부 과대 추정).
- flip은 백본 간 정확도 순위를 따른다: flip과 정확도의 Spearman −0.99 / −0.91 (confidence 평균과 정확도는 +0.89 / +0.81). flip 평균 11.6 / 12.8% (ID 2.1 / 1.0%).
- flip 추정(기울기 k는 합성 손상에서 맞춘 값 그대로): 평균 절대 오차 1.7 / 5.2pt, 10pt 넘는 오차 0 / 1개 (ATC는 4 / 5개).
  UCM→R45에서는 정렬 후에도 과신이 남아(signed ECE +0.145 → +0.128, 정렬 후 ATC 오차 5.9) 추정이 양쪽으로 흩어진다(satdino_b16 −12.8, mae_in +8.8).
- **정렬 이득은 작다**: +2.2 / +2.1pt (합성 +5.5 / +5.9). 0.5pt 넘게 오른 백본 10 / 15개, 내린 백본 3 / 0개. 큰 이득은 MIM에 몰림
  (mae_in +7.9 / +4.9, scalemae +7.8 / +6.3, satmae +5.6 / +0.3). 평균만 맞춰도 같다(+2.3 / +2.1). 정렬 뒤에도 하락 13.1 / 28.8pt가 남는다.
  frac_mean(이동 중 공통 평균 이동의 몫)이 0.03~0.27로 작다 → 자연 shift의 대부분은 공통 이동이 아니라서 통계 정렬로 못 되돌린다(해석).
- 위성 사전학습의 효과는 방식마다 반대: DINO 쌍 +1.1~+3.8 / +6.2~+9.4, GASSL +8.0 / +13.5 (MoCo-ImageNet 대비)로 이득이지만
  SatMAE −14.1 / −0.9, ScaleMAE −18.4 / −3.4 (MAE-ImageNet 대비), DINOv3 sat −29.8 / −28.2 (web 대비)로 손해. 원인은 확인하지 않음.
- 읽기: (1) "confidence는 과신, flip은 정확도를 따름"은 자연 shift에서도 선다. (2) "정렬로 회복"은 자연 shift에서 약하다. 정렬은 공통 이동이 큰
  합성 손상용 처방이고, 자연 shift에서는 신호(flip, 정렬 후 ATC)로서의 가치가 더 크다. (3) 한계: 데이터 쌍 1개, 19클래스, seed 1개,
  백본 17개가 표본(백본 간 순위 상관)이라 조건 간 추정이 아니다. 두 데이터셋의 클래스 정의·GSD 차이가 섞여 있다.

### 2026-11-24 Potsdam 분할: 신호 측정과 11월 방법 (중간, `experiments/2026-11-23_seg_signals/RESULTS.md`)
SatMAE B / ScaleMAE B, clean + 손상 8종 × 5단계, 배치 8, stream마다 초기화. 수치는 SatMAE / ScaleMAE 순. tmux `segnov`(11-23 17:36 시작).
완료: 신호 측정, flipf, dsbr, dsbr_src, flipf_bnbatch. 진행 중: dsbr_src_bnbatch, dsbr_bnbatch. 빠진 손상 4종(rotate, scale, translate, data gaps)은 미생성.
- 학습 없는 정렬(손상 평균 mIoU): none 50.6 / 51.3, bn(decoder BN을 test 통계로) 52.5 / 54.4, isa(입력 정렬) 53.8 / 53.9, isa_bn 55.1 / 56.3.
- flip(none과 bn 예측이 다른 픽셀 비율)은 손상 stream 40개에서 픽셀 정확도를 따른다: Spearman −0.94 / −0.97 (평균 confidence +0.86 / +0.94).
  isa로 잰 flip은 약하다(−0.52 / −0.69): 입력 정렬은 radiometric 손상에서만 예측을 바꾼다.
- 과신은 분류보다 작다: confidence − 픽셀 정확도 평균 +3.3 / +4.8pt (JPEG +10.9 / +11.2, ScaleMAE noise +9.3, S&P +8.7).
- 픽셀 필터는 분류와 반대: 엔트로피 필터(< 0.4 ln 5)는 47 / 54%를 남기고 그 정확도 89.2 / 90.6%, flip 필터는 88%를 남기고 76.2 / 77.7% (전체 71.5 / 72.4).
  flip한 픽셀의 정확도는 36%. 남기는 비율이 달라 직접 비교는 아니다.
- 예측 쏠림 tv: none 11.4 / 12.3% → bn 5.9 / 5.6% (clean 4.1 / 3.6). none의 tv와 픽셀 정확도의 Spearman −0.89 / −0.89.
- 학습하는 방법(none 대비 손상 평균 mIoU): tent −1.96 / −1.19, sar −1.14 / −0.71, flipf −0.97 / −0.63, dsbr +0.85 / +0.56, dsbr_src +0.02 / +0.09,
  bnadapt +1.87 / +3.08, tent_bnbatch +2.73 / +3.79, flipf_bnbatch +2.60 / +3.67.
- 읽기: (1) "flip이 정확도를 따른다"는 분할에서도 선다. (2) BN 고정 상태에서 양수인 학습 방법은 DSBR뿐이고, flip 필터는 Tent의 손해를 절반으로 줄이지만
  이득은 못 낸다(분류와 다름). (3) source 비율을 목표로 한 DSBR 변형은 균등 목표보다 나쁘다. (4) 이득은 여전히 통계 정렬에서 온다
  (flipf_bnbatch ≈ tent_bnbatch ≈ bnadapt + 0.7~0.9). seed 1개, lr은 10월과 같은 값(7.5e-6) 하나.


### 2026-11-24 Potsdam 분할: 손상 8종 실행 완료 (`experiments/2026-11-23_seg_signals/RESULTS.md`)
tmux `segnov` 종료(16:46, 실패 없음). 추가된 것: dsbr_bnbatch +2.84 / +3.88, dsbr_src_bnbatch +2.80 / +3.90 (none 대비 손상 평균 mIoU, SatMAE / ScaleMAE).
- BN 정렬 위에서는 손실 종류가 거의 차이를 만들지 않는다: tent / flipf / dsbr / dsbr_src가 +2.60 ~ +2.84 / +3.67 ~ +3.90에 모인다.
  정렬만(bnadapt) +1.87 / +3.08이므로 학습이 더하는 몫은 1pt 미만.
- DSBR의 효과는 BN을 고정했을 때만 뚜렷하다(tent −1.96 / −1.19, dsbr +0.85 / +0.56). 분류(AID-C 정렬 위 tent 88.1, dsbr 88.1)와 같은 모양.
- 빠진 손상 4종(rotate, scale, translate, data gaps)은 2026-11-24 12:42에 생성 완료(각 10,080장, 기하 3종은 라벨도 생성, translate 방향은 배포 라벨과 전부 일치).
  이 4종의 실행은 tmux `seggeo`에서 진행 중.

### 2026-11-24 Entropy Enigma를 우리 백본에서 확인: COT, 군집 지표, EM 두 단계 (`experiments/2026-11-24_cluster_cot/`)
사용자가 Entropy Enigma를 읽고 낸 질문(WF 대신 COT, Silhouette / Shift distance를 우리도 재야 하나, AID처럼 비슷한 클래스가 많은 데서도 성립하나)에 대한 측정.
스크립트: `scripts/analysis/cluster_cot.py`, `cluster_subspace.py`, `em_phases.py`, `plot_feature_dist.py`. 17개 백본, head 입력 z, 학습 없음(EM 추적 제외).
- **COT baseline** (공식 코드 확인: exact OT, 비용 1 − p[label], source 라벨 비율): 2,040개 stream 평균 절대 오차 AC 7.6, DoC 7.0, ATC 5.2,
  COT 3.0, COT + temperature 2.3, flip 추정 2.4pt. 10pt 넘는 오차는 COT 0.7%, COT_ts 0.4%, flip 추정 4.6%, ATC 12.8%. 최대 13.1 / 13.5 / 26.1 / 84.4.
  중앙값은 flip 추정이 낮다(1.2 대 2.0~2.6). → "정확도 추정기로서 flip이 가장 좋다"는 주장은 할 수 없다. COT가 평균은 같고 꼬리는 짧다.
  COT에 유리한 조건: 손상 stream의 클래스 비율이 source와 정확히 같고 stream 전체(2,000 / 6,300장)를 쓴다. 배치 단위·치우친 stream은 미검증.
- **Silhouette(k-means, 논문 정의)은 정확도를 따르지 않는다**: 60개 stream에서의 Spearman 평균 AID +0.21 (−0.83 ~ +0.87), RESISC45 +0.28 (−0.87 ~ +0.84).
  군집이 원래 뚜렷한 백본(SatDINO, DINOv3 web)에서만 +0.6 이상. satmae, moco_in은 손상될수록 Silhouette이 오른다(feature가 몇 덩어리로 몰림).
  예측 기준 Silhouette +0.83 / +0.87, Shift distance −0.90 / −0.93, flip −0.96 / −0.97, confidence +0.96 / +0.91.
- **클래스는 갈라지지만 덩어리는 아니다** (clean test, 17개 평균 AID / RESISC45): 선형 head 94.8 / 90.6, 5-NN 88.5 / 83.3, k-means 59.1 / 51.3,
  k-means(PCA C−1차원) 58.6 / 50.4, k-means(클래스 평균 부분공간) 72.0 / 62.8. 전체 분산 중 클래스 간 분산 0.27 / 0.27.
  MAE 계열이 가장 낮고(k-means 30~56) SatDINO, DINOv3 web이 가장 높다(68~81). 그림: `docs/figs/2026-11-24_feature_dist/`.
- **EM 두 단계** (AID-C, head 입력 affine만 Adam lr 1e-3으로 2,000 step, stream 반복): tent 81.8 → 최고 83.0(중앙값 약 230 step) → 70.5, 끝에서 예측되는
  클래스 수 30 → 23.5 (satmae 16.5). DSBR 81.8 → 88.0, 예측 클래스 29.8, 실패 단계 없음. Shift distance 증가는 tent +5.8, DSBR +5.3으로 비슷하다
  → 우리 설정에서 실패를 가르는 것은 source 클래스 평균까지의 거리가 아니라 예측 클래스 수. k-means Silhouette 변화는 둘 다 +0.001.
  논문의 두 상관은 방향은 같고 작다(1단계 Δsil–Δacc +0.38 대 논문 +0.70, 2단계 Δshift–Δacc −0.59 대 −0.79).
  주의: tent의 1단계 상승 +1.2pt는 checkpoint 9개 중 최고점을 고른 값이라 부풀려져 있다. 거리 증가에는 γ 확대가 섞여 있다. online 한 번 통과가 아니다.
- **사전학습 augmentation 표** (`docs/SETTINGS.md` §1): Gaussian blur augmentation이 있는 11개 백본은 Gaussian blur 손상 하락이
  AID 0.4~3.9 / RESISC45 1.0~6.2pt, 없는 6개(MAE 3, supervised 3)는 4.5~25.1 / 6.5~33.2pt. motion blur로는 넘어가지 않는다. scale 손상은 crop 범위가 같은
  쌍 안에서 fMoW 쪽이 5쌍 모두 높다(DINOv3만 반대). DINOv3 web / sat은 논문상 hyper-parameter가 같아 차이는 데이터에서 온다.
- 읽기: (1) Entropy Enigma의 군집 설명("정확도가 높으면 잘 군집")은 end-to-end 분류기의 것이고 frozen 백본에서는 군집형 feature를 내는 계열에만 맞는다.
  원인 설명은 군집보다 source 통계에서의 이동·예측 쏠림으로 쓰는 편이 안전하다. (2) 분할에는 현상(EM이 해롭다, 쏠림을 누르면 낫다, flip이 정확도를 따른다)만
  주장하고 geometry 설명은 분류에 한정한다(사용자 지적: neural collapse류 구조는 분할에서 약하다). (3) 추정기 비교에는 COT를 baseline으로 넣어야 한다.

### 2026-11-24 SenseBench 읽기 (`docs/reading/sensebench.md`)
사용자가 papers에 추가(NeurIPS 2026 E&D). VLM이 RS 열화를 알아보는지 재는 벤치마크(22종, 10,815문항)이고 task 정확도는 재지 않는다.
- 합성 엔진은 공개되지 않음(저장소는 추론·평가 코드만, 논문에 파라미터 없음) → 분류 체계만 빌릴 수 있다.
- REOBench에 대한 문장("treat image quality as a mere confounding variable")은 열화를 평가 대상으로 삼지 않았다는 뜻이고 열화의 고유성·현실성 비판은 아니다.
- 복합 열화 붕괴(Gemini 79.68 → 41.17)는 열화를 맞히는 객관식 정확도. 복합 stream 평가의 동기로는 쓸 수 있고 task 정확도 근거로는 못 쓴다.
- "열화를 알아보고 대처를 고른다"의 상한을 기존 분할 결과로 계산: 고정 isa_bn 55.81 / 57.41, 손상별 최선 56.52 / 58.05 mIoU (+0.7 / +0.6, 손상 10종).
  이득은 정렬이 필요 없는 손상과 clean에서 건드리지 않는 데서 온다.
- 제안(미실행): 복합 stream, 전송·보정 단계 열화 소수(band attenuation, band switch, stripe, tile missing)를 통제 세트로.

### 2026-11-24 설정 문서 통합, F3 판단
- `WRAPUP_backbones_datasets_2026-11-17.md` + `DATASETS_2026-11.md` → `docs/SETTINGS.md` 하나로 합침(사용자 요청: 설정은 한 파일에서 관리).
  §2.5에 REOBench 12종의 단계별 배치 표(SenseBench 분류 기준)와 손상 생성 방법을 모음.
- F3 판단(사용자와 합의): 지금 가진 것(AID-C, RESISC45-C, Potsdam-C, RESISC45 ↔ UCM)으로 충족. RSPDBench·SenseBench 방식의 새 벤치마크는 만들지 않고
  motivation·related work에서 인용한다. REOBench의 가치가 떨어진 것은 아니다.
- 복합 stream과 통제 2종(band attenuation, band switch)은 차선. 사용자 의견: 성능을 올리는 방법이 아직 없는데 더 어려운 설정을 먼저 돌리는 것은 순서가 맞지 않다.
  통계 정렬은 source 통계를 쓰고 novelty가 약하다는 것이 사용자의 평가.
