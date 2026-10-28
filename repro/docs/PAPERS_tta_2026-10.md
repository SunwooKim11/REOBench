# 면담용 논문 정리: Panda, FOA, DAF, Hybrid-TTA (2026-10-28)

> 전문을 읽고 핵심 수치는 추출한 원문 텍스트에서 다시 확인함. [not found] = 원문에서 확인 못 함.
> 우리 결과는 `NOTES_tta.md` §4.2, §5.5–5.7, 흐름은 `STUDY_alignment.md`.

## 0. 한눈에 비교

| | Panda (AAAI 2026) | FOA (ICML 2024) | DAF (arXiv 2026.08, 심사 중) | Hybrid-TTA (ICCV 2025) |
|---|---|---|---|---|
| 모델 | CLIP zero-shot 분류 (ViT-B/32, B/16) | ViT-B/16 (전통, 8/6-bit 양자화 포함) | CLIP zero-shot 분할 (NACLIP ViT-B/32) | SegFormer MiT-B5 분할 |
| 설정 | 단일 도메인, 배치 100 | 단일 도메인, 배치 64 | **continual**, 리셋 없음, 배치 8 | **continual**, 배치 1 |
| 무엇을 바꾸나 | 최종 임베딩에서 negative 평균을 뺌 (학습 없음) | 입력 프롬프트(CMA-ES, 역전파 없음) + 최종 CLS를 소스 평균 쪽으로 이동 | 이미지 인코더 LayerNorm (SGD) | 전체 또는 adapter (Adam) |
| 핵심 신호 | 패치 셔플 negative = corruption 성분 추정 | **CLS 통계(평균·표준편차)를 소스에 맞추기** | 예측 다양성 + 텍스트 prototype anchor | 손실 급등 감지 + 마스킹 복원 |
| 왜곡별 결과 | 있음 (부록 Table 5) | 있음 (Table 2) | 없음 | 없음 |
| RS | 없음 | 없음 | LoveDA 1개 (일반 왜곡) | 없음 |
| 코드 | github.com/ruxideng/Panda | github.com/mr-eggplant/FOA | github.com/chandlerbing65nm/DAF | [not found] |

---

## 1. Panda: Test-Time Adaptation with Negative Data Augmentation (Deng, Bao, Wei, He; AAAI 2026; arXiv 2511.10481)

**문제의식**: corruption이 이미지 임베딩에 섞이면 텍스트 임베딩과 우연히 상관되어 특정 클래스로 예측이 쏠림(prediction bias,
예측 분포와 정답 분포의 L1 거리). entropy TTA는 편향된 예측을 pseudo-label로 써서 편향이 커지다 붕괴. AugMix 같은 positive
augmentation(TPT, Zero, TPS)은 corruption을 그대로 두므로 편향을 못 줄이고 비용도 큼(이미지당 63 view).

**방법**
1. 배치 B장을 32×32 패치(224 입력에서 7×7)로 자르고 **배치 전체의 패치 풀**에서 무작위로 다시 조립해 negative M = B/10장 생성
   (패치는 한 번씩만 사용). 객체 형태는 깨지고 corruption 질감은 남는다는 가정.
2. `n̄ = mean(E(x⁻_j))`, `d_i = v_i − β·n̄` (Eq. 1–2). 특징은 L2 정규화된 CLIP 최종 이미지 임베딩. 학습 없음.
3. 다른 TTA(Tent, SAR, DeYO, TPT …)에 붙일 때는 v 대신 d로 logit과 손실을 계산 (Algorithm 1).
- 근거(Theorem 4.1): `v = v_cls + v_corr`, n이 v_corr와 상관 r, v_cls와 상관 0이면 βn을 빼서 corruption 성분이 최소 √(1−r²)배.
- β는 0.1–1.0 grid search로 baseline·데이터셋마다 최적값 보고 (CLIP: CIFAR-10-C 0.7, ImageNet-C 0.4) → **test에서 고름**.
- 대조학습 아님.

**결과**: 모든 baseline에 +1~8 (CIFAR-10-C Tent 62.8→71.1, SAR 63.3→70.7). 오버헤드 < 10%.
**왜곡별 (부록 Table 5, CIFAR-10-C, CLIP 단독)**: Gaussian +7.8, Shot +6.8, Glass +3.8, Pixelate +3.0, Fog +2.4,
**Brightness +1.5, Contrast +1.7**, JPEG +1.7 → noise에 가장 강하고 radiometric에 가장 약함.

**DeYO와의 차이**: DeYO는 이미지 *내부* 셔플로 예측 변화(PLPD)를 재서 샘플 선택에 씀. Panda는 *배치 간* 셔플로 offset을 만듦.
같은 조건 비교(Table 4) CIFAR-10-C 65.5 vs 68.3. ablation에서 이미지 내부 셔플은 확실히 나쁨.

**비판 포인트**
- "원본 특징의 배치 평균만 빼기" baseline이 없음. CLIP 임베딩의 공통 성분 제거 효과와 negative의 기여가 분리 안 됨.
- β를 test에서 선택. clean 정확도 보고 [not found]. 섞인 corruption 배치, 작은 배치 검증 없음(배치 크기는 Tent만 10–500).
- 표현이 실제로 corruption만 담는지 직접 분석 없음(1차원 가우시안 정리와 L1 편향만).

**우리와의 연결**: Panda = 최종 층에서 평균 하나를 빼는 1차 모멘트 보정. 우리 진단상 degradation은 특징 평균 이동이 되고(§5.7)
radiometric은 입력 스케일 변화(§4.2) → Panda가 noise에 강하고 밝기·대비에 약한 패턴과 일치. 선형 head 앞 특징에서 n̄를 빼면
텍스트 없이도 되므로 MIM에도 적용 가능 → "Panda vs head 입력 정렬 vs 배치 평균 빼기" 비교 실험 후보.

---

## 2. FOA: Test-Time Model Adaptation with Only Forward Passes (Niu, Miao, Chen, Wu, Zhao; ICML 2024; arXiv 2404.01650)

**문제의식**: 엣지 기기의 양자화 모델은 가중치를 못 바꾸고 기울기도 안 나옴 → 역전파 없는 TTA.

**방법**
1. **프롬프트 탐색**: 입력 앞에 프롬프트 3개 추가, CMA-ES(population 28)로 배치당 1회 갱신. 가중치는 고정.
   fitness (Eq. 5) = 엔트로피 + λ·Σ_{12개 층} (‖μ_i(X_t) − μ_i^S‖₂ + ‖σ_i(X_t) − σ_i^S‖₂).
   통계는 **각 층 CLS 토큰만**의 평균·표준편차. 소스 통계는 ImageNet val의 라벨 없는 샘플(32장이면 충분, Fig. 2c).
2. **back-to-source activation shifting** (Eq. 7–9): 마지막 층 CLS(= 분류 head 입력)를 `e ← e + γ(μ^S − μ_test)` 이동,
   γ = 1, μ_test는 EMA(α = 0.1). **평균만** 이동.

**결과 (ImageNet-C 강도 5, ViT-B/16, Table 2)**: NoAdapt 55.5, TENT 59.6, CoTTA 61.7, SAR 62.7, **FOA 66.3** (ECE 3.2).
8-bit 모델에서도 54.1→63.5.
**왜곡별 이득**: Contrast **+35.1**, Glass +21.1, Elastic +16.7, Zoom +12.9, Defocus +12.4 / Brightness +3.4,
Gaussian +4.7, JPEG +5.4, Shot +6.4.

**ablation (Table 5, 9) — 우리 주장의 직접 근거**
| 구성 | 정확도 |
|---|---|
| NoAdapt | 55.5 |
| CMA + 엔트로피만 | 44.9 (손해) |
| CMA + 통계 맞추기만 | **63.4** |
| 활성 이동만 | 59.1 |
| 엔트로피 + 통계 | 65.4 |
| 전체 FOA | 66.3 |
| 통계 손실(Eq. 5)을 SGD로 norm 층에 적용 | **70.5** (논문 최고) |
→ 이득의 대부분이 "통계를 소스에 맞추기"에서 나옴.

**비판 포인트**
- 소스 통계를 ImageNet val에서 계산 = ImageNet-C가 왜곡을 입힌 바로 그 이미지들.
- 민감도 분석 대부분 Gaussian noise 하나. 어느 층 통계를 쓸지 ablation 없음(항상 12층 전부).
- CNN(ResNet-50)에서는 약함(저자 인정). 비용은 배치당 forward 28회.

**우리와의 연결**
- 같은 점: head 입력(CLS)을 소스 통계 쪽으로 학습 없이 옮김 = 우리 "head 입력 정렬". 패치 토큰은 안 건드림(우리: 전체 토큰 정렬은 해로움).
- 다른 점: FOA는 평균만, 우리는 평균+표준편차. FOA는 중간 층 통계를 목표로만 쓰고 직접 고치지 않으며 **깊이를 고르지 않음**
  → 우리의 기여 자리("왜곡 종류에 맞는 깊이에서 직접 정렬").
- 왜곡별: noise·JPEG 이득이 작음 = 입력+최종 층만으로는 degradation이 잘 안 고쳐짐 → 중간 블록이 필요하다는 우리 결과와 맞는 정황(대조 실험은 아님).

---

## 3. DAF: Towards Continual TTA of Vision-Language Models in Open-Vocabulary Semantic Segmentation (Doloriel et al.; arXiv 2608.29923, 심사 중)

본문 10쪽만 공개, 보충자료(ViT-B/16·L/14, LoveDA ablation 등) [not found].

**설정**: CLIP zero-shot 분할(NACLIP ViT-B/32), continual(리셋 없음), 이미지 인코더 LayerNorm만 SGD lr 1e-3, 배치 8, 1 step.

**진단 (Fig. 1, MLMP on VOC20-C)**: 엔트로피 최소화로 활성 클래스 수가 ~20 → 1~3 (클래스 붕괴), 패치 특징이 배정된 텍스트
prototype에서 멀어지고 다른 prototype에 가까워짐, 특징 변화가 작은 샘플도 역전파를 소비.

**방법**: L = L_base(TENT 또는 MLMP) + 2.0·MDIV + 0.5·CMAC, 그리고 SAFS.
- MDIV: 배치 평균 예측의 엔트로피 최대화(한 클래스로 몰리지 않게).
- CMAC: 원본 모델이 패치마다 가장 가까운 텍스트 prototype을 배정해 두고, 적응 중 배정 클래스 유사도가 원본보다 떨어지거나
  다른 클래스 유사도가 원본보다 오르면 hinge 페널티.
- SAFS: 원본 대비 특징 변화 Δ가 μ − 0.5σ 이하인 샘플은 손실에서 제외.

**결과 (Table 1, mIoU)**: VOC20-C 소스 54.4 / TENT 3.3 / MLMP 9.7 / **DAF-M 62.7**.
LoveDA 평균 소스 22.7 / TENT 20.3 / **DAF-M 31.8**. LoveDA-C 소스 9.2 / TENT 1.2 / SAR 7.6 / **DAF-M 14.9**. 왜곡별 결과 없음.

**주의할 반대 증거 (Table 3, lr sweep)**: lr 1e-4에서 TENT 55.1, MLMP 57.0으로 소스(54.4)를 넘음 → "붕괴"의 상당 부분은 기본
lr 1e-3 때문. DAF-M도 1e-2에서 41.0으로 소스 아래. 하이퍼파라미터 전부 VOC20-C test에서 고름. 대부분 seed 0.

**우리와의 연결**
- 엔트로피 TTA 손해: LoveDA TENT −2.4 ≈ 우리 BN 고정 Tent −1~−2.
- text anchor는 closed-set(CLIP 인코더 + UperNet)에 바로 못 씀(패치–텍스트 유사도가 필요). 텍스트 없는 대안인
  "원본 특징과의 코사인 일관성"도 붕괴를 막음(51.7 vs 9.7, Table 6) → MIM에도 적용 가능.
- RS 특화 요소 없음(LoveDA는 5개 중 하나, ImageNet-C식 일반 왜곡, OpenAI CLIP). 학습 없는 통계 정렬 baseline과 비교 없음.

---

## 4. Hybrid-TTA: Continual TTA via Dynamic Domain Shift Detection (Park, Park, Ko, Min; ICCV 2025; arXiv 2409.08566)

**설정**: continual 분할, SegFormer MiT-B5, Cityscapes→ACDC(fog→night→rain→snow ×10 round), 배치 1.

**방법**
- DDSD: mean-teacher 손실 L_t가 그 EMA 임계값 τ_t(α = 0.999, τ_0 = 0)의 λd(=1.1)배를 넘으면 Full Tuning, 아니면 Adapter Tuning
  (AdaptMLP, 0.17M = 0.1%). 인스턴스마다 결정.
- MIMA: 마스킹 비율 0.6, 패치 32(SimMIM), RGB L1 복원. 디코더는 랜덤 초기화, test 때만 사용(MIM 사전학습 백본 불필요). λr = 0.3.

**결과 (Table 1, 10 round 평균 mIoU)**: 소스 56.7, TENT 52.3, CoTTA 58.6, SVDP 61.0, ViDA 61.6, **Hybrid-TTA 62.2**, 2.0 FPS
(CoTTA 0.10). ablation(Table 3): FT만 59.3, FT+MIMA 61.9, DDSD만 59.8, DDSD+MIMA 62.2 → 이득은 주로 MIMA(+2.6).

**비판 포인트**: DDSD는 변화의 *크기*(손실 급등)만 봄, 감지 정확도 검증 없음, 스트림 순서·continual에 의존.
λd, λr를 첫 target 도메인 일부로 고름. 표 사이 수치 불일치 여러 곳(Tab. 1 vs 4 등).

**우리와의 연결 (약함)**: 왜곡 *종류*를 골라 정렬 깊이를 정하는 라우터와는 다른 문제. mild(매번 리셋)에서는 DDSD가 의미 없음.
MIMA는 occlusion 대응이 아니고, MAE 백본을 활용하지도 않음(그 방향은 TTT-MAE, Continual-MAE에 가까움).

---

## 5. 면담 추천 구성
- 1편 Panda(최신), 2편 FOA(지금 방향과 가장 밀접). DAF는 향후 CLIP 방향에서 한 줄 언급("엔트로피 붕괴는 lr 의존적"), Hybrid-TTA는 제외.
- 흐름: Panda = 최종 특징에서 평균 하나 빼기(noise 강함, 밝기 약함) → FOA = CLS 통계 맞추기가 이득의 핵심
  → 우리 = 왜곡 종류에 따라 정렬 깊이가 달라야 함(입력 / 중간 블록), 두 논문의 왜곡별 패턴이 이것으로 설명됨.
