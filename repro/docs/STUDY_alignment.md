# 통계 정렬(statistics alignment) 연구 흐름 정리 — 왜 / 어떻게 / 근거

> 목적: 지금까지 Claude가 주도해 온 "통계 정렬" 줄기를 직접 파악하기 위한 문서(2026-10-22).
> 결과 수치는 `NOTES_tta.md` §3, §5.4–5.6. 선행 연구 상세는 `PRIOR_ART_stat_alignment.md`.
> 각 절 끝의 **[직접 확인]** 은 손으로 확인해 볼 항목.

---

## 0. 한 장 요약

```
P→PN 차이(+7)          "입력 통계만 맞춰도 큰 이득"         → ISA (입력 정렬)       [분류, 10-13]
엔트로피 TTA ±1~2      "학습하는 TTA는 거의 효과 없음"
분할 TTA (요청)        bnadapt·tent+BN batch만 이득          → "통계 맞추기"가 이득의 원천 [분할, 10-20]
  └ 왜곡별로 보완적: ISA는 radiometric, BN 통계는 degradation
"어느 깊이에서 정렬?"  depth sweep (fsa)                     → radiometric=입력, degradation=중간 블록 [10-22]
"둘 다 하면?"          조합 (ISA + 블록 k)                   → 효과가 그룹별로 거의 그대로 더해짐 [10-22]
"왜 그 깊이?"          feature shift 진단                    → radiometric은 입력의 스케일 변화, degradation은 중간 블록의 평균 이동
```

핵심 주장 후보: **affine radiometric 왜곡(밝기·대비·haze)은 픽셀 공간에서 되돌릴 수 있고, 비 affine 국소 열화(noise·blur·JPEG)는
네트워크 중간에서 재정규화해야 한다.** 학습 없는 정렬이 엔트로피 기반 TTA보다 RS 기반 모델에서 낫다.

---

## 1. 출발점: 왜 "통계 정렬"인가

### 1.1 관찰 1 — 입력 정규화 하나로 +7 (분류, `NOTES_tta.md` §3)
- REOBench 원 프로토콜(P)은 `ToTensor()`만 하고 Normalize를 빼먹음. 넣으면(PN) 왜곡 평균이 약 +7.
- 정규화는 고정 상수 affine 변환일 뿐인데 이만큼 차이 → **모델이 입력 채널 통계에 매우 민감**.
- 그래서: "test 배치의 통계를 소스(train) 통계로 맞추면 radiometric 왜곡도 되돌아오지 않을까?" → ISA.

### 1.2 관찰 2 — 학습하는 TTA는 거의 효과 없음
- AID: Tent / SAR / DeYO / EATA / CoTTA 모두 ±1~2 (§3.1). CoTTA는 큰 lr에서 붕괴.
- Potsdam 분할: BN 고정 Tent −1.2/−2.0, SAR −0.7/−1.1, CoTTA ≈ none (§5.4).
- 이득이 난 것은 **BN을 test 배치 통계로 바꾼 것**(tent + BN batch, bnadapt)과 **ISA**뿐 → "이득의 원천은 통계 맞추기".
  - tent+BN batch(+3.8)와 bnadapt(+3.1)의 차이가 작음 → Tent 이득의 대부분이 엔트로피 최소화가 아니라 BN 통계에서 옴.

### 1.3 관찰 3 — 왜곡 종류별로 서로 다른 정렬이 맞음 (분할, §5.4)
- ISA: brightness/contrast, haze 회복 (ScaleMAE BC 55.5→64.6).
- BN 통계: gaussian noise, S&P, JPEG 회복 (GNoise 39.4→49.1).
- → "그럼 어느 **깊이**에서 정렬해야 하나?" 라는 질문이 생김 → depth sweep.

**[직접 확인]** `experiments/2026-10-20_seg_tta/summary.txt`에서 isa와 bnadapt의 per-corruption 열을 비교.

---

## 2. 방법별: 무엇을, 어떻게 구현했나

모두 **학습 파라미터 없음**. 공통 식(채널 c별):

```
h' = (h − μ_test) / σ_test · σ_src + μ_src
```
- μ_test, σ_test: 지금 들어온 test 배치(8장)에서 계산 → 배치마다 새로 계산, 다음 배치로 상태 이월 없음.
- μ_src, σ_src: clean **train** 이미지에서 한 번 계산 (test 데이터는 절대 안 씀).
- 바꾸는 것은 "어디의 h냐"뿐.

| 방법 | 정렬 위치 h | 채널 / 통계 범위 | 코드 | 근거 논문 |
|---|---|---|---|---|
| ISA | 입력 픽셀 (정규화 전) | RGB 3채널, 배치 8장의 모든 픽셀 | `src/reobench_tta/tta/isa.py`, 분할 `scripts/seg/seg_tta.py:165` `isa_batch` | 논문 방법 아님. 같은 연산: histogram moment matching, Reinhard 색 전이, AdaIN을 0층에 적용, RS 상대 방사 정규화 |
| bnadapt | 분할 디코더의 BN 약 13개 | 각 BN 채널, 배치 8장의 모든 픽셀 | `seg_tta.py:153` `configure_norm`, `:208` | Schneider et al. NeurIPS 2020; Nado et al. 2020 (BN adaptation) |
| tent_bnbatch | bnadapt + LN/BN affine을 엔트로피로 학습 | | `seg_tta.py` tent 분기 | Wang et al. Tent, ICLR 2021 원문 설정 |
| fsa | 임의 모듈의 출력(또는 `:in`이면 입력) | ViT 토큰 D=1024 채널, 배치의 모든 토큰(8×257) | `src/reobench_tta/tta/stat_align.py` | AdaBN을 임의 층으로 일반화. 관련: TTN(ICLR 2023, 층별 통계 혼합), FOA(ICML 2024, ViT activation shift), test-time style shifting(ICML 2023), surgical fine-tuning(ICLR 2023, 어느 층을 적응할지) |
| 조합 | 입력 ISA + fsa 한 곳 | | `--with_isa` (cls `cls_tta.py:204`, seg `:189`) | 우리 실험 |

### 2.1 ISA 구현 세부
- 분류: 픽셀 [0,1] 공간에서 정렬한 뒤 PN 정규화 (`cls_tta.py:95` `PIXEL_SPACE`에 해당하면 정규화를 데이터 변환이 아니라 `Net.forward` 안에서 함).
  소스 통계: 무작위 clean AID train 2,000장 (`cls_tta.py:156`, seed 0).
- 분할: 0–255 공간, 소스 통계는 train 패치 500장 (`seg_tta.py:173`).
- 2,000 / 500이라는 숫자는 **근거 없는 내 선택**(통계가 안정되기에 충분하다고 판단). 민감도 확인은 안 함.

### 2.2 fsa(`stat_align.py`) 동작
1. `source_stats`: clean train 배치를 모델에 통과시키며 forward hook으로 해당 모듈 출력의 채널별 합·제곱합을 누적 → μ_src, σ_src.
2. `StatAlign`: 같은 모듈에 hook을 달아 test 때 출력을 위 식으로 바꿔치기. 이후 층은 바뀐 값을 받음.
3. 채널 축: (B,N,D) 토큰 → D, (B,C,H,W) 맵 → C, (B,D) 벡터 → D (`_flat`, `stat_align.py:17`).
4. `"head:in"`, `"decode_head.conv_seg:in"`처럼 `:in`을 붙이면 forward **pre**-hook으로 모듈의 **입력**을 정렬.
- 단위 테스트: 정렬 후 출력의 평균/표준편차 = 소스 값 (확인함).
- 조합 시 소스 통계는 **ISA를 먼저 적용한 clean 배치**에서 계산 (test 때와 같은 입력 경로) — `cls_tta.py:204`, `seg_tta.py:189`.

### 2.3 "head 입력"과 "최종 norm"이 다른 이유 (`NOTES_tta.md` §5.5)
- 분류 모델은 최종 LayerNorm 뒤 **CLS 토큰만** 선형 head로 감 (`common.py` `features`, global_pool=False).
- final norm 정렬: 통계를 CLS + 패치 256개 전부로 계산 → CLS에는 맞지 않는 통계로 CLS가 변형됨.
- head 입력 정렬: CLS 8개(배치)만으로 통계 → 실제 분류에 쓰이는 분포를 직접 맞춤.

**[직접 확인]**
- `stat_align.py` 전체(110줄)를 읽고, `_apply`의 식이 위 공통 식과 같은지.
- 아무 블록이나 하나 골라 `python -c`로 hook 전후 평균/표준편차를 찍어 보기.

### 2.4 head 입력 정렬을 처음부터 다시 (2026-11-23 문답 정리)

11월의 17개 백본 실험(`scripts/analysis/feat_metrics.py`, `feat_tta.py`의 `align_ms`, `_al`; 배치 64) 기준이다.

**(1) 어디에 넣나.** 백본과 Linear 사이다.

```
이미지 → ViT 블록들 → final LayerNorm → h          ← 여기까지 백본 (고정). h = CLS 벡터 (DINOv3는 [CLS, patch 평균], ResNet은 GAP)
       → 1단계: z = (h − μ_src) / σ_src            ← head를 학습할 때 우리가 넣은 표준화 (고정)
       → Linear: z W + b
```
- final LayerNorm은 실제로 있다. 블록 안의 LayerNorm은 attention / MLP의 **입력**을 정규화하고(pre-norm), 잔차를 더한 블록 출력은
  정규화되지 않으므로 마지막 블록 뒤에 norm을 하나 더 둔다(timm `model.norm`, lucidrains `Transformer.norm`).
- 블록 안의 MLP(FFN)와 분류 head(Linear 하나)는 다른 것이다.

**(2) 통계량의 축.** μ, σ는 모두 h의 통계이고, **채널마다 따로, 여러 장의 h에 걸쳐** 낸다. LayerNorm과 축이 다르다.

| | 무엇의 평균·표준편차 | 크기 |
|---|---|---|
| LayerNorm | 이미지 한 장의 h 안 D개 원소 | 이미지마다 숫자 2개 |
| μ_src, σ_src | 채널 c마다, clean train 이미지 전체의 h[c] | D차원 벡터 2개, 한 번 계산 |
| μ_test, σ_test | 채널 c마다, test 배치 64장의 h[c] | D차원 벡터 2개, 배치마다 계산 |

여기서 src는 head를 학습한 clean train(AID / RESISC45 train)이고, 백본의 사전학습 데이터(ImageNet, fMoW)가 아니다.
LayerNorm을 지나도 "채널 c의 값이 배치 안에서 어떤 분포인가"는 정해지지 않으며, 손상이 바꾸는 것이 이 채널별 분포다.

**(3) 1단계는 우리가 정한 head 학습 방식이다** (ViT에도 TTA 방법에도 없음).
a. clean train 이미지를 동결한 백본에 통과시켜 h를 얻는다. b. 채널별 μ_src, σ_src를 낸다. c. z = (h − μ_src) / σ_src.
d. (z, 라벨)로 로지스틱 회귀(= Linear + softmax + cross-entropy인 선형 분류기)를 L-BFGS로 학습해 W, b를 얻는다. e. W, b, μ_src, σ_src를 함께 저장한다.
- 이유: 백본마다 feature 크기가 달라서, 표준화해야 같은 λ(L2 규제) 격자와 같은 절차를 17개 백본에 쓸 수 있다.
- L-BFGS는 head를 학습하는 optimizer일 뿐이다(전체 데이터를 한 번에 놓고 푸는 볼록 문제라 lr·epoch 선택이 필요 없음). 정렬의 근거가 아니다.
- 1단계는 상수로 하는 선형 변환이라 학습 뒤 W′ = W / σ_src, b′ = b − (μ_src / σ_src)W로 Linear에 합칠 수 있다.

**(4) 정렬 = 2단계.** 1단계를 지난 z를 배치 통계로 다시 표준화한다. 학습 없음.

```
z_al = (z − mean_batch(z)) / std_batch(z) = (h − μ_test) / σ_test        (μ_src, σ_src는 약분되어 사라짐)
```
- 직관: W, b는 채널마다 평균 0·표준편차 1인 입력으로 학습됐다. 손상이 오면 z가 그 분포를 벗어나고, 정렬은 head 입력을 다시 평균 0·표준편차 1로 만든다.
- BatchNorm으로 말하면 1단계 = eval 모드(저장된 통계), 정렬 = train 모드(지금 배치의 통계). 이것이 AdaBN이고, Tent가 하는 두 가지
  (test 배치 통계로 정규화 + γ, β 학습) 중 앞의 것만 한 것이다. Tent는 백본 안의 BN에, 우리는 백본 뒤의 이 단계에 한다.
- 2단계는 Linear에 합칠 수 **없다**. μ_test, σ_test가 배치마다 바뀌므로, 합치면 배치마다 다른 W″, b″가 된다. 즉 정렬은
  "학습 없이 test 배치를 보고 head를 배치마다 다시 계산하는 것"이고, 이것이 TTA인 이유다.
- 대가: 한 이미지의 예측이 같은 배치의 다른 이미지에 의존한다(작은 배치에서 불안정, 한 클래스로 몰린 배치에서 위험).

**(5) 일반 규칙: head 입력의 채널별 통계를, head가 학습 때 본 입력의 통계에 맞춘다.**
test 통계로 표준화하는 것((h − μ_test) / σ_test)은 어떤 head든 항상 한다. 그 뒤 src 통계를 식에 넣을지는 head가 무엇을 보고 학습됐는지에 달렸다.

| head | 학습 때 본 입력의 통계 | 정렬 식 |
|---|---|---|
| raw (h 그대로 학습) | μ_src, σ_src | (h − μ_test) / σ_test · σ_src + μ_src |
| std (표준화한 z로 학습, 우리 방식) | 0, 1 | (h − μ_test) / σ_test |

- 맨 위의 공통 식 h′는 raw head용 일반식이고, std head에서는 σ_src = 1, μ_src = 0이 되어 뒷부분이 사라진다.
- 블록 사이에 넣는 fsa가 항상 일반식을 쓰는 이유도 같다. 다음 블록은 원래 크기의 값을 기대한다.
- 따라서 필요한 정보는 "h의 src 통계"가 아니라 **head 학습 입력의 통계**이고, head와 함께 저장돼 있어야 한다(BN이 running 통계를 들고 다니는 것과 같음).
  식을 잘못 고르면(raw head에 z_al을 그대로 넣기 등) head가 학습 때와 다른 크기의 입력을 받는다.

**(6) 식을 맞게 쓰면 이득은 head 학습 방식에 거의 무관하다** (`scripts/analysis/head_nostd.py`, `experiments/2026-11-23_head_nostd/RESULTS.md`).

| 17개 백본 평균 손상 정확도 (%) | head | clean | 정렬 전 | 정렬 후 | 이득 |
|---|---|---:|---:|---:|---:|
| AID-C | std | 94.8 | 81.8 | 87.3 | +5.5 |
| | raw | 94.6 | 81.6 | 86.9 | +5.4 |
| RESISC45-C | std | 90.6 | 75.2 | 81.2 | +5.9 |
| | raw | 90.4 | 74.9 | 80.7 | +5.8 |

raw head에서도 17/17 백본에서 이득. moco_in, moco_fmow만 raw head의 정렬 전 정확도가 1.4~2.5pt 높아 이득이 조금 작다.

**(7) src 통계를 모를 때** (`NOTES_tta.md` §5.8, 10월 SatMAE / ScaleMAE, AID): 라벨 없는 clean 이미지 32장으로 낸 통계면 head 입력 정렬의 이득이
+3.71 / +7.22로, 2,000장(+5.24 / +7.97)보다 0.7~1.5pt만 낮다. 다운스트림 통계가 전혀 없는 경우는 head 입력에서 시험하지 않았다
(픽셀 정렬에서는 사전학습 Normalize 상수로 대신하면 clean −2.6~−3.5).

**(8) 근거와 구분.**
- 2단계의 근거: AdaBN / BN adaptation(정규화 통계를 test 배치 것으로 교체). 평균만 맞추는 `align_m`은 FOA의 activation shifting과 같은 위치·같은 종류
  (FOA는 EMA로 test 평균을 추적, 우리는 배치마다 계산). 연산 자체는 새롭지 않다.
- 정렬과 **feature 단계 TTA**는 다른 실험이다. `feat_tta.py`의 tent / flipf / dsbr는 head 입력에 z′ = γ ⊙ z + β를 붙이고 γ, β를 엔트로피 손실로
  **학습**한다(ViT에서는 final norm의 γ, β만 학습하는 TENT와 같음). `_al`은 정렬한 z_al 위에서 그 학습을 한 것.

---

## 3. 실험 설계 결정과 그 이유

| 결정 | 이유 | 근거 강도 |
|---|---|---|
| mild 프로토콜 ((왜곡, 강도)별 독립, 매번 리셋) | 사용자 지시("continual은 필요없음"). 의료 분할 TTA(VPTTA, GraTa)의 사이트별 평가와 같은 구조 | 지시 |
| 배치 8 | 분할 학습 배치와 같음. 분류도 같은 값 | 관례 |
| 분류 PN 헤드 사용 | P는 정규화 누락이라 비정상 기준선. PN이 주 결과, P는 논문 재현 보충 | 앞선 분석 |
| 분할 B 설정(블록 7/11/15/23) | ViT-L 관례, 사용자 선택 | 지시 |
| sweep 블록 0,3,7,11,15,19,23 | 24층을 대략 4층 간격으로. 7·11·15·23은 분할 디코더가 가져가는 블록과 겹침 | 내 선택 |
| 분할은 강도 3·5만 | 비용(1회 15분). CoTTA도 같은 부분집합이라 비교 가능 | 내 선택 |
| 조합 블록 3/7/11 (분할), 7/11/head (분류) | sweep에서 degradation이 좋았던 위치 | **test 결과를 보고 고름** |

### 3.1 "test에서 골랐다"는 문제
- 블록 7을 test 표를 보고 골랐으므로 그 점수는 상한(낙관적).
- 정당한 방법: (a) 왜곡 일부로 위치를 고르고 나머지 왜곡으로 보고(leave-out),
  (b) 분류에서 고른 위치를 분할에 그대로, (c) "깊이 1/3" 같은 규칙을 미리 고정.
- 완화 요인: 하이퍼파라미터가 위치 하나뿐이고, 고정 처방 "ISA + 블록 7"이 4개 설정 모두 ISA 단독보다 좋음.

### 3.2 알려진 약점
- seed 0 한 번. 배치는 i.i.d. 셔플(같은 왜곡만 들어옴) → 배치에 왜곡이 섞이거나 배치 크기 1이면 무너질 수 있음.
- ISA는 분할 occlusion(clouds)에서 −1.4 ~ −2.0.
- Yi et al.(ACM MM 2024)은 분할에서 BN 통계 갱신 효과가 거의 없다고 보고 → 우리 +5~10과 상충, 설명 필요.

---

## 4. 선행 연구와의 관계 (`PRIOR_ART_stat_alignment.md` 요약)

| 우리 요소 | 이미 있는 것 | 아직 못 찾은 것 |
|---|---|---|
| ISA | moment/histogram matching, Reinhard, AdaIN@0층 — **새롭지 않음** | RS 기반 모델 TTA에서 엔트로피 TTA보다 낫다는 보고 |
| bnadapt | Schneider 2020 등 — 표준 baseline | |
| 어느 깊이에서 정렬? | Surgical fine-tuning(입력 수준 왜곡 → 앞층), TTN, FOA, style shifting | **왜곡 메커니즘별 최적 깊이가 다르다**(radiometric=입력, degradation=중간)는 정량 sweep |
| 왜곡 판별 후 처방 | PAN, FROST (BN 통계만 선택) | 입력 vs 특징 정렬을 고르는 라우터 |

- Surgical fine-tuning 논리로는 noise·JPEG도 "입력 수준" 왜곡이라 앞층이 맞아야 하는데, 우리는 중간 블록이 최선 → 이 긴장이 기여 포인트.

**[직접 확인]** 먼저 읽을 논문 3편: Lee et al. *Surgical Fine-Tuning* (ICLR 2023), Lim et al. *TTN* (ICLR 2023),
Schneider et al. *Improving robustness against common corruptions by covariate shift adaptation* (NeurIPS 2020).

---

## 5. feature shift 진단 (`scripts/analysis/feature_shift.py`)
- 질문: 왜 radiometric은 입력, degradation은 중간 블록인가? 왜 ScaleMAE 분류는 블록 정렬에 약한가?
- 방법: clean test 400장과 같은 이미지의 왜곡본(강도 5)을 통과시켜 위치별로
  - d_mu, d_sd: 채널 평균·스케일이 얼마나 움직였나
  - cos → cos_al: 같은 이미지·같은 토큰의 clean/왜곡 특징 코사인 유사도가 정렬로 얼마나 회복되나
  - outlier: 소수 채널이 에너지를 독점하는지(표준화가 깨지기 쉬운 구조)
- 출력: `experiments/2026-10-22_feature_shift/summary.txt`
- 결과 요약 (상세는 `NOTES_tta.md` §5.7):
  - radiometric: 입력에서 스케일(d_sd)이 크게 바뀌고 깊어질수록 줄어듦 → 입력 정렬이 맞음.
  - degradation: 입력에서는 평균이 안 움직여 ISA로 못 고침. 중간 블록에서 평균 이동으로 나타남 → 중간 블록 정렬이 맞음.
  - 소수 채널(상위 1%)이 에너지의 60~95%를 차지(massive activation). 원 코사인 지표는 이 채널에 좌우됨.
  - 분류 ScaleMAE가 블록 정렬에 약한 이유는 아직 설명 못 함.

---

## 6. 스스로 점검할 질문 (답할 수 있으면 이 줄기를 파악한 것)
1. ISA와 bnadapt는 각각 어떤 공간의 어떤 채널을 정렬하나? 왜 둘의 효과가 왜곡별로 다른가?
2. fsa에서 μ_test는 몇 개 값으로 계산되나? (분류 블록: 8×257 토큰, head 입력: 8개)
3. tent+BN batch와 bnadapt의 차이가 작다는 것은 무엇을 뜻하나?
4. 조합의 소스 통계를 ISA 후에 계산한 이유는?
5. "ISA + 블록 7" +10.8 (분할 ScaleMAE)을 논문에 그대로 쓰면 왜 안 되나?
6. 이 결과가 Surgical fine-tuning의 주장과 어디서 맞고 어디서 다른가?
