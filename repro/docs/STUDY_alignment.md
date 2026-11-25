# 통계 정렬(statistics alignment) 정리 — 무엇을, 어디서, 어떤 통계로

> 목적: "통계 정렬" 줄기를 직접 파악하기 위한 문서. 2026-10-22에 쓰고, 2026-11-25에 11월 실험 기준으로 전체를 다시 정리했다.
> 수치의 출처: 10월(SatMAE·ScaleMAE 두 모델)은 `NOTES_tta.md` §3, §5.4–5.8. 11월(백본 17개, 분할 12종)은 각 실험의 `RESULTS.md`와
> `WRAPUP_midterm_2026-11-26.md` §4. 선행 연구 상세는 `PRIOR_ART_stat_alignment.md`.
> 각 절 끝의 **[직접 확인]** 은 손으로 확인해 볼 항목.

---

## 0. 한 장 요약

**정렬이란.** 모델 안의 한 위치에서, 지금 들어온 test 배치의 채널별 평균·표준편차를 그 위치가 학습 때 갖던 평균·표준편차로
되돌리는 것이다. 가중치는 하나도 바꾸지 않는다(학습 없음). 방법들의 차이는 "어느 위치냐"뿐이다.

| 위치 | 이 문서의 이름 | 듣는 손상 | 확인한 범위 |
|---|---|---|---|
| 입력 픽셀 (encoder 앞) | 입력 정렬 (ISA) | 밝기·대비, haze | 분류 2개 모델(10월), 분할 2개 모델 × 12종(11월) |
| encoder 중간 블록 | 블록 정렬 (fsa) | noise, blur, JPEG | 분류·분할 2개 모델(10월). 위치를 test를 보고 골라 상한 |
| 분할 decoder의 BN | BN 정렬 (bnadapt) | noise, JPEG, data gaps | 분할 2개 모델 × 12종 |
| head 입력 (encoder 뒤, Linear 앞) | head 입력 정렬. 11월 문서에서 그냥 "정렬"이라 하면 이것 | 분류 평균 +5.5 / +5.9pt, 17개 백본 전부 이득 | 분류 17개 백본 × AID-C, RESISC45-C |

**흐름.**

```
10-13  P→PN 차이(+7)             모델이 입력 통계에 민감                 → 입력 정렬(ISA)
10-13  엔트로피 TTA ±1~2          학습하는 TTA는 거의 효과 없음
10-20  분할 TTA                  이득은 BN 통계 교체와 ISA뿐              → "통계 맞추기"가 이득의 원천
10-22  어느 깊이에서 정렬?         depth sweep                           → 밝기류 = 입력, noise류 = 중간 블록, 분류 평균은 head 입력
10-22  둘 다 하면?                입력 + 블록 조합                        → 손상 그룹별로 거의 그대로 더해짐
10-28  소스 통계가 꼭 필요한가      32장 / 사전학습 상수로 대체              → 라벨 없는 clean 32장이면 대부분 유지, 사전학습 상수는 손해
11-17  백본 17개로 넓히면          head 입력 정렬                         → 81.8 → 87.3 / 75.2 → 81.1, 엔트로피 방법보다 큼
11-23  head 학습 방식에 달렸나      표준화 있는 head / 없는 head            → 식을 맞게 쓰면 이득이 같음 (+5.4 ~ +5.9)
11-24  head에서 더 얻을 것은?      head를 라벨로 고친 상한                 → 정렬 뒤 남는 여지는 motion blur, data gaps, scale
11-25  분할 12종                  BN 정렬 / 입력 정렬                     → 손상이 세 묶음으로 갈림
```

**지금 말할 수 있는 것.**
1. 학습 없는 정렬이 엔트로피 기반 TTA보다 이득이 크다(분류 17개 백본, 분할 2개 모델).
2. 어디서 정렬해야 하는지는 손상에 달렸다. 밝기·haze는 입력에서, noise·JPEG는 feature 쪽(중간 블록, BN, head 입력)에서 듣는다.
3. 정렬로 안 되는 손상이 있다. 분할의 blur·구름·기하 3종, 분류의 motion blur·data gaps·scale이다. 이쪽은 방법 후보 1
   ("촬영 조건 되돌리기", `WRAPUP_midterm_2026-11-26.md` §6)에서 다룬다.

---

## 1. 전체 그림: 무엇이 고정이고, 정렬은 어디에 들어가나

### 1.1 모델의 세 부분과 고정 / 학습

```
이미지 x → [전처리: resize, Normalize] → encoder(백본) → h → head 또는 decoder → 예측
```

| 설정 | encoder | encoder 뒤 | 학습 | 쓰는 코드 |
|---|---|---|---|---|
| 분류, 10월 (SatMAE, ScaleMAE, 256 px) | 고정 | Linear 하나. 표준화 없음 (raw head) | `scripts/cls/ablation.py`, Adam | `scripts/tta/cls_tta.py` |
| 분류, 11월 (백본 17개, 224 px) | 고정 | 표준화 + Linear (std head) | L-BFGS 로지스틱 회귀 (§2.4 (3)) | `scripts/analysis/feat_metrics.py`, `feat_tta.py`, `scripts/tta/cls_tta_fm.py` |
| 분할 UPerNet (SatMAE B, ScaleMAE B) | **고정 아님. 전체 fine-tune** | UPerNet decoder (BN 약 13개) | AdamW, 12 epoch | `scripts/seg/seg_tta.py`, `seg_signals.py` |
| 분할 linear probe (백본 17개, 2026-11-25 실행 중) | 고정 | 표준화 + 1×1 conv | AdamW, 20 epoch | `scripts/seg/seg_probe.py` |

- "encoder는 고정, 뒤쪽만 학습"은 분류와 분할 linear probe에 맞는 말이다. 발표 표 T5의 Potsdam 결과(UPerNet)는 REOBench 설정대로
  encoder까지 fine-tune한 모델이다.
- 분류에서 "decoder"는 Linear 하나(head)다. 블록 안의 MLP와는 다른 것이다.
- 위 표는 **source 모델을 만드는 단계**다. TTA는 그 다음, 라벨 없는 test 배치가 들어올 때 하는 일이다. TENT·SAR 같은 방법은
  encoder 안 norm 층의 γ, β를 test 때 학습하고, 정렬은 아무것도 학습하지 않는다.

### 1.2 정렬이 들어가는 네 위치

```
x (픽셀)
 │   ← ① 입력 정렬 (ISA)           RGB 3채널
 ▼
Normalize: (x − m) / s             사전학습 때 쓴 고정 상수. 정렬이 아니라 전처리
 ▼
encoder 블록 0 … L−1
 │   ← ② 블록 정렬 (fsa)            토큰의 D채널
 ▼
final norm → h                     분류: CLS 벡터 (DINOv3는 [CLS, patch 평균], ResNet은 GAP)
 │   ← ④ head 입력 정렬             h의 D채널
 ▼
head (Linear)                      분할 UPerNet은 이 자리에 decoder가 있고, ③ BN 정렬은 decoder 안의 BN들
```

### 1.3 헷갈리기 쉬운 셋: Normalize, 입력 정렬, head 입력 정렬

| | 어디에 | 무엇을 무엇으로 | 목표 통계의 출처 | 배치마다 바뀌나 |
|---|---|---|---|---|
| Normalize (P → PN의 N) | 픽셀, encoder 바로 앞 | (x − m) / s. 모든 이미지에 같은 상수 | 사전학습 데이터(ImageNet, fMoW)의 상수. 모델에 딸려 옴 | 아니다. 적응이 아니다 |
| 입력 정렬 (ISA) | 픽셀, Normalize **앞** | test 배치의 RGB 평균·표준편차 → clean train의 RGB 평균·표준편차 | **다운스트림** clean train 이미지(AID train 2,000장, Potsdam train 500 패치). 숫자 6개 | 그렇다 |
| head 입력 정렬 | encoder 출력 h, Linear 앞 | test 배치의 채널별 평균·표준편차 → head가 학습 때 본 통계 | head를 학습한 clean train의 h. 숫자 2 × D개 | 그렇다 |

- **ISA는 모델 전체의 입력(encoder에 들어가는 사진)을 고친다.** head 입력을 가리키는 말이 아니다. 둘은 식의 모양만 같고 위치가 양 끝이다.
- ISA는 Normalize를 대신하지 않는다. 순서는 `ISA → Normalize → encoder`이고, Normalize는 평소와 똑같이 한다.
- ISA의 목표 통계는 사전학습 데이터의 것이 아니다. 사전학습 상수(Normalize의 m, s)를 목표로 써 본 적이 있는데
  (`cls_tta.py --isa_target pretrain`, 10월) clean 정확도가 2.6–3.5pt 떨어졌다. 이유는 §2.1.
- PN은 방법 이름이 아니라 10월 분류 head의 설정 이름이다. P = REOBench 공개 코드대로 Normalize 없이 학습·평가, PN = Normalize를 넣은 것.

### 1.4 정렬은 "주어진 모델"에 붙인다: 필요한 것과 한계

정렬은 이미 학습이 끝난 모델을 받아서 쓰는 방법이다. 그래서 그 모델과 함께 무엇이 주어졌는지가 한계가 된다.

- 필요한 것은 하나다. **정렬하는 위치의 값이 clean train에서 갖던 채널별 평균·표준편차**(ISA는 6개, head 입력은 2 × D개).
- head가 어떻게 학습됐는지(표준화를 따로 뒀는지, W와 b에 합쳐졌는지, 아예 안 했는지)는 **몰라도 된다.** head를 "h를 받는 상자"로 보고,
  h를 clean train의 h 통계로 되돌려서 넣으면 어느 경우든 같은 결과가 된다(§2.4 (5)).
- 통계가 없으면 라벨 없는 clean 다운스트림 이미지 몇 장으로 다시 낼 수 있다(10월 두 모델: 32장이면 이득의 대부분, §2.4 (7)).
- 통계도 clean 이미지도 없으면 할 수 없다. W, b만 보고는 통계를 되찾을 수 없기 때문이다.
- 따라서 이 방법은 "source-free"가 아니라 **"소스 통계를 쓰는 방법"**이라고 말해야 한다. BN이 running 통계를 들고 다니는 것,
  EATA가 Fisher를 내려고 소스 2,000장을 쓰는 것, FOA가 소스 CLS 통계를 쓰는 것과 같은 수준의 가정이다.

---

## 2. 방법별: 무엇을, 어떻게 구현했나

모두 학습 파라미터가 없다. 공통 식(채널 c마다):

```
h′ = (h − μ_test) / σ_test · σ_src + μ_src
```
- μ_test, σ_test: 지금 들어온 test 배치에서 계산한다. 배치마다 새로 계산하고 다음 배치로 넘기는 상태가 없다.
- μ_src, σ_src: clean **train** 이미지에서 한 번 계산한다. test 데이터는 쓰지 않는다.
- 배치 크기: 분류 64, 분할 8.
- 바뀌는 것은 "어디의 h냐"뿐이다.

| 방법 | 정렬 위치 | 채널 / 통계를 내는 범위 | 코드 | 근거 |
|---|---|---|---|---|
| 입력 정렬 (ISA) | 입력 픽셀, Normalize 앞 | RGB 3채널 / 배치의 모든 픽셀 | `src/reobench_tta/tta/isa.py`. 분류 `scripts/tta/cls_tta.py --method isa_batch`, 분할 `seg_tta.py`·`seg_signals.py`·`seg_probe.py`의 `isa` | 논문 방법 아님. 같은 연산: histogram moment matching, Reinhard 색 전이, AdaIN을 0층에 적용, RS 상대 방사 정규화 |
| 블록 정렬 (fsa) | 임의 모듈의 출력 (`:in`을 붙이면 입력) | 토큰의 D채널 / 배치의 모든 토큰 | `src/reobench_tta/tta/stat_align.py`, `--method fsa --fsa_layer <모듈>` | AdaBN을 임의 층으로 일반화. 관련: TTN(ICLR 2023), FOA(ICML 2024), test-time style shifting(ICML 2023), surgical fine-tuning(ICLR 2023) |
| BN 정렬 (bnadapt) | 분할 decoder의 BN 약 13개 | 각 BN 채널 / 배치의 모든 픽셀 | `seg_tta.py`의 `configure_norm`, `seg_signals.py`의 `bn` | Schneider et al. NeurIPS 2020; Nado et al. 2020 |
| head 입력 정렬 | Linear 앞의 h | D채널 / 배치의 벡터 64개 | 10월 `--fsa_layer head:in`, 11월 `feat_metrics.align "ms"`, `cls_tta_fm.py`의 `none_al`, 분할 `seg_probe.py`의 `fa` | AdaBN / BN adaptation을 head 앞 한 곳에 적용 (§2.4) |
| 조합 | 입력 정렬 + 위 중 하나 | | `--with_isa`, 분할 `isa_bn`, `isa_fa` | 우리 실험 |
| (비교) tent + BN batch | BN 정렬 + norm의 γ, β를 엔트로피로 **학습** | | `seg_tta.py --method tent --bn batch` | Wang et al. Tent, ICLR 2021 원문 설정 |

### 2.1 입력 정렬 (ISA)

```
x′ = (x − μ_test) / σ_test · σ_train + μ_train        (채널 = R, G, B)
그 뒤는 평소대로: (x′ − m) / s → encoder → head
```
- μ_test, σ_test: test 배치의 모든 픽셀(배치 단위 `isa_batch`). 이미지 한 장씩 하는 `isa_image`도 있다.
- μ_train, σ_train: 분류는 무작위 clean AID train 2,000장(`--n_src`, seed 0)의 [0, 1] 픽셀, 분할은 Potsdam train 패치 500장의 0–255 픽셀.
  2,000과 500은 근거 없는 내 선택이다. 32장으로 줄여도 ±0.9 이내였다(10월, §2.4 (7)의 표).
- 구현: 정렬을 픽셀 공간에서 해야 하므로 Normalize를 데이터 변환에서 빼고 모델 forward 안에서 한다(`cls_tta.py`의 `PIXEL_SPACE`, `Net.in_norm`).
- 목표를 사전학습 상수(m, s)로 두면: ISA 뒤에 Normalize를 하면 식이 (x − μ_test) / σ_test만 남는다. 즉 입력을 채널별 평균 0·표준편차 1로
  만드는 것이다. 그런데 head가 학습 때 본 입력은 그렇지 않았다(AID 픽셀의 표준편차는 0.19, SatMAE의 fMoW 상수는 0.28이라 Normalize 뒤
  표준편차가 약 0.68). 그래서 clean이 떨어진다. 규칙은 head 입력과 같다. **뒤쪽이 학습 때 보던 통계로 되돌려야 한다.**
- 한계: 채널별 a·x + b 꼴의 변화만 되돌린다. noise·blur·JPEG는 픽셀 평균·표준편차가 거의 안 변해서 고칠 것이 없다
  (`scripts/analysis/pixel_affine.py`). 구름이 섞인 배치에서는 통계가 틀어져 분할에서 −1.4 ~ −2.0.
- 11월 범위: 분할(SatMAE·ScaleMAE × 12종)에서 돌렸다. 17개 백본 분류의 AID-C / RESISC45-C에서는 돌리지 않았고,
  실제 haze(RRSHID)의 예측 일치율에서만 17개 백본으로 봤다(`scripts/analysis/rrshid_analysis.py`).

### 2.2 블록 정렬(fsa)과 BN 정렬

`stat_align.py`의 동작:
1. `source_stats`: clean train 배치를 통과시키며 forward hook으로 해당 모듈 출력의 채널별 합·제곱합을 누적 → μ_src, σ_src.
2. `StatAlign`: 같은 모듈에 hook을 달아 test 때 출력을 공통 식으로 바꿔치기한다. 뒤쪽 층은 바뀐 값을 받는다.
3. 채널 축: (B, N, D) 토큰 → D, (B, C, H, W) 맵 → C, (B, D) 벡터 → D (`_flat`).
4. `"head:in"`, `"decode_head.conv_seg:in"`처럼 `:in`을 붙이면 forward **pre**-hook으로 모듈의 **입력**을 정렬한다.
- 단위 테스트: 정렬 후 출력의 평균·표준편차 = 소스 값(확인함).
- 조합(`--with_isa`)에서는 소스 통계를 **ISA를 먼저 적용한 clean 배치**에서 낸다. test 때와 같은 입력 경로여야 하기 때문이다.

BN 정렬은 BatchNorm이 저장해 둔 running 통계 대신 test 배치의 통계를 쓰게 하는 것이다. ViT encoder에는 BN이 없어서
(LayerNorm만 있음) UPerNet decoder에만 해당한다. BN이 여러 개이므로 여러 위치를 한꺼번에 정렬하는 셈이다.

### 2.3 "final norm"과 "head 입력"이 다른 이유 (`NOTES_tta.md` §5.5)
- 분류 모델은 final LayerNorm 뒤 **CLS 토큰만** Linear로 간다.
- final norm 정렬: 통계를 CLS + 패치 토큰 전부로 낸다 → CLS에는 맞지 않는 통계로 CLS가 변형된다(10월 평균 −0.1 / −4.7).
- head 입력 정렬: 배치의 CLS 64개만으로 통계를 낸다 → 실제 분류에 쓰이는 분포를 직접 맞춘다(+5.2 / +8.0).
- 같은 깊이인데 결과가 반대다. 정렬의 깊이만큼 "통계를 어떤 토큰에서 내느냐"가 중요하다.

**[직접 확인]**
- `stat_align.py` 전체(110줄)를 읽고 `_apply`의 식이 공통 식과 같은지.
- 아무 블록이나 하나 골라 hook 전후의 평균·표준편차를 찍어 보기.

### 2.4 head 입력 정렬을 처음부터 (2026-11-23 문답 정리, 11-25 보강)

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
d. (z, 라벨)로 로지스틱 회귀(= Linear + softmax + cross-entropy인 선형 분류기)를 L-BFGS로 학습해 W, b를 얻는다. e. W, b, μ_src, σ_src를 함께 저장한다
(`/data/rs/feats/<model>/head_aid.pt`, `head_r45.pt`).
- 이유: 백본마다 feature 크기가 달라서, 표준화해야 같은 λ(L2 규제) 격자와 같은 절차를 17개 백본에 쓸 수 있다.
- L-BFGS는 head를 학습하는 optimizer일 뿐이다(전체 데이터를 한 번에 놓고 푸는 볼록 문제라 lr·epoch 선택이 필요 없음). 정렬의 근거가 아니다.
- 1단계는 상수로 하는 선형 변환이라 학습 뒤 W′ = W / σ_src, b′ = b − (μ_src / σ_src)W로 Linear에 합칠 수 있다.
- 선행 연구의 linear probe 코드는 어떻게 하나 (2026-11-23, 공식 repo의 해당 파일을 읽어 확인; 백본 안의 final norm은 별개):

  | 코드 | head 앞의 feature 정규화 | optimizer |
  |---|---|---|
  | MAE `main_linprobe.py` | 있음: `BatchNorm1d(affine=False, eps=1e-6)` + Linear | LARS |
  | Scale-MAE `mae/main_linprobe.py` | 있음: 같은 BatchNorm1d(affine=False) | AdamW |
  | DINO `eval_linear.py`, DINOv2 / DINOv3 `eval/linear.py` | 없음 (CLS 등을 그대로 Linear에) | SGD |
  | MoCo v3 `main_lincls.py` | 없음 | SGD |
  | SatDINO `satdino/eval_linear.py` | 없음 | AdamW |
  | CLIP README의 linear-probe 예시 | 없음 (feature를 그대로 sklearn LogisticRegression에) | sklearn 기본 solver |
  | SatMAE, MoCo v1 | 확인 못 함 (해당 파일 404) | – |

  즉 관행은 갈린다. MIM 계열(MAE, Scale-MAE)은 affine 없는 BatchNorm으로 표준화하고, DINO·MoCo v3·CLIP 계열은 하지 않는다.
  affine 없는 BatchNorm은 eval 때 저장된 train 통계로 (h − μ) / σ를 하므로 우리 1단계와 같은 계산이다. 우리는 이 MAE식 표준화를
  17개 백본 전부에 똑같이 적용한 것이고, DINO류의 공식 head는 (5)의 raw head에 해당한다. 10월의 분류 head(`cls_tta.py`가 읽는 P / PN head)도
  표준화 없는 Linear라 raw head다.

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

**(5) 일반 규칙: 정렬하는 지점의 값을, 그 지점이 clean train에서 갖던 통계로 되돌린다.**
test 통계로 표준화하는 것((h − μ_test) / σ_test)은 항상 한다. 그 뒤에 무엇을 곱하고 더할지는 "어느 지점을 보고 있느냐"로 정해진다.

| 정렬하는 지점 | 그 지점의 clean train 통계 | 정렬 식 |
|---|---|---|
| h (표준화 앞, 또는 표준화가 없는 head의 입력) | μ_src, σ_src | (h − μ_test) / σ_test · σ_src + μ_src |
| z (표준화 뒤, Linear 바로 앞) | 0, 1 | (h − μ_test) / σ_test |

- 두 식은 같은 계산이다. 위 식으로 되돌린 h′를 1단계에 넣으면 (h′ − μ_src) / σ_src = (h − μ_test) / σ_test가 되어 아래 식과 똑같다.
- 그래서 **head를 "h를 받는 상자"로 보고 위 식을 쓰면 head의 학습 방식을 몰라도 된다.** 표준화 층이 따로 있든(우리 std head, MAE식 BN head),
  W′, b′로 합쳐졌든, 처음부터 없었든(DINO식 raw head) 결과가 같다. 알아야 하는 것은 h의 clean train 통계 μ_src, σ_src뿐이다.
- 블록 사이에 넣는 fsa가 항상 위 식을 쓰는 이유도 같다. 다음 블록은 원래 크기의 값을 기대한다.
- 틀리는 경우는 하나다. 통계 없이 아래 식만 쓰는 것, 곧 표준화가 없거나 합쳐진 head에 (h − μ_test) / σ_test를 그대로 넣는 것이다.
  head가 학습 때와 다른 크기의 입력을 받는다.

**(6) 식을 맞게 쓰면 이득은 head 학습 방식에 거의 무관하다** (`scripts/analysis/head_nostd.py`, `experiments/2026-11-23_head_nostd/RESULTS.md`).

| 17개 백본 평균 손상 정확도 (%) | head | clean | 정렬 전 | 정렬 후 | 이득 |
|---|---|---:|---:|---:|---:|
| AID-C | std | 94.8 | 81.8 | 87.3 | +5.5 |
| | raw | 94.6 | 81.6 | 86.9 | +5.4 |
| RESISC45-C | std | 90.6 | 75.2 | 81.2 | +5.9 |
| | raw | 90.4 | 74.9 | 80.7 | +5.8 |

raw head에서도 17/17 백본에서 이득. moco_in, moco_fmow만 raw head의 정렬 전 정확도가 1.4~2.5pt 높아 이득이 조금 작다.

**(7) 주어진 것에 따라 할 수 있는 일** (§1.4의 상세).

| 모델과 함께 받은 것 | 정렬할 수 있나 |
|---|---|
| 표준화 층이 따로 있는 head (통계가 모델 안에 들어 있음) | 된다. 저장된 통계를 배치 통계로 바꾸면 끝 |
| 표준화가 없거나 W′, b′에 합쳐진 head + 따로 저장된 μ_src, σ_src | 된다. (5)의 위 식 |
| 같은 head, 통계는 없고 라벨 없는 clean 다운스트림 이미지 몇 장 | 된다. 그 이미지로 μ_src, σ_src를 다시 낸다 |
| W′, b′뿐 | 안 된다 |

- W′, b′만으로는 μ_src, σ_src를 되찾을 수 없다. W′ = W / σ_src는 어떤 σ_src를 넣어도 성립하는 식이라, 같은 W′, b′를 만드는 (W, b, μ_src, σ_src)가 무수히 많다.
  표준화가 합쳐졌는지 아닌지도 W′, b′만 봐서는 구분되지 않는다. 다만 (5)에 따라 구분할 필요는 없다.
- clean 이미지 몇 장이면 얼마나 되나 (`NOTES_tta.md` §5.8, 10월 SatMAE / ScaleMAE, AID, raw head, 평균 이득):

  | 통계를 낸 clean train 이미지 | 입력 정렬 | 블록 7 | head 입력 | 입력 + 블록 7 |
  |---|---|---|---|---|
  | 2,000장 | +1.67 / +1.32 | +4.89 / +1.02 | +5.24 / +7.97 | +7.04 / +3.12 |
  | 32장 | +1.88 / +0.58 | +4.91 / +0.95 | +3.71 / +7.22 | +7.18 / +2.22 |
  | 사전학습 Normalize 상수 (이미지 없음) | −0.16 / −2.39 (clean −2.60 / −3.05) | – | – | +2.22 / −2.59 |

  32장이면 head 입력 정렬이 0.7~1.5pt만 낮다(1,024차원 통계의 추정 오차). 다운스트림 통계가 전혀 없는 경우는 head 입력에서 시험하지 않았다.
- 17개 백본에서는 이 줄이기 실험을 하지 않았다(전부 clean train 전체로 낸 통계).

**(8) 근거와 구분.**
- 2단계의 근거: AdaBN / BN adaptation(정규화 통계를 test 배치 것으로 교체). 평균만 맞추는 `align_m`은 FOA의 activation shifting과 같은 위치·같은 종류
  (FOA는 EMA로 test 평균을 추적, 우리는 배치마다 계산). 연산 자체는 새롭지 않다.
- 정렬과 **feature 단계 TTA**는 다른 실험이다. `feat_tta.py`의 tent / flipf / dsbr는 head 입력에 z′ = γ ⊙ z + β를 붙이고 γ, β를 엔트로피 손실로
  **학습**한다(ViT에서는 final norm의 γ, β만 학습하는 TENT와 같음). `_al`은 정렬한 z_al 위에서 그 학습을 한 것.
- 분할 linear probe(`seg_probe.py`)의 `fa`는 같은 계산을 패치 feature 맵에 한 것이다. 통계는 배치 8장의 모든 위치에서 낸다.

---

## 3. 근거가 된 관찰과 지금의 수치

### 3.1 입력 Normalize 하나로 +7 (분류, 10월, `NOTES_tta.md` §3)
- REOBench 공개 프로토콜(P)은 `ToTensor()`만 하고 Normalize가 없다. 넣으면(PN) 손상 평균이 약 +7.
- Normalize는 고정 상수로 하는 선형 변환일 뿐인데 이만큼 차이가 난다 → 모델이 입력 채널 통계에 민감하다.
- 그래서 "test 배치의 통계를 train 통계로 맞추면 밝기류 손상도 되돌아오지 않을까"가 입력 정렬의 출발이었다.

### 3.2 학습하는 TTA는 거의 효과가 없고, 이득은 통계 교체에서 온다

| | 설정 | 엔트로피 방법 | 정렬 |
|---|---|---|---|
| 분류 10월 | AID, 2개 모델 | Tent / SAR / DeYO / EATA / CoTTA ±1~2, CoTTA는 큰 lr에서 붕괴 | head 입력 +5.2 / +8.0 |
| 분류 11월 | 17개 백본, AID-C / RESISC45-C | none 81.8 / 75.2 → tent 81.4 / 71.2, sar 80.6 / 71.8, eata 81.9 / 71.1 | 87.3 / 81.1 |
| 분할 11월 | Potsdam 12종, SatMAE / ScaleMAE | none 52.8 / 54.0, tent −1.6 / −1.0 | BN 54.3 / 56.3, 입력 54.7 / 55.6, 둘 다 55.9 / 57.5 |

- 분할에서 tent에 BN 정렬을 얹으면 +2.2 / +3.0이고, 그중 BN 정렬만의 몫이 +1.5 / +2.3이다. 학습이 더하는 것은 +0.7 안팎이다.
- 분류에서도 정렬 위에 학습을 얹은 것(tent + 정렬 88.1 / 83.0)이 정렬만(87.3 / 81.1)보다 조금 높은 정도다.
- 출처: `experiments/2026-11-17_fm_tta/RESULTS.md`, `2026-11-19_dsbr/RESULTS.md`, `2026-11-23_seg_signals/RESULTS.md`.

### 3.3 손상마다 맞는 위치가 다르다 (depth sweep, 10월, `NOTES_tta.md` §5.5–5.6)

none 대비 변화, 평균 / 밝기류(radiometric) / noise류(degradation):

| 위치 | 분류 SatMAE | 분류 ScaleMAE | 분할 ScaleMAE | 분할 SatMAE |
|---|---|---|---|---|
| 입력 (ISA) | +1.7 / **+11.1** / +0.1 | +1.3 / **+8.4** / −0.3 | +4.9 / **+19.1** / +0.5 | +5.8 / **+25.4** / −0.7 |
| 블록 7 | +4.9 / −2.1 / **+12.0** | +1.0 / −4.5 / +3.9 | +8.0 / +9.3 / **+9.3** | +4.5 / +13.3 / +2.1 |
| 블록 23 | −0.5 / −2.4 / −0.3 | −2.9 / −5.2 / −4.3 | +2.7 / +3.5 / +3.0 | +1.6 / +2.8 / +1.5 |
| head 입력 / 분류기 입력 | **+5.2** / +6.0 / +8.5 | **+8.0** / +5.6 / **+13.7** | +4.2 / +4.8 / +5.0 | +2.6 / +4.4 / +2.5 |
| decoder BN | | | +4.5 / +5.8 / +5.0 | +3.0 / +5.5 / +2.7 |
| 입력 + 블록 7 | +7.0 / +11.5 / +12.0 | +3.1 / +8.8 / +3.6 | +10.8 / +19.2 / +9.9 | +7.2 / +25.6 / +1.7 |

- 밝기류는 네 설정 모두 입력 정렬이 최선이다. noise류는 중간 블록(3–11)이 좋고, 분류 평균은 head 입력이 가장 높다.
- 조합은 그룹별로 거의 그대로 더해진다(밝기류 ≈ 입력 단독, noise류 ≈ 블록 단독).
- 11월 분할 12종에서도 같은 그림이다. 밝기·haze는 입력 정렬이 clean 수준까지 되돌리고(SatMAE 49.1 → 61.5, 44.2 → 61.5),
  noise·JPEG·data gaps는 BN 정렬이 듣고(+2.4 ~ +9.7), blur·구름·기하 3종은 하락이 작고 어떤 정렬도 듣지 않는다.

**[직접 확인]** `experiments/2026-10-20_seg_tta/summary.txt`에서 isa와 bnadapt의 손상별 열을 비교. 11월은 `docs/figs/midterm_2026-11-26/fig9_seg.png`.

### 3.4 정렬 뒤에 남는 것 (11월)
- head를 라벨로 다시 맞춘 상한과 비교하면(3·5단계, 17개 백본) AID-C는 그대로 77.1 → 정렬 84.2 → 상한 89.2, RESISC45-C는 70.5 → 78.1 → 84.2.
  남는 여지는 motion blur, data gaps, scale에 몰려 있다(`experiments/2026-11-24_cluster_cot/RESULTS_head_oracle.md`).
- data gaps는 입력에서 무효 영역을 채우면 head 상한보다 높다(AID의 data gaps 3·5단계: head 상한 85.0, 보간 88.8, 보간 + 정렬 90.8;
  `experiments/2026-11-25_undo_acquisition/RESULTS_gaps.md`). 정렬이 못 고치는 손상은 입력 쪽에서 원인을 되돌리는 편이 낫다는 근거다.

---

## 4. 실험 설계 결정과 약점

| 결정 | 이유 | 근거 강도 |
|---|---|---|
| mild 프로토콜 ((손상, 단계)별 독립, 매번 리셋) | 사용자 지시("continual은 필요없음"). 의료 분할 TTA(VPTTA, GraTa)의 사이트별 평가와 같은 구조 | 지시 |
| 배치 분류 64, 분할 8 | 분류는 DeYO / SAR의 값, 분할은 학습 배치와 같은 값 | 관례 |
| 10월 분류는 PN head | P는 Normalize가 빠진 비정상 기준선. PN이 주 결과, P는 논문 재현 보충 | 앞선 분석 |
| 11월 분류는 표준화 + L-BFGS head | 17개 백본에 같은 절차를 쓰기 위해 (§2.4 (3)) | 내 선택. raw head로 바꿔도 이득 같음 (§2.4 (6)) |
| 분할 B 설정 (블록 7 / 11 / 15 / 23) | ViT-L 관례, 사용자 선택 | 지시 |
| sweep 블록 0, 3, 7, 11, 15, 19, 23 | 24층을 대략 4층 간격으로. 7·11·15·23은 분할 decoder가 가져가는 블록과 겹침 | 내 선택 |
| 10월 분할은 3·5단계만 | 비용(1회 15분). 11월 12종 실험은 5단계 전부 | 내 선택 |
| 조합 블록 3 / 7 / 11 (분할), 7 / 11 / head (분류) | sweep에서 noise류가 좋았던 위치 | **test 결과를 보고 고름** |

### 4.1 "test에서 골랐다"는 문제
- 블록 7을 test 표를 보고 골랐으므로 그 점수는 상한(낙관적)이다.
- 정당한 방법: (a) 손상 일부로 위치를 고르고 나머지 손상으로 보고(leave-out), (b) 분류에서 고른 위치를 분할에 그대로, (c) "깊이 1/3" 같은 규칙을 미리 고정.
- 완화 요인: 하이퍼파라미터가 위치 하나뿐이고, 고정 처방 "입력 + 블록 7"이 4개 설정 모두 입력 단독보다 좋다.
- head 입력 정렬과 BN 정렬은 고를 위치가 없어서 이 문제가 없다. 11월에 이 둘을 주 결과로 쓰는 이유 중 하나다.

### 4.2 알려진 약점
- seed 0 한 번. 배치는 한 손상만 들어오는 i.i.d. 셔플이다 → 배치에 손상이 섞이거나 배치 크기 1이면 무너질 수 있다.
- 소스 통계가 필요하다(§1.4). "source-free"라고 쓰면 안 된다.
- 입력 정렬은 분할의 구름에서 −1.4 ~ −2.0.
- Yi et al.(ACM MM 2024)은 분할에서 BN 통계 갱신 효과가 거의 없다고 보고 → 우리 결과와 상충, 설명 필요.
- 분할 결과는 encoder까지 fine-tune한 두 모델뿐이다. encoder를 고정한 17개 백본 분할(linear probe)은 2026-11-25에 실행 중.

---

## 5. 선행 연구와의 관계 (`PRIOR_ART_stat_alignment.md` 요약)

| 우리 요소 | 이미 있는 것 | 아직 못 찾은 것 |
|---|---|---|
| 입력 정렬 | moment / histogram matching, Reinhard, AdaIN@0층 — **새롭지 않음** | RS 기반 모델 TTA에서 엔트로피 TTA보다 낫다는 보고 |
| BN 정렬 | Schneider 2020 등 — 표준 baseline | |
| head 입력 정렬 | AdaBN을 head 앞에 적용. FOA의 activation shifting과 같은 위치 — **새롭지 않음** | |
| 어느 깊이에서 정렬? | Surgical fine-tuning(입력 수준 손상 → 앞층), TTN, FOA, style shifting | **손상 메커니즘별로 맞는 깊이가 다르다**(밝기류 = 입력, noise류 = 중간)는 정량 sweep |
| 손상 판별 후 처방 | PAN, FROST (BN 통계만 선택) | 입력 정렬과 feature 정렬 중에서 고르는 라우터 |

- Surgical fine-tuning 논리로는 noise·JPEG도 "입력 수준" 손상이라 앞층이 맞아야 하는데, 우리는 중간 블록이 최선 → 이 긴장이 기여 포인트.

**[직접 확인]** 먼저 읽을 논문 3편: Lee et al. *Surgical Fine-Tuning* (ICLR 2023), Lim et al. *TTN* (ICLR 2023),
Schneider et al. *Improving robustness against common corruptions by covariate shift adaptation* (NeurIPS 2020).

---

## 6. feature shift 진단 (10월, `scripts/analysis/feature_shift.py`)
- 질문: 왜 밝기류는 입력, noise류는 중간 블록인가? 왜 ScaleMAE 분류는 블록 정렬에 약한가?
- 방법: clean test 400장과 같은 이미지의 손상본(5단계)을 통과시켜 위치별로
  - d_mu, d_sd: 채널 평균·스케일이 얼마나 움직였나
  - cos → cos_al: 같은 이미지·같은 토큰의 clean / 손상 feature 코사인 유사도가 정렬로 얼마나 회복되나
  - outlier: 소수 채널이 에너지를 독점하는지(표준화가 깨지기 쉬운 구조)
- 출력: `experiments/2026-10-22_feature_shift/summary.txt`
- 결과 요약 (상세는 `NOTES_tta.md` §5.7):
  - 밝기류: 입력에서 스케일(d_sd)이 크게 바뀌고 깊어질수록 줄어든다 → 입력 정렬이 맞다.
  - noise류: 입력에서는 평균이 안 움직여 입력 정렬로 못 고친다. 중간 블록에서 평균 이동으로 나타난다 → 중간 블록 정렬이 맞다.
  - 소수 채널(상위 1%)이 에너지의 60~95%를 차지한다(massive activation). 원 코사인 지표는 이 채널에 좌우된다.
  - 분류 ScaleMAE가 블록 정렬에 약한 이유는 아직 설명하지 못했다.
- 11월의 17개 백본 feature 지표(rel_drift, frac_common, CKA 등)는 `STUDY_stage1_metrics.md` §4.

---

## 7. 스스로 점검할 질문 (답할 수 있으면 이 줄기를 파악한 것)
1. Normalize, 입력 정렬, head 입력 정렬은 각각 어디에서, 어느 데이터의 통계로 하는가? (§1.3)
2. 입력 정렬의 목표를 사전학습 상수로 두면 왜 clean이 떨어지는가? (§2.1)
3. 입력 정렬과 BN 정렬은 각각 어떤 공간의 어떤 채널을 정렬하나? 왜 둘의 효과가 손상별로 다른가?
4. head 입력 정렬에서 μ_test는 몇 개 값으로 계산되나? final norm 정렬과 무엇이 다른가? (§2.3)
5. 표준화가 W, b에 합쳐진 head를 받았다. 무엇이 있으면 정렬할 수 있고, 어떤 식을 쓰나? (§2.4 (5), (7))
6. tent + BN batch와 BN 정렬의 차이가 작다는 것은 무엇을 뜻하나?
7. 조합의 소스 통계를 입력 정렬 뒤에 낸 이유는?
8. "입력 + 블록 7" +10.8(분할 ScaleMAE)을 논문에 그대로 쓰면 왜 안 되나? (§4.1)
9. 이 결과가 Surgical fine-tuning의 주장과 어디서 맞고 어디서 다른가?
10. 분류의 "encoder 고정"이 Potsdam T5 결과에는 왜 해당하지 않는가? (§1.1)
