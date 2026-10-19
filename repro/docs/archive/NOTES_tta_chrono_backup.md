# TTA 코드 분석: DeYO 공식 repo (TENT / SAR / DeYO)

- repo: https://github.com/Jhyun17/DeYO (commit 3d7d289, 2024-05-21). README: "This implementation is based on SAR implementation".
- `methods/sar.py`의 `collect_params` / `configure_model`은 SAR 공식 repo(mr-eggplant/SAR)와 diff 결과 동일합니다.
- 세 방법이 같은 `main.py`, 같은 파라미터 선택, 같은 optimizer 규칙을 공유합니다. 그래서 이 repo를 기반으로 하면 구현 차이 없이 공정하게 비교할 수 있습니다.

## 1. 파일 구조

| 파일 | 역할 |
|---|---|
| `main.py` | 왜곡별 루프, 모델 생성, lr 규칙, 시나리오(normal / mix_shifts / label_shifts / bs1) 처리, online 정확도 측정 |
| `config.py` | 하이퍼파라미터 기본값 |
| `methods/tent.py` | TENT |
| `methods/sar.py` + `methods/sam.py` | SAR (SAM optimizer 포함) |
| `methods/deyo.py` | DeYO |
| `dataset/generate_shifted_sample_indices.py` | label shift 시나리오용 샘플 순서 생성 (ImageNet용 .npy 파일이 미리 들어 있음) |

## 2. 세 방법의 공통 뼈대

```python
net = method.configure_model(net)        # train(), 전체 requires_grad=False, norm layer만 True
params, names = method.collect_params(net)  # 업데이트할 파라미터 = norm layer의 weight(γ), bias(β)
optimizer = SGD(params, lr, momentum=0.9)   # SAR만 SAM(SGD)으로 감쌈
for x, y in loader:                      # 테스트 스트림 (한 번만 지나감 = online)
    out = adapt_model(x)                 # ① 예측 → ② loss 계산 → ③ 업데이트
    acc += (out.argmax(1) == y)          # 정확도는 업데이트 "전" 예측(①)으로 측정
```

- **online:** 테스트 데이터를 한 번만 지나가면서 매 batch마다 바로 업데이트합니다.
- **정확도 측정 시점:** 각 batch는 자기 자신으로 업데이트되기 직전의 예측으로 채점됩니다.
- **리셋:** `main.py`가 왜곡 종류마다 `net`을 새로 만듭니다. 즉 왜곡 종류가 바뀔 때마다 사전학습 상태로 돌아갑니다.
- **업데이트하는 파라미터:** LayerNorm/GroupNorm/BN의 affine(γ, β)뿐입니다. ViT-B에서는 전체 파라미터의 극히 일부입니다.
- **상위 블록 제외:** `collect_params`가 `blocks.9`, `blocks.10`, `blocks.11`(ViT-B 12블록 중 마지막 3개)과 마지막 `norm`을 업데이트 대상에서 뺍니다. SAR 논문 기준으로 상위 층을 건드리면 불안정해서입니다. **이 repo에서는 TENT도 같은 함수를 쓰므로**, 원래 TENT(BN 전체 업데이트)와 다릅니다. 세 방법이 같은 파라미터를 업데이트한다는 점에서는 공정합니다.

## 3. 방법별 loss

### TENT (`methods/tent.py`)
```python
loss = softmax_entropy(outputs).mean(0)   # 모든 샘플의 예측 엔트로피 평균
loss.backward(); optimizer.step()
```
- 예측을 더 확신하는 방향(엔트로피를 낮추는 방향)으로 norm 파라미터를 움직입니다.
- 틀린 예측도 확신하게 만들 수 있어서, 심한 shift나 label shift에서는 붕괴(모든 샘플을 한 클래스로 예측)가 일어납니다.

### SAR (`methods/sar.py`, `methods/sam.py`)
```python
ent = softmax_entropy(model(x))
keep = ent < E0                          # ① 신뢰할 수 있는 샘플만 (E0 = 0.4·ln C)
loss = ent[keep].mean(); loss.backward()
optimizer.first_step()                   # ② SAM: 가중치를 ρ만큼 loss가 커지는 방향으로 이동 (ρ = 0.05)
ent2 = softmax_entropy(model(x[keep]))   #    그 위치에서 다시 forward
loss2 = ent2[ent2 < E0].mean(); loss2.backward()
optimizer.second_step()                  #    원위치로 돌아와 ②의 gradient로 업데이트
ema = 0.9·ema + 0.1·loss2                # ③ model recovery: ema < 0.005면 리셋
```
- ① 엔트로피가 큰(불확실한) 샘플은 gradient가 크고 해로워서 제외합니다.
- ② sharpness-aware minimization으로 평평한 최솟값을 찾아, 노이즈 gradient에 덜 민감하게 만듭니다.
- ③ 엔트로피가 0 근처로 붕괴하면 모델을 원래대로 되돌립니다.
- forward와 backward가 두 번씩이라 TENT보다 약 2배 느립니다.

### DeYO (`methods/deyo.py`)
```python
ent = softmax_entropy(model(x))
keep1 = ent < τEnt                                   # ① 엔트로피 필터 (τEnt = 0.5·ln C)
x' = patch_shuffle(x[keep1])                         # ② 이미지를 4×4 조각으로 섞음 (patch_len=4)
plpd = p(ŷ | x) - p(ŷ | x')                          #    같은 클래스 ŷ의 확률이 얼마나 떨어지나
keep2 = plpd > τPLPD                                 # ③ PLPD 필터 (τPLPD = 0.2, mild 0.3)
w = 1/exp(ent - Ent0) + 1/exp(-plpd)                 # ④ 재가중: 확신 높고 PLPD 클수록 큰 가중치 (Ent0 = 0.4·ln C)
loss = (ent * w).mean()
```
- 핵심 아이디어: 엔트로피가 낮아도 배경 같은 엉뚱한 단서(spurious feature)로 확신하는 샘플일 수 있습니다.
- 이미지 조각을 섞으면 물체의 모양(shape) 정보가 깨집니다. 그래도 예측이 별로 안 바뀐다면(PLPD 작음), 모양이 아니라 색이나 질감으로 판단했다는 뜻입니다. 이런 샘플은 업데이트에서 제외합니다.
- 추가 forward(x')는 `no_grad`라서 비용은 TENT의 약 1.5배입니다.

## 4. 하이퍼파라미터 (ViT, `main.py` 기준)

| 항목 | 값 | 코드 |
|---|---|---|
| optimizer | SGD, momentum 0.9 (SAR은 SAM으로 감쌈, ρ=0.05) | L387, L466, L588 |
| lr | `0.001 / 64 × bs` → bs 64면 0.001 | L352 |
| bs 1 | TENT: 0.001/64, **SAR·DeYO: 2배 → 0.001/32** | L373-379 |
| threshold | `margin × ln(num_class)`: SAR E0 = 0.4, DeYO τEnt = 0.5, Ent0 = 0.4 | L268-271 |
| τPLPD | wild 0.2, mild 0.3 (`exp_deyo.sh`) | |
| 시나리오 | `normal`(mild), `mix_shifts`, `label_shifts`, `bs1` | L219-247 |
| label shift | 사전 생성된 index 파일. 클래스 순서를 섞고 imbalance ratio 500,000(≈순수 클래스 순서), 복원추출로 100,000개 | L335-346 |
| mix shift | 15개 왜곡을 이어 붙이고 shuffle | L219-239 |

## 5. 우리 모델(ViT-L/16, AID 30클래스)에 그대로 쓰면 안 되는 부분 ⚠️

1. **업데이트 제외 블록이 ViT-B 기준으로 하드코딩됨.**
   `'blocks.9' / 'blocks.10' / 'blocks.11'`을 문자열로 빼기 때문에, 24블록인 ViT-L에 그대로 쓰면 **중간 블록 9~11을 빼고 최상위 12~23을 업데이트**합니다. 의도와 정반대입니다.
   → 같은 비율(상위 25%)로 바꿔서 ViT-L에서는 blocks 18~23을 제외해야 합니다.
2. **`configure_model`이 `model.train()`을 부름.**
   우리 backbone은 drop_path 0.1~0.2로 만들었기 때문에, TTA 중에 무작위로 블록을 건너뛰게 됩니다(NOTES_concepts.md 3절).
   → TTA용 모델은 `drop_path=0`으로 만들어야 합니다. timm 공식 ViT-B는 기본값이 0이라 원 논문에는 이 문제가 없었습니다.
3. **클래스 수 C가 1000으로 고정.**
   threshold가 `× ln(num_class)`로 계산되므로 `num_class = 30`으로 바꿔야 합니다.
   DeYO의 `filter_ent=0` 분기에는 `math.log(1000)`이 하드코딩돼 있지만 기본값(filter_ent=1)에서는 쓰이지 않습니다.
4. **label shift index 파일이 ImageNet(1000클래스, 50,000장)용.**
   → AID용(30클래스, 2,000장)을 같은 방식으로 새로 생성해야 합니다.
5. **우리 모델 구조.** backbone(`forward_features`)과 Linear head가 분리돼 있으므로, 하나의 `nn.Module`로 감싸서 넘겨야 합니다. head는 LayerNorm이 아니라서 자동으로 업데이트 대상에서 빠집니다(원 논문과 같음).
6. **step 수.** 왜곡당 2,000장 ÷ 64 ≈ 31 step으로, ImageNet-C(50,000장, 약 780 step)보다 25배 적습니다. TTA 효과가 작게 나올 수 있습니다.
7. **head 가중치가 아직 없음.** TTA는 학습된 head가 필요합니다. C_ablation 스크립트는 head를 저장하지 않으므로 A0ep(논문 프로토콜)와 norm 설정의 head를 다시 학습해서 저장해야 합니다.
   - SatMAE는 캐시를 써서 몇 분이면 됩니다.
   - Scale-MAE A0ep는 drop path를 켜고 학습하므로 약 40분이 걸립니다.

## 6. 첫 결과 (mild, seed 0, 2026-10-13) — `D_tta/summary_tta.txt`

왜곡 평균(12종 × severity 1~5)과 none 대비 변화. `S5`는 severity 5 평균입니다.

| | none | TENT | SAR | DeYO |
|---|---|---|---|---|
| SatMAE P | 64.9 | **+1.6** | −1.5 | −10.1 |
| SatMAE PN | 72.4 | 0.0 | **+0.8** | +0.2 |
| Scale-MAE P | 68.0 | −14.1 | **−47.1 (붕괴)** | **−39.0 (붕괴)** |
| Scale-MAE PN | 74.8 | **+1.1** | +0.1 | −7.7 |

- 입력 정규화 하나(P → PN)의 효과(+7.5 / +6.8)가 어떤 TTA 방법보다 훨씬 큽니다.
- Scale-MAE P 붕괴 진단(`D_tta/diag.py`, clean 스트림):
  - 첫 step 뒤 파라미터 이동량(drift)이 SAR 1.14로, PN 0.0006·SatMAE P 0.002와 비교해 수백 배입니다. 둘째 batch 정확도는 95 → 2%로 떨어집니다.
  - DeYO는 첫 step drift가 0.028(TENT의 10배, 재가중 때문)이고, 직후 `blocks.0.norm1.weight` gradient가 2.8 → 1,549로 폭발합니다.
  - TENT는 step당 drift 약 0.003~0.009로 안정적입니다.
  - 즉 첫 블록 LayerNorm 근처에 매우 가파른 방향이 있고, SAM 섭동(ρ=0.05)과 DeYO의 큰 step이 그 방향으로 들어가면 발산합니다.
  - 정규화가 빠지면 첫 블록 입력에서 이미지 정보 비중이 두 모델 모두 약 35% 줄지만(token-to-token 변동 / token std: 0.63 → 0.41~0.43), SatMAE P는 안정적이었습니다. Scale-MAE에만 나타나는 이유는 아직 모릅니다.
- DeYO가 업데이트에 쓴 샘플 비율(τPLPD = 0.3): SatMAE 34%, Scale-MAE 6~10%.
  - 가설: AID 장면 클래스(숲, 사막, 농지 등)는 모양보다 질감(texture)으로 정의되는 경우가 많습니다. patch를 섞어도 예측이 잘 안 바뀌므로 PLPD가 작아지고, 대부분의 샘플이 걸러집니다. DeYO의 전제("모양 정보가 중요하다")가 원격탐사 장면 분류에는 약할 수 있습니다. (아직 검증하지 않음)
- 주의: seed 1개, 스트림당 약 31 step, 하이퍼파라미터는 ImageNet ViT-B 값(lr 0.001)을 그대로 사용했습니다.

## 7. DeYO 질감 가설 검증 (`D_tta/plpd_analysis.py`, clean test, 적응 없는 원래 모델)

클래스 그룹은 결과를 보기 전에 코드에 고정했습니다.
- texture 6: BareLand, Desert, Farmland, Forest, Meadow, Mountain
- shape 8: Airport, BaseballField, Bridge, Church, Playground, Stadium, StorageTanks, Viaduct
- other 16: 나머지

지표 정의:
- PLPD: p(ŷ|x) − p(ŷ|x'), x'는 patch_len × patch_len 격자로 섞은 이미지
- sel: PLPD > 0.3 비율 (DeYO mild 기준선이 업데이트에 쓰는 샘플)
- kept: 섞은 뒤에도 예측 클래스가 그대로인 비율

patch_len = 4 (DeYO 기본값), 클래스 평균:

| | texture PLPD / sel / kept | shape PLPD / sel / kept | other PLPD / sel |
|---|---|---|---|
| SatMAE P | 0.039 / 5% / 94% | 0.481 / 55% / 50% | 0.257 / 34% |
| SatMAE PN | 0.024 / 4% / 97% | 0.453 / 50% / 54% | 0.259 / 33% |
| Scale-MAE P | 0.012 / 2% / 97% | 0.132 / 19% / 83% | 0.071 / 12% |
| Scale-MAE PN | 0.015 / 2% / 98% | 0.068 / 10% / 93% | 0.063 / 9% |

- **질감형 클래스는 조각을 섞어도 예측이 94~98% 유지되고, DeYO 선택률은 2~5%입니다.** 사실상 DeYO 업데이트에서 빠집니다.
  - Mountain: PLPD 0.000, 선택 0%
  - Desert: PLPD 0.006, 선택 2%
- **모양형 클래스는 예측이 크게 바뀝니다(SatMAE).**
  - Stadium: 선택 97%, 유지 3%
  - Playground: 선택 94%, 유지 9%
  - BaseballField: 선택 80%, 유지 24%
- other 그룹 중 주거지·상업지·주차장·항구도 질감형처럼 동작합니다(PLPD 약 0.05). 반면 Pond, Center, Park는 모양형처럼 동작합니다.
- 그룹 순서(texture < other < shape)는 patch_len 2/4/8 모두에서 같습니다. patch를 잘게 쪼갤수록 PLPD가 커집니다.
- **Scale-MAE는 모든 그룹에서 조각 섞기에 둔감합니다**(모양형도 선택 10~19%). DeYO 선택률이 6~10%로 낮았던 이유가 이것입니다. 패치들을 순서 없이 모아 보는 방식(bag-of-patches)에 가깝게 판단한다는 의미일 수 있습니다.
- 함의: DeYO의 업데이트는 일부 모양형 클래스 샘플에 치우칩니다. 이것이 성능 하락의 원인인지(예: 예측이 모양형 클래스 쪽으로 쏠림)는 아직 검증하지 않았습니다.

## 8. TRAP 쪽 확인: 왜곡 이미지에서 PLPD가 정답·오답을 가르는가 (`D_tta/trap_analysis.py`, PN, 적응 없음)

- AUROC: 점수가 정답 샘플을 오답 샘플보다 높게 매기는 확률입니다(0.5 = 신호 없음).
- acc_sel: DeYO 필터(엔트로피 < 0.5 ln C **그리고** PLPD > 0.3)를 통과한 샘플의 정확도입니다.
- acc_ent: 엔트로피 필터만 통과한 샘플의 정확도입니다.

| stream | SatMAE AUROC PLPD / 엔트로피 | SatMAE acc_sel / acc_ent | Scale-MAE AUROC PLPD / 엔트로피 | Scale-MAE acc_sel / acc_ent |
|---|---|---|---|---|
| clean | **0.44** / 0.93 | 85.8 / 92.8 | **0.46** / 0.95 | 87.1 / 95.5 |
| haze s3 | 0.52 / 0.89 | 79.8 / 84.8 | 0.55 / 0.90 | 84.3 / 90.8 |
| haze s5 | 0.64 / 0.78 | **66.1 / 60.1** | 0.49 / 0.83 | 69.6 / 75.4 |
| brightness s3 | 0.46 / 0.91 | 83.2 / 90.0 | 0.45 / 0.93 | 89.7 / 94.8 |
| brightness s5 | 0.66 / 0.76 | **54.5 / 44.5** | 0.53 / 0.78 | 52.5 / 54.1 |
| clouds s5 | 0.49 / 0.85 | 74.6 / 80.2 | 0.44 / 0.88 | 71.8 / 84.0 |
| gauss noise s5 | 0.58 / 0.78 | 55.7 / 55.9 | 0.53 / 0.79 | 66.7 / 60.2 |
| gauss blur s5 | 0.66 / 0.77 | 52.9 / 45.2 | 0.50 / 0.78 | 39.0 / 44.1 |
| motion blur s3 | 0.57 / 0.67 | 30.9 / 26.3 | 0.43 / 0.79 | 23.0 / 30.8 |

- **PLPD는 정답 여부를 거의 가르지 못합니다**(AUROC 0.43~0.66). clean과 약한 왜곡에서는 0.5 아래로, 오히려 반대 방향입니다. 질감형 클래스는 대부분 맞히는데 PLPD가 낮기 때문입니다.
- **엔트로피가 훨씬 나은 신호입니다**(AUROC 0.67~0.95).
- **PLPD 필터를 추가하면 고른 샘플의 정확도가 대부분 떨어집니다**(acc_sel < acc_ent). 예외: SatMAE의 강한 방사 왜곡(haze s5 +6, brightness s5 +10, gauss blur s5 +8, gauss noise 동률)에서는 PLPD가 도움이 됩니다. Scale-MAE에서는 gauss noise s5만 예외(+6.5)입니다.
- 결론: "PLPD로 CPR(질감)과 TRAP(연무 등)을 구분할 수 없다"는 주장은 대체로 맞습니다. 다만 SatMAE의 심한 방사 왜곡에서는 일부 신호가 있습니다. 하나로 뭉뚱그리지 말고 이 예외도 함께 보고해야 합니다.

## 9. 왜곡 분류 체계와 그룹별 TTA 가설 (2026-10-13 논의)

REOBench 논문(2.1절)의 분류는 environmental(cloud, brightness, haze) / sensor-induced(blur, noise, gap, JPEG) / geometric입니다.
TTA 관점에서는 "원인 메커니즘"으로 나누는 게 낫다고 판단해 아래 4그룹을 씁니다.
(사용자 제안 3그룹에서 '나머지'를 radiometric과 degradation으로 분리)

| 그룹 | 왜곡 | 메커니즘 | PN none (SatMAE / Scale-MAE) | 기존 TTA(TENT/SAR/DeYO) 효과 | 가설: 맞는 처방 |
|---|---|---|---|---|---|
| geometry | rotate, scale, translate | 픽셀 배치 변화, 통계 거의 불변 | 86.3 / 91.3 | ±0.2 (DeYO만 손해) | 회전·뒤집기 test-time augmentation 평균 (불변성) |
| occlusion | clouds, data_gaps | 일부 영역 정보 소실 | 79.7 / 80.0 | +1 내외 | **MAE 마스킹**: 가려진 패치를 토큰에서 제외 (MIM 사전학습 모델 특화) |
| radiometric | brightness_contrast, haze | 전역 방사 변환, 정보는 보존 | 80.3 / 85.9 | **불안정**: S5에서 +11.9 ~ −22.8 | **입력 통계 정렬** (smoke test: haze s5 60→91) |
| degradation | gauss/S&P noise, gauss/motion blur, JPEG | 고주파 정보 소실·오염 | 58.1 / 58.4 (최대 손실) | ±1, S5에서는 모두 손해 | 뚜렷한 답 없음. 회복 한계 자체가 결과 |

- 그룹별로 처방을 다르게 하려면 왜곡 종류를 테스트 때 **자동으로 판별**해야 합니다(정답 그룹을 알고 고르는 건 반칙).
  판별 신호 후보: 채널 평균·분산 이동(radiometric), 고주파 에너지 감소·증가(blur·noise), 거의 단색인 패치 비율(occlusion).
  판별기 정확도도 함께 보고해야 합니다.
- 진행 순서: 입력 통계 정렬 전체 결과로 radiometric 가설 확인 → occlusion용 MAE 마스킹 실험 설계.

## 10. Segmentation 확장 검토 (2026-10-13)

### 10.1 방법별 segmentation 적용 가능성
| 방법 | 원 논문의 segmentation 실험 | 적용 방식 | 비고 |
|---|---|---|---|
| TENT | ✅ GTA → Cityscapes | 픽셀별 엔트로피 평균 | 바로 적용 가능 |
| CoTTA | ✅ Cityscapes → ACDC (SegFormer, 연속 10라운드) | 다중 스케일 + 뒤집기 augmentation 평균 | segmentation용 augmentation은 분류용과 다름 (논문 기준) |
| EATA | (EATA-C 논문의 ACDC 실험에 baseline으로 등장) | 엔트로피 필터를 픽셀 단위로 | 중복 샘플 필터(batch 평균 확률과의 코사인 유사도)를 이미지 단위로 할지 정해야 함 |
| SAR | ❌ 분류만 | 픽셀 단위 엔트로피 필터 + SAM | 일반화는 쉽지만 검증된 적 없음 |
| DeYO | ❌ 분류만 | 조각을 섞으면 픽셀 위치가 바뀌므로, 예측 맵을 다시 원래 순서로 되돌려야 비교 가능 | 8절 결과상 PLPD는 RS에서 신호가 약함 → 제외하거나 한계로 보고 |
| LSCD-TTA | ❌ 분류만 (데이터셋 간 이동) | 클래스 균형·다양성 항이 batch 단위 분류를 전제 | 일반화가 애매함 |
| 입력 통계 정렬 | – | 입력 단계라 task와 무관 | 바로 적용 가능 |
| MAE 마스킹 (아이디어) | – | dense 출력에는 가려진 위치의 토큰이 필요 → 토큰을 빼는 대신 mask token으로 대체해야 함 | segmentation backbone은 mask token 없이 fine-tune되었으므로 추가 설계 필요 |

→ 현재 목록 중 segmentation에서 검증된 방법은 TENT와 CoTTA뿐입니다. segmentation 고유 TTA 후보(예: ViDA, ICLR 2024; SVDP, AAAI 2024)는 코드 공개 여부를 확인한 뒤 결정합니다(아직 확인하지 않음).

### 10.2 REOBench Segmentation 코드에서 확인한 사항 (`Segmentation/configs/vit_base_win/`)
- 설정: UperNet(+FCN 보조 head), 512 crop, AdamW lr 6e-5, 학습 batch 2. **backbone 전체를 fine-tune**합니다(분류는 backbone 고정).
  → TTA의 출발 모델 성격이 분류와 다르므로 결과를 직접 비교할 때 주의해야 합니다.
- decoder(UPerHead, FCNHead)는 **BatchNorm**을 씁니다(`norm_cfg = BN`).
  TTA 때 BN을 batch 통계로 바꾸면(TENT 원 방식) 작은 batch에서 불안정합니다.
  → 권장: decoder BN은 학습 통계(eval)로 고정하고, backbone LayerNorm만 adapt. 원 방식과의 차이로 명시합니다.
- **SatMAE 정규화 단위 오류 의심:** `SatMAE_Potsdam.py`가 `mean = [0.418, 0.421, 0.399], std = [0.288, 0.275, 0.276]`(0~1 단위)를 mmseg `Normalize`에 넘깁니다. 그런데 이 함수는 0~255 픽셀에 적용됩니다. 그러면 입력 값이 최대 약 (255 − 0.42) / 0.28 ≈ 900이 됩니다.
  - Scale-MAE와 CLIP 설정은 0~255 단위 ImageNet 통계를 써서 단위가 맞습니다. CLIP은 원래 CLIP 통계가 아니라 ImageNet 통계를 쓴다는 문제는 남습니다.
  - (재현으로 확인 필요)
- SatMAE segmentation도 `pretrained = pretrain-vit-large-e199.pth`(Sentinel 체크포인트)를 가리킵니다. 분류와 같은 문제입니다.
- 결론: 분류처럼 segmentation도 먼저 baseline을 재현·점검한 뒤에 TTA를 얹어야 합니다.

## 11. 방법 우선순위와 segmentation 공식 코드 조사 (2026-10-13)

결정: 1순위 4개 = **TENT, CoTTA, ViDA, ISA** (분류·분할 공통). 2순위 = EATA, LSCD, SVDP/Hybrid-TTA 등.
SAR, DeYO는 지금까지 한 분류 실험까지만. 분할 구현 원칙: 저자/공개 repo 코드 1순위, 없으면 논문 기술대로 직접 구현.

| 방법 | 분류 코드 | 분할 코드 (공식/준공식) |
|---|---|---|
| TENT | DequanWang/tent (분류만) | CoTTA ACDC 코드의 `single_gpu_tent` (SegFormer, norm/bn affine, Adam 6e-5/8, bs 1). 원본은 OneDrive(브라우저 필요). SVDP repo `mmseg/apis/test.py`에 같은 코드 사본 있음 |
| CoTTA | qinenergy/cotta | 같은 ACDC 코드의 `single_gpu_cotta` (SVDP repo에 사본, 주석 "Batchsize=1 now, was 8 during cityscapes training") |
| ViDA | Yangsenqiao/vida (cifar/imagenet만) | **비공개**. issue #1: 저자가 준 Drive 링크는 CoTTA 코드였고 현재 404. 공동저자 공지(2024-04-16): "CoTTA segmentation 프레임워크에 ViDA를 SegFormer에 주입해 쓰라". issue #6: 분할 ViDA는 SegFormer처럼 ImageNet 사전학습 시 encoder에 ViDA만 추가해 학습. issue #4: ViDA 분할 lr 7.5e-6, CoTTA 숫자는 원 논문에서 복사 → 공정성 논란 |
| ISA | 자체 baseline | 입력 변환뿐이라 그대로 적용 |
| SAR (참고) | mr-eggplant/SAR | 공식 없음. SPEGC repo·VPTTA repo 모두 SAR 코드 없음, 논문에도 설정 없음 (ResUNet-50, bs 1 추정) |
| EATA (참고) | mr-eggplant/EATA (분류만) | 공식 없음. EATA-C 논문 Table IV(ACDC): 모든 norm affine, lr 7.5e-5, Fisher 20장, β 500. EATA-C 분할에선 sample selection·min-max entropy 끔. 코드 미공개 |

참고 repo: Hybrid-TTA(hhhyyeee/Hybrid-TTA, mmseg, 자체 adapter, ViDA 없음), SVDP(Anonymous-012/SVDP), BeCoTTA(daeunni/BeCoTTA, ViDA 없음).
ViDA 분할은 "CoTTA 분할 프레임워크 + ViDA 분류 코드의 adapter 주입"으로 저자 공지대로 구현해야 함.

### 11-1. SAR segmentation 구현 추적 (2026-10-14)
- 공식 SAR(mr-eggplant/SAR): 분류만.
- **PASS** (EndoluminalSurgicalVision-IMR/PASS, 의료 분할 TTA 공식 repo) `models/TENT/sar.py`: 유일하게 찾은 SAR 분할 이식본.
  sigmoid(binary) 출력에서 픽셀별 max 확률로 이미지 단위 entropy를 만들고 `/200`으로 스케일, **이미지 단위** 필터(margin 직접 조정, 주석본엔 0.004),
  reset 기준 ema<1 (원본 0.2), BN/LN/GN affine. 주석 처리된 이전 버전 2개가 같이 있음 → binary 의료 분할 전용, 다중 클래스 RS 분할에 그대로 못 씀.
- GraTa(AAAI'25): 논문에 "모든 비교 방법을 같은 baseline으로 재구현" 명시, repo에는 GraTa 코드만.
- SPEGC, VPTTA, Döbler test-time-adaptation(segmentation: tent/cotta/gtta/asm/memo 등, SAR 없음), TTAP, TEA(분류): SAR 분할 코드 없음.
- 결론: SAR 분할은 논문 기술대로 우리가 이식(softmax 픽셀 entropy, 필터 단위 명시)하고 PASS 방식은 참고로만 기록.

## 12. ImageNet-C 구현 검증 (`E_verify/`, 2026-10-14)
- 목적: D_tta/methods 코드(REOBench에 실제 쓰는 그대로)를 논문 ImageNet-C 표와 비교해 구현 정확성 확인.
- `imagenetc.py`(harness), `REFERENCES.md`(비교 대상 표, PDF 렌더로 확인한 수치), `summarize.py`, `run_verify.sh`(detached, run_new 이후 실행).
- 비교 세트: DeYO T1 (R50-BN mild: tent/eata/sar/deyo), DeYO T5 + SAR T2 (ViT-B label shift), ViDA T1 (continual: R50 CoTTA 62.7, ViT ViDA 43.4),
  EATA-C Table A (ViT mild, 우리 프로토콜과 가장 가까움 — 체크포인트 미상이라 source 후보 3개 먼저 확인).
- 공용 코드 수정: params.py ResNet이면 layer4 skip(원본 규칙), cotta.IMG_SIZE 변수화. ViT/AID 동작은 동일.
- ViDA 이식: methods/vida.py (공식 코드 + HKA 버그 재현, `fix_hka` 옵션), 명세 E_verify/VIDA_SPEC.md.
  smoke: ViDA 원본 모델 source err gauss 55.7 / bright 25.8 / contrast 90.7 (논문 53.0/26.4/91.4, 1000장) → 정규화 없는 raw 입력이 맞음.
- 세 논문의 ViT-B source가 서로 다름: SAR/DeYO 29.9 (augreg + ImageNet 정규화), ViDA 44.2 (augreg + 정규화 없음), EATA-C 39.84 (미상).

### 12-1. ImageNet-C 검증 결과 (2026-10-14, `E_verify/summary.txt`)
- 통과(평균 차이 ≤0.35): DeYO T1 R50-BN mild — none 18.15/18.0, Tent 43.29/43.1, EATA 47.88/47.8, SAR 44.22/43.9, DeYO 48.73/48.6.
  ViDA ViT-B continual — source err 56.15/55.8, ViDA err 43.29/43.4 (repo log 43.37).
- 대체로 통과: ViT-B label shift — SAR 54.63/54.8; DeYO 64.7/62.3, EATA 51.4/49.2, Tent 53.3/47.7.
  차이는 붕괴형 열(shot, snow/frost, contrast: 논문 std ±25까지)과 noise 열에 몰림. 나머지 열은 ±1~2.
- noise 열: 원본 모델부터 논문보다 쉬움(ViT gauss 16.6 vs 9.4, R50 err 94 vs 98); 다른 열은 일치. Zenodo 공식 파일이라 JPEG 디코딩(Pillow/libjpeg) 차이로 추정.
- 미통과: CoTTA R50 continual err 66.73 vs 62.7 (+4, 모든 열에서 3~7 나쁨). 코드는 공식과 API 패치 외 동일, source는 일치(81.95/82.0).
  → 다음: 공식 CoTTA repo 코드를 우리 환경에서 그대로 실행해 구현 vs 환경(torchvision 증강 연산) 분리.
- EATA-C ViT 체크포인트 미확인: augreg2+ImageNet 정규화가 평균 39.84로 같지만 열별 패턴이 전혀 다름(gauss 35 vs 12.9) → 우연. EATA-C 표 검증은 보류.
- RS CoTTA (p_th 5% quantile, lr 0.01, 전체 파라미터): 모든 설정에서 붕괴 (SatMAE PN clean 92.8→49.5, avg −47). CoTTA ViT-L 설정 재검토 필요.
- 공식 CoTTA repo를 우리 환경에서 실행 (`F_cotta_official/`, RandomAffine API 한 줄만 수정, 2026-10-15):
  source 오류율은 우리 harness와 소수점까지 동일. 표준 순서 CoTTA 66.56 (우리 66.73) → **우리 이식은 공식 코드와 같음**.
  표준 순서 Tent 67.28 (ViDA T1 62.6). CoTTA 논문 순서 0: 59.50 (논문은 10개 순서 평균 63.0±1.8만 보고).
  → 순서에 따라 7%p 차이. ViDA T1의 62.7은 제3자 실행이라 절대값 비교는 불확실, 이식 정확성은 확인됨.

## 13. RS CoTTA 재실행 · 분할 검증 · issue 진행 (2026-10-15)

### 13-1. RS CoTTA 재실행 (EATA-C ViT 설정: lr 0.005, p_th 0.1, `D_tta/run_cotta_vit.sh`)
- 증강은 한 번도 켜지지 않음 (aug_batches 0/32) → 그래도 SatMAE PN clean 92.75 → 82.6, 스트림마다 붕괴 여부가 갈림
  (clouds s1 40.6 / s2 91.8, motion_blur 7~27). 원인은 증강이 아니라 ViT-L 전체 파라미터(~3억) 매 배치 업데이트.
- SatMAE PN 1개만 완주: clean 82.6, avg 55.2 (source 92.75 / 72.43, 첫 실행 49.5 / 25.0). 나머지 3개(ScaleMAE PN, P×2) 취소.
  결론: CoTTA 원 설정은 ViT-L RS에서 불안정 (설정을 바꿔도 source보다 −17 avg).
- tta.py: `--cotta_ap` 추가(출력 태그 `_ap<val>`), 저장 시 경로 재계산(폴더 rename 대비),
  cotta_ref 메모리 누수 수정(스트림마다 ViT-L 3개 복사본이 쌓여 OOM).

### 13-2. 분할 검증 Cityscapes→ACDC (`G_seg_verify/`, 패치 목록 `README_REOBench.md`)
- 코드: CoTTA ACDC 코드 사본(SVDP repo). 원본 OneDrive는 브라우저 필요. env `segtta` (torch 2.0.1 cu118, mmcv-full 1.7.2).
- Source mIoU: fog 69.08 / night 40.28 / rain 59.72 / snow 57.75 (CoTTA T5: 69.1 / 40.3 / 59.7 / 57.8) → 일치.
- Tent 1라운드: 69.07 / 40.19 / 60.02 / 57.43 (논문 69.0 / 40.2 / 60.1 / 57.3), 4라운드 66.76 / 36.71 / 58.97 / 54.31
  (논문 66.5 / 36.3 / 58.7 / 54.0), 10라운드 62.48 / 30.26 / 52.54 / 48.62 (논문 61.8 / 29.8 / 51.9 / 47.8).
  **10라운드 전체 평균 52.74 (논문 52.3)** → 통과.
- **분할 "TENT"는 entropy가 아님**: argmax pseudo-label CE, norm 파라미터, Adam 6e-5/8, bs 1. CoTTA/ViDA/SVDP/Hybrid-TTA 모두 이 변형을 TENT로 보고.
  → RS 분할에서 이 관례를 따를지, 픽셀 entropy Tent를 쓸지 결정 필요.
  원문 Tent: 목적함수 = Shannon entropy (논문 §3, 공식 tent.py `softmax_entropy(outputs).mean(0)`), BN affine만 갱신.
  논문 분할 실험(HRNet-W18 GTA→Cityscapes: 28.8 → BN 31.4 → Tent 35.8)은 픽셀 entropy로 읽히나 코드 미공개.
  hard PL은 Tent 논문에서 별도 baseline(PL, 임계값 있음)으로 비교됨 → CoTTA 코드의 "TENT"는 임계값 없는 PL에 가까움.
  제안: RS 분할은 원문대로 픽셀 entropy Tent를 기본, 필요 시 "TENT (CoTTA-codebase PL variant)"를 따로 표기.
- CoTTA 분할 augmentation = scale 7개 × flip = 14 forward (분류는 32개 랜덤 증강). anchor 확률 > 0.69인 픽셀은 1.0배 단일 예측, 나머지는 14개 평균.
  이미지당 ~20 s → 4조건×10라운드는 80시간 이상이라 fog 1라운드(400장, 논문 70.9)만 확인.

### 13-3. REOBench issue #4
- 저자 답변(10-13): 체크포인트는 코드 그대로, 전처리는 ToTensor만, MIM/CL 정규화 차이는 원 코드가 그렇다고 함. 성능 차이 원인은 미설명.
- 확인: e199 = Sentinel(`args.dataset_type='sentinel'`, grouped patch_embed) → RGB ViT에서 patch_embed 랜덤 → clean 18.2.
  Scale-MAE는 체크포인트 정상 로드(L52 assert 통과)인데도 91.4 vs 75.75 → 남은 원인 후보는 비공개 `data_type/*.pth`.
  두 원 repo 모두 사전학습·평가에서 정규화함 → 저자 답변 3번은 사실과 다름.
- 재답글 게시(사용자, 정규화 항목 포함). 근거 코드: fork `SunwooKim11/REOBench` 브랜치 `repro_aid` (`repro_aid/README.md`, `issue_check.py`).

### 13-4. 엔트로피 Tent 분할 기준 수치 조사 (agent, 2026-10-15)
- SegFormer-B5 + Cityscapes→ACDC + 10라운드 + 코드 공개인 엔트로피 Tent 기준은 **없음**.
- 가장 가까운 것: ATASeg (arXiv 2312.01835, T1; 코드 없음): 같은 960×540, bs1, Adam 6e-5/8. source 56.9 → TENT-CTTA(1라운드) 53.2,
  fog −6.1. source 행이 CoTTA와 달라 절대값 말고 차이로 비교. Yi et al. (ACM MM'24, T1, 도메인별 리셋): fog −4.9.
  같은 표의 "TENT (BN-fixed)" 68.1/39.4/60.1/57.1 ≈ source → 큰 하락은 decode head BN을 test batch 통계로 바꾼 데서 옴.
- 코드로 엔트로피 확인된 것은 Döbler tta-bench(GTTA, CarlaTTA, DeepLabV2)뿐 — 백본·데이터 다름.
- CoTTA 숫자를 그대로 쓰는 곳(PL 변형, 비교 금지): SVDP, BeCoTTA, Hybrid-TTA 코드; DAT·Continual-MAE·OCL 등은 CoTTA 수치 복사; EATA-C Tent도 CoTTA 궤적과 동일.
- 우리 엔트로피 Tent 1라운드: 69.06 / 40.16 / 60.11 / 57.38 ≈ PL TENT·source. CoTTA 코드는 model.eval()이라
  decode head BN이 running stats 고정 → Yi et al.의 "BN-fixed" 조건과 같고 그 결과(≈ source)와 일치.
- 엔트로피 Tent + BN을 test 통계로(원문 configure_model, `TENT_BN=batch`, SyncBN→BN 변환), 1라운드:
  64.54 / 39.76 / 57.38 / 55.35 (mean 54.26). source 대비 −4.5 / −0.5 / −2.3 / −2.4 (mean −2.5).
  Yi et al.(도메인별 리셋) −4.9 / −0.3 / −2.1 / −2.7 과 거의 같고, ATASeg(−6.1 / −2.1 / −3.3 / −3.6)와 같은 양상.
  → 원문 Tent 분할 구현 검증 통과. 성능 하락의 원인은 decode head BN을 이미지 1장 통계로 바꾸는 것.

### 13-5. 코드 리뷰 가이드 (`review/`, 2026-10-15)
- `review/REVIEW_GUIDE.md`: 폴더 역할, 원본 파일 고정본(`upstream/`, commit 기록), 우리 변경 diff(`diffs/`), 논문 대조 체크리스트.
- 리뷰 중 발견: **SAR 복구 임계값** — 공식 SAR는 `ema < 0.2` (sar.py L102 하드코딩), DeYO repo 사본(우리 사용)은
  `reset_constant_em=0.005` 기본값 → 복구가 거의 안 걸림. DeYO 논문 수치는 0.005로 재현됨. SAR 원 논문 설정은 0.2.
  → RS SAR 결과(특히 ScaleMAE P 붕괴)를 0.2로 다시 볼 가치 있음.
- 분할 CoTTA 1라운드 fog: **70.93 (논문 70.9)** → 통과. (14-scale aug, ~10–20 s/img라 fog 400장만.)
- 엔트로피 Tent(BN 고정) 10라운드: R1 69.06/40.16/60.11/57.38, R10 62.06/29.69/51.93/47.93, 평균 52.28
  (PL "TENT" 52.74, CoTTA 논문 52.3). BN을 고정하면 entropy와 PL은 장기적으로도 거의 같은 궤적.
- SAR 복구 임계값 0.2 재실행 (clean / avg, `sar_s0_rst0.2.json`):
  | | none | SAR 0.005 (DeYO repo) | SAR 0.2 (원 논문) | reset 횟수 (0.2) |
  | SatMAE PN | 92.8/72.4 | 92.8/73.3 | 92.8/72.7 | 895 |
  | SatMAE P | 90.5/64.9 | 90.0/63.4 | 90.5/64.7 | 193 |
  | ScaleMAE PN | 95.2/74.8 | 95.2/74.9 | 95.2/75.0 | 312 |
  | ScaleMAE P | 91.5/68.0 | 21.4/20.9 | 17.6/18.8 | 25 |
  해석: AID(C=30)는 확신도가 높아 엔트로피 EMA가 정상 상태에서도 0.2 아래 → 전체 ~1950 배치 중 최대 895번 리셋 →
  사실상 source에 가까워짐(SatMAE PN +0.3, P −0.2). ScaleMAE P 붕괴는 저엔트로피형이 아니라(리셋 25번) 복구가 못 막음
  (diag.py: blocks.0.norm1 gradient 폭발). 0.2는 ImageNet(1000클래스) 기준 절대값이라 클래스 수가 다른 RS에는 그대로 안 맞음.
