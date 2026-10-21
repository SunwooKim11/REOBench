# 실험 중 나온 개념 정리 (REOBench 재현)

## 1. epoch / iteration / step / scheduler

| 용어 | 뜻 | AID (7,999장, batch 16) |
|---|---|---|
| batch | 한 번에 모델에 넣는 이미지 묶음 | 16장 |
| iteration | batch 1개로 "예측 → loss → 가중치 업데이트" 1회 | – |
| epoch | 학습 데이터 전체를 한 바퀴 | ≈ 500 iteration |
| 전체 학습 | 100 epoch | ≈ 50,000 iteration |
| learning rate (lr) | 업데이트 1회에 가중치를 얼마나 크게 움직일지 | 시작값 1e-3 |
| scheduler | 학습 도중 lr을 자동으로 바꾸는 장치 | CosineAnnealingWarmRestarts |

"step"은 두 가지를 가리킵니다.
- `optimizer.step()`: 가중치 업데이트. 항상 iteration마다 부릅니다.
- `scheduler.step()`: lr 시계를 한 칸 넘깁니다. 언제 부를지는 설계자가 정합니다.

```python
for epoch in range(100):            # epoch 루프
    for x, y in loader:             # iteration 루프 (~500회)
        loss.backward()
        optimizer.step()            # 매 iteration
        scheduler.step()            # ← REOBench 원본 코드 (iteration마다)
    # scheduler.step()              # ← REOBench 논문 App. F.1 (epoch마다)
```

REOBench 원본 코드는 5 iteration마다, 논문 설명대로라면 5 epoch마다 lr이 재시작합니다.
실험 결과 차이는 0.6점 미만이었습니다(`experiments/2026-10-12_cls_ablation/summary.txt`의 A0 vs A0ep).

## 2. CosineAnnealingWarmRestarts (SGDR, Loshchilov & Hutter, ICLR 2017)

- **Cosine Annealing:** lr을 코사인 곡선 모양으로 최댓값에서 최솟값까지 줄입니다.
- **Warm Restarts:** 주기(`T_0`칸)마다 lr을 다시 최댓값으로 올립니다. 가중치는 유지하므로 "warm"입니다.
- 수식: `lr(t) = lr_min + ½(lr_max − lr_min)(1 + cos(π·t/T))`
- 목적: lr을 주기적으로 올려서 local minimum을 벗어나 다른 해를 탐색합니다.
- warmup(초반에 lr을 0에서 서서히 올리는 것)과는 다른 기법입니다.

`T_0=5`, `lr_max=1e-3`이면 칸마다 lr은 1e-3 → 9.0e-4 → 6.5e-4 → 3.5e-4 → 9.5e-5 → (재시작) 1e-3 …

## 3. drop_path (Stochastic Depth)

- ViT 블록은 `x = x + Block(x)` 형태입니다. drop_path는 학습 중에 블록 출력을 무작위로 0으로 만들어 그 블록을 건너뛰게 합니다.
- dropout이 뉴런 단위라면, drop_path는 블록(경로) 단위입니다.
- `drop_path_rate=0.1`이면 블록 1은 0%, 블록 24는 10% 확률로 건너뜁니다(`linspace(0, 0.1, 24)`).
- `model.train()`이면 켜지고, `model.eval()`이면 꺼집니다.
- REOBench 원본 Scale-MAE 코드는 backbone을 고정(`requires_grad=False`)해 놓고 `model.train()`을 불러서 drop_path가 켜진 상태로 head를 학습합니다.
  → 같은 이미지라도 매번 feature가 조금씩 달라져서 일종의 feature augmentation이 됩니다.
  → 실험: 켜짐(A0) 91.1 / 67.4, 끔(bbeval) 91.1 / 65.3. clean은 같고 robustness는 켜진 쪽이 2점 높습니다.
- SatMAE는 `eval()`로 학습해서, 두 모델의 조건이 다릅니다.

## 4. Scale-MAE의 GSD 위치 인코딩 (GSDPE)

### 4.1 원리
- GSD(Ground Sample Distance)는 픽셀 하나가 땅 위에서 몇 m인지입니다.
- 일반 ViT의 위치 인코딩은 patch 번호 0, 1, 2, …만 씁니다. 해상도가 달라도 같은 값입니다.
- Scale-MAE 논문 식 (4): `sin((g/G) · pos / 10000^(2i/D))`
  - g = 이미지의 GSD, G = 기준 GSD("nominally 1m")
  - 위치에 배율 g/G를 곱해서, patch 간격이 땅 위 거리에 비례하게 만듭니다.
- 코드: `util/pos_embed_scale.py` → `grid = einsum("chw,n->cnhw", grid, res)`. 위치 × `res`이고, `res`가 g/G 역할입니다.

### 4.2 공식 코드의 `res` 계산 (fMoW 사전학습)
```python
# lib/transforms.py: RandomResizedCrop으로 원본에서 정사각형을 잘라 448px로 리사이즈
ratio = dx / h          # dx = 잘라낸 영역의 원본 픽셀 폭, h = 출력 크기(448)
# dataloaders/utils.py
res = ratio * base_resolution                  # base_resolution = 2.5 (체크포인트 args)
imgs_src_res = res * (448 / 224)               # 인코더 입력은 224로 한 번 더 줄임
```
- `ratio`: "모델 입력 픽셀 1개가 원본 픽셀 몇 개를 덮는가"입니다.
  예를 들어 원본에서 672px 폭을 잘라 448로 줄이면 ratio=1.5입니다. 이를 다시 224로 줄이면 인코더 입력 기준 3.0입니다.
- `base_resolution = 2.5`: "원본 픽셀 1개 = 2.5 단위"라고 가정하는 상수입니다. 모든 fMoW 원본 이미지에 같은 값을 씁니다.
- 결론: `res` = (확대/축소 배율) × 상수입니다. 이미지의 실제 GSD(m)는 들어가지 않습니다.
  fMoW-RGB는 이미지마다 실제 GSD가 다르지만(메타데이터에 있음) 코드는 이를 쓰지 않습니다.
  학습되는 것은 같은 이미지를 얼마나 확대/축소했는가에 대한 상대적인 scale 정보입니다.

### 4.3 평가할 때 `res = 1.0`이 뜻하는 것
- 공식 eval 기본값(`--eval_base_resolution 1.0`, `--eval_gsd` 미사용)과 REOBench는 모든 이미지에 `res = 1`을 넣습니다.
- 위치 × 1이므로 일반 MAE와 똑같은 위치 인코딩이 됩니다.
- 해상도가 다른 이미지도 같은 위치 값을 받으므로, 해상도 정보가 모델에 전달되지 않습니다. 이것이 "GSD를 안 쓴다"는 뜻입니다.
- 공식 `--eval_gsd` 옵션은 `res = base × (224 / eval_scale)`, 즉 이미지를 몇 배로 줄였는지만 반영합니다. 이것도 상대 배율입니다.

### 4.4 AID의 GSD
- AID 논문(Xia et al., 2017): Google Earth, 600×600, "약 0.5m ~ 8m"이고 이미지별 GSD는 제공되지 않습니다.
- REOBench는 600 → 256으로 줄였으므로 실제 GSD는 약 2.34배 커서 약 1.2m ~ 19m입니다.
- 이미지별 값을 알 수 없어 상수 하나를 쓰는 것이 현실적입니다. Scale-MAE 논문도 GSD가 섞여 있고 명시되지 않은 데이터셋은 "approximate constant GSD"로 가정했다고 썼습니다.
- `scale_image` 왜곡은 축소 배율을 알 수 있으므로, 그 배율만큼 `res`를 조정하는 실험이 가능합니다(아직 하지 않음).

### 4.5 "거짓 보고인가?"
- 논문 본문은 "g is the GSD of the image"라고 써서 실제 GSD를 쓰는 것처럼 읽힙니다. 하지만 구현은 상대 배율만 씁니다. 표현과 구현 사이에 괴리가 있습니다.
- 다만 논문의 핵심 실험(Fig. 5 등)은 x축이 "Relative GSD"입니다. 같은 검증셋을 일정 비율로 줄여 가며 평가하는데, 이 경우 상대 배율이 곧 정확한 GSD 비율입니다. 그래서 주요 결과 자체는 구현과 모순되지 않습니다.
- 즉 "모호하고 과장된 서술"이지 "결과 조작"은 아닙니다.
- 이 점은 이미 Scale-MAE repo 이슈 #14 "Question about GSDPE"에서 제기됐습니다. 사용자들이 "실제 GSD는 쓰이지 않고 해상도 스케일링으로 시뮬레이션한다"고 확인했고, 저자의 공식 답변 없이 닫혔습니다.
- REOBench 이슈와는 별개입니다. REOBench는 공식 eval 기본값(`res=1`)을 그대로 따른 것이라 REOBench의 잘못이 아닙니다.

## 5. 수치 정밀도 (fp32 / TF32 / bf16)

### 5.1 부동소수점 형식
숫자 하나를 몇 비트로, 어떻게 나눠 저장하는지의 차이입니다.
지수(exponent) 비트는 표현 가능한 범위를, 가수(mantissa) 비트는 유효숫자 정밀도를 정합니다.

| 형식 | 전체 비트 | 지수 | 가수 | 유효숫자 | 특징 |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | ~7자리 | PyTorch 기본값. 가장 정확하지만 느림 |
| TF32 | (19) | 8 | 10 | ~3자리 | NVIDIA Ampere 이상 GPU의 Tensor Core가 행렬곱 내부에서만 쓰는 형식. 저장은 fp32 그대로 |
| bf16 | 16 | 8 | 7 | ~2~3자리 | 범위는 fp32와 같고 정밀도만 낮음. 메모리 절반, 속도 빠름 |
| fp16 | 16 | 5 | 10 | ~3자리 | 정밀도는 bf16보다 높지만 범위가 좁아 overflow 위험 (우리 실험은 feature 캐시 저장에만 사용) |

- **TF32:** 입력과 출력은 fp32인데, 행렬곱(matmul/conv) 계산 중간에만 가수를 10비트로 잘라서 계산합니다. 코드를 바꿀 필요 없이 플래그로 켜고 끕니다.
  - `torch.backends.cuda.matmul.allow_tf32` (행렬곱): PyTorch 1.12 이후 기본값 False
  - `torch.backends.cudnn.allow_tf32` (convolution): 기본값 True
- **bf16 autocast:** `with torch.autocast("cuda", dtype=torch.bfloat16):` 블록 안에서 matmul 같은 연산은 bf16으로, softmax·LayerNorm 같은 민감한 연산은 fp32로 자동 선택해 계산합니다.

### 5.2 각 설정의 정밀도

| 설정 | backbone 계산 | head 학습 | 근거 |
|---|---|---|---|
| REOBench 원본 | fp32 (conv만 TF32, PyTorch 기본값) | fp32 | 별도 설정 없음. RTX 4090 |
| A (`scripts/cls/train_released.py`) | fp32 + **TF32 matmul 켬** | fp32 | 속도 때문에 `allow_tf32=True`로 설정 |
| B (`scripts/cls/train_official.py`) | **bf16 autocast** | fp32 (feature를 `.float()`로 변환) | 공식 MAE/Scale-MAE도 AMP(혼합 정밀도) 사용 |
| C (`scripts/cls/ablation.py`) | **bf16 autocast** | fp32 | 모든 ablation 설정을 같은 정밀도로 맞추려고 bf16으로 통일 |

C의 feature 캐시는 bf16으로 계산한 feature를 **fp16으로 저장**하고(`.half()`), 학습할 때 fp32로 되돌립니다. 저장 공간을 절반으로 줄이기 위해서입니다.

### 5.3 결과에 미친 영향
같은 프로토콜(A0)을 정밀도만 달리 해서 비교했습니다.

| | A (fp32+TF32, seed 1개) | C A0 (bf16, seed 3개) |
|---|---|---|
| SatMAE clean / 왜곡 평균 | 90.40 / 64.67 | 90.5 ± 0.2 / 65.0 ± 0.2 |
| Scale-MAE clean / 왜곡 평균 | 91.25 / 67.56 | 91.1 ± 0.2 / 67.4 ± 0.2 |

차이가 seed 오차 범위 안이라 정밀도는 결과에 영향이 없다고 볼 수 있습니다.
backbone이 고정이고 학습 대상이 Linear head 하나라서 수치 오차가 쌓일 여지가 적습니다.
backbone까지 학습하는 full fine-tuning이나 TTA(테스트 시 파라미터 업데이트)에서는 영향이 커질 수 있어서, 그때는 정밀도를 명시하고 통일하는 게 좋습니다.
