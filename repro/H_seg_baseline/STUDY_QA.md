# 분할 코드 Q&A (2026-10-21, 하루 읽어 보고 막힌 12가지)

기준 코드: `repro/H_seg_baseline/seg/` (REOBench Segmentation 사본). 숫자는 512×512 Potsdam 패치, ViT-L/16 기준입니다.
이미지 1장의 텐서 모양 전체는 `trace_scalemae_B_12ep.log`에 있습니다(`trace_shapes.py`가 자동 저장).

---

## 1. 왜 train / test pipeline이 따로 있나? Resize는 왜 하나?

pipeline = "디스크의 이미지 파일 → 모델 입력 텐서"로 가는 전처리 단계 목록입니다. `configs/vit_base_win/scale_MAE.py` L13-56에 있습니다(train L13, test L43).

| 단계 | train_pipeline | test_pipeline |
|---|---|---|
| 읽기 | 이미지 + **라벨** | 이미지만 (라벨은 평가 함수가 따로 읽음) |
| `Resize` | `img_scale=(512,512), ratio_range=(0.5, 2.0)` → **매번 무작위로 0.5~2배 확대·축소** | `img_scale=None, img_ratios=[1.0]` → **크기 그대로** |
| `RandomCrop` | 512×512를 무작위 위치에서 잘라냄 (라벨도 같은 위치) | 없음 |
| `RandomFlip` | 50% 확률 좌우 반전 | 없음 (`flip=False`) |
| `PhotoMetricDistortion` | 밝기·대비·채도·색조를 무작위로 바꿈 | 없음 |
| `Normalize` | 평균·표준편차 정규화 | **같은 값으로** 정규화 |
| `Pad` | 512보다 작으면 채움 (라벨은 5 = ignore로 채움) | 없음 |

- **train에서 Resize를 하는 이유:** 데이터 증강입니다. 같은 이미지를 크기를 바꿔 가며 보여 주면, 모델이 물체 크기 변화에 강해집니다. 0.5배로 줄이면 건물이 작게, 2배로 키우면 크게 보입니다. 키운 뒤에는 `RandomCrop`으로 512만 잘라 씁니다.
- **test는 무작위 요소가 없어야 합니다.** 매번 같은 입력에 같은 답이 나와야 평가가 의미 있습니다. 그래서 크기를 그대로 두고 정규화만 합니다.
- 분류 코드와 같은 원리입니다. 분류 `scripts/cls/train_official.py`의 `--aug`(RandomResizedCrop + Flip)가 train_pipeline에 해당합니다.

## 2. `scale_MAE_vit_large.py`의 encoder/decoder, num_heads, mlp_ratio는?

**이 파일은 encoder(백본)만 있습니다.** 분할에서 "decoder"는 UperNet(`uper_head.py`)입니다.
- 체크포인트를 불러올 때 `unexpected=98`이 찍힙니다. 이 98개가 MAE **사전학습용 decoder**(가려진 패치를 복원하던 부분)의 가중치입니다. 분할에서는 쓰지 않아서 버려집니다.

ViT-L 설정(`configs/_base_/models/scale_MAE_vit_large.py` L19-24):

| 인자 | 값 | 의미 |
|---|---|---|
| `patch_size=16` | 16 | 이미지를 16×16 조각으로 자름 → 512/16 = 32, 32×32 = 1024개 토큰 |
| `embed_dim=1024` | 1024 | 토큰 하나를 표현하는 벡터 길이 |
| `depth=24` | 24 | Transformer block 수 |
| `num_heads=16` | 16 | **multi-head attention**: 1024차원을 16개 head × 64차원으로 나눠, 16가지 관점의 attention을 따로 계산한 뒤 이어 붙임 |
| `mlp_ratio=4` | 4 | block 안의 MLP 은닉층 = 1024 × 4 = **4096**차원 (1024 → 4096 → 1024) |
| `qkv_bias=True` | | attention의 Q, K, V 선형층에 bias 사용 |

Transformer block 하나(timm `Block`)의 계산:
```
x = x + Attention(LayerNorm1(x))   # 토큰끼리 정보 교환 (16 heads)
x = x + MLP(LayerNorm2(x))         # 토큰마다 독립적으로 1024→4096→1024
```
이 block이 24번 반복됩니다. 분류든 분할이든 이 부분은 똑같습니다.

## 3. 핵심만 알면 되나, 하나하나 다 알아야 하나?

결론: **우리가 만들거나 고친 코드는 한 줄도 빠짐없이**, **프레임워크(mmseg) 코드는 "입력과 출력이 무엇인지" 수준까지** 알면 됩니다.

학부 과제는 코드 전체를 내가 썼으니 전부 아는 게 당연했습니다. 연구 코드는 수만 줄짜리 프레임워크 위에 수백 줄을 얹는 구조입니다. mmseg의 dataloader, runner, hook을 다 읽는 사람은 거의 없습니다. 그 대신 경계를 정확히 압니다.

| 꼭 다 알아야 하는 것 (내 연구의 주장이 여기에 달림) | 인터페이스만 알면 되는 것 |
|---|---|
| `seg_tta.py` 전체 (TTA 방법, 평가 프로토콜, mIoU 계산) | mmseg `Runner`, hook, registry, config 상속 메커니즘 |
| 우리가 고친 곳 (`[REOBench repro]` 표시, 7곳) | `LoadImageFromFile` 같은 개별 pipeline 클래스 내부 |
| 데이터 경로와 라벨 처리 (255 → ignore) | `ConvModule` 내부 (= conv + BN + ReLU라는 것만) |
| 모델의 입출력 모양 (이 문서의 7, 8, 11번) | 분산 학습, 로깅 |

확인하는 방법: "이 숫자가 왜 이 값인지 다른 사람에게 설명할 수 있는가?" 설명할 수 없는 부분이 결과에 영향을 준다면 그 부분은 읽어야 합니다.

## 4. `encode_decode`의 bilinear는?

`encoder_decoder.py` L87-97:
```python
x = self.extract_feat(img)                          # 백본 → 4개 해상도 피라미드
out = self._decode_head_forward_test(x, img_metas)  # UperNet → (B, 5, 128, 128)  ← 1/4 해상도
out = resize(out, size=img.shape[2:], mode='bilinear', align_corners=False)  # → (B, 5, 512, 512)
```
- UperNet의 출력은 입력의 **1/4 해상도(128×128)**입니다. 정답은 픽셀(512×512)마다 있으니 크기를 4배로 키워야 합니다.
- **bilinear 보간:** 새 픽셀의 값을 주변 원래 픽셀 4개의 거리 가중 평균으로 정합니다. 가장 가까운 값을 그대로 복사하는 nearest보다 경계가 부드럽습니다.
- 키우는 대상은 **logits(클래스별 점수)**이고, 키운 다음에 argmax로 클래스를 정합니다. 라벨(정수)을 키우는 게 아닙니다.
- `align_corners=False`: 픽셀을 "격자의 칸 중앙"으로 보고 보간한다는 PyTorch의 관례 옵션입니다. 결과에 영향은 미미합니다.

## 5. `slide_inference`의 `preds += F.pad(...)` 이후

큰 이미지를 512 창(window)으로 잘라 따로 예측하고 다시 붙이는 코드입니다(`encoder_decoder.py` L175-219). config는 `crop_size=(512,512), stride=(384,384)`입니다.
```python
preds = 0 텐서 (B, 5, H, W)          # 전체 크기의 빈 캔버스
count_mat = 0 텐서 (B, 1, H, W)      # 픽셀마다 몇 번 예측됐는지 셀 카운터
for 각 창 (y1:y2, x1:x2):
    crop_logit = encode_decode(img[..., y1:y2, x1:x2])   # (B, 5, 512, 512)
    preds += F.pad(crop_logit, (x1, W - x2, y1, H - y2)) # 창 결과를 0으로 둘러싸서 캔버스 크기로 만든 뒤 더함
    count_mat[..., y1:y2, x1:x2] += 1
preds = preds / count_mat            # 창이 겹친 곳은 평균
```
- `F.pad(t, (왼쪽, 오른쪽, 위, 아래))`는 텐서 주변을 0으로 채웁니다. 창이 (x1, y1) 위치에 있으면 왼쪽에 x1칸, 위에 y1칸을 채워, 창 결과가 캔버스의 제자리에 놓이게 합니다.
- 예: 1000×1000 이미지라면 창이 (0~512, 384~896, 488~1000) × 같은 세로 3개 = 9개입니다. 겹친 띠는 2~4번 예측되어 평균됩니다.
- **우리 Potsdam 패치는 512×512라서 창이 1개**입니다. `pad`는 (0, 0, 0, 0)이고 `count_mat`은 전부 1이라, 결국 `encode_decode` 한 번과 같습니다. 그래서 `seg_tta.py`는 `encode_decode`를 바로 부릅니다(`none` 결과가 mmseg 평가와 64.97로 일치한 것으로 확인).

## 6. `get_2d_sincos_*` 함수는 왜 이렇게 많고, axis와 dim은 무엇인가

다 같은 수식 하나를 조금씩 다른 조건으로 계산하는 버전입니다.

| 함수 | 차이 |
|---|---|
| `get_1d_sincos_pos_embed_from_grid` | **1차원** 위치 목록 → sin/cos 벡터. **numpy** 버전 |
| `get_1d_sincos_pos_embed_from_grid_torch` | 같은 계산의 **torch** 버전 (GPU, 배치 처리용) |
| `get_2d_sincos_pos_embed_from_grid` (+ `_torch`) | 2차원 격자: 세로 좌표로 512차원, 가로 좌표로 512차원을 만들어 이어 붙임 → 1024 |
| `get_2d_sincos_pos_embed` | 격자 크기만 받아 위 함수를 부르는 진입점. **실제로 쓰이는 것** (`__init__`에서 한 번) |
| `get_2d_sincos_pos_embed_with_resolution` | Scale-MAE의 **GSD 버전**: 좌표에 해상도 `res`를 곱함. 정의만 있고 **분할 forward에서는 호출되지 않음** |

같은 계산이 numpy와 torch 두 벌 있는 건 원본 MAE 코드(numpy)와 Scale-MAE 코드(torch)를 각각 복사해 와서입니다. 정리되지 않은 연구 코드에서 흔한 모습입니다.

수식(1차원): 위치 `pos`, 차원 `i = 0..D/2−1`에 대해
```
ω_i = 1 / 10000^(i / (D/2))
emb = [sin(pos·ω_0), …, sin(pos·ω_{D/2−1}), cos(pos·ω_0), …, cos(pos·ω_{D/2−1})]
```
위치마다 주파수가 다른 sin·cos 값들이 붙어서, 위치를 고유한 벡터로 표현합니다. 학습하지 않는 고정값입니다.

**axis와 dim:** 둘 다 "다차원 배열에서 몇 번째 축을 따라 계산하나"를 뜻합니다. numpy는 `axis`, torch는 `dim`이라고 부를 뿐 같은 개념입니다.
- 예: 모양이 (1024, 512)인 두 배열을 `concatenate(..., axis=1)` → (1024, 1024), 즉 두 번째 축으로 이어 붙임. `axis=0`이면 (2048, 512).
- `np.stack(grid, axis=0)`: 가로·세로 좌표 격자 두 장을 맨 앞에 새 축을 만들어 쌓음 → (2, 32, 32).

## 7. 분류와 무엇이 다른가: `forward_features`

| | 분류 (`repro/common.py` L86 `features()` → timm `forward_features`) | 분할 (`scale_MAE_vit_large.py` `forward_features`) |
|---|---|---|
| 토큰 | [CLS] + 패치 256개 (256px) | **패치 1024개만** (512px, [CLS] 안 씀: `pos_embed[:, 1:]`) |
| 블록 출력 | **마지막 블록만** | **중간 블록 4개** (A: 3, 5, 7, 11 / B: 7, 11, 15, 23) |
| 정리 | 마지막 `norm` → [CLS] 벡터 1개 (1024) | 토큰 1024개를 32×32 지도로 다시 배치 (1024채널) |
| 이후 | Linear (1024 → 30) | `fpn1~4`로 해상도 4개 → UperNet |

분할은 "어디에 무엇이 있는지"가 필요하므로 위치를 가진 패치 토큰을 쓰고, 여러 깊이의 특징을 함께 씁니다. 얕은 블록은 경계 같은 세부를, 깊은 블록은 의미를 담고 있습니다.

## 8. FPN, `lateral_convs`

FPN(Feature Pyramid Network): 해상도가 다른 특징 지도 여러 장을 합쳐, **고해상도이면서 의미도 풍부한** 지도를 만드는 구조입니다.
- **lateral conv** (`uper_head.py` 생성 L47-68, 실행 L99-102): 각 해상도 지도에 **옆으로(lateral)** 1×1 conv를 걸어 채널을 1024 → 512로 맞춥니다. 크기는 그대로입니다.
- **top-down** (L106-112): 가장 거친 지도부터 2배씩 키워(bilinear) 바로 위 단계 지도에 **더합니다**. 1/32 → 1/16 → 1/8 → 1/4 순서이고, 고운 지도에 깊은 의미 정보가 섞입니다.
- **fpn_convs** (L115-118): 더한 결과에 3×3 conv를 한 번 더 걸어 다듬습니다.

## 9. "`D_tta/common.py`가 없다"

가이드의 오타였습니다. 정확한 위치는 **`repro/common.py`**입니다(`features()`는 L86). `scripts/tta/cls_tta.py`가 `import common`으로 이 파일을 불러옵니다. 가이드도 고쳤습니다.

## 10. `auxiliary_head`는 무엇이고 왜 있나

`FCNHead`(config `_base_/models/scale_MAE_vit_large.py` L39-51, `in_index=2` = 1/16 지도, `loss_weight=0.4` L51). 중간 특징에서 바로 클래스를 예측하는 **보조 출력**입니다.
- **학습 때만** 씁니다. loss = 주 head의 CE × 1.0 + 보조 head의 CE × 0.4입니다.
- 이유: 중간 층에도 직접 학습 신호를 주어 학습을 안정시킵니다(deep supervision, PSPNet에서 시작).
- **테스트 때는 계산하지 않습니다**(`encode_decode`는 decode_head만 부름). 그래서 TTA에서도 영향이 없습니다.

## 11. UperNet 논문 그림 ↔ 코드

`uper_head.py` `forward` (L89-131)를 논문 그림 순서대로 주석 달면:
```python
inputs = self._transform_inputs(inputs)        # 백본 피라미드 4장: 1/4, 1/8, 1/16, 1/32 (각 1024채널)

laterals = [lateral_conv(inputs[i]) for i in 0..2]   # [FPN] 1/4, 1/8, 1/16을 1×1 conv로 512채널
laterals.append(self.psp_forward(inputs))            # [PPM] 1/32 지도 → 아래 psp_forward

# psp_forward (L79-87): [PPM] 장면 전체 맥락
#   x = 1/32 지도 (16×16)
#   1×1, 2×2, 3×3, 6×6으로 평균 풀링 → 각각 1×1 conv → 다시 16×16으로 키움 (psp_head.py L36-46, PPM.forward)
#   [x, 풀링 4개]를 채널로 이어 붙임 → 3×3 conv(bottleneck) → 512채널

for i in 3, 2, 1:                                    # [FPN top-down] 거친 것부터
    laterals[i-1] += resize(laterals[i], laterals[i-1] 크기)

fpn_outs = [fpn_convs[i](laterals[i]) for i in 0..2] + [laterals[3]]   # [FPN] 3×3 conv로 다듬기
fpn_outs = [모두 1/4 크기(128×128)로 resize]                            # [Fuse] 크기 통일
output = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))               # [Fuse] 512×4=2048채널 → 3×3 conv → 512
output = self.cls_seg(output)                                          # dropout + 1×1 conv → 5채널 = 클래스 점수
```
논문 그림의 네 덩어리가 코드에서는 `psp_forward`(PPM), `lateral_convs` + top-down 루프(FPN), `torch.cat` + `fpn_bottleneck`(Fuse), `cls_seg`(Head)입니다.
- 모양 변화 전체는 `trace_scalemae_B_12ep.log`의 "lateral …" ~ "cls_seg …" 줄이 정확히 이 순서입니다.

## 12. `trace_shapes.py` 출력 저장

이제 실행할 때마다 `repro/H_seg_baseline/trace_<config 이름>.log`에 자동 저장됩니다(`--log`로 경로 지정 가능).
예: `trace_scalemae_B_12ep.log`.
