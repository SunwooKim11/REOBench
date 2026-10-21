# 분할 코드 공부 가이드 (분류 코드를 아는 상태에서 출발)

## 0. 한 장 요약: 분류와 무엇이 다른가

| | 분류 (AID, 우리 TTA 코드) | 분할 (Potsdam, REOBench) |
|---|---|---|
| 입력 | 256×256 | 512×512 |
| 백본 출력에서 쓰는 것 | 마지막 블록의 **[CLS] 토큰 1개** (1024) | 블록 3, 5, 7, 11의 **패치 토큰 전부** (1024개 × 1024) |
| 그 뒤 | Linear (1024 → 30) | 토큰을 32×32 지도로 → 4개 해상도 피라미드 → **UperNet** → 픽셀마다 5개 클래스 점수 |
| 출력 | (B, 30) | (B, 5, 128, 128) → 512×512로 업샘플 |
| loss | CE(이미지 1개당 정답 1개) | CE(**픽셀마다**, 라벨 5와 255는 무시) + 보조 head CE × 0.4 |
| 백본 학습 | 고정 | **전체 fine-tune** (AdamW 6e-5, layer decay) |
| norm 층 | LayerNorm만 | 백본 LayerNorm + **head의 BatchNorm** ← TTA에서 중요 |

**TTA 관점에서의 핵심:** logits가 (B, C) 대신 (B, C, H, W)라서, 엔트로피를 **픽셀마다** 계산해 평균내면 Tent가 됩니다.
그리고 BN이 생깁니다. ACDC 검증에서 결과를 가장 크게 바꾼 것이 이 BN 처리였습니다(−2.5).

## 1. 먼저 돌려 보기 (10분)

```bash
cd ~/REOBench/repro/H_seg_baseline/seg
PYTHONPATH=. TIMM_FUSED_ATTN=0 /opt/conda/envs/segtta/bin/python ../trace_shapes.py --ckpt ../work_dirs/scalemae_A_12ep/latest.pth
```
- 진짜 Potsdam 이미지 1장을 학습된 모델에 넣고, 단계마다 텐서 모양을 출력합니다.
- 마지막 줄 `check vs model.encode_decode` = 0이면 이 스크립트가 실제 모델과 똑같이 계산한다는 뜻입니다.
- 출력 (ScaleMAE, 2026-10-16):
```
input image (normalized)                         (1, 3, 512, 512)
patch_embed: 16x16 patches -> tokens             (1, 1024, 1024)     # 32×32=1024 토큰, 각 1024차원
  block  3/5/7/11 output (kept)                  (1, 1024, 1024)
  ... blocks 12-23 are computed but not used
tokens -> 2D map (B, C, H/16, W/16)              (1, 1024, 32, 32)
fpn1 -> scale 1/4                                (1, 1024, 128, 128)  # ConvTranspose ×2
fpn2 -> scale 1/8                                (1, 1024, 64, 64)    # ConvTranspose ×1
fpn3 -> scale 1/16                               (1, 1024, 32, 32)    # 그대로
fpn4 -> scale 1/32                               (1, 1024, 16, 16)    # MaxPool
lateral 0..2 (1x1 conv)                          (1, 512, 128/64/32)
lateral 3 (PPM)                                  (1, 512, 16, 16)
concat 4 levels at 1/4 -> fpn_bottleneck         (1, 512, 128, 128)
cls_seg (dropout + 1x1 conv) = logits            (1, 5, 128, 128)
aux FCNHead logits (from fpn3, train only)       (1, 5, 32, 32)
upsampled to label size                          (1, 5, 512, 512)
```

## 2. 코드 읽는 순서 (`seg/` 기준, 위에서 아래로 따라가기)

| 순서 | 파일 : 위치 | 무엇을 보나 |
|---|---|---|
| 1 | `configs/repro/scalemae_A_12ep.py` → `configs/vit_base_win/scale_MAE.py` → `configs/_base_/models/scale_MAE_vit_large.py` | config 상속 구조. 모델 = backbone + decode_head(UPerHead) + auxiliary_head(FCNHead), 데이터 파이프라인, optimizer |
| 2 | `mmseg/models/segmentors/encoder_decoder.py` `forward_train` (L139) | 학습 1 step: `extract_feat` → `decode_head.forward_train` + `auxiliary_head.forward_train` → loss dict |
| 3 | 같은 파일 `encode_decode` (L87), `slide_inference` (L175) | 테스트: logits를 이미지 크기로 키움. 512 crop slide지만 패치가 512라 창 1개 |
| 4 | `mmseg/models/backbones/scale_MAE_vit_large.py` `forward_features` (L315) | 패치 토큰 → 블록 3, 5, 7, 11 → 2D 지도 → `fpn1~4`. **분류의 `forward_features`와 비교하며 읽기** |
| 5 | `mmseg/models/decode_heads/uper_head.py` `forward` (L89), `psp_forward` (L79) | UperNet 본체. 아래 3절 |
| 6 | `mmseg/models/decode_heads/decode_head.py` `forward_train` (L201), `losses` (L246), `cls_seg` (L238) | logits를 라벨 크기로 업샘플 → CE(ignore_index) |
| 7 | `mmseg/datasets/pipelines/` (`LoadAnnotations`, `Resize`, `RandomCrop`, `PhotoMetricDistortion`, `Normalize`) | 학습 증강. 라벨도 같은 기하 변환을 받음 |

## 3. UperNet을 코드로 이해하기 (`uper_head.py`)

UperNet = **PPM** (Pyramid Pooling Module, PSPNet) + **FPN** (Feature Pyramid Network).
1. 입력: 해상도가 다른 4장의 특징 지도 (1/4, 1/8, 1/16, 1/32).
2. **PPM** (`psp_forward`): 가장 거친 1/32 지도를 1×1, 2×2, 3×3, 6×6으로 평균 풀링 → 다시 키워 이어 붙임 → 장면 전체 맥락.
3. **lateral 1×1 conv**: 나머지 3장을 채널 512로 맞춤.
4. **top-down** (L109): 거친 지도를 키워 더 고운 지도에 더함 → 고운 지도에 의미 정보가 섞임.
5. **fpn_convs** (3×3 conv) → 4장을 모두 1/4로 키워 이어 붙임(512×4) → `fpn_bottleneck` (3×3 conv, 512).
6. `cls_seg`: dropout + 1×1 conv → 클래스 수(5) 채널 = 픽셀별 logits.
- 보조 head(FCNHead): 1/16 지도에서 바로 분류. 학습을 돕는 용도이고 테스트에선 안 씀(loss 가중치 0.4).
- ConvModule = conv + **BN** + ReLU. 그래서 head 곳곳에 BN이 있습니다.

## 4. TTA를 얹을 때 바뀌는 곳 (이미 검증한 코드와 연결)

| 분류 TTA | 분할 TTA | 검증된 참고 코드 |
|---|---|---|
| `softmax_entropy(logits).mean(0)` | 픽셀 엔트로피 `-(p·log p).sum(1).mean()` | `G_seg_verify/mmseg/apis/test.py` `single_gpu_tent_entropy` |
| LayerNorm affine 업데이트 | + head BN을 **고정** 또는 **test 통계로** (선택 필요) | 같은 함수, `TENT_BN=batch` |
| CoTTA 32개 랜덤 증강 | 다중 스케일 7개 × flip = 14 view, 픽셀 확신도 0.69 | `single_gpu_cotta` |
| 샘플 필터 (SAR/EATA) | 픽셀 단위 필터로 바꿔야 함 | (미구현) |

## 5. 추천 학습 경로
1. `trace_shapes.py` 실행 → 출력과 2절 표를 대조 (모양이 왜 그렇게 되는지).
2. `scale_MAE_vit_large.py forward_features`를 분류 쪽 `repro/common.py`의 `features()`(L86; `D_tta/tta.py`가 이걸 import)와 나란히 읽기.
3. `uper_head.py forward`를 3절과 대조. `trace_shapes.py` 4번 블록이 같은 계산을 풀어 쓴 것.
4. `encoder_decoder.py forward_train` → `decode_head.py losses` 순으로 학습 1 step 따라가기.
5. 마지막으로 `G_seg_verify/mmseg/apis/test.py`의 `single_gpu_tent`와 `single_gpu_tent_entropy`를 비교 → TTA가 이 구조 어디에 끼어드는지.
- 논문 대신 코드로 먼저 보고 싶으면: UperNet 원 논문(Xiao et al., ECCV 2018) 3절의 그림과 `uper_head.py`를 같이 보는 것이 가장 빠릅니다.

## 6. SatMAE와 ScaleMAE 분할 백본은 무엇이 다른가 (2026-10-20 추가)

`mmseg/models/backbones/SatMAE_vit_large.py`와 `scale_MAE_vit_large.py`를 diff하면 **클래스 본체는 사실상 같습니다**.
- 둘 다 timm `VisionTransformer`(ViT-L/16, 24블록, 1024차원)를 상속하고, `forward_features`, `fpn1~4`, `PatchEmbed`가 같습니다.
- 위치 임베딩: 둘 다 **고정 2D sin-cos**(`__init__`에서 생성)를 씁니다. `scale_MAE_vit_large.py`에 `get_2d_sincos_pos_embed_with_resolution`(GSD 반영 함수)이 정의돼 있지만, **forward에서는 호출하지 않습니다**. 분류 쪽 `input_res=1`과 같은 결론으로, 분할에서도 Scale-MAE의 GSD 기능은 쓰이지 않습니다.
- 실제 차이는 세 가지뿐입니다.
  1. **가중치**: SatMAE는 fMoW 사전학습 (A: Sentinel `e199` / B: RGB `fmow_pretrain.pth`), ScaleMAE는 `scalemae-vitlarge-800.pth`.
  2. **키 제거 규칙** (`init_weights`): SatMAE는 모양이 다르면 `patch_embed`까지 지웁니다(그래서 Sentinel 체크포인트에서 조용히 랜덤이 됨). ScaleMAE는 head와 pos_embed만 지웁니다.
  3. **정규화** (config의 `img_norm_cfg`): SatMAE는 fMoW 통계(A는 0~1 단위 오류, B는 ×255로 수정), ScaleMAE는 ImageNet 통계(0~255).

우리가 고친 곳(모두 `[REOBench repro]` 표시): 체크포인트 경로를 config로 받기(`ckpt`), feature 블록을 config로 받기(`out_indices`, A = [3, 5, 7, 11], B = [7, 11, 15, 23]), 512가 아닌 입력에서 위치 임베딩 보간(CoTTA 다중 스케일용).

B 설정으로 따라가 보려면:
```bash
PYTHONPATH=. TIMM_FUSED_ATTN=0 /opt/conda/envs/segtta/bin/python ../trace_shapes.py \
  --config configs/repro/satmae_B_12ep.py --ckpt ../work_dirs/satmae_B_12ep/latest.pth
```
(`trace_shapes.py`는 config의 `out_indices`를 따라갑니다.)
