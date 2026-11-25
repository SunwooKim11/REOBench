# GRAM: "Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts" (arXiv 2511.10300) — RS에서 TTA를 쓴 선행 연구

Lee, Park, Yang, Kim, Cha (MPI-SP, KAIST). arXiv 2025-11-13. 사용자가 찾은 논문(2026-11-26)이고 venue는 사용자 말로 AAAI 2026이다.
arXiv HTML에는 venue 표기가 없다. 코드: github.com/DS4H-GIS/GRAM.
출처: 요약 agent가 arXiv HTML(v1) 본문과 부록 전체를 읽고 보고한 내용(2026-11-26). Claude는 원문을 직접 읽지 않았다. 그림은 캡션만 읽혔다.
인용문은 agent가 옮긴 것이다. 발표에 쓰기 전에 표 1, 표 2의 숫자를 원문에서 한 번 확인할 것.

## 무엇을 했나
- task: 슬럼(비공식 주거지) 이진 분할. 지표는 mIoU, F1.
- shift: **도시 간 자연 shift**. source 12개 도시(아프리카 4, 아시아 3, 남미 3, 중미 2) → target 3개 도시(Dar es Salaam, Kampala, Maputo).
  손상(noise, blur, haze 등)은 다루지 않는다.
- 스스로 붙인 이름은 test-time adaptation이다("we propose GRAM, a test-time adaptation (TTA) framework for slum segmentation").
  다만 target 전체에 10 epoch 동안 self-training을 하므로 stream을 한 번 지나가는 online TTA(Tent, 우리 설정)와는 다르다.
  source 데이터를 적응 중에 쓰는지는 논문에 적혀 있지 않다.
- 데이터: ESRI World Imagery Wayback, 256×256 타일. 해상도는 본문 안에서 약 1.2 m와 10 m 두 값이 나온다(논문이 정리하지 않음).

## 방법 (GRAM)
1. **source 학습**: SegFormer encoder의 중간 층에 가벼운 MLP expert 12개(Mixture-of-Experts)를 넣는다. source 도시마다 gate가 하나 있고
   expert 중 2개를 고른다(noisy top-k). 손실은 분할 + 상호정보(도시와 expert의 대응) + 도시 분류.
2. **test**: 외부 region classifier가 target 이미지와 가장 비슷한 source 도시를 고르고, 그 도시의 gate로 예측한 것을 pseudo label로 쓴다.
3. **pseudo label 고르기**: 같은 이미지를 **다른 source 도시의 gate들로 예측한 mask와 얼마나 일치하나**(mIoU)를 점수로 삼아 상위 50%만 남긴다.
4. 남긴 pseudo label로 cross-entropy self-training(SGD, lr 1e-4, 10 epoch).
- 엔트로피 최소화, BN 통계 교체, feature 통계 정렬은 쓰지 않는다. 어떤 파라미터를 갱신하는지, 배치 크기는 적혀 있지 않다.

## 결과 (표 1, mIoU; target 3개 도시)

| 방법 | Dar es Salaam | Kampala | Maputo |
|---|---:|---:|---:|
| source (SegFormer 그대로) | 0.681 | 0.716 | 0.800 |
| MoE source (적응 없음) | 0.806 | 0.800 | 0.900 |
| SHOT | 0.712 | 0.713 | 0.813 |
| TENT | 0.691 | 0.716 | 0.802 |
| SAR | 0.700 | 0.748 | 0.804 |
| CoTTA | 0.762 | 0.821 | 0.821 |
| BECoTTA | 0.741 | 0.844 | 0.904 |
| GRAM | 0.859 | 0.870 | 0.907 |

- TENT는 source 대비 +0.010 / 0.000 / +0.002로 거의 변화가 없다. SAR은 +0.019 / +0.032 / +0.004.
- 이득의 큰 몫은 적응이 아니라 구조(MoE source)에서 온다: +0.125 / +0.084 / +0.100. 적응이 더하는 것은 +0.053 / +0.070 / +0.007.
- 각 baseline이 어느 source 모델에서 출발했는지는 적혀 있지 않다.

**pseudo label 고르는 법의 비교 (표 2, Dar es Salaam mIoU)**

| 고르는 법 | mIoU |
|---|---:|
| confidence로 고름 | 0.463 |
| 고르지 않음 | 0.818 |
| 학습 checkpoint 사이의 일치 | 0.837 |
| 도시별 gate 사이의 일치 (GRAM) | 0.859 |

저자의 설명: "Confidence-based filtering performs the worst (mIoU: 0.463), likely due to overconfidence under domain shift."
calibration을 따로 재지는 않았다(ECE, reliability 분석 없음).

## 우리 연구와의 접점

| 그쪽 | 우리 | 같은 점 / 다른 점 |
|---|---|---|
| confidence로 고르면 안 고르는 것보다 나쁘다(0.463 대 0.818). 원인은 "아마 과신"이라고만 적음 | 손상에서 confidence − 정확도가 최대 +50pt, 엔트로피 필터가 남긴 표본의 오류율 23–27% | 같은 현상. 그쪽은 추정, 우리는 17개 백본에서 측정 |
| 두 예측의 일치로 고른다(도시별 gate 사이) | flip 필터: head 입력 그대로와 표준화 뒤의 예측이 같은 표본만 | 같은 착안(confidence 대신 예측 일치). 그쪽은 일치를 만들려고 expert 구조를 새로 학습하고, 우리는 학습 없이 통계만 바꾼다 |
| TENT는 변화 없음, SAR은 작은 이득 | Potsdam에서 tent −1.6 / −1.0, sar −1.0 / −0.7 (T5) | 엔트로피 방법이 RS 분할에서 듣지 않는다는 방향이 같다. 그쪽은 손해까지는 아님 |
| 도시 간 자연 shift, 분할, 모델 1개(SegFormer) | 합성 손상 + 자연 shift 1쌍, 분류와 분할, 백본 17개 | 그쪽은 손상을 다루지 않는다. REOBench, LSCD-TTA, EarthShift를 인용하지 않는다 |

- F2의 "RS에서 TTA를 돌린 선행 연구" 목록에 들어간다. 분류하면 **자연 shift + 분할 + self-training 계열**이고, "손상된 RS 분류에 Tent·EATA·SAR·CoTTA를 돌린 연구를 찾지 못했다"는 문장은 그대로 유지된다.
- flip 필터의 선행 연구로 적어야 한다. "confidence 대신 두 예측의 일치로 표본을 고른다"는 생각은 우리만의 것이 아니다.
- 방법 후보 2(라벨 없는 head 적응)에서 표본 고르기의 baseline으로 쓸 수 있다.

## 주의 (agent 보고에서 드러난 논문 안의 불일치)
- 해상도가 두 값(약 1.2 m, 10 m)으로 적혀 있다.
- 안정성 점수가 본문은 평균, 식은 합으로 적혀 있다.
- 타일 수가 본문(2,714,489 / 라벨 86,752)과 부록 표 S7의 합(2,339,390 / 87,094)이 다르다.
- Limitations 절이 없다.
