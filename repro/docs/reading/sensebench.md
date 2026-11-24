# SenseBench (NeurIPS 2026 Evaluations & Datasets, poster) — 읽기 메모와 판단

Zhong, An, Sun, Gui, Yang, He (Wuhan Univ., Shanghai AI Lab). "SenseBench: A Benchmark for Remote Sensing Low-Level Visual
Perception and Description in Large Vision-Language Models". PDF: `papers/benchmark/SenseBench_ …pdf` (43쪽).
코드: github.com/Zhong-Chenchen/SenseBench, 데이터: HuggingFace `Zhongchenchen/SenseBench`.
출처: Claude가 PDF 본문과 부록 A·B·F의 텍스트를 읽음(2026-11-24). 그림은 캡션만. 저장소는 하위 agent가 README와 파일 목록을 확인.

## 1. 무엇을 하는 논문인가
- **대상은 VLM이고, 재는 것은 "열화를 알아보는가"이다.** downstream 정확도가 열화에서 얼마나 떨어지는지는 재지 않는다.
  RSPDBench·REOBench(열화 아래의 task 정확도)와 목적이 다르다.
- 10,815개 문항: 단일 이미지 7,300 + 이미지 쌍 3,515. 객관식(Whether / What / How, exact match)과 서술(LLM judge, 0–2점 × 3항목).
- 열화 분류: 6개 대분류 22종. ISO 19157-1, ISO 19115-2, GB/T 24356 품질 표준과 촬영–전송–지상 처리 단계에 맞춰 정리.

| 대분류 | 세부 | REOBench 12종과 겹침 |
|---|---|---|
| Noise (7) | Gaussian, impulse, 공간 상관 노이즈, stripe, deadline, SAR speckle, SAR sidelobe | Gaussian, salt & pepper |
| Blur (2) | Gaussian, motion | 둘 다 |
| Cloud (2) | cloud, haze | 둘 다 |
| Compression (4) | JPEG, JPEG2000, WebP, SPIHT | JPEG |
| Correction (4) | band attenuation, band switch, 기하 압축, 기하 늘림 | scale과 일부 |
| Missing (3) | tile missing, 점 dead pixel(표 5에서는 "blind flickering"), linear blindness | data gaps와 일부 |

- SAR 2종을 빼면 RGB는 20종, 그중 REOBench에 없는 것은 13종 안팎.
- 원본: Google Earth 타일(6개 대륙 809개 도시, 인구 가중 표본, 512 × 512, 0.3–10 m). 구름·SAR 잡음은 실제 데이터(RICE, Sentinel-2 등)에서 수집.
- severity: 연속 값 s를 열화별 임계값으로 3단계(없음/약함, 중간, 심함)로 나눔. 임계값은 "사람 눈에 보이는가"로 맞춤(센서 사양 기준이 아님).
- 모델 29개(본문 22개) zero-shot. 단일 이미지 평균 정확도: 사람 89.8, Gemini-3.1-pro 75.8, LLaVA2-ov-7B 64.2, GPT-5.4 62.5,
  나머지 open-source 43–47 수준(Table 2에서 읽은 일부).

## 2. 주요 결과
1. **도메인 편향**: 일반 VLM은 일반 열화에 강하고 RS 고유 열화에 약하다(Qwen3VL-32B 70.06 → 56.40). RS VLM은 반대.
2. **복합 열화에서 붕괴**: Gemini 79.68 → 41.17, EarthDial 48.89 → 20.50. 복합 문항은 300 + 300개뿐이고, 낮은 모델은 바닥 효과(저자 언급).
3. **유창성 착시**: 서술이 그럴듯하지만 이미지와 맞지 않는다. 쌍 입력에서는 객관식과 서술 성적이 뒤집힌다.
- REOBench에 대한 문장(§2): "Even rare exceptions like REOBench treat image quality as a mere confounding variable rather than the true subject of evaluation."
  뜻: REOBench는 열화를 task 견고성을 재는 교란 요인으로만 쓰고, 모델이 열화 자체를 지각하는지는 재지 않는다. 열화의 현실성에 대한 비판은 아니다.
- 한계(부록 F): 합성 열화가 실제 분포와 다를 수 있음, LLM이 만든 정답 서술, LLM judge.

## 3. 다른 세션 요약과의 대조 (2026-11-24)
| 요약의 주장 | 확인 결과 |
|---|---|
| 6개 대분류 22종 | 맞음. 요약은 4개 대분류만 나열(Blur, Cloud 빠짐) |
| 생성 엔진 공개 여부 미확인 | **공개되지 않음.** 저장소에는 추론·평가 코드만 있고(`src/infer.py`, `eval.py` 등) 합성 코드는 보이지 않음. 논문에도 파라미터 값이 없음(haze 식만 있음) |
| "생성 방식을 가져와 AID·Potsdam에 적용" | 가져올 코드와 파라미터가 없다. 분류 체계만 빌려 우리가 구현해야 하고, "SenseBench 열화"라고 부를 수 없다 |
| REOBench 비판이 "RS 손상의 고유성"의 근거 | 문장은 맞게 인용됨. 다만 고유성이 아니라 "열화를 평가 대상으로 삼지 않았다"는 비판이다 |
| Gemini 79.68 → 41.17 | 맞음. 단 task 정확도가 아니라 열화를 맞히는 객관식 정확도 |
| EO-Jev·DORA와 엮은 결정 모델 | 우리 노트에 해당 항목이 없어 판단하지 못함 |

## 4. 우리에게 주는 것
**인용할 수 있는 것**
- 촬영–전송–지상 처리 단계와 품질 표준에 맞춘 열화 분류. 손상 목록을 "어느 단계에서 생기는가"로 정리하는 틀.
- "기존 견고성 벤치마크는 열화를 교란 요인으로만 본다" → 라벨 없이 shift를 감지하는 신호(flip, source 통계)의 동기.
- 복합 열화가 단일 열화의 단순 확장이 아니라는 관찰 → 복합 stream 평가의 동기(REOBench 심사평과 같은 지점).

**인용하면 안 되는 것**
- 복합 열화에서 task 정확도가 무너진다는 근거로는 쓸 수 없다(잰 것이 다르다).
- RSPDBench처럼 "물리 사양에 고정한 severity"의 근거로도 쓸 수 없다(가시성 기준).

**실험으로 옮길 것 (제안, 우선순위 순)**
1. 복합 stream: 기존 REOBench 연산을 둘씩 합친 stream(예: haze + noise, blur + JPEG, cloud + noise)에서 정렬 이득이 더해지는지, flip·COT 추정이 유지되는지.
2. 전송·보정 단계 열화 소수(우리 구현): band attenuation(채널별 gain, 통계 정렬이 정확히 되돌림), band switch(채널 순서 바뀜, 통계 정렬로 못 되돌림),
   stripe noise, tile missing(정보 손실, 어떤 정렬도 못 고침). "정렬로 고쳐지는가 × 신호가 감지하는가" 2 × 2를 채우는 통제 세트.
3. "열화를 먼저 알아보고 대처를 고른다": 기존 분할 결과로 상한을 계산함. 손상 10종(8종 + rotate, scale) × 5단계에서 고정 선택 isa_bn 55.81 / 57.41 mIoU,
   손상별 최선 선택 56.52 / 58.05 (SatMAE / ScaleMAE) → 상한 +0.7 / +0.6. 이득의 대부분은 정렬이 필요 없는 손상(구름, blur, 기하)과 clean에서
   "건드리지 않기"에서 온다. 지금 가진 도구(통계 정렬) 안에서는 선택기의 여지가 작다.
