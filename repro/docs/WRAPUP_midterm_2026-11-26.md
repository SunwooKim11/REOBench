# 중간 발표 준비용 통합 정리 (2026-11-26 발표, 2026-11-24 작성, 2026-11-25 분할 12종 반영)

발표 자료를 만들 때 이 파일 하나만 보면 되도록 모았다. 수치의 원본 위치는 각 표 아래에 적었다.
PPT skill에는 **3절(슬라이드 구성)과 4절(표)** 을 그대로 주면 된다. 그림 파일 경로는 5절에 있다.

## 0. 발표 전체 구성 (네 부분)

| 부분 | 내용 | 시간(안) | 슬라이드 | 재료 |
|---|---|---|---|---|
| 1 | 논문 2편 발표: Entropy Enigma, DSBR | 10분 | 10장 | `docs/reading/dsbr.md`, `docs/reading/entropy_enigma.md` |
| 2 | 피드백 반영, 실험 진행 상황과 분석 | 12–15분 | S1–S15 (필수 8장 중심) | 이 문서 2–4절, `docs/figs/midterm_2026-11-26/` |
| 3 | NeurIPS 2026 논문 3편 (선정 완료, 다음 주 발표) | 1분 | 1장 | 아래 표 |
| 4 | 연구계획서 방향 | 3분 | 2장 | 이 문서 6절(방법 후보), 7절(한계) |

**부분 1 (논문 2편, 10분, 10장): 논문 내용 중심**

순서는 Entropy Enigma(ICML 2024) → DSBR(arXiv 2606.02339)이다. DSBR이 Entropy Enigma 위에 서 있기 때문이다.
DSBR의 표현: Entropy Enigma는 "EM이 군집을 조이는 과정이고, 군집이 원래 클래스 중심에서 떠나면 실패한다"를 보였고,
DSBR은 "그 실패의 씨앗은 적응 전에 이미 있다: shift가 서로 다른 클래스의 군집을 합친다"를 보인다(DSBR §2).

| 장 | 논문 | 내용 | 논문에서 가져올 그림·표 |
|---|---|---|---|
| 1 | – | 두 논문의 관계. 질문은 하나다: 엔트로피 최소화(EM)는 왜 무너지는가. Entropy Enigma는 "어떻게 무너지는가", DSBR은 "왜 무너지고 어떻게 막는가" | – |
| 2 | Entropy Enigma | 질문과 첫 발견(§2). "확신 있는 표본이 맞으니까 더 확신시키면 된다"는 통념은 일부만 맞다. 처음에 맞게 분류된 이미지(45%)를 다 빼고 TENT를 돌려도 이득이 +10.50pt로, 안 뺀 경우 +12.38pt와 비슷하다 | Fig. 2 |
| 3 | Entropy Enigma | EM은 embedding을 군집화하고, 두 단계가 있다(§3). 1단계: 군집이 뚜렷해지고 학습 데이터 위치에 머문다(정확도 상승). 2단계: 군집은 계속 뚜렷해지는데 학습 데이터 위치에서 멀어진다(정확도 하락). Silhouette 변화와 정확도 변화의 상관 +0.70(1단계), Shift distance 변화와의 상관 −0.79(2단계). ResNet-50, TENT 50,000 iteration | Fig. 3, Fig. 4 |
| 4 | Entropy Enigma | 응용: Weighted Flips(§4). 적응 전후로 예측이 바뀐 표본을 초기 confidence 순위로 가중해 세고, 2차식으로 정확도를 추정한다. 23개 데이터셋 평균 오차 5.75(AC 26.02, ATC 11.81, COT 8.17) | Fig. 5 또는 6, Table 1 |
| 5 | Entropy Enigma | 한계. 합성 손상 4개 묶음에서는 COT가 낫다(2.6 대 4.7). 클래스가 적으면 flip이 거의 없어 실패한다(CIFAR-10 오차 16.64). 구조가 바뀌면 2차식을 다시 맞춰야 한다(ViT-B/16 8.34 → 1.64) | Table 1의 Noises 행 |
| 6 | DSBR | 문제(§3.1). shift가 오면 서로 다른 클래스의 feature 군집이 움직여 합쳐지는데 결정 경계는 그대로다. 그래서 예측 클래스 분포가 한쪽으로 쏠린다(prediction bias). 쏠림은 붕괴의 결과가 아니라 원인이다 | Fig. 1, Table 1 |
| 7 | DSBR | EM이 쏠림을 키운다(§3.2). EM은 같은 예측을 받은 무리를 통째로 조인다. 다른 클래스 표본이 이미 섞여 있으면 그 덩어리를 한 군집으로 보고 더 조이고, 다음 step은 더 쏠린 상태에서 시작한다. Camelyon 병원 2→5: Normal 예측 85.0% → EM 후 100%. ImageNet-C Fog 5: 한 클래스 25.5% → 99.3% | Fig. 2, Table 1 |
| 8 | DSBR | 방법(§4). 예측 클래스 빈도 P를 EMA로 세고(시작값 1/K, α = 0.9), 표본의 엔트로피에 1 / (K · P[예측 클래스])를 곱한다. 많이 예측된 클래스의 표본은 가중이 작아진다. 표본을 버리지 않고, 손실 항은 하나, 하이퍼파라미터는 α 하나 | 식, Algorithm 1 |
| 9 | DSBR | 결과와 한계. 의료 4종에서 무적응 대비 ResNet +4.2pt(Tent +0.8, ROID +1.7), ViT +2.6pt. ImageNet-C ViT-B/16: 무적응 39.8, Tent 51.9, DeYO 61.8, DSBR 64.4. 한계: 클래스 비율이 치우친 stream에서는 약하다(label shift 설정에서 α 0.9는 42.2로 Tent 53.0보다 낮음). 공개 코드 없음 | Fig. 3, ImageNet-C 표 |
| 10 | – | 두 논문에서 가져가는 것. (1) EM의 실패는 표본 하나의 문제가 아니라 분포(군집, 클래스 비율) 수준의 문제다. (2) 붕괴는 예측 분포를 보면 라벨 없이 미리 보인다. (3) 우리 10월 관찰(손상에서 예측이 한 클래스로 몰림: clean 4% → SatMAE motion blur 60%)은 DSBR의 prediction bias와 같은 현상이다. EO 백본에서의 확인은 부분 2에서 | – |

주의할 점 세 가지.
- **DSBR의 venue**: PDF와 arXiv 페이지에는 "Preprint"만 있고 NeurIPS 표기가 없다(2026-11-25 확인). 예전 검색에서 NeurIPS 2026 poster 페이지가 보였지만 열어 보지 않았다.
  "oral"은 neurips.cc에서 확인한 뒤 슬라이드에 적을 것.
- **10월 관찰과 일치하는 논문은 DSBR 쪽이다.** "예측이 한 클래스로 몰린다"를 직접 다루는 것은 DSBR(prediction bias)이다.
  Entropy Enigma가 다루는 것은 "처음엔 오르다가 떨어진다"는 두 단계이고, 예측 클래스 분포는 다루지 않는다.
- **우리 실험 결과는 부분 1에 넣지 않는다.** 10장에서 한 줄로 예고만 하고 부분 2의 S5, S6에서 보인다.

**부분 2 (3절의 S1–S15를 그대로 쓰고, 비중만 다르게)**

| 비중 | 슬라이드 | 다루는 법 |
|---|---|---|
| 필수 | S1 피드백 대응, S3 백본, S5 기존 TTA는 손해, S6 왜 실패하나, S7 정렬이 하는 일, S8 flip 신호, S11 RS 손상의 고유성, S13 남은 여지 | 그림을 띄우고 한 줄 메시지를 말한다 |
| 짧게 | S2 연구 질문, S4 데이터, S9 자연 shift, S10 분할, S15 한계 | 표나 그림만 띄우고 한 문장으로 넘어간다 |
| 건너뜀 | S12 feature의 모양 | 질문이 나오면 보여 줄 예비 장 |

S14(방법 후보)는 부분 4에서 쓴다.

**부분 3 (1장)**: 사용자가 고른 NeurIPS 2026 논문 3편(2026-11-24 선정, 훑어본 단계). 아래 "우리 연구와의 접점"은 사용자의 요약만 보고 Claude가 적은 것이고 원문은 읽지 않았다.

| 논문 | 한 줄 요약 (사용자) | 우리 연구와의 접점 (Claude, 요약 기준) |
|---|---|---|
| How Much of a Model Do We Need? Redundancy and Slimmability in Remote Sensing Foundation Models | RS FM 8개의 블록 폭을 사후에 줄여도 상대 정확도 69–109% 유지, 자연영상 MAE·DINOv2는 급락. 과제 관련 분산이 소수 주성분에 몰려 중복 인코딩됨 | "head가 읽는 방향은 소수"라는 우리 측정(이동의 83–94%가 head 부분공간 밖)과 같은 쪽의 관찰. 방법 후보 3(부분공간 안의 정렬)의 근거 후보 |
| Bridging CLIP with DINO: Cross-Modal Information Maximization for Online Test-Time Adaptation (CDIM) | 손상 입력에서 CLIP이 틀린 텍스트 앵커에 높은 확신으로 붙고, 엔트로피 최소화가 이를 강화해 예측이 붕괴. DINO 공간으로 투영한 logit 앙상블 + 믿을 수 없는 표본을 빼는 gate + 온라인 logit 보정 | 실패 양상(과신 → 엔트로피 최소화 → 예측 붕괴)이 우리 관찰과 같다. gate는 우리 flip 필터, 정보 최대화는 DSBR과 같은 계열. 방법 후보 2의 직접적인 선행 연구 |
| Who Needs Labels? Adapting Vision Foundation Models With the Metadata You Already Have (META-DINO) | 라벨이 부족한 영역(현미경, 지구관측 등)에서 메타데이터로 자기지도 학습을 유도해 유용한 요인은 남기고 허위 요인은 억제 | "RS에는 촬영 메타데이터(GSD, 센서, 시각)가 있다"는 점을 쓴다는 것이 방법 후보 1(촬영 조건 되돌리기)과 같은 착안. 그쪽은 사전학습, 우리는 test 시점 |

**부분 4 (2장)**: 연구계획서의 뼈대로 쓴다.

| 항목 | 지금 채울 수 있는 내용 |
|---|---|
| 문제 | EO 기초모델은 shift에서 confidence가 틀리고 예측이 쏠린다. 기존 TTA는 그 confidence에 기대어 실패한다 |
| 지금까지 확인한 것 | 17개 백본, 합성 손상 3개 데이터셋과 자연 shift 1쌍에서의 분석 (부분 2) |
| 남은 여지 | head 쪽 5–6pt, 자연 shift는 거의 전부 head (fig5, T12) |
| 방법 후보 | 1. 촬영 조건 되돌리기(입력 단계, 학습 없음), 2. 라벨 없는 head 적응 (6절) |
| 첫 실험 | 배율 탐색의 상한, data gaps에서 token 제외, head 적응의 가능성 확인 |
| 위험 | 후보 1의 조각들은 기존 기법과 닮았고, 후보 2는 선행 연구가 많다. 교수님 의견을 구할 지점 |

## 0.1 PPT 제작 절차 (로컬에서 저장소를 받은 뒤)

| 단계 | 할 일 | 입력으로 넣을 것 |
|---|---|---|
| 0 | 로컬에서 `main` 브랜치 pull (서버의 `repro`와 같은 내용, 2026-11-25부터) | – |
| 1 | 논문 그림 캡처 | 두 PDF에서 0절 부분 1 표의 마지막 열에 적힌 그림·표를 캡처(Entropy Enigma Fig. 2, 3, 4, Table 1 / DSBR Fig. 1, 2, 3, Table 1, ImageNet-C 표) |
| 2 | 부분 1 덱 생성 (PPT skill) | (a) 논문 PDF 2개, (b) `repro/docs/reading/entropy_enigma.md` §1, `repro/docs/reading/dsbr.md` §1, (c) 이 문서 0절의 "부분 1" 표와 주의할 점, (d) 1단계의 캡처 |
| 3 | 부분 2 덱 생성 (PPT skill) | (a) 이 문서 3절의 슬라이드 표(비중 열 포함), (b) 4절의 표 T1–T12 중 각 슬라이드가 가리키는 것, (c) 2절의 F1–F5 상세(S1, S3, S11용), (d) 그림 `repro/docs/figs/midterm_2026-11-26/fig1`–`fig10`, `repro/docs/figs/2026-11-24_feature_dist/`의 2장(잘라서) |
| 4 | 부분 3, 4 슬라이드 (PPT skill 또는 직접) | 이 문서 0절의 "부분 3" 표, "부분 4" 표, 6절(방법 후보) |
| 5 | 합치기와 한글 설명 | 그림 안 글자가 영어이므로 각 그림 아래에 3절의 "한 줄 메시지"와 그림 목록의 "한 줄로 읽는 법"을 한글로 붙임 |
| 6 | 검수 | 7절 "말하면 안 되는 것" 일곱 가지, 슬라이드의 숫자를 4절 표와 대조, DSBR venue 표기 확인 |

skill에 줄 지시문의 예(부분 2): "3절 표의 각 행을 슬라이드 한 장으로 만든다. 제목은 '제목' 열, 제목 아래 한 문장은 '한 줄 메시지' 열,
본문은 '넣을 것' 열의 그림과 표. 비중이 '짧게'인 장은 그림이나 표 하나만 둔다. 비중이 '건너뜀'인 장은 맨 뒤 부록으로 보낸다."

## 1. 발표에서 할 말 (세 문장)

1. RS 기초모델은 shift 아래에서 confidence를 믿을 수 없다. 원인은 feature가 집단으로 이동해 예측이 몇 클래스로 쏠리는 것이고,
   그래서 엔트로피 기반 TTA는 17개 백본 평균에서 손해다.
2. confidence 대신 쓸 수 있는 라벨 없는 신호(source 통계 정렬, 정렬 전후 flip)가 있다. 학습 없는 정렬이 어떤 기존 TTA보다 크고,
   flip은 합성 손상, 자연 shift, 분할에서 모두 정확도를 따라간다.
3. 다만 정렬은 새로운 방법이 아니고 자연 shift에서는 약하다. 남은 여지가 어디 있는지 쟀고(head 쪽 5–6pt, 자연 shift는 거의 전부 head),
   그 위에서 방법 후보를 정했다.

## 2. 피드백 F1–F5 대응 현황

| 항목 | 상태 | 한 줄 답 | 근거 (슬라이드) |
|---|---|---|---|
| F1. LSCD-TTA 한계를 설계 원칙으로 | 원칙은 반영, 방법은 아직 없음 | 표본별 예측이 아니라 분포(배치·stream) 수준 신호를 쓰고, 손실 항을 붙이지 않는다 | 정렬과 flip은 배치 통계만 쓰고 학습이 없다. 표본별 엔트로피 방법은 손해 (S5–S8) |
| F2. RS corruption의 고유성 | 완료 (선행 연구 조사는 일부 미확인) | "RS 이미지라서 다르다"는 증거는 약하다. 고유성은 **손상의 종류**와 **촬영 단위로 공유되는 shift**에 있다. RS 성질이 TTA를 깬다는 주장은 선행 연구에서 손상으로 시험된 적이 없다 | 같은 손상을 ImageNet과 RS에 적용한 비교, 12종의 단계별 배치 표, 선행 연구 조사 (S11) |
| F3. 벤치마크 2개 이상, 논문 읽기 | 완료 | AID-C, RESISC45-C, Potsdam-C, RESISC45 ↔ UC Merced. 네 논문 모두 노트 있음 | S4 |
| F4. fMoW 사전학습 ResNet / ViT, head 학습 방식 | 완료 | fMoW 백본 7개와 ImageNet 짝. head는 frozen feature 위의 선형 head이고 학습 방식의 영향도 통제함 | S3, 4절 표 T10 |
| F5. 백본 선택 근거 | 완료 | 구조와 방법이 같고 데이터만 다른 쌍으로 구성해 "위성 사전학습 때문인가"를 가른다 | S3. 선행 연구 4편의 선택 방식 정리(아래 F5 상세), Beyond Accuracy가 요청한 통제 연구에 해당 |

### F1 상세
- 교수님 지적: LSCD-TTA의 두 항은 전체 클래스 분포가 아니라 표본별 예측으로 분포를 조정한다. 손실을 여러 개 붙이는 것은 요즘 방식이 아니다.
- 우리 쪽 증거 세 가지.
  - LSCD 공식 코드에는 논문에 없는 "배치 평균 예측" diversity 항이 있고, 빼면 손상에서 붕괴한다(10월, `NOTES_tta.md` §1.2).
  - 표본별 엔트로피를 쓰는 Tent, SAR, EATA는 17개 백본 평균에서 손해다. stream 수준 클래스 빈도를 쓰는 DSBR은 이득이다.
  - 가장 큰 이득은 학습과 손실이 아예 없는 배치 통계 정렬에서 온다.
- 말할 수 있는 원칙: (1) 분포 수준 신호, (2) forward만, (3) confidence에 의존하지 않기.

### F2 상세
- **같은 손상 7종을 ImageNet, AID, RESISC45에 적용** (`2026-11-16_backbone_feats/ANALYSIS.md` §5).
  픽셀 수준 변화는 거의 같다(MS-SSIM: noise 0.755 / 0.710 / 0.716, blur 0.849 / 0.834 / 0.831, JPEG 0.932 / 0.932 / 0.931).
  정확도 유지율도 blur, JPEG, brightness는 같고 noise와 contrast만 RS가 낮다(62 대 48, 69 대 61).
- **REOBench 12종을 촬영–전송–처리 단계에 배치** (`SETTINGS.md` §2.5). 7종은 ImageNet-C와 같은 일반 열화이고,
  RS 고유는 data gaps, clouds, scale, rotate다.
- **confidence가 무너지는 손상은 3종에 몰려 있다**: motion blur, data gaps, scale. 이 중 둘이 RS 고유 유형이다.
  분할(Potsdam)에서는 이 셋 중 data gaps만 크게 떨어지고(−14.9 / −18.4pt) scale, rotate, translate는 거의 떨어지지 않는다(T5).
- **실제 haze(RRSHID)는 합성 haze와 다르다**: 채널별 선형 변환이 아니다(R² 0.35–0.43, 합성은 1.0).
- **사전학습 augmentation이 일부를 설명한다**: Gaussian blur augmentation이 있는 11개 백본은 Gaussian blur 손상 하락이 평균 1.3 / 2.6pt,
  없는 6개는 14.1 / 16.3pt. scale 손상은 fMoW로 학습한 쪽이 5쌍 모두 강하다.
- **선행 연구 조사** (`docs/PRIOR_ART_rs_tta_corruption.md`, 웹 조사는 agent가 수행해 일부는 원문 미확인):
  - RS 성질이 TTA를 깬다는 주장(LSCD-TTA)은 논증 수준이고 측정된 적이 없다. 그 논문의 표에서도 Tent는 source보다 높다(74.19 → 76.08).
  - 그 주장을 손상에서 시험한 연구, 손상된 RS 장면 분류에 Tent·EATA·SAR·CoTTA를 돌린 연구를 찾지 못했다.
    REOBench, Beyond Accuracy, RSPDBench, EarthShift는 TTA를 돌리지 않았고 뒤의 둘은 "필요하다 / 탐구되지 않았다"고 적었다.
  - RS에서 손상 + TTA는 분할(DeepGlobe, Tent가 도움), 검출(DIOR-C, Tent가 해로움), 초분광, LiDAR 몇 편뿐이고 결론이 엇갈린다.
- **LSCD-TTA의 주장을 손상에서 대조하면**:

| LSCD-TTA의 주장 | 손상에서 우리가 본 것 | 판정 |
|---|---|---|
| 클래스 간 유사성 때문에 확신 있는 클래스만 강조된다 | 예측이 몇 클래스로 쏠리고, tent를 오래 돌리면 예측 클래스가 30개 → 23.5개 | 이어진다 (클래스 합쳐짐) |
| 확신이 커지면 gradient가 사라지고 저신뢰 표본이 지배한다 | 정확도 15–60%인 손상에서 91–99%가 확신으로 분류된다 | 이어지지 않는다 (문제는 과신) |
| 클래스 불균형 | 직접 실험 없음 | 미검증 |
| 정보 밀도, 복잡한 장면 | DeYO의 패치 섞기 신호가 질감형 클래스에서 거의 0 | clean에서만 확인 |

  - 주의: "선행 연구가 없다"는 조사 범위 안에서의 말이다. 출판사 페이지가 열리지 않아 TGRS·JSTARS 논문은 빠졌을 수 있고,
    Murugesan 외(Frontiers in AI 2026, AID·UCM에 TENT·SAR)가 손상을 다루는지 확인하지 못했다. 발표에서는 "찾지 못했다"로 말할 것.

### F4 상세
- fMoW 사전학습 백본: SatMAE, Scale-MAE (ViT-L), MoCo-v2 fMoW, GASSL (ResNet-50), SatDINO ×3 (ViT-S/16, S/8, B/16).
  각각 ImageNet 짝이 있다. 사전학습 데이터, epoch, augmentation 표는 `SETTINGS.md` §1.
- head: 모든 백본에 같은 절차. frozen feature를 clean train 통계로 표준화하고 선형 head를 L-BFGS로 학습, L2 세기는 검증 NLL로 선택.
- 공식 linear probe 관행을 확인했다. MAE와 Scale-MAE는 head 앞에 affine 없는 BatchNorm을 두고(우리 표준화와 같은 계산),
  DINO, DINOv3, MoCo v3, SatDINO는 두지 않는다.
  matched-pair 연구(arXiv 2611.13849)의 선형 probe도 "train 통계로 표준화 → L-BFGS 로지스틱 회귀"로 우리와 같은 방식이다.
- head 학습 방식에 따른 차이를 통제했다. 표준화 없이 학습한 head에서도 정렬 이득이 같다(AID-C +5.4 대 +5.5, RESISC45-C +5.8 대 +5.9).

### F5 상세
- 선택 원칙: (구조, 사전학습 방식)이 같고 데이터만 다른 쌍 6개 + 지도학습 ImageNet 기준점 3개.
  위성 백본만 쓰면 결과가 위성 사전학습 때문인지 구조 때문인지 가를 수 없다.
- ResNet을 넣은 이유: BatchNorm이 있어 기존 TTA의 원래 대상이다.
- 선행 연구: "Satellite Pretraining Is Not One Thing"(arXiv 2611.13849, GeoCV@WACV 2027)이 같은 쌍 설계로 clean 정확도를 비교했다
  (encoder 14개, 쌍 5개, 데이터 5개에 UC Merced와 RESISC45 포함). 노트: `docs/reading/matched_pair_study.md`.
  - 쌍으로 비교하는 이유를 그쪽이 명시했다: 기존 비교는 objective, 구조, 크기, 데이터, 전처리가 한꺼번에 달라 "위성 데이터 때문인지"를 말할 수 없다.
  - 결론: 위성 사전학습은 세 쌍에서 돕고 두 쌍에서 해친다(선형 probe 평균: SatDINO +1.0, Scale-MAE +1.0, RemoteCLIP +1.2, SatMAE −4.2, DINOv3-SAT −10.3).
    "위성으로 사전학습했다"는 선택 기준이 못 된다. 우리 clean 결과와 방향이 같다.
  - 차이는 작은 인공 구조물 클래스(court, palace, storage tank)에 몰리고, 저자들은 GSD와의 관련을 추측하며 "배율을 통제한 실험"을 future work로 남겼다.
  - DINOv3-SAT은 CLS token이 약하다(patch 평균으로 바꾸면 UC Merced 선형 72.7 → 84.9). 가장 좋은 feature로도 web에 10.6pt 뒤진다.
  - 그쪽이 다루지 않은 것: 손상, 데이터셋 간 shift, calibration, TTA, 라벨 없는 신호. 우리 기여는 이쪽이다. MoCo ResNet-50 쌍도 그쪽에는 없다.
- "GeoFM의 강건성을 보완한다"에 대한 지금의 답: 약점은 shift에서 confidence가 틀리고 예측이 쏠리는 것이다.
  라벨 없는 신호로 감지할 수 있고, 공통 이동은 정렬로 되돌릴 수 있다. 남는 것은 head 불일치와 백본에서 사라진 정보다.
- 선행 연구 저자들의 선택 방식 (각 논문의 모델 선택 절을 직접 확인, 전체 표는 `SETTINGS.md` §5):

| 논문 | 고른 방식 | 일반 영상 모델 | 스스로 적은 한계 |
|---|---|---|---|
| REOBench | RS 최신 모델 12개를 세 유형(MIM, CLIP 계열, MLLM)으로. 코드와 가중치가 공개된 것 | 없음 | 모델 구성에 대한 언급 없음 |
| Beyond Accuracy | GeoFM 8 + 일반 7 + 무학습 1. 일반 모델은 "EO 모델이 더 잘 calibrate되는지 보려고 의도적으로" | 있음 (구조가 다름) | 구조·크기·recipe가 함께 달라 도메인만 따로 볼 수 없다. **구조와 용량을 고정하고 데이터만 바꾸는 통제 연구가 필요하다** |
| EarthShift | GFM 8 + 일반 3 + 무작위 초기화 2. 공개된 전 지구 사전학습 모델 | 있음 (구조가 다름) | 모델 구성에 대한 언급 없음 |
| matched-pair 연구 | 구조와 objective가 같은 쌍 5개 | 있음 (쌍) | 데이터 규모와 recipe는 미통제 |

  - 우리 구성은 Beyond Accuracy가 "필요하다"고 적은 통제 연구에 해당한다(쌍 6개, MoCo ResNet-50 쌍 포함).
  - Beyond Accuracy가 배제하지 못한 설명(일반 영상 pipeline의 blur·색 augmentation)은 우리 augmentation 표(T7)가 직접 본다.
  - REOBench는 일반 모델이 없어서 "RS 모델이라서 약한가"를 말할 기준이 없다. 우리가 ImageNet 짝을 넣은 이유다.

## 3. 슬라이드 구성 (안)

| 번호 | 비중 | 제목 | 넣을 것 | 한 줄 메시지 |
|---|---|---|---|---|
| S1 | 필수 | 지난 피드백과 대응 | 2절의 표 | 다섯 항목 중 넷은 완료(F2–F5), F1은 원칙만 반영(방법은 아직 없음) |
| S2 | 짧게 | 연구 질문 | 질문 한 문장, 10월 관찰(엔트로피 필터가 91–99%를 확신으로 분류) | confidence를 못 믿으면 무엇을 믿을 것인가 |
| S3 | 필수 | 백본 17개 (F4, F5) | **fig6**, T10, 선행 연구 4편의 선택 방식 표(2절 F5 상세) | 선행 연구가 요청한 통제 비교: 데이터만 다른 쌍 |
| S4 | 짧게 | 데이터와 벤치마크 (F3) | T11 | 합성 손상 3개, 자연 shift 1개 |
| S5 | 필수 | 기존 TTA는 평균적으로 손해 | **fig1**, T1 | 엔트로피 기반은 손해, 정렬이 가장 큼 |
| S6 | 필수 | 왜 실패하나: 과신과 예측 쏠림 | **fig4**, T2, T9 | 과신은 3종 손상에 몰려 있고 예측 클래스가 합쳐진다 |
| S7 | 필수 | 통계 정렬이 하는 일 | **fig10**, `docs/figs/2026-11-24_feature_dist/fig2_shift_align_motion_blur_s3.png`에서 필요한 행만 잘라 씀, 식 한 줄, head 통제 결과 | head 입력을 test 배치 통계로 다시 표준화 |
| S8 | 필수 | flip 신호와 정확도 추정 | **fig3**, **fig2**, T3 | flip은 정확도를 따르고, 추정 오차는 COT와 같은 수준 |
| S9 | 짧게 | 자연 shift | T4 | 신호는 유지, 정렬은 약함 |
| S10 | 짧게 | 분할 (Potsdam, 12종) | **fig9**, T5의 첫 표 | 엔트로피 방법은 손해, 정렬이 이득, flip이 정확도를 따른다. 기하 손상은 분할에서 거의 떨어지지 않는다 |
| S11 | 필수 | RS 손상의 고유성 (F2) | **fig8**, **fig7**, T6, T7, LSCD-TTA 주장 대조표(2절 F2 상세) | 고유성은 손상 종류에 있다. RS 성질과 TTA 실패의 연결은 손상에서 시험된 적이 없다 |
| S12 | 건너뜀 | feature의 모양 (선택) | `docs/figs/2026-11-24_feature_dist/fig1_clean_geometry.png`에서 필요한 행만 잘라 씀, T8 | 갈라지지만 덩어리는 아니다 |
| S13 | 필수 | 남은 여지 | **fig5**, T12 | head 쪽에 5–6pt, 자연 shift는 거의 전부 head |
| S14 | 부분 4 | 방법 후보와 계획 | 6절 | 촬영 조건 되돌리기 + 라벨 없는 head 적응 |
| S15 | 짧게 | 한계 | 7절 | seed 1개, 자연 shift 쌍 1개 |

"비중" 열: 필수는 그림을 띄우고 설명, 짧게는 한 문장으로 넘어감, 건너뜀은 예비 장.
feature 분포 그림은 2026-11-24에 그린 원본(백본 4개)에서 필요한 행만 잘라 쓴다(사용자 결정). 같은 내용을 백본 2개로 다시 그린 fig11, fig12는 대안으로만 남겨 둔다.

### 발표용 그림 (`docs/figs/midterm_2026-11-26/`, 굵게 표시한 fig 번호)

기존 결과 파일에서 그렸다(`scripts/analysis/plot_midterm_figs.py`). 서버에 한글 글꼴이 없어 그림 안 글자는 영어다.

| 파일 | 내용 | 한 줄로 읽는 법 |
|---|---|---|
| `fig1_methods.png` | 방법별 손상 정확도. 점 = 백본 17개, 막대 = 평균 | 엔트로피 기반(주황)은 일부 백본에서 크게 무너지고, 정렬(초록)은 전부 오른쪽으로 모인다 |
| `fig2_estimation.png` | 추정 정확도 대 실제 정확도, stream 2,040개. ATC / COT / flip 추정 | ATC는 대각선 위로 크게 벗어난다(과신). COT와 flip 추정은 대각선을 따른다 |
| `fig3_signals.png` | RESISC45-C에서 confidence 대 정확도, flip 대 정확도. 백본 3개 | 과신 백본(moco_in, sup_in_r50)에서 confidence는 정확도를 못 따르고 flip은 따른다 |
| `fig4_em_phases.png` | 2,000 step 동안의 정확도와 예측 클래스 수. tent 대 DSBR | tent는 클래스가 합쳐지며 떨어지고 DSBR은 30개를 유지하며 오른다 |
| `fig5_headroom.png` | 손상별로 그대로(회색) → 정렬(초록, 우리가 실제로 쓰는 것, 라벨 없음) → head 상한(파랑, 손상 stream의 라벨로 head를 다시 맞춘 가상의 값, 방법이 아님) → clean(점선) | 남은 여지는 motion blur, data gaps, scale에 몰려 있다 |
| `fig6_pairs.png` | 쌍 7개의 위성 − 일반 차이. clean, 손상, 자연 shift 양방향 | 위성 사전학습의 효과는 방식마다 반대다 |
| `fig7_blur_aug.png` | Gaussian blur 손상에서의 하락. blur augmentation 유무로 묶음 | 두 묶음의 범위가 겹치지 않는다 |
| `fig8_corruptions.png` | AID 이미지 한 장에 12종 손상(3단계), 단계별 표기 | 굵은 이름이 RS 고유 유형 |
| `fig9_seg.png` | Potsdam 손상 12종의 mIoU. none / BN 정렬 / 입력 정렬 / 둘 다. 세 묶음으로 나눠 그림(* = 우리가 생성한 4종) | 밝기·haze는 입력 정렬, 노이즈·JPEG·data gaps는 BN 정렬이 듣고, blur·구름·기하 3종은 하락이 작고 아무것도 안 듣는다 |
| `fig10_alignment.png` | 정렬이 하는 일: 흐름도(학습 때의 표준화 대 test 배치 통계로 표준화), feature 채널 하나의 분포, 클래스별 예측 비율. mae_in, AID, scale 3단계 | 손상이 feature를 한쪽으로 통째로 밀고(주황) 정렬이 제자리로 되돌린다(파랑). 예측이 한 클래스에 58% 쏠리던 것이 다시 퍼진다. 정확도 23.0 → 64.5 (clean 94.3) |
| `fig11_feature_geometry.png` | clean feature의 모양, 백본 2개(satmae, satdino_s8). 왼쪽 막대는 선형 head / 5-NN / k-means 정확도, 오른쪽은 같은 feature의 2차원 보기 세 가지 | 선형 head는 둘 다 90%대인데 k-means는 44% 대 81%. satmae는 한 클래스가 여러 섬으로 쪼개지고, satdino_s8은 섬이 뚜렷하지만 비슷한 세 클래스가 붙어 있다 |
| `fig12_shift_alignment.png` | clean / 손상(motion blur 3단계) / 정렬 후를 같은 축에. 백본 2개(mae_in, dinov3_web_l16). 삼각형 = stream 평균 | 손상이 전체를 한 방향으로 민다. 정렬은 평균을 되돌리지만 퍼진 모양은 남는다(mae_in 36.8 → 70.8, clean 94.3). dinov3_web은 밀려도 클래스 배치가 유지된다(83.0) |

fig6에는 MAE → Scale-MAE 쌍을 넣어 7개다. DINOv3의 AID-C 값은 반올림 전 계산이라 −14.4로 찍힌다(표 T10은 −14.3).

## 4. 슬라이드에 넣을 표

### T1. 방법 비교 (17개 백본 평균 손상 정확도 %, 백본 전체 통과)

| 방법 | AID-C | RESISC45-C |
|---|---:|---:|
| none | 81.8 | 75.2 |
| tent | 81.4 | 71.2 |
| sar | 80.6 | 71.8 |
| eata | 81.9 | 71.1 |
| dsbr | 83.6 | 77.7 |
| flip 필터 | 84.3 | 79.1 |
| 정렬만 (학습 없음) | 87.3 | 81.1 |
| tent + 정렬 | 88.1 | 83.0 |
| dsbr + 정렬 | 88.1 | 83.1 |

출처: `experiments/2026-11-17_fm_tta/RESULTS.md`, `2026-11-19_dsbr/RESULTS.md`. clean 정확도 평균은 94.8 / 90.6.

### T2. 과신이 몰린 곳 (sup_in_r50, AID-C)

| 손상 | confidence − 정확도 (pt) | 가장 많이 예측된 클래스의 비율 |
|---|---:|---:|
| data gaps | +50.4 | 52% |
| scale | +31.2 | 32% |
| motion blur | +29.9 | 31% |
| 나머지 9종 평균 | +5.2 | clean 약 5% |

- confidence로 정확도를 추정하면(ATC) 10pt 넘게 틀리는 stream이 12.8%, 정렬하면 1.4%.
- 과신 백본에서 SAR의 엔트로피 필터가 남긴 표본의 오류율 23–27%, flip 필터는 7–13%.
- 출처: `docs/feedback/2026-11-16_direction.md` 11-17 항목.

### T3. 라벨 없는 정확도 추정 (17개 백본, 손상 stream 2,040개)

| 추정 | 평균 절대 오차 (pt) | 중앙값 | 10pt 초과 (%) | 최대 |
|---|---:|---:|---:|---:|
| AC (평균 confidence) | 7.6 | 2.3 | 20.0 | 83.4 |
| DoC | 7.0 | 1.2 | 19.9 | 83.2 |
| ATC | 5.2 | 1.1 | 12.8 | 84.4 |
| COT | 3.0 | 2.6 | 0.7 | 13.1 |
| COT + temperature | 2.3 | 2.0 | 0.4 | 13.5 |
| flip 추정 (우리) | 2.4 | 1.2 | 4.6 | 26.1 |

정확도와의 순위상관(60개 stream, 17개 백본 평균, AID / RESISC45): flip −0.96 / −0.97, confidence +0.96 / +0.91.
RESISC45에서 confidence는 세 백본에서 0.54–0.64로 떨어지고, flip은 17개 모두 −0.94 이하.
출처: `experiments/2026-11-24_cluster_cot/RESULTS.md`.

### T4. 자연 shift (RESISC45 ↔ UC Merced, 17개 백본 평균)

| | R45 → UCM | UCM → R45 |
|---|---:|---:|
| 정확도 하락 (pt) | 15.3 | 30.9 |
| 정확도 추정 오차: ATC (pt) | 6.0 | 7.6 |
| 정확도 추정 오차: flip 추정 (pt) | 1.7 | 5.2 |
| flip과 정확도의 순위상관 (백본 17개) | −0.99 | −0.91 |
| 정렬 이득 (pt) | +2.2 | +2.1 |
| 정렬 뒤 남는 하락 (pt) | 13.1 | 28.8 |

합성 손상의 정렬 이득은 +5.5 / +5.9. 출처: `experiments/2026-11-17_natural_shift/RESULTS.md`.

### T5. 분할 (Potsdam, 5단계 평균 mIoU)

손상 12종 = REOBench 배포 8종 + 우리가 생성한 4종(rotate, scale, translate, data gaps). 실행은 두 모델 모두 끝났다(2026-11-25 02:51).
clean은 61.9 / 65.0. 위 네 행은 mIoU, 아래 행들은 none 대비 변화(pt)다.

| | SatMAE 8종 | SatMAE 4종 | SatMAE 12종 | ScaleMAE 8종 | ScaleMAE 4종 | ScaleMAE 12종 |
|---|---:|---:|---:|---:|---:|---:|
| none | 50.6 | 57.3 | 52.8 | 51.3 | 59.2 | 54.0 |
| BN 정렬 | 52.5 | 57.9 | 54.3 | 54.4 | 60.1 | 56.3 |
| 입력 정렬 | 53.8 | 56.6 | 54.7 | 53.9 | 59.0 | 55.6 |
| 입력 + BN 정렬 | 55.1 | 57.5 | 55.9 | 56.3 | 60.0 | 57.5 |
| tent | −1.96 | −0.89 | −1.60 | −1.19 | −0.69 | −1.02 |
| sar | −1.14 | 실행 안 함 | – | −0.71 | 실행 안 함 | – |
| flip 필터 | −0.97 | −0.48 | −0.81 | −0.63 | −0.26 | −0.51 |
| dsbr | +0.85 | +0.82 | +0.84 | +0.56 | +0.57 | +0.56 |
| tent + BN 정렬 | +2.73 | +1.06 | +2.17 | +3.79 | +1.33 | +2.97 |
| flip 필터 + BN 정렬 | +2.60 | +1.03 | +2.08 | +3.67 | +1.28 | +2.87 |
| dsbr + BN 정렬 | +2.84 | 실행 안 함 | – | +3.88 | 실행 안 함 | – |

12종 열의 변화량은 8종과 4종 값을 손상 수로 가중 평균한 것이다.

생성한 4종의 손상별 값 (none은 mIoU, 나머지는 none 대비 변화):

| | SatMAE rotate | scale | translate | data gaps | ScaleMAE rotate | scale | translate | data gaps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 61.7 | 58.7 | 61.7 | 47.0 | 64.5 | 61.3 | 64.5 | 46.6 |
| BN 정렬 | −0.9 | −1.3 | −0.6 | +5.3 | −0.7 | −1.1 | −0.7 | +5.9 |
| tent | −0.9 | −1.4 | −0.9 | −0.3 | −0.6 | −0.6 | −0.5 | −1.0 |
| dsbr | −0.1 | −0.2 | −0.1 | +3.6 | −0.2 | −0.2 | −0.2 | +2.8 |
| tent + BN 정렬 | −0.7 | −1.1 | −0.5 | +6.7 | −0.7 | −1.0 | −0.6 | +7.6 |

라벨 없는 신호 (none의 픽셀 정확도와의 순위상관, 손상 stream 40개 / 60개):

| 신호 | SatMAE 8종 | SatMAE 12종 | ScaleMAE 8종 | ScaleMAE 12종 |
|---|---:|---:|---:|---:|
| flip (BN 정렬 전후) | −0.94 | −0.94 | −0.97 | −0.96 |
| 평균 confidence | +0.86 | +0.90 | +0.94 | +0.96 |
| 예측 쏠림 (예측 클래스 비율과 실제 비율의 차이, 라벨 필요) | −0.89 | −0.87 | −0.89 | −0.81 |

읽는 법.
- **손상이 세 묶음으로 갈린다** (fig9). 밝기·haze는 입력 정렬이 clean 수준까지 되돌린다(SatMAE 49.1 → 61.5, 44.2 → 61.5).
  노이즈·JPEG·data gaps는 BN 정렬이 듣는다(+2.4에서 +9.7). blur, 구름, 기하 3종은 하락이 작고(none이 clean보다 0.2–4.0pt 낮음) 어떤 방법도 듣지 않는다.
- **엔트로피 방법은 12종에서도 손해다** (tent −1.6 / −1.0). 쏠림을 누르는 dsbr은 작은 이득(+0.8 / +0.6)이고, 가장 큰 이득은 학습 없는 BN 정렬에서 온다.
  BN 정렬 위에 학습을 얹은 것의 추가 이득은 12종 평균 +0.7pt 안팎이다(SatMAE +1.5 → +2.2, ScaleMAE +2.3 → +3.0).
- **분류와 다른 점 두 가지.**
  - 분류에서 confidence가 무너지던 세 손상(motion blur, data gaps, scale) 중 분할에서 크게 떨어지는 것은 data gaps뿐이다(−14.9 / −18.4pt).
    motion blur와 scale은 3–4pt, rotate와 translate는 0.5pt 이하다. scale은 512px 패치를 줄인 뒤 reflect 패딩으로 채우므로
    패치 안의 내용이 남는다는 것이 우리 해석이고, 따로 검증하지는 않았다.
  - 분할에서는 confidence도 정확도를 따른다(+0.90 / +0.96). flip이 confidence보다 낫다고 말할 근거는 SatMAE의 작은 차이뿐이다.
    과신(confidence − 정확도)이 큰 곳은 JPEG(+10.9 / +11.2), data gaps(+6.9 / +8.9), haze(+7.1 / +4.4)이고 평균은 +2.8 / +3.9pt다.
- 예측 쏠림은 BN 정렬로 절반이 된다(12종 평균 9.8 → 5.5, 10.4 → 5.1).

출처: `experiments/2026-11-23_seg_signals/RESULTS.md`.

### T6. REOBench 12종의 단계별 배치

| 단계 | 손상 | RS 고유인가 |
|---|---|---|
| 센서 | Gaussian noise, salt & pepper | 일반 |
| 센서 (검출기 고장) | data gaps | 고유 |
| 광학, 플랫폼 | Gaussian blur, motion blur | 일반 |
| 대기 | clouds, haze | 고유 / 일부 고유 |
| 방사 보정 | brightness / contrast | 일반 |
| 전송, 저장 | JPEG | 일반 |
| 기하 | scale, rotate, translate | 고유 (scale, rotate) / 일반 |

단계 분류는 SenseBench(NeurIPS 2026)의 것이고 "고유인가"는 우리 판단이다. 전체 표는 `SETTINGS.md` §2.5.

### T7. 사전학습과 손상 (AID / RESISC45)

| 묶음 | Gaussian blur 손상에서의 하락 (pt) |
|---|---:|
| Gaussian blur augmentation 있음 (MoCo 3, DINO 3, SatDINO 3, DINOv3 2) | 평균 1.3 / 2.6 |
| 없음 (MAE 3, supervised 3) | 평균 14.1 / 16.3 |

| 쌍 (AID scale 손상 정확도) | 일반 이미지 | fMoW / 위성 |
|---|---:|---:|
| MAE / SatMAE | 25.7 | 78.9 |
| MoCo / MoCo-fMoW | 46.3 | 77.7 |
| DINO / SatDINO (S/16, S/8, B/16) | 75.8, 70.8, 67.1 | 89.6, 91.3, 89.4 |
| DINOv3 web / sat | 92.1 | 80.5 |

출처: `SETTINGS.md` §1.

### T8. feature의 모양 (clean test, 17개 백본 평균)

| | 선형 head | 5-NN | k-means | 클래스 간 분산 비율 |
|---|---:|---:|---:|---:|
| AID | 94.8 | 88.5 | 59.1 | 0.27 |
| RESISC45 | 90.6 | 83.3 | 51.3 | 0.27 |

k-means Silhouette과 정확도의 순위상관은 평균 +0.21 / +0.28이고 백본에 따라 −0.87에서 +0.87까지 갈린다.
matched-pair 연구도 같은 방향을 보고했다: 선형 probe − k-NN 차이가 MAE ViT-L 22.9, SatMAE 19.0pt, DINOv3 ViT-L 5.5pt.
그쪽의 "Silhouette이 k-NN 정확도를 예측한다(ρ 0.96–0.98)"는 encoder 사이의 비교이고 라벨로 잰 값이라 우리 결과(한 백본 안의 손상 stream 사이)와 모순이 아니다.
출처: `experiments/2026-11-24_cluster_cot/RESULTS_subspace.md`, `RESULTS.md`.

### T9. 엔트로피 최소화를 오래 돌리면 (AID-C, head 입력만 2,000 step, 17개 백본 평균)

| | 시작 | 최고 | 끝 | 끝에서 예측되는 클래스 수 (전체 30) |
|---|---:|---:|---:|---:|
| tent | 81.8 | 83.0 | 70.5 | 23.5 |
| dsbr | 81.8 | 88.1 | 88.0 | 29.8 |

source 클래스 평균에서 멀어지는 정도는 둘이 비슷하다(+5.8 대 +5.3). 둘을 가르는 것은 예측 클래스 수다.
출처: `experiments/2026-11-24_cluster_cot/RESULTS_em_phases.md`.

### T10. 백본 17개

| 방식 | 구조 | ImageNet / web | fMoW / 위성 |
|---|---|---|---|
| MAE | ViT-L/16 | mae_in | satmae, scalemae |
| MoCo-v2 | ResNet-50 | moco_in | moco_fmow, gassl_fmow |
| DINO | ViT-S/16, S/8, B/16 | dino_in ×3 | satdino ×3 |
| DINOv3 | ViT-L/16 | dinov3_web (16.9억 장) | dinov3_sat (4.9억 장) |
| 지도학습 | ResNet-50, ViT-B/16, ViT-L/16 | sup_in ×3 | 없음 (기준점) |

| 쌍 (위성 − 일반, pt) | AID clean | AID-C | 정렬 후 | R45 → UCM | UCM → R45 |
|---|---:|---:|---:|---:|---:|
| MAE → SatMAE | −1.6 | +1.9 | −6.4 | −14.1 | −0.9 |
| MoCo → MoCo-fMoW | −1.3 | +3.1 | −1.8 | +2.9 | −1.7 |
| DINO-S16 → SatDINO | +1.1 | +4.2 | +2.4 | +3.8 | +8.0 |
| DINO-S8 → SatDINO | +1.8 | +2.8 | +1.2 | +1.1 | +6.2 |
| DINO-B16 → SatDINO | +0.2 | +3.9 | +1.2 | +3.8 | +9.4 |
| DINOv3 web → sat | −2.3 | −14.3 | −8.6 | −29.8 | −28.2 |

위성 사전학습의 효과는 방식마다 반대다. 원인은 확인하지 못했다. 출처: `ANALYSIS.md` §4, `WRAPUP_progress_2026-11-23.md` §3.5.

### T11. 데이터

| 데이터 | task | 종류 | 크기 |
|---|---|---|---|
| AID-C | 분류 30클래스 | 합성 손상 12종 × 5단계 | test 2,000 × 60 |
| RESISC45-C (우리가 생성) | 분류 45클래스 | 합성 손상 12종 × 5단계 | test 6,300 × 60 |
| Potsdam-C | 분할 5클래스 | 합성 손상 12종(배포 8종 + 우리가 생성한 4종) × 5단계 | test 2,016 패치 × 60 |
| RESISC45 ↔ UC Merced | 분류 19클래스 | 자연 shift (EarthShift 방식) | 2,659 / 1,900 |

### T12. head만 고쳐서 얻을 수 있는 상한 (손상 12종의 3·5단계, 17개 백본 평균)

| | 그대로 | 정렬 | head를 라벨로 고침 (상한) | clean |
|---|---:|---:|---:|---:|
| AID-C | 77.1 | 84.2 | 89.2 | 94.3 |
| RESISC45-C | 70.5 | 78.1 | 84.2 | 90.8 |

여지(상한 − 정렬)가 몰린 곳: motion blur +20.8 / +22.1, data gaps +9.6 / +11.9, scale +8.4 / +8.7.
자연 shift에서는 target 라벨로 학습한 head가 91–99%이고 source head는 평균 15 / 31pt 낮다.
이 표는 3·5단계만이라 T1의 숫자(전체 5단계)와 다르다. 출처: `experiments/2026-11-24_cluster_cot/RESULTS_head_oracle.md`.

## 5. 볼 것 (순서와 시간)

| 순서 | 무엇 | 어디 | 시간 |
|---|---|---|---|
| 1 | 이 문서 전체 | – | 20분 |
| 2 | 정렬이 하는 일 (Normalize·입력 정렬·head 입력 정렬의 구분, 식과 두 단계, 필요한 소스 통계) | `docs/STUDY_alignment.md` §1, §2.4 | 20분 |
| 3 | 백본 표, augmentation 표, 손상 배치 표 | `docs/SETTINGS.md` §1, §2.5 | 15분 |
| 4 | feature 그림 두 장 | `docs/figs/2026-11-24_feature_dist/fig1_clean_geometry.png`, `fig2_shift_align_motion_blur_s3.png` | 10분 |
| 5 | Geometry 세 가지 (T8, Silhouette 상관, T9) | 이 문서 4절 | 15분 |
| 5 | Geometry의 설명 (k-means와 선형 head가 왜 다른가, EM 두 단계) | `docs/reading/entropy_enigma.md` §5 | 15분 |
| 5 | confidence 분석의 설명 (ECE, 과신, ATC) | `docs/STUDY_stage1_metrics.md` §3, §5.3, 수치는 `experiments/2026-11-16_backbone_feats/ANALYSIS.md` §1 | 15분 |
| 5 | feature 이동 지표 (rel_drift, frac_common, CKA). 이 문서에는 표가 없고 결론 한 문장만 있다 | `docs/STUDY_stage1_metrics.md` §4, §7, 수치는 `ANALYSIS.md` §3.5 | 필요할 때 |
| 6 | 10월 결과가 필요하면 | `docs/WRAPUP_progress_2026-11-23.md` §1 | 10분 |
| 7 | F2 선행 연구 조사 | `docs/PRIOR_ART_rs_tta_corruption.md` §1, §3 | 10분 |
| 7 | 백본 선택 근거의 선행 연구 (F5) | `docs/reading/matched_pair_study.md`의 "원문 직접 확인" 절 | 10분 |
| 8 | 질문이 나올 만한 논문 | `docs/reading/dsbr.md`, `entropy_enigma.md` §5, `sensebench.md` | 필요할 때 |

Geometry에서 이해해야 할 것은 세 문장이다.
- 선형 head는 95%를 맞히는데 k-means는 59%다. 클래스는 직선으로 갈라지지만 둥근 덩어리 하나를 이루지 않는다.
- 그래서 "잘 뭉쳐 있으면 정확도가 높다"(Entropy Enigma의 전제)는 우리 백본에서 일부에만 맞는다.
- 엔트로피 최소화가 실패할 때 일어나는 일은 예측 클래스가 합쳐지는 것이다(30개 → 23.5개).

10월 그림은 `docs/figs/meeting_2026-10/`에 있다(왜 TTA가 실패하나, 깊이별 정렬, 손상 예시 등).

## 6. 방법 후보 (S14)

| 순위 | 후보 | 한 줄 설명 | 근거 | 약점 |
|---|---|---|---|---|
| 1 | 촬영 조건 되돌리기 | stream이 공유하는 촬영 조건(밝기, 배율, 무효 영역)을 source 통계에 맞춰 추정하고 입력에서 되돌림. 학습 없음 | 밝기 쪽(입력 정렬)은 이미 거의 완전히 회복. scale과 data gaps는 여지가 크고 원인이 분명 | 조각 각각은 기존 기법과 닮음 |
| 2 | 라벨 없는 head 적응 | 클래스 비율을 강제하고 flip으로 고른 표본으로 head만 갱신 | 여지가 가장 크게 측정된 곳(자연 shift, motion blur) | 선행 연구가 많음(T3A, LAME, SHOT) |
| 3 | head가 읽는 부분공간 안의 정렬 (GOLD) | 29–44차원 안에서 공분산까지 정렬 | 이동의 83–94%는 head가 읽지 않는 방향 | 부품 수준 |
| 4 | 붕괴 감지로 적응 제어 | 예측 클래스 수와 flip으로 멈추거나 되돌림 | T9 | 이득이 작음 |

1번의 외부 근거: matched-pair 연구가 "위성 사전학습의 효과는 겉보기 크기가 GSD에 달린 작은 구조물에 몰린다"고 보고하고 배율을 통제한 실험을 future work로 남겼다.

첫 실험(발표 뒤): 배율 탐색의 상한(scale 손상, RESISC45 ↔ UCM), data gaps에서 token 제외, 2번의 가능성 확인.
2번의 가능성 확인은 시작했다가 중단했다(결과 없음).

## 7. 한계와 주의 (S15, 질문 대비)

**한계**
- seed는 대부분 1개다. 자연 shift는 데이터 쌍 하나, 분할은 모델 2개다.
- 정렬과 flip 결과는 배치 64, 한 종류의 손상만 있는 stream 기준이다. 배치 16에서는 정렬 이득이 줄고 clean 비용이 생긴다.
- 방법은 아직 없다. 지금까지는 분석과 기존 방법의 적용이다.

**말하면 안 되는 것**
- "flip이 가장 좋은 정확도 추정기다": COT와 평균이 같고 큰 오차는 COT가 적다.
- "통계 정렬이 우리 방법이다": head 입력에 AdaBN을 적용한 것이고 FOA와도 겹친다. baseline이다.
- "DSBR Table 3을 재현했다": 이득의 크기가 비슷한 것까지만 확인했다.
- "SenseBench가 REOBench 손상의 고유성을 비판했다": 그 문장은 열화를 평가 대상으로 삼지 않았다는 뜻이다.
- "위성 사전학습이 낫다 / 못하다": 방식마다 반대이고 원인은 모른다.
- "손상된 RS 분류에 TTA를 적용한 선행 연구는 없다": 조사 범위 안에서 찾지 못했다는 것까지만 말한다.
- "분할에서도 flip이 confidence보다 낫다": 분할에서는 둘 다 정확도를 따른다(−0.94 / −0.96 대 +0.90 / +0.96).

**예상 질문과 답**

| 질문 | 답 |
|---|---|
| 정렬은 source 통계를 쓰는데 반칙 아닌가 | 표준화해서 학습한 head에서는 test 식에 source 통계가 들어가지 않는다. 그대로 학습한 head에서 쓰는 통계도 BatchNorm의 running 통계와 같은 종류다. 다만 새롭지는 않다 |
| RS만의 문제가 무엇인가 | 이미지 도메인이 아니라 손상의 종류(data gaps, scale, clouds)와, shift가 촬영 단위로 stream 전체에 공유된다는 점이다 |
| 왜 DINOv3 web이 가장 강한가 | 같은 방법의 위성 모델은 14pt 낮으므로 방법이 아니라 데이터나 규모다. 어느 쪽인지는 가르지 못했다. matched-pair 연구와 DINOv3 저자들의 GEO-Bench 평가도 web이 위성을 이긴다고 보고했다 |
| DINOv3-SAT이 낮은 것은 feature를 잘못 뽑아서가 아닌가 | 우리는 CLS와 patch 평균을 이어 붙여 쓴다. matched-pair 연구가 세 가지 feature를 모두 시험했고 가장 좋은 것으로도 web에 뒤졌다 |
| 분할에서도 되나 | 현상(엔트로피 방법은 손해, 쏠림을 누르면 이득, 정렬이 가장 큼, flip이 정확도를 따름)은 12종 전체에서 같다. 다른 점은 기하 손상이 분할에서는 거의 문제가 안 되고, confidence도 정확도를 따른다는 것이다(T5). feature geometry 설명은 분류에서만 확인했다 |
| 다음은 무엇인가 | 6절의 1번과 2번. 여지가 어디 있는지는 T12로 쟀다 |

## 8. 로컬로 가져갈 것
- 이 파일과 그림은 모두 저장소에 commit돼 있다. 로컬에서는 `main`을 pull하면 된다(서버의 `repro`와 같은 내용).
- 그림: `docs/figs/midterm_2026-11-26/` 12장(발표용), `docs/figs/2026-11-24_feature_dist/` 2장, `docs/figs/meeting_2026-10/` 7장.
- PPT skill에 줄 입력: 3절(슬라이드 구성) + 4절의 표 중 해당 번호 + 7절의 한계. 슬라이드마다 "한 줄 메시지"를 제목 아래 문장으로 쓰면 된다.
