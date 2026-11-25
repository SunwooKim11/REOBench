# 중간 발표 준비용 통합 정리 (2026-11-26 발표, 2026-11-24 작성, 2026-11-25 분할 12종 반영, 2026-11-26 수치 재점검, 같은 날 부분 2 구성을 다시 짬)

발표 자료를 만들 때 이 파일 하나만 보면 되도록 모았다. 수치의 원본 위치는 각 표 아래에 적었다.
PPT skill에는 **3절(슬라이드 구성)과 4절(표)** 을 그대로 주면 된다. 그림 파일 경로는 5절에 있다.

## 0. 발표 전체 구성 (네 부분)

| 부분 | 내용 | 시간(안) | 슬라이드 | 재료 |
|---|---|---|---|---|
| 1 | 논문 2편 발표: Entropy Enigma, DSBR | 10분 | 10장 | `docs/reading/dsbr.md`, `docs/reading/entropy_enigma.md` |
| 2 | 피드백 반영, 실험 진행 상황과 분석 | 12–15분 | A1–D2 17장 (3절) | 이 문서 2–4절, `docs/STUDY_backbones.md`, `docs/figs/midterm_2026-11-26/`, `docs/figs/2026-11-24_feature_dist/` |
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
  "oral"은 neurips.cc에서 확인한 뒤 슬라이드에 적을 것. → 2026-11-26 사용자가 직접 확인함.
- **10월 관찰과 일치하는 논문은 DSBR 쪽이다.** "예측이 한 클래스로 몰린다"를 직접 다루는 것은 DSBR(prediction bias)이다.
  Entropy Enigma가 다루는 것은 "처음엔 오르다가 떨어진다"는 두 단계이고, 예측 클래스 분포는 다루지 않는다.
- **우리 실험 결과는 부분 1에 넣지 않는다.** 10장에서 한 줄로 예고만 하고 부분 2의 B3, B6, C1에서 보인다.

**부분 2 (2026-11-26 다시 짬: "무엇이 듣나"가 아니라 "피드백에 어떻게 답했고, 왜 실패하나" 순서)**

사용자 지적(2026-11-26): 이전 구성은 피드백 대응이 두 장으로 끝나고, geometry 분석이 빠졌고, 결론이 "flip과 DSBR을 적용했고 표준화가 좋더라"로 읽혔다.
그래서 네 묶음으로 다시 짰다. 장별 내용은 3절.

| 묶음 | 장 | 무엇을 말하나 |
|---|---|---|
| A. 피드백과 읽은 것 | A1–A5 | 피드백 다섯 항목, 선행 연구가 백본을 고른 방식과 스스로 적은 한계, 그래서 정한 백본 17개와 그 사전학습, RS에서 TTA를 돌린 선행 연구, 데이터 |
| B. 어떻게 실패하나 (분석) | B1–B6 | 손상의 고유성, 사전학습이 설명하는 부분, confidence가 틀리는 곳, clean feature의 모양, 손상이 feature를 옮기는 방식, 엔트로피 최소화를 오래 돌리면 |
| C. 분석에서 나온 도구 | C1–C4 | 방법 비교, DSBR을 구현해 적용한 결과, flip 신호, 자연 shift와 분할 |
| D. 남은 여지와 방향 | D1–D2 | head 쪽 여지, 방법 후보와 한계 |

시간이 모자라면 C4와 A5를 한 문장으로 줄인다. B4(feature의 모양)는 이전에 "건너뜀"이었으나 본론으로 올렸다. D2의 방법 후보는 부분 4와 겹치므로 한쪽에서만 다룬다.

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
| 첫 실험 | 배율 탐색의 상한, data gaps에서 token 제외(둘 다 2026-11-25에 예비 실행, 6절 아래 참고), head 적응의 가능성 확인 |
| 위험 | 후보 1의 조각들은 기존 기법과 닮았고, 후보 2는 선행 연구가 많다. 교수님 의견을 구할 지점 |

## 0.1 PPT 제작 절차 (로컬에서 저장소를 받은 뒤)

| 단계 | 할 일 | 입력으로 넣을 것 |
|---|---|---|
| 0 | 로컬에서 `main` 브랜치 pull (서버의 `repro`와 같은 내용, 2026-11-25부터) | – |
| 1 | 논문 그림 캡처 | 두 PDF에서 0절 부분 1 표의 마지막 열에 적힌 그림·표를 캡처(Entropy Enigma Fig. 2, 3, 4, Table 1 / DSBR Fig. 1, 2, 3, Table 1, ImageNet-C 표) |
| 2 | 부분 1 덱 생성 (PPT skill) | (a) 논문 PDF 2개, (b) `repro/docs/reading/entropy_enigma.md` §1, `repro/docs/reading/dsbr.md` §1, (c) 이 문서 0절의 "부분 1" 표와 주의할 점, (d) 1단계의 캡처 |
| 3 | 부분 2 덱 생성 (PPT skill) | (a) 이 문서 3절의 슬라이드 표, (b) 4절의 표 T1–T12 중 각 슬라이드가 가리키는 것, (c) 2절의 F1–F5 상세(A1–A4, B1용)와 `repro/docs/STUDY_backbones.md` §0(A3용), (d) 그림 `repro/docs/figs/midterm_2026-11-26/fig1`–`fig10`, `repro/docs/figs/2026-11-24_feature_dist/`의 fig1, fig2(잘라서)와 정사영 설명 fig0, 클래스 모양 설명 fig3(B4용), 한글 캡션은 같은 폴더 `README.md` |
| 4 | 부분 3, 4 슬라이드 (PPT skill 또는 직접) | 이 문서 0절의 "부분 3" 표, "부분 4" 표, 6절(방법 후보) |
| 5 | 합치기와 한글 설명 | 그림 안 글자가 영어이므로 각 그림 아래에 3절의 "한 줄 메시지"와 그림 목록의 "한 줄로 읽는 법"을 한글로 붙임 |
| 6 | 검수 | 7절 "말하면 안 되는 것" 열한 가지, 슬라이드의 숫자를 4절 표와 대조 (DSBR venue는 2026-11-26 사용자 확인 완료) |

skill에 줄 지시문의 예(부분 2): "3절 표의 각 행을 슬라이드 한 장으로 만든다. 제목은 '제목' 열, 제목 아래 한 문장은 '한 줄 메시지' 열,
본문은 '넣을 것' 열의 그림과 표. 묶음(A–D)이 바뀔 때 구분 장을 넣지 말고 제목 위의 작은 글씨로 묶음 이름만 적는다."

## 1. 발표에서 할 말 (2026-11-26 고침: 결론은 진단이고, 표준화와 flip은 도구다)

1. **어떻게 실패하나.** RS 기초모델은 shift 아래에서 feature가 집단으로 이동하고 예측이 몇 클래스로 쏠린다. 그래서 confidence가 틀리고(sup_in_r50의 data gaps에서 +50pt 과신),
   이것은 손상 12종 중 셋(motion blur, data gaps, scale)에 몰려 있다. 17개 백본을 합쳐도 그렇다(AID-C에서 confidence 기반 추정의 편향이 이 셋은 +11 ~ +14pt, 나머지 9종은 +2.4pt 이하).
2. **엔트로피 최소화는 왜 실패하나.** 클래스는 선형으로 갈라지지만(95%) 둥글게 뭉친 덩어리가 아니다(k-means 59%). 클래스마다 덩어리는 하나인데
   클래스를 가르지 않는 방향으로 넓게 퍼져 있다. 이 상태에서 엔트로피를 낮추면 클래스가 합쳐진다(예측 클래스 30개 → 23.5개).
3. **사전학습이 무엇을 정하나.** 위성 사전학습의 효과는 방식마다 반대다. 강건성은 "위성 데이터인가"보다 augmentation(blur)과 목적함수(MAE 대 DINO)가 더 잘 설명한다.

이 진단에서 나온 도구 둘(head 입력 표준화, 표준화 전후 flip)은 학습 없이 기존 TTA보다 크게 회복하고 정확도를 추정하지만, 새로운 방법은 아니다
(표준화는 AdaBN과 같은 계산이고, "예측 일치로 고른다"는 GRAM 등에 선례가 있다). **방법은 아직 없다.** 남은 여지가 어디 있는지 쟀고(head 쪽 5–6pt,
자연 shift는 거의 전부 head) 그 위에서 방법 후보를 정했다.

## 2. 피드백 F1–F5 대응 현황

| 항목 | 상태 | 한 줄 답 | 근거 (슬라이드) |
|---|---|---|---|
| F1. LSCD-TTA 한계를 설계 원칙으로 | 원칙은 반영, 방법은 아직 없음 | 표본별 예측이 아니라 분포(배치·stream) 수준 신호를 쓰고, 손실 항을 붙이지 않는다 | head 입력 표준화와 flip은 배치 통계만 쓰고 학습이 없다. 표본별 엔트로피 방법은 손해, 분포 수준 신호를 쓰는 DSBR은 이득 (B3, B6, C1, C2, C3) |
| F2. RS corruption의 고유성 | 완료 (선행 연구 조사는 일부 미확인) | "RS 이미지라서 다르다"는 증거는 약하다. 고유성은 **손상의 종류**와 **촬영 단위로 공유되는 shift**에 있다. RS 성질이 TTA를 깬다는 주장은 선행 연구에서 손상으로 시험된 적이 없다 | 같은 손상을 ImageNet과 RS에 적용한 비교, 12종의 단계별 배치 표, 선행 연구 조사 (A4, B1, B2) |
| F3. 벤치마크 2개 이상, 논문 읽기 | 완료 | AID-C, RESISC45-C, Potsdam-C, RESISC45 ↔ UC Merced. 네 논문 모두 노트 있음 | A5 |
| F4. fMoW 사전학습 ResNet / ViT, head 학습 방식 | 완료 | fMoW 백본 7개와 ImageNet 짝. head는 frozen feature 위의 선형 head이고 학습 방식의 영향도 통제함 | A3, 4절 표 T10, `docs/STUDY_backbones.md` |
| F5. 백본 선택 근거 | 완료 | 구조와 방법이 같고 데이터만 다른 쌍으로 구성해 "위성 사전학습 때문인가"를 가른다 | A2, A3. 선행 연구 4편의 선택 방식 정리(아래 F5 상세), Beyond Accuracy가 요청한 통제 연구에 해당 |

### F1 상세
- 교수님 지적: LSCD-TTA의 두 항은 전체 클래스 분포가 아니라 표본별 예측으로 분포를 조정한다. 손실을 여러 개 붙이는 것은 요즘 방식이 아니다.
- 우리 쪽 증거 세 가지.
  - LSCD 공식 코드에는 논문에 없는 "배치 평균 예측" diversity 항이 있고, 빼면 손상에서 붕괴한다(10월, `NOTES_tta.md` §1.2).
  - 표본별 엔트로피를 쓰는 Tent, SAR, EATA는 17개 백본 평균에서 이득이 없거나 손해다(AID-C −1.2 ~ +0.1pt, RESISC45-C −3.4 ~ −4.1pt). stream 수준 클래스 빈도를 쓰는 DSBR은 이득이다(+1.8 / +2.5pt).
  - 가장 큰 이득은 학습과 손실이 아예 없는 통계 정렬(분류는 head 입력 표준화, 분할은 BN 통계 교체와 입력 정렬)에서 온다.
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
  - **GRAM** (Lee 외, arXiv 2511.10300, 사용자가 찾음 2026-11-26; venue는 사용자 확인으로 AAAI 2026): 슬럼 분할, 도시 간 자연 shift, SegFormer + Mixture-of-Experts.
    손상은 다루지 않는다. TENT는 source 대비 mIoU +0.010 / 0.000 / +0.002로 변화가 없고, pseudo label을 **confidence로 고르면 0.463으로 안 고른 것(0.818)보다 나쁘고**
    두 예측의 일치로 고르면 0.859다. 저자는 "likely due to overconfidence under domain shift"라고만 적었다. 우리 flip 필터와 같은 착안의 선례다.
    노트: `docs/reading/gram_slum_tta.md` (요약 agent가 읽은 내용이고 Claude는 원문을 직접 읽지 않음).
- **LSCD-TTA의 주장을 손상에서 대조하면**:

| LSCD-TTA의 주장 | 손상에서 우리가 본 것 | 판정 |
|---|---|---|
| 클래스 간 유사성 때문에 확신 있는 클래스만 강조된다 | 예측이 몇 클래스로 쏠리고, tent를 오래 돌리면 예측 클래스가 30개 → 23.5개 | 이어진다 (클래스 합쳐짐) |
| 확신이 커지면 gradient가 사라지고 저신뢰 표본이 지배한다 | SatMAE(10월, AID-C 5단계, 적응 없음)에서 정확도가 15–60%인 손상 8종의 이미지 91–99%가 엔트로피 필터를 통과한다(ScaleMAE는 58–92%). 아래 설명 참고 | 이어지지 않는다 (문제는 과신) |
| 클래스 불균형 | test stream을 long-tail로 만들면 head 입력 표준화의 이득이 줄고(+5.5 → +2.1pt, 100배 불균형) clean 비용이 생긴다(−3.0pt). 클래스 순서로 정렬된 stream에서는 표준화가 무너진다(−54pt). T14 | 이어진다. 단 무너지는 것은 엔트로피 방법이 아니라 배치 통계를 쓰는 방법이다 |
| 정보 밀도, 복잡한 장면 | 픽셀 복잡도(JPEG 비트/픽셀, edge 밀도)는 RS가 ImageNet보다 높지 않다(1.63–1.67 대 1.74). 이미지 단위로는 복잡도와 하락·과신의 상관이 0에 가깝다. 클래스 단위로는 복잡한 클래스가 손상에서 더 과신한다(ρ +0.4). T14. DeYO의 패치 섞기 신호는 질감형 클래스에서 거의 0(clean) | 약하게만 이어진다 (클래스 단위, 인과는 모름) |

  - **"91–99%가 확신으로 분류된다"의 뜻** (10월 진단, `scripts/analysis/entropy_bias.py` → `experiments/2026-10-28_entropy_bias/summary.txt`, 설명은 `docs/NOTES_tta.md` §4.3, 그림은 `docs/figs/meeting_2026-10/fig1_why_tta_fails.png`).
    REOBench가 공개한 SatMAE 분류 모델을 **적응 없이** AID-C 5단계에 돌리고, 이미지마다 예측의 엔트로피가 SAR·EATA가 "믿을 만한 표본"으로 쓰는 기준(엔트로피 < 0.4 ln 30)보다 낮은지 셌다.
    정확도가 15–60%로 떨어진 손상 8종에서도 이미지의 91–99%가 그 기준을 통과한다(motion blur: 정확도 15.2%, 통과 95.0%, 통과한 것의 정확도 15.7%).
    즉 필터가 거의 아무것도 거르지 못하고, 통과한 표본의 정확도가 전체와 같다. LSCD-TTA는 "저신뢰 표본이 손실을 지배한다"고 했지만 손상에서는 저신뢰 표본이 거의 없다.
    ScaleMAE는 같은 8종에서 58–92%가 통과하고 통과한 것의 정확도는 27–78%다. 17개 백본에서의 같은 종류 숫자는 T2(엔트로피 필터가 남긴 표본의 오류율 23–27%)다.
  - 주의: "선행 연구가 없다"는 조사 범위 안에서의 말이다. 출판사 페이지가 열리지 않아 TGRS·JSTARS 논문은 빠졌을 수 있고,
    Murugesan 외(Frontiers in AI 2026, AID·UCM에 TENT·SAR)가 손상을 다루는지 확인하지 못했다. 발표에서는 "찾지 못했다"로 말할 것.

### F4 상세
- fMoW 사전학습 백본: SatMAE, Scale-MAE (ViT-L), MoCo-v2 fMoW, GASSL (ResNet-50), SatDINO ×3 (ViT-S/16, S/8, B/16).
  각각 ImageNet 짝이 있다. 사전학습 데이터, epoch, augmentation 표는 `SETTINGS.md` §1.
- head: 모든 백본에 같은 절차. frozen feature를 clean train 통계로 표준화하고 선형 head를 L-BFGS로 학습, L2 세기는 검증 NLL로 선택.
- 공식 linear probe 관행을 확인했다. MAE와 Scale-MAE는 head 앞에 affine 없는 BatchNorm을 두고(우리 표준화와 같은 계산),
  DINO, DINOv3, MoCo v3, SatDINO는 두지 않는다.
  matched-pair 연구(arXiv 2611.13849)의 선형 probe도 "train 통계로 표준화 → L-BFGS 로지스틱 회귀"로 우리와 같은 방식이다.
- head 학습 방식에 따른 차이를 통제했다. 표준화 없이 학습한 head에서도(이때는 표준화 뒤 source 통계로 되돌리는 정렬이 필요) 이득이 같다(AID-C +5.4 대 +5.5, RESISC45-C +5.8 대 +5.9).

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
  라벨 없는 신호로 감지할 수 있고, 공통 이동은 head 입력 표준화로 되돌릴 수 있다. 남는 것은 head 불일치와 백본에서 사라진 정보다.
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

## 3. 슬라이드 구성 (2026-11-26 다시 짬, 17장)

"이전" 열은 2026-11-25까지 쓰던 번호(S1–S15)다. 이미 만든 슬라이드를 옮길 때 쓴다. feature 그림 폴더 = `docs/figs/2026-11-24_feature_dist/`.

**A. 피드백과 읽은 것**

| 번호 | 이전 | 제목 | 넣을 것 | 한 줄 메시지 |
|---|---|---|---|---|
| A1 | S1 | 지난 피드백과 대응 | 2절의 표 | 다섯 항목 중 넷은 완료(F2–F5), F1은 원칙만 반영(방법은 아직 없음) |
| A2 | 새로 | 선행 연구는 백본을 어떻게 골랐나 (F5) | 2절 F5 상세의 표(네 논문의 고른 방식과 **스스로 적은 한계**), Beyond Accuracy의 문장 "구조와 용량을 고정하고 데이터만 바꾸는 통제 연구가 필요하다", matched-pair 연구의 설계(쌍 5개, 표준화 → L-BFGS 선형 probe)와 결론(위성 사전학습은 세 쌍에서 돕고 두 쌍에서 해친다) | 선행 벤치마크 셋은 구조가 섞여 있어 "위성 사전학습 때문인가"를 말하지 못한다. matched-pair 연구의 쌍 설계를 기준으로 삼고, 그쪽이 다루지 않은 손상·shift·TTA로 넓혔다 |
| A3 | S3 | 백본 17개와 사전학습 (F4) | `docs/STUDY_backbones.md` §0의 표(방식별로 무엇을 맞히도록 학습하나, 쌍 안에서 다른 점), head 절차 한 줄(2절 F4 상세), T10 아래 표 | 방식 네 가지마다 일반 이미지와 위성의 쌍이 있고 지도학습 기준점이 셋이다. 쌍 안에서 다른 것은 데이터이고, epoch와 규모도 함께 다르다는 한계가 있다 |
| A4 | 새로 | RS에서 TTA를 돌린 선행 연구 (F2, F1) | `docs/PRIOR_ART_rs_tta_corruption.md` §2의 표에서 여섯 행(LSCD-TTA, DeepGlobe 분할, DIOR-C 검출, GRAM, RSPDBench, EarthShift), LSCD-TTA 주장 대조표(2절 F2 상세; 불균형과 정보 밀도 행의 숫자는 T14) | 손상된 RS 장면 분류에 Tent·EATA·SAR을 돌린 연구를 찾지 못했다. 분할과 검출에서는 결론이 엇갈리고, RSPDBench와 EarthShift는 "필요하다 / 탐구되지 않았다"고 적었다. GRAM은 confidence로 고른 pseudo label이 가장 나쁘다고 보고했다 |
| A5 | S2, S4 | 연구 질문과 데이터 (F3) | 질문 한 문장, T11 | confidence를 못 믿으면 무엇을 믿을 것인가. 합성 손상 3개 데이터셋과 자연 shift 1쌍에서 본다 |

**B. 어떻게 실패하나 (분석)**

| 번호 | 이전 | 제목 | 넣을 것 | 한 줄 메시지 |
|---|---|---|---|---|
| B1 | S11 | RS 손상의 고유성 (F2) | **fig8**, T6, 같은 손상을 ImageNet과 RS에 적용한 비교(2절 F2 상세의 MS-SSIM과 유지율) | 같은 손상은 ImageNet과 RS 이미지를 거의 같게 망가뜨린다. 고유성은 이미지가 아니라 손상의 종류(data gaps, clouds, scale, rotate)에 있다 |
| B2 | S11, S3 | 사전학습이 설명하는 부분 | **fig7**, T7, **fig6** | Gaussian blur augmentation이 있는 11개와 없는 6개의 하락 범위가 겹치지 않는다. scale은 5쌍에서 위성 쪽이 강하다. 위성 사전학습의 전체 효과는 방식마다 반대다 |
| B3 | S6 | confidence는 어디서 틀리나 | T2, **fig3**의 윗줄, **fig10**의 오른쪽(예측 비율) | 과신은 손상 3종에 몰려 있고, 그때 예측이 한두 클래스로 쏠린다(mae_in, AID scale 3단계에서 한 클래스가 58%) |
| B4 | S12 | clean feature의 모양 | feature 그림 폴더의 `fig3_class_shape.png`(장난감 세 경우와 실제 측정을 같은 네 방법으로 비교), `fig1_clean_geometry.png`에서 한두 행, T8. 질문 대비 예비 장: `fig3b_class_shape_tests.png`(확인 측정 두 가지) | 클래스는 선형으로 갈라지고 이웃도 같은 클래스인데 k-means는 못 찾는다. 클래스마다 덩어리는 하나지만 클래스와 무관한 방향으로 넓게 퍼져 있기 때문이다. 뭉친 정도는 목적함수를 따른다(MAE 낮음, DINO 높음) |
| B5 | S7 | 손상이 feature를 옮기는 방식 | feature 그림 폴더의 `fig2_shift_align_motion_blur_s3.png`에서 필요한 행, **fig10**의 왼쪽(채널 분포), 숫자 한 줄(이동의 83–94%는 head가 읽지 않는 방향) | 손상은 모든 이미지를 한 방향으로 통째로 민다. head의 경계는 제자리에 있으므로 예측이 쏠린다. test 배치 통계로 다시 표준화하면 평균이 돌아오지만 퍼진 모양은 남는다 |
| B6 | S6 | 엔트로피 최소화를 오래 돌리면 (Tent protocol, 2026-11-26 05:30 중간 결과) | **fig4b**(`fig4b_em_long_tent.png`), T15, T13 (3) | 백본 norm 층을 학습하는 원래 Tent 설정에서, tent는 MAE·MoCo 계열에서 수십 step 만에 예측이 한두 클래스로 붕괴한다(SatMAE motion blur 29.6 → 4.0%, 클래스 25 → 1개). DSBR은 그 붕괴 10건 중 6건을 막고 82–87%까지 올린다. 다만 DSBR도 4건에서는 붕괴한다(ScaleMAE motion blur, MoCo-ImageNet). DINO 계열은 tent도 무너지지 않는다. feature 단계 fig4는 본문에서 뺀다 |

**C. 분석에서 나온 도구**

| 번호 | 이전 | 제목 | 넣을 것 | 한 줄 메시지 |
|---|---|---|---|---|
| C1 | S5 | 방법 비교 | **fig1**, T1 | 엔트로피 기반은 이득이 없거나 손해(RESISC45-C −4pt)이고, 학습 없는 head 입력 표준화(+5.5 / +5.9pt)가 가장 크다. 표준화는 AdaBN과 같은 계산이라 baseline이다 |
| C2 | 새로 | DSBR을 구현해 적용한 결과 | T13 (구현 확인, 17개 백본, 쏠림이 큰 손상, 분할), **fig4** 다시 언급 | 부분 1에서 발표한 DSBR을 공개 코드 없이 구현했다. ImageNet-C에서 논문과 비슷한 크기의 이득을 확인했고, EO 백본 17개 중 16개에서 none보다 높다(tent는 10–11개). 다만 이득(+1.8 / +2.5pt)은 학습 없는 표준화(+5.5 / +5.9pt)보다 작다 |
| C3 | S8 | flip 신호와 정확도 추정 | **fig3**의 아랫줄, **fig2**, T3 | 표준화 전후로 예측이 바뀐 비율(flip)은 17개 백본 모두에서 정확도를 따른다. 추정 오차는 COT와 같은 수준이다. Entropy Enigma의 Weighted Flips와는 다른 양이다 |
| C4 | S9, S10 | 자연 shift와 분할 | T4, **fig9**, T5 | 자연 shift에서 신호는 유지되고 표준화의 이득은 작다(+2pt). 분할에서도 엔트로피 방법은 손해이고 통계를 바꾸는 쪽이 이득이다 |

**D. 남은 여지와 방향**

| 번호 | 이전 | 제목 | 넣을 것 | 한 줄 메시지 |
|---|---|---|---|---|
| D1 | S13 | 남은 여지 | **fig5**, T12 | 표준화 뒤에도 head 쪽에 5–6pt가 남고, motion blur, data gaps, scale에 몰려 있다. 자연 shift의 하락은 거의 전부 head 쪽이다 |
| D2 | S14, S15 | 방법 후보와 한계 | 6절, 7절 | 촬영 조건 되돌리기와 라벨 없는 head 적응. 방법은 아직 없고, seed 1개와 자연 shift 1쌍이라는 한계가 있다 |

- B4와 B5의 feature 그림은 2026-11-24에 그린 원본(백본 4개)에서 필요한 행만 잘라 쓴다(사용자 결정). 같은 내용을 백본 2개로 다시 그린 fig11, fig12는 대안으로만 남겨 둔다.
- B4는 사용자 제안(2026-11-26)으로 설명 그림을 새로 만들었다. 읽는 법과 한글 캡션은 feature 그림 폴더의 `README.md`.
- A2–A4는 사용자가 직접 읽은 논문(EarthShift, Beyond Accuracy, RSPDBench, matched-pair 연구, GRAM)이 중심이므로 노트(`docs/reading/`)의 문장을 그대로 쓴다.

### 발표용 그림 (`docs/figs/midterm_2026-11-26/`, 굵게 표시한 fig 번호)

기존 결과 파일에서 그렸다(`scripts/analysis/plot_midterm_figs.py`). 서버에 한글 글꼴이 없어 그림 안 글자는 영어다.

| 파일 | 내용 | 한 줄로 읽는 법 |
|---|---|---|
| `fig1_methods.png` | 방법별 손상 정확도. 점 = 백본 17개, 막대 = 평균. 초록은 head 입력 표준화(①)뿐이고 입력 정렬·블록 정렬·BN 통계 교체는 들어 있지 않다(그림 아래 주석) | 엔트로피 기반(주황)은 일부 백본에서 크게 무너지고, head 입력 표준화(초록)는 전부 오른쪽으로 모인다 |
| `fig2_estimation.png` | 추정 정확도 대 실제 정확도, stream 2,040개. ATC / COT / flip 추정 | ATC는 대각선 위로 크게 벗어난다(과신). COT와 flip 추정은 대각선을 따른다 |
| `fig3_signals.png` | RESISC45-C에서 confidence 대 정확도, flip 대 정확도. 백본 3개 | 과신 백본(moco_in, sup_in_r50)에서 confidence는 정확도를 못 따르고 flip은 따른다 |
| `fig4_em_phases.png` | 2,000 step 동안의 정확도와 예측 클래스 수. tent 대 DSBR | tent는 클래스가 합쳐지며 떨어지고 DSBR은 30개를 유지하며 오른다 |
| `fig4b_em_long_tent.png` | fig4를 원래 Tent 설정(백본 norm 층 학습)으로 다시 그린 것. AID-C 3단계, 끝난 백본과 손상만(그림 제목에 개수 표시). 얇은 선 = 백본 | tent(주황)는 백본에 따라 0 근처로 붕괴하고, DSBR(파랑)은 대부분 오르지만 두 백본에서는 같이 무너진다 |
| `fig5_headroom.png` | 손상별로 그대로(회색) → head 입력 표준화(초록, 우리가 실제로 쓰는 것, 라벨 없음) → head 상한(파랑, 손상 stream의 라벨로 head를 다시 맞춘 가상의 값, 방법이 아님) → clean(점선) | 남은 여지는 motion blur, data gaps, scale에 몰려 있다 |
| `fig6_pairs.png` | 쌍 7개의 위성 − 일반 차이. clean, 손상, 자연 shift 양방향 | 위성 사전학습의 효과는 방식마다 반대다 |
| `fig7_blur_aug.png` | Gaussian blur 손상에서의 하락. blur augmentation 유무로 묶음 | 두 묶음의 범위가 겹치지 않는다 |
| `fig8_corruptions.png` | AID 이미지 한 장에 12종 손상(3단계), 단계별 표기 | 굵은 이름이 RS 고유 유형 |
| `fig9_seg.png` | Potsdam 손상 12종의 mIoU. none / BN 통계 교체(③) / 입력 정렬(②) / 둘 다. 세 묶음으로 나눠 그림(* = 우리가 생성한 4종) | 밝기·haze는 입력 정렬, 노이즈·JPEG·data gaps는 BN 통계 교체가 듣고, blur·구름·기하 3종은 하락이 작고 아무것도 안 듣는다 |
| `fig10_alignment.png` | head 입력 표준화가 하는 일: 흐름도(학습 때의 표준화 대 test 배치 통계로 표준화), feature 채널 하나의 분포, 클래스별 예측 비율. mae_in, AID, scale 3단계 | 손상이 feature를 한쪽으로 통째로 밀고(주황) 표준화가 제자리로 되돌린다(파랑). 예측이 한 클래스에 58% 쏠리던 것이 다시 퍼진다. 정확도 23.0 → 64.5 (clean 94.3) |
| `fig11_feature_geometry.png` | clean feature의 모양, 백본 2개(satmae, satdino_s8). 왼쪽 막대는 선형 head / 5-NN / k-means 정확도, 오른쪽은 같은 feature의 2차원 보기 세 가지 | 선형 head는 둘 다 90%대인데 k-means는 44% 대 81%. t-SNE에서 satmae는 한 클래스가 여러 섬처럼 보이지만 측정으로는 다봉이 아니다(feature 그림 폴더 fig3). satdino_s8은 섬이 뚜렷하지만 비슷한 세 클래스가 붙어 있다 |
| `fig12_shift_alignment.png` | clean / 손상(motion blur 3단계) / 표준화 후를 같은 축에. 백본 2개(mae_in, dinov3_web_l16). 삼각형 = stream 평균 | 손상이 전체를 한 방향으로 민다. 표준화는 평균을 되돌리지만 퍼진 모양은 남는다(mae_in 36.8 → 70.8, clean 94.3). dinov3_web은 밀려도 클래스 배치가 유지된다(83.0) |

feature 그림 폴더(`docs/figs/2026-11-24_feature_dist/`)의 네 장: `fig0_projection.png`(정사영 설명), `fig1_clean_geometry.png`(B4), `fig2_shift_align_motion_blur_s3.png`(B5),
`fig3_class_shape.png`(B4, 2026-11-26 추가: k-means가 낮은 이유의 설명 그림과 측정). 읽는 법은 그 폴더의 `README.md`.

fig6에는 MAE → Scale-MAE 쌍을 넣어 7개다(T10의 참고 행). fig6의 숫자는 T10과 같다. fig9와 fig10은 2026-11-26에 글자 겹침만 고쳐 다시 그렸다(숫자는 그대로).

## 4. 슬라이드에 넣을 표

> **용어 (2026-11-25 정리, 2026-11-26 통일).** 통계학과 선행 연구의 쓰임에 맞춰 두 단어를 나눠 쓴다.
>
> | 용어 | 뜻 | 근거 |
> |---|---|---|
> | 표준화 (standardization) | (x − 평균) / 표준편차. 평균 0, 표준편차 1로 만든다 | scikit-learn StandardScaler "Standardize features by removing the mean and scaling to unit variance", Wikipedia "Standardization (Z-score Normalization)" |
> | 정렬 (alignment) | test 쪽 통계를 source 쪽 통계에 맞춘다. 식은 test 통계로 표준화한 뒤 source 표준편차를 곱하고 source 평균을 더하는 것 | AdaIN "aligns the mean and variance of the content features with those of the style features", CORAL "aligning the second-order statistics", FOA "making them align with the source training domain" |
> | (쓰지 않음) 정규화 (normalization) | 통계학에서는 보통 [0, 1]로 맞추는 min-max scaling(Wikipedia "min-max normalization"). 딥러닝은 표준화를 normalization이라 부른다(BatchNorm 논문 "normalize each scalar feature ... mean of zero and the variance of 1", torchvision `Normalize`) | 두 뜻이 섞이므로 설명에는 쓰지 않고, BatchNorm, LayerNorm, `Normalize` 같은 고유 이름에만 남긴다 |
>
> 이 문서에 나오는 다섯 가지. 모두 학습 파라미터가 없다.
>
> | 이름 | 위치 | 하는 일 | 쓰인 곳 |
> |---|---|---|---|
> | 전처리 표준화 (`Normalize`) | 입력 픽셀, encoder 바로 앞 | 사전학습 데이터(ImageNet, fMoW)의 고정 상수로 표준화. 모든 이미지에 같은 상수라 적응이 아니다 | 모든 실험의 전처리 |
> | ① head 입력 표준화 | encoder 출력 h, Linear 바로 앞 | h를 test 배치(64장)의 채널별 평균·표준편차로 표준화. head가 표준화된 입력으로 학습됐으므로 source 통계가 (0, 1)이라 표준화만으로 정렬이 된다 | 분류 17개 백본. T1–T4, T12, fig1·3·5·10·12 |
> | ② 입력 정렬 (ISA) | 입력 픽셀, 전처리 표준화 앞 | test 배치의 RGB를 표준화한 뒤 clean train의 RGB 평균·표준편차로 되돌림 | 분할 T5, fig9 |
> | ③ BN 통계 교체 (AdaBN) | 분할 UPerNet decoder의 BN | BN이 표준화에 쓰는 running 통계를 test 배치 통계로 바꿈. 되돌리기는 BN의 γ, β가 맡는다 | 분할 T5, fig9 |
> | 블록 정렬 | encoder 중간 블록 | 블록 출력을 test 배치로 표준화한 뒤 clean train 통계로 되돌림 | 10월 실험(SatMAE, ScaleMAE)만. 이번 그림·표에는 없음 |
>
> ①–③과 블록 정렬은 모두 "test 배치로 표준화하고, 그 위치가 학습 때 갖던 통계로 되돌린다"는 같은 일을 한다. 되돌리는 몫을 ①은 head 학습 방식이, ③은 BN의 γ, β가
> 이미 갖고 있어서 표준화만 하면 되고, ②와 블록 정렬은 따로 곱하고 더해야 한다. 이 셋을 묶어 말할 때만 "통계 정렬"이라 쓴다. 상세는 `docs/STUDY_alignment.md` §1.3.

> **학습하는 방법(tent, sar, eata, dsbr, flip 필터)의 실행 방식은 두 가지다 (2026-11-26 사용자 지적으로 명시).** 발표의 방법 비교는 모두 위쪽 방식이다.
>
> | 방식 | 학습하는 파라미터 | 쓰인 곳 |
> |---|---|---|
> | **Tent protocol** (이미지를 백본에 통과) | 백본 안 norm 층(LayerNorm, BatchNorm)의 γ, β. 분류는 SAR·DeYO 규칙대로 상위 25% 블록과 마지막 norm을 뺀다. 분할은 모델의 모든 norm 층. SGD(분류) / Adam(분할), 공식 lr | **T1, fig1, T5, T13 (1)(2)(3)(5)**, 10월의 SatMAE·ScaleMAE 결과 |
> | feature 단계 (분석용 지름길) | 저장해 둔 feature 위에서, 표준화와 선형 head 사이에 새로 넣은 채널별 affine z′ = γz + β 하나. encoder, μ_src·σ_src, head는 그대로. Adam | T9, fig4, T13 (4)(6), T14의 tent·dsbr 열 |
>
> - feature 단계를 쓴 이유는 속도다. 백본을 다시 통과시키지 않아 17개 백본 × 60 stream × 2,000 step 같은 계산이 몇 분에 끝난다. Tent protocol로는 같은 계산이 며칠 걸린다.
> - 근거: ViT의 CLS feature는 마지막 LayerNorm의 출력이므로, 그 뒤의 채널별 affine을 학습하는 것은 "마지막 LayerNorm의 γ, β만 학습하는 Tent"와 표현할 수 있는 함수가 같다
>   (ResNet의 평균 pooling feature와 DINOv3의 CLS + patch 평균에는 이 대응이 정확히 성립하지 않는다).
>   frozen feature 위에서만 적응하는 선례는 있다: T3A(NeurIPS 2021, 선형 분류기만 조정), TAST(ICLR 2023, frozen extractor 위에 붙인 모듈만 학습), LAME(CVPR 2022, 출력만 보정).
>   다만 "채널별 affine 하나를 엔트로피로 학습한다"를 그대로 쓴 논문은 확인하지 못했다. 논문의 방법이 아니라 Claude가 분석용으로 만든 단순화다.
> - 공평성: feature 단계 표 안에서는 tent, dsbr, flip 필터가 같은 파라미터와 optimizer를 쓰므로 서로 비교할 수 있다. **feature 단계의 숫자를 Tent protocol의 결과처럼 말하거나 T1과 나란히 놓으면 안 된다.**
>   같은 방법이라도 숫자가 다르다(균형 stream의 tent, none 대비: feature 단계 lr 0.001에서 −0.1 / −1.2pt, lr 0.01에서 −3.2 / −11.6pt, Tent protocol −0.4 / −4.0pt).
> - 영향받는 주장: "tent는 오래 돌리면 클래스가 합쳐지고 DSBR은 오른다"(fig4, T9)는 feature 단계에서만 확인했다. T9의 주의 참고.

### T1. 방법 비교 (17개 백본 평균 손상 정확도 %, 백본 전체 통과 = Tent protocol)

| 방법 | AID-C | RESISC45-C |
|---|---:|---:|
| none | 81.8 | 75.2 |
| tent | 81.4 | 71.2 |
| sar | 80.6 | 71.8 |
| eata | 81.9 | 71.1 |
| dsbr | 83.6 | 77.7 |
| flip 필터 | 84.3 | 79.1 |
| head 입력 표준화만 (학습 없음) | 87.3 | 81.1 |
| tent + head 입력 표준화 | 88.1 | 83.0 |
| dsbr + head 입력 표준화 | 88.1 | 83.1 |

none 대비 변화로 읽으면: 엔트로피 기반 셋(tent, sar, eata)은 AID-C에서 −0.4 / −1.2 / +0.1pt, RESISC45-C에서 −4.0 / −3.4 / −4.1pt다.
평균에서 "손해"가 뚜렷한 것은 RESISC45-C이고, AID-C의 평균은 거의 0이다(eata는 +0.1). AID-C에서도 일부 백본은 크게 무너진다(fig1의 왼쪽으로 떨어진 점: sar의 moco_in 71.3 → 48.5).
dsbr +1.8 / +2.5, flip 필터 +2.5 / +3.9, head 입력 표준화 +5.5 / +5.9.

출처: `experiments/2026-11-17_fm_tta/RESULTS.md`, `2026-11-19_dsbr/RESULTS.md`. clean 정확도 평균은 94.8 / 90.6.

### T2. 과신이 몰린 곳 (sup_in_r50, AID-C)

| 손상 | confidence − 정확도 (pt) | 가장 많이 예측된 클래스의 비율 |
|---|---:|---:|
| data gaps | +50.4 | 52% |
| scale | +31.2 | 32% |
| motion blur | +29.9 | 31% |
| 나머지 9종 평균 | +5.2 | clean 약 5% |

- confidence로 정확도를 추정하면(ATC) 10pt 넘게 틀리는 stream이 12.8%, head 입력을 표준화한 뒤의 ATC는 1.4%.
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
**flip 추정의 식 (발표에서 Entropy Enigma의 Weighted Flips와 다르다고 밝힐 것).**

```
추정 정확도 = ATC_al − k · max(0, flip − clean flip),   k = 0.472
```
- flip = head 입력을 그대로 넣은 예측과 test 배치로 표준화해 넣은 예측이 다른 표본의 비율(%). 학습이 없다.
- ATC_al = 표준화한 head의 confidence에 ATC를 적용한 값. 표준화 뒤의 정확도를 잘 맞춘다(오차 1.6pt). 임계값은 라벨 있는 clean stream에서 정한다.
- 추정 대상("원래 정확도") = 그 손상 stream에서 아무 적응 없이(head 입력 표준화 전) 낸 정확도. Entropy Enigma의 WF도 적응 전 모델의 정확도를 추정한다.
  표준화한 모델 자체의 정확도가 궁금하면 ATC_al만 쓰면 된다(오차 1.6pt). 원래 head의 ATC는 과신 때문에 오차가 크다(5.2pt, 최대 84pt).
- clean flip = 손상이 없는 clean test stream에서 잰 flip(0.8–5.7%). shift가 없어도 배치 통계의 잡음으로 생기는 바닥값이라 뺀다.
- k·(flip − clean flip) = 표준화로 오른 만큼을 빼서 원래 정확도로 되돌리는 항. k = 0.47은 "초과로 뒤집힌 표본 중 순수하게 맞게 바뀐 몫이 약 절반"이라는 뜻. 2,040개 stream에서 (표준화 후 정확도 − 원래 정확도)와 초과 flip의
  상관이 0.94이고, 원점을 지나는 최소제곱 기울기가 0.472다.
- k를 맞추는 데 손상 stream의 라벨이 쓰인다. 그래서 T3의 값은 **백본 하나를 빼고 나머지 16개로 k를 맞춰 뺀 백본에 적용**한 것이다(leave-one-backbone-out).
  자연 shift(T4)에는 합성 손상 2,040개로 맞춘 k = 0.472를 그대로 썼다.

| | Entropy Enigma (Weighted Flips) | 우리 flip 추정 |
|---|---|---|
| flip | 적응 전 모델 대 EM 1,000 step 뒤 모델 | head 입력 그대로 대 표준화 뒤, 학습 없음 |
| 가중 | 초기 confidence의 percentile | 없음. clean stream의 flip을 뺌 |
| 정확도로 바꾸는 식 | 2차식 f(WF) = 0.00036·WF² − 0.32·WF + 75.66 (계수 3개) | ATC_al − k · 초과 flip (계수 1개) |
| 식을 맞춘 곳 | ImageNet-C의 보류 손상에서 한 번, 다른 데이터셋에 그대로 | 다른 백본들의 합성 손상 stream (T3), 합성 손상 전체 (T4) |

출처: `scripts/analysis/eval_labelfree.py`, `experiments/2026-11-16_backbone_feats/signals/summary.txt`.

### T4. 자연 shift (RESISC45 ↔ UC Merced, 17개 백본 평균)

| | R45 → UCM | UCM → R45 |
|---|---:|---:|
| 정확도 하락 (pt) | 15.3 | 30.9 |
| 정확도 추정 오차: ATC (pt) | 6.0 | 7.6 |
| 정확도 추정 오차: flip 추정 (pt) | 1.7 | 5.2 |
| flip과 정확도의 순위상관 (백본 17개) | −0.99 | −0.91 |
| head 입력 표준화 이득 (pt) | +2.2 | +2.1 |
| 표준화 뒤 남는 하락 (pt) | 13.1 | 28.8 |

합성 손상에서의 이득은 +5.5 / +5.9. 출처: `experiments/2026-11-17_natural_shift/RESULTS.md`.

### T5. 분할 (Potsdam, 5단계 평균 mIoU)

손상 12종 = REOBench 배포 8종 + 우리가 생성한 4종(rotate, scale, translate, data gaps). 모든 방법을 12종 전부에서 실행했다(마지막 실행 2026-11-25 22:46).
none 행은 mIoU, 나머지 행은 none 대비 변화(pt)다. 12종 평균은 8종과 4종을 손상 수로 가중 평균한 것이다.

**SatMAE** (clean 61.9)

| | 배포 8종 평균 | rotate | scale | translate | data gaps | 생성 4종 평균 | 12종 평균 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 50.6 | 61.7 | 58.7 | 61.7 | 47.0 | 57.3 | 52.8 |
| BN 통계 교체 | +1.9 | −0.9 | −1.3 | −0.6 | +5.3 | +0.6 | +1.5 |
| 입력 정렬 | +3.2 | −0.6 | −0.5 | −0.6 | −1.0 | −0.7 | +1.9 |
| 입력 정렬 + BN 통계 교체 | +4.5 | −1.3 | −1.5 | −1.0 | +4.6 | +0.2 | +3.0 |
| tent | −2.0 | −0.9 | −1.4 | −0.9 | −0.3 | −0.9 | −1.6 |
| sar | −1.1 | −0.7 | −1.1 | −0.8 | 0.0 | −0.7 | −1.0 |
| flip 필터 | −1.0 | −0.9 | −1.4 | −0.9 | +1.2 | −0.5 | −0.8 |
| dsbr | +0.9 | 0.0 | −0.2 | −0.1 | +3.6 | +0.8 | +0.8 |
| tent + BN 통계 교체 | +2.7 | −0.7 | −1.1 | −0.5 | +6.7 | +1.1 | +2.2 |
| flip 필터 + BN 통계 교체 | +2.6 | −0.7 | −1.1 | −0.5 | +6.5 | +1.0 | +2.1 |
| dsbr + BN 통계 교체 | +2.8 | −0.7 | −1.0 | −0.5 | +7.0 | +1.2 | +2.3 |

**ScaleMAE** (clean 65.0)

| | 배포 8종 평균 | rotate | scale | translate | data gaps | 생성 4종 평균 | 12종 평균 |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 51.3 | 64.5 | 61.3 | 64.5 | 46.6 | 59.2 | 54.0 |
| BN 통계 교체 | +3.1 | −0.7 | −1.1 | −0.7 | +5.9 | +0.9 | +2.3 |
| 입력 정렬 | +2.6 | −0.5 | −0.4 | −0.4 | +0.3 | −0.3 | +1.6 |
| 입력 정렬 + BN 통계 교체 | +5.0 | −1.0 | −1.2 | −0.8 | +6.2 | +0.8 | +3.6 |
| tent | −1.2 | −0.6 | −0.6 | −0.5 | −1.0 | −0.7 | −1.0 |
| sar | −0.7 | −0.6 | −0.6 | −0.5 | −0.5 | −0.5 | −0.7 |
| flip 필터 | −0.6 | −0.6 | −0.6 | −0.5 | +0.7 | −0.3 | −0.5 |
| dsbr | +0.6 | −0.2 | −0.2 | −0.2 | +2.8 | +0.6 | +0.6 |
| tent + BN 통계 교체 | +3.8 | −0.7 | −1.0 | −0.6 | +7.6 | +1.3 | +3.0 |
| flip 필터 + BN 통계 교체 | +3.7 | −0.7 | −1.0 | −0.6 | +7.4 | +1.3 | +2.9 |
| dsbr + BN 통계 교체 | +3.9 | −0.6 | −1.0 | −0.6 | +7.8 | +1.4 | +3.0 |

라벨 없는 신호 (none의 픽셀 정확도와의 순위상관, 손상 stream 40개 / 60개):

| 신호 | SatMAE 8종 | SatMAE 12종 | ScaleMAE 8종 | ScaleMAE 12종 |
|---|---:|---:|---:|---:|
| flip (BN 통계 교체 전후) | −0.94 | −0.94 | −0.97 | −0.96 |
| 평균 confidence | +0.86 | +0.90 | +0.94 | +0.96 |
| 예측 쏠림 (예측 클래스 비율과 실제 비율의 차이, 라벨 필요) | −0.89 | −0.87 | −0.89 | −0.81 |

읽는 법.
- **손상이 세 묶음으로 갈린다** (fig9). 밝기·haze는 입력 정렬이 clean 수준까지 되돌린다(SatMAE 49.1 → 61.5, 44.2 → 61.5).
  노이즈·JPEG·data gaps는 BN 통계 교체가 듣는다(+2.4에서 +9.7). blur, 구름, 기하 3종은 하락이 작고(none이 clean보다 0.2–4.0pt 낮음) 어떤 방법도 듣지 않는다.
- **엔트로피 방법은 12종에서도 손해다** (tent −1.6 / −1.0, sar −1.0 / −0.7). 쏠림을 누르는 dsbr은 작은 이득(+0.8 / +0.6)이고, 가장 큰 이득은 학습 없는 BN 통계 교체에서 온다.
  BN 통계 교체 위에 학습을 얹은 것의 추가 이득은 12종 평균 +0.7pt 안팎이다(SatMAE +1.5 → +2.1~2.3, ScaleMAE +2.3 → +2.9~3.0). 세 학습 방법(tent, flip 필터, dsbr)의 차이는 0.2pt 이내다.
- **분류와 다른 점 두 가지.**
  - 분류에서 confidence가 무너지던 세 손상(motion blur, data gaps, scale) 중 분할에서 크게 떨어지는 것은 data gaps뿐이다(−14.9 / −18.4pt).
    motion blur와 scale은 3–4pt, rotate와 translate는 0.5pt 이하다. scale은 512px 패치를 줄인 뒤 reflect 패딩으로 채우므로
    패치 안의 내용이 남는다는 것이 우리 해석이고, 따로 검증하지는 않았다.
  - 분할에서는 confidence도 정확도를 따른다(+0.90 / +0.96). flip이 confidence보다 낫다고 말할 근거는 SatMAE의 작은 차이뿐이다.
    과신(confidence − 정확도)이 큰 곳은 두 모델 공통으로 JPEG(+10.9 / +11.2)와 data gaps(+6.9 / +8.9)다. 그다음은 모델마다 달라서
    SatMAE는 haze(+7.1), ScaleMAE는 Gaussian noise(+9.3)와 salt & pepper(+8.7)다. 12종 평균은 +2.8 / +3.9pt다.
- 예측 쏠림은 BN 통계 교체로 절반이 된다(12종 평균 9.8 → 5.5, 10.4 → 5.1).

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

**k-means는 왜 낮은가 (2026-11-26 측정, `RESULTS_shape.md`, 그림은 feature 그림 폴더 `fig3_class_shape.png`)**

| clean test, 17개 백본 평균 | AID | RESISC45 |
|---|---:|---:|
| k-means (라벨 없음) | 59.1 | 51.3 |
| 가장 가까운 클래스 평균 (클래스당 중심 1개, 중심은 train 라벨로 계산) | 85.1 | 75.8 |
| 같은 방법, 클래스들에 공통인 퍼진 모양을 편 뒤 (클래스 내 공분산으로 whitening) | 92.3 | 83.9 |
| 5-NN | 88.5 | 83.3 |
| 선형 head | 94.8 | 90.6 |
| 클래스 하나를 둘로 쪼갰을 때 줄어드는 분산: 실제 클래스 / 같은 퍼짐의 Gaussian 하나 / 실제 클래스 둘을 합친 것 | 8.8 / 8.1 / 17.6% | 9.6 / 8.6 / 18.0% |
| 클래스 내 분산 중 클래스 평균 부분공간 밖의 몫 | 76% | 69% |

- 읽는 법: 클래스마다 덩어리는 **하나**다(중심 하나로 85%, 둘로 쪼개도 Gaussian 하나와 같은 정도로만 쪼개짐). 그런데 **둥글지 않다**. 퍼짐의 대부분(76%)이
  클래스를 가르지 않는 방향이고, 그 모양이 클래스들에 공통이라 그것만 펴 주면 92%가 된다. k-means는 모든 방향을 똑같이 재므로 퍼진 방향을 따라 자른다.
- "한 클래스가 여러 덩어리(섬)"라는 해석은 측정과 맞지 않는다. 2026-11-24 노트와 그림 제목("not one blob per class")이 그렇게 읽혔는데 고쳤다.
  t-SNE에서 섬이 여러 개로 보이는 것(satmae)은 근거가 못 된다.
- k-means 59%에는 알고리즘의 불안정도 섞여 있다. 클래스 평균에서 시작하면 77.8 / 62.4인데 목적함수 값은 무작위 시작과 1% 이내로 같고,
  시작점을 40번 바꾸면 정확도가 한 백본 안에서 9–19pt 흔들린다(4개 백본에서 확인). 백본끼리 k-means 숫자를 몇 pt 단위로 비교하지 말 것.
- whitening과 클래스 평균은 라벨을 쓰므로 방법이 아니라 진단이다.

**뭉친 정도는 무엇이 정하나 (쌍 비교, 위성 − 일반, pt; 5-NN은 k-means보다 안정적이라 이쪽으로 본다)**

| 쌍 | 선형 head (AID) | 5-NN (AID) | 5-NN (RESISC45) |
|---|---:|---:|---:|
| MAE → SatMAE | −1.6 | +0.7 | −4.7 |
| MoCo → MoCo-fMoW | −1.3 | +4.2 | +6.4 |
| DINO-S16 → SatDINO | +1.1 | +3.8 | +5.2 |
| DINO-S8 → SatDINO | +1.8 | +3.1 | +4.0 |
| DINO-B16 → SatDINO | +0.2 | +3.4 | +4.3 |
| DINOv3 web → sat | −2.3 | −9.4 | −10.6 |

5-NN의 절대값은 방식을 따른다: AID에서 MAE 계열 83–92, MoCo 계열 81–88, DINO 계열 91–95, DINOv3 web 96.3, 지도학습 83–84.
위성 사전학습은 5-NN을 AID 5쌍, RESISC45 4쌍에서 올린다. "RS 고유의 feature 모양"이라고 말할 근거는 없고, 말할 수 있는 것은 "모양은 목적함수가 정하고 위성 데이터는 그 안에서 조금 더 뭉치게 한다(DINOv3 제외)"까지다.

출처: `experiments/2026-11-24_cluster_cot/RESULTS_subspace.md`, `RESULTS.md`, `RESULTS_shape.md`.

### T9. 엔트로피 최소화를 오래 돌리면 (AID-C, head 입력만 2,000 step, 17개 백본 평균)

| | 시작 | 최고 | 끝 | 끝에서 예측되는 클래스 수 (전체 30) |
|---|---:|---:|---:|---:|
| tent | 81.8 | 83.0 | 70.5 | 23.5 |
| dsbr | 81.8 | 88.1 | 88.0 | 29.8 |

source 클래스 평균에서 멀어지는 정도는 둘이 비슷하다(+5.8 대 +5.3). 둘을 가르는 것은 예측 클래스 수다.
**이 표와 fig4는 feature 단계 실행이다(head 앞 affine 하나만 학습). Tent protocol(백본 norm 층)로 2,000 step을 돌린 적은 없다.**
Tent protocol에서 가진 가장 가까운 증거는 lr을 10배로 올린 실행이다(AID-C, 17개 백본): none 81.8, tent 76.1, dsbr 80.4, flip 필터 83.8.
즉 Tent protocol에서 적응을 세게 하면 **DSBR도 none 아래로 내려간다**(tent보다는 덜). "DSBR은 오래 돌려도 계속 오른다(88.0)"는 feature 단계에서만 본 것이므로 그렇게 한정해서 말할 것.
"최고"는 stream마다 정확도가 가장 높았던 checkpoint의 값을 평균한 것이다. 최고점의 step이 stream마다 달라서(tent 중앙값 약 230 step)
fig4의 평균 곡선에는 tent의 봉우리가 보이지 않고 처음부터 내려가는 것처럼 보인다. fig4를 띄우고 "오르다가 떨어진다"고 말하지 말 것. 말할 수 있는 것은 "stream별로는 평균 +1.2pt 오른 뒤 떨어진다"까지다.
출처: `experiments/2026-11-24_cluster_cot/RESULTS_em_phases.md`.

### T10. 백본 17개

| 방식 | 구조 | ImageNet / web | fMoW / 위성 |
|---|---|---|---|
| MAE | ViT-L/16 | mae_in | satmae, scalemae |
| MoCo-v2 | ResNet-50 | moco_in | moco_fmow, gassl_fmow |
| DINO | ViT-S/16, S/8, B/16 | dino_in ×3 | satdino ×3 |
| DINOv3 | ViT-L/16 | dinov3_web (16.9억 장) | dinov3_sat (4.9억 장) |
| 지도학습 | ResNet-50, ViT-B/16, ViT-L/16 | sup_in ×3 | 없음 (기준점) |

| 쌍 (위성 − 일반, pt) | AID clean | AID-C | 표준화 후 | R45 → UCM | UCM → R45 |
|---|---:|---:|---:|---:|---:|
| MAE → SatMAE | −1.6 | +2.0 | −6.4 | −14.1 | −0.9 |
| MAE → Scale-MAE (참고) | +1.2 | +4.1 | −2.7 | −18.4 | −3.4 |
| MoCo → MoCo-fMoW | −1.3 | +3.1 | −1.8 | +2.9 | −1.7 |
| DINO-S16 → SatDINO | +1.1 | +4.2 | +2.4 | +3.8 | +8.0 |
| DINO-S8 → SatDINO | +1.8 | +2.8 | +1.2 | +1.1 | +6.2 |
| DINO-B16 → SatDINO | +0.2 | +3.9 | +1.2 | +3.8 | +9.4 |
| DINOv3 web → sat | −2.3 | −14.4 | −8.6 | −29.8 | −28.2 |

위성 사전학습의 효과는 방식마다 반대다. 원인은 확인하지 못했다. 출처: `ANALYSIS.md` §4, `WRAPUP_progress_2026-11-23.md` §3.5.
"데이터만 다른 쌍"으로 세는 것은 Scale-MAE를 뺀 6개다. Scale-MAE는 MAE와 구조는 같지만 학습 방식(GSD 위치 인코딩, 다중 배율 복원)이 달라 참고 행으로만 두었고, fig6에는 이 행까지 7개가 그려져 있다.
AID-C 열의 MAE → SatMAE와 DINOv3는 2026-11-26에 반올림 전 값으로 고쳤다(+1.9 → +2.0, −14.3 → −14.4). fig6과 같은 값이다.

### T11. 데이터

| 데이터 | task | 종류 | 크기 |
|---|---|---|---|
| AID-C | 분류 30클래스 | 합성 손상 12종 × 5단계 | test 2,000 × 60 |
| RESISC45-C (우리가 생성) | 분류 45클래스 | 합성 손상 12종 × 5단계 | test 6,300 × 60 |
| Potsdam-C | 분할 5클래스 | 합성 손상 12종(배포 8종 + 우리가 생성한 4종) × 5단계 | test 2,016 패치 × 60 |
| RESISC45 ↔ UC Merced | 분류 19클래스 | 자연 shift (EarthShift 방식) | 2,659 / 1,900 |

### T12. head만 고쳐서 얻을 수 있는 상한 (손상 12종의 3·5단계, 17개 백본 평균)

| | 그대로 | head 입력 표준화 | head를 라벨로 고침 (상한) | clean |
|---|---:|---:|---:|---:|
| AID-C | 77.1 | 84.2 | 89.2 | 94.3 |
| RESISC45-C | 70.5 | 78.1 | 84.2 | 90.8 |

여지(상한 − 표준화)가 몰린 곳: motion blur +20.8 / +22.1, data gaps +9.6 / +11.9, scale +8.4 / +8.7.
자연 shift에서는 target 라벨로 학습한 head가 91–99%이고 source head는 평균 15 / 31pt 낮다.
이 표는 3·5단계만이고 평가도 test 이미지의 절반(나머지 절반의 라벨로 head를 고치므로)이라 T1의 숫자(전체 5단계, test 전체)와 다르다. clean이 94.3 / 90.8로 T1의 94.8 / 90.6과 다른 것도 그래서다. 출처: `experiments/2026-11-24_cluster_cot/RESULTS_head_oracle.md`.

### T13. DSBR: 구현 확인과 적용 결과 (C2)

DSBR(arXiv 2606.02339)은 공개 코드가 없어 Algorithm 1을 보고 구현했다(분류 `scripts/tta/cls_tta_fm.py`, 분할 `scripts/seg/seg_tta.py`, 구현 확인 `scripts/verify/dsbr_imagenet_c.py`).
tent와 학습 파라미터, optimizer, lr이 같고 손실만 다르다(엔트로피에 1 / (K · 예측 클래스 빈도의 EMA)를 곱함, α = 0.9).

**(1) 구현 확인: ImageNet-C 5단계 15종 평균 정확도 (%), ViT-B/16, 배치 64**

| | 무적응 | Tent | DSBR | DSBR − 무적응 |
|---|---:|---:|---:|---:|
| 논문 Table 3 | 39.8 | 51.9 | 64.4 | +24.6 |
| 우리 A: 무적응이 논문과 같아지는 설정 (augreg2 체크포인트, ImageNet 정규화) | 39.8 | 47.0–49.7 | 61.2 | +21.4 |
| 우리 B: 그 체크포인트의 원래 설정 (augreg, 0.5 / 0.5 정규화) | 51.6 | 59.3–59.9 | 66.2 | +14.6 |

논문이 체크포인트, 학습 파라미터, lr을 밝히지 않아 가정한 값이다. "재현했다"가 아니라 "이득의 크기가 비슷한 것까지 확인했다"로 말한다. Tent의 범위는 Adam / SGD.

**(2) 분류: 17개 백본, 손상 60개 stream 평균 정확도 (%)**

| | AID-C | none보다 높은 백본 수 | RESISC45-C | none보다 높은 백본 수 |
|---|---:|---:|---:|---:|
| none | 81.8 | – | 75.2 | – |
| tent | 81.4 | 11 / 17 | 71.2 | 10 / 17 |
| dsbr | 83.6 | 16 / 17 | 77.7 | 16 / 17 |
| head 입력 표준화 (학습 없음) | 87.3 | 17 / 17 | 81.1 | 17 / 17 |
| dsbr + head 입력 표준화 | 88.1 | 17 / 17 | 83.1 | 17 / 17 |

**(3) 예측 쏠림이 큰 손상에서 (AID-C, 17개 백본 × 5단계 평균)**

| 손상 | none | tent | dsbr | 가장 많이 예측된 클래스의 비율: none → dsbr |
|---|---:|---:|---:|---:|
| motion blur | 49.7 | 46.5 | 54.1 | 23.8% → 17.6% |
| data gaps | 66.0 | 63.9 | 68.6 | 19.9% → 16.9% |
| scale | 73.1 | 71.0 | 76.0 | 16.2% → 12.1% |

**(4) 오래 돌리면** (T9, 2,000 step, **feature 단계 실행**): tent 81.8 → 70.5 (예측 클래스 23.5개), dsbr 81.8 → 88.0 (29.8개). Tent protocol에서 lr을 10배로 올리면(AID-C) tent 76.1, dsbr 80.4로 둘 다 none(81.8) 아래다.
**(5) 분할** (T5, 12종 평균 mIoU 변화, SatMAE / ScaleMAE): tent −1.6 / −1.0, dsbr +0.8 / +0.6, dsbr + BN 통계 교체 +2.3 / +3.0.

읽는 법: DSBR은 tent가 무너지는 곳(쏠림이 큰 손상, 긴 적응)에서 안전하고 거의 모든 백본에서 이득이다. 그러나 이득의 크기는 학습 없는 head 입력 표준화보다 작고,
표준화 위에 얹으면 +0.8 / +2.0pt를 더한다(tent를 얹은 것과 같은 수준). 논문이 스스로 적은 한계(클래스 비율이 치우친 stream에서 약함)는 (6)에서 확인했다.

출처: `experiments/2026-11-19_dsbr/RESULTS.md` §2, §3, `2026-11-24_cluster_cot/RESULTS_em_phases.md`, `2026-11-23_seg_signals/RESULTS.md`.

**(6) 클래스 비율이 치우친 stream에서** (T14의 설정, head 입력 affine만 학습하는 feature 단계 실행이라 (2)의 숫자와 다르다; none 대비 변화 pt, AID-C / RESISC45-C)

| stream 구성 | tent | dsbr | head 입력 표준화 |
|---|---:|---:|---:|
| 균형 (섞음) | −3.2 / −11.6 | +1.8 / +3.6 | +5.5 / +5.9 |
| long-tail 10배 | −0.7 / −4.3 | +0.8 / +2.2 | +4.5 / +5.3 |
| long-tail 100배 | +0.2 / −0.5 | +0.8 / +2.1 | +2.1 / +4.0 |
| 클래스 순서로 정렬 | −3.7 / −13.7 | −1.1 / −1.2 | −54.3 / −60.5 |

DSBR은 long-tail에서 이득이 절반쯤으로 줄지만 손해로 넘어가지는 않고, 정렬된 stream에서는 −1pt다. tent와 dsbr의 이 숫자는 **백본 안의 norm 층이 아니라 head 바로 앞에 새로 넣은 채널별 affine(z′ = γz + β) 하나만 학습**한 것이다(저장된 feature를 써서 백본을 다시 통과시키지 않으려고). encoder, 학습 때의 표준화 상수(μ_src, σ_src), 선형 head는 그대로다. T1과 (2)의 tent / dsbr은 백본 안의 norm 층을 학습한 것이라 숫자가 다르다.
주의: long-tail stream은 표본 수가 적어(AID 788 / 445장) step 수도 적다.
tent의 손해가 줄어 보이는 것은 그 영향일 수 있다. 표준화는 상태가 없어서 이 영향을 받지 않는다.

### T14. LSCD-TTA가 든 RS 성질 두 가지를 손상에서 확인 (2026-11-26, A4의 대조표용)

저장된 feature로 계산했다(`scripts/analysis/rs_properties.py`, `experiments/2026-11-26_rs_properties/RESULTS.md`). 17개 백본 평균.

**(가) 클래스 불균형: test stream의 클래스 구성을 바꾼다** (손상 60개 stream 평균 정확도 %, 괄호 = none 대비)

| stream 구성 | AID-C none | 표준화 | clean에서 표준화의 비용 | RESISC45-C none | 표준화 | clean에서 표준화의 비용 | 표준화가 none보다 높은 백본 수 (17개 중, AID / R45) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 균형 (test 전체를 섞음, 다른 표의 설정) | 81.8 | 87.3 (+5.5) | −0.3 | 75.2 | 81.2 (+5.9) | −0.4 | 17 / 17 |
| long-tail 10배 | 82.6 | 87.1 (+4.5) | −1.0 | 75.3 | 80.6 (+5.3) | −1.2 | 16 / 17 |
| long-tail 100배 | 82.1 | 84.2 (+2.1) | −3.0 | 75.2 | 79.1 (+4.0) | −2.6 | 14 / 16 |
| 클래스 순서로 정렬 (배치마다 클래스 한두 개) | 81.8 | 27.5 (−54.3) | −63.6 | 75.2 | 14.7 (−60.5) | −74.2 | 0 / 0 |

- **stream 구성의 정확한 정의.** 모든 구성에서 손상 60종 + clean의 61개 stream은 같은 test 이미지로 이루어지고, 배치는 64장, 표준화는 배치마다 따로 계산한다.
  - 균형: test 전체(AID 2,000장, 클래스당 47–86장 / RESISC45 6,300장, 클래스당 115–169장)를 무작위로 섞는다.
  - long-tail ρ배: 클래스에 무작위 순위 r = 0 … C−1을 매기고, 순위 r인 클래스는 자기 이미지의 ρ^(−r / (C−1)) 만 남긴다(1등 100%, 꼴등 1/ρ). 남긴 것을 섞는다.
    예(AID, ρ = 100): 많은 쪽 61, 50, 40장 … 가운데 7장 … 적은 쪽 1, 1, 1장, 합 약 440장(배치 7개). ρ = 10이면 61 … 22 … 6장, 합 약 790장(배치 13개). 순위를 3번 바꿔 평균했다.
    지수 꼴 long-tail은 Cui 외(CVPR 2019, Class-Balanced Loss)의 long-tailed CIFAR 구성(클래스 i의 표본 수 n_i·μ^i, imbalance factor = 가장 큰 클래스 / 가장 작은 클래스)과 같은 식이다.
  - 클래스 순서로 정렬: test 전체를 클래스 0번부터 차례로 늘어놓는다. 배치 하나에 클래스가 평균 1.9개(AID) / 1.4개(RESISC45) 들어간다.
    TTA 논문에서 "non-i.i.d. stream" 또는 "label shift"라 부르는 설정의 극단이다(SAR의 imbalance ratio ∞, NOTE의 시간 상관 stream. 이 둘의 세부는 요약 agent가 읽은 것이고 원문 대조는 안 함).
- **"clean에서의 비용"** = 손상이 없는 clean stream에서 (방법을 켠 정확도 − 아무것도 안 한 정확도). 손상이 있는지 모르고 항상 켜 두었을 때 잃는 양이다.
- 정확도는 그 stream에 들어 있는 표본 기준이다(long-tail에서는 많은 클래스의 비중이 크다).
- 읽는 법: head 입력 표준화는 "배치의 평균이 source 평균과 같아야 한다"를 가정한다. 배치의 클래스 구성이 치우치면 손상이 없어도 평균이 어긋나므로 이득이 줄고 clean에서 손해가 난다.
  배치가 한두 클래스뿐이면 완전히 무너진다. **이것은 표준화와 flip의 한계로 발표에서 밝혀야 한다**(7절).
- LSCD-TTA의 주장과의 관계: "불균형이 TTA를 깬다"는 손상에서도 선다. 다만 깨지는 것은 엔트로피 방법보다 배치 통계를 쓰는 방법이다(tent와 dsbr은 T13 (6)).

**(나) 정보 밀도와 장면 복잡도: 이미지의 픽셀 복잡도를 재서 결과와 견준다**

| 224 px 입력의 복잡도 (평균) | ImageNet val 5,000장 | AID test | RESISC45 test |
|---|---:|---:|---:|
| JPEG(품질 75) 비트/픽셀 | 1.74 | 1.63 | 1.67 |
| edge 밀도 (밝기 gradient 크기의 평균) | 0.092 | 0.083 | 0.085 |
| 밝기 히스토그램 엔트로피 (bit) | 7.21 | 6.66 | 6.68 |

| JPEG 비트/픽셀과의 순위상관 (AID / RESISC45) | 이미지 단위 (2,000 / 6,300장) | 클래스 평균 단위 (30 / 45개) |
|---|---:|---:|
| clean 정확도 | +0.08 / −0.09 | −0.54 / −0.33 |
| 손상에서의 하락 (clean − 손상) | 0.00 / +0.02 | +0.23 / +0.22 |
| 손상에서의 과신 (confidence − 정답) | +0.01 / +0.08 | +0.43 / +0.39 |
| flip 비율 | +0.05 / +0.09 | +0.40 / +0.32 |
| head 입력 표준화 이득 | −0.11 / −0.01 | 0.00 / +0.05 |

- 이 세 지표로는 **RS 이미지가 ImageNet보다 복잡하지 않다**(모델이 보는 224 px 기준). "RS는 정보 밀도가 높다"는 픽셀 수준에서는 확인되지 않는다.
- 이미지 단위로는 복잡한 이미지가 더 떨어지거나 더 과신한다는 관계가 없다(|ρ| 0.11 이하). 사분위로 나눠도 단조롭지 않다.
- 클래스 단위로는 복잡한 클래스가 clean에서도 어렵고 손상에서 과신과 flip이 크다(ρ +0.3 ~ +0.4, 30개 / 45개에서 5% 수준으로 유의한 크기는 0.36 / 0.29).
  하락 자체와의 상관(+0.2)은 유의하지 않다. 복잡한 클래스는 서로 비슷한 클래스(주거지, 상업지 등)이기도 해서 복잡도가 원인인지는 가를 수 없다.
- 한계: 픽셀 복잡도는 "물체가 많다, 장면이 복잡하다"의 대용일 뿐이다. 원본 해상도(AID 600 px)가 아니라 224 px로 줄인 입력에서 쟀다.
- **정의의 출처 (2026-11-26 확인).**
  - LSCD-TTA는 "high information density"를 서론과 4.3절에서 세 번 쓰지만 **정의도 측정도 하지 않는다**(요약 agent가 arXiv v2 전체를 읽음). 붙인 인용은 코코넛 나무 조사 논문(Zheng 외, RSE 2023) 하나다.
    클래스 불균형도 2.4절의 논증뿐이고, 불균형 stream이나 label shift 평가는 없다. RESISC45는 클래스당 700장으로 균형이다.
  - 그래서 "정보 밀도"를 어떻게 잴지는 Claude가 정한 것이다. 고른 세 지표의 근거:
    JPEG 비트/픽셀과 gradient 크기 평균은 Yu & Winkler, "Image complexity and spatial information"(QoMEX 2013)에 선례가 있다(Claude가 PDF 본문 확인).
    그 논문은 압축률의 역수를 이미지 복잡도로 정의하고(Kolmogorov 복잡도의 근사), Sobel gradient 크기의 평균(SI_mean)이 그것과 0.91–0.97로 상관한다고 보고했다.
    우리 edge 밀도는 Sobel이 아니라 단순 중앙 차분으로 계산했다(같은 종류의 양이지만 같은 식은 아님). 밝기 히스토그램 엔트로피는 확인한 선례가 없다.
  - "복잡한 클래스"는 따로 정의한 것이 아니라 그 클래스 이미지들의 복잡도 평균이 높은 클래스라는 뜻이다.
  - RS 장면 복잡도를 사람의 판단으로 정의한 연구가 있다는 검색 결과(AID에서 만든 AID-22, Remote Sensing 2021)는 페이지가 열리지 않아 확인하지 못했다. 발표에 쓰려면 원문 확인이 필요하다.

### T15. 엔트로피 최소화를 오래 돌리면: Tent protocol (B6, 2026-11-26 05:30 중간 결과, 실행 계속 중)

사용자 지적으로 fig4(feature 단계)를 원래 Tent 설정으로 다시 돌렸다. 이미지를 백본에 통과시키고 백본 norm 층의 γ, β를 학습한다(T1과 같은 파라미터, SGD, lr).
AID-C 3단계 stream(2,000장)을 반복해 2,000 step, 배치 64. checkpoint에서 stream 전체를 갱신 없이 예측. `scripts/tta/cls_tta_long.py`, `experiments/2026-11-26_em_long_tent/RESULTS.md`.
**지금까지 끝난 것: 백본 7개 × 손상 3–4종(motion blur, data gaps, scale, Gaussian noise).** 나머지 백본 10개와 손상 8종은 실행 중이라 숫자가 바뀐다.

| 백본 (손상 3종 평균: motion blur, data gaps, scale) | 시작 | tent 끝 | dsbr 끝 | 예측 클래스 수: 시작 → tent / dsbr |
|---|---:|---:|---:|---:|
| satmae | 62.2 | 59.7 | 86.6 | 28.3 → 20.3 / 30.0 |
| scalemae | 59.9 | 33.0 | 59.1 | 28.3 → 12.3 / 20.3 |
| moco_in | 29.2 | 3.3 | 3.1 | 24.0 → 3.0 / 1.0 |
| moco_fmow | 43.1 | 29.6 | 80.6 | 25.3 → 11.7 / 30.0 |
| gassl_fmow | 53.6 | 43.2 | 86.7 | 29.0 → 19.7 / 30.0 |
| dino_in_s16 | 66.8 | 65.8 | 85.1 | 30.0 → 30.0 / 30.0 |
| satdino_s16 | 76.2 | 89.9 | 91.3 | 29.7 → 30.0 / 30.0 |
| **평균 (7개)** | 55.9 | 46.4 | 70.4 | 27.8 → 18.1 / 24.5 |

**붕괴(끝에서 예측 클래스 5개 이하)가 일어난 (백본, 손상) 쌍: 끝난 27쌍 중 tent 10쌍, dsbr 4쌍.**

| 손상 3단계 | tent가 붕괴한 백본 | 그중 dsbr이 막은 것 (끝 정확도) | dsbr도 붕괴한 것 |
|---|---|---|---|
| motion blur | satmae, scalemae, moco_fmow, gassl_fmow, moco_in | satmae 82.2, moco_fmow 81.3, gassl_fmow 87.0 | scalemae (3.5), moco_in (3.5) |
| data gaps | scalemae, moco_fmow, moco_in | scalemae 83.3, moco_fmow 79.2 | moco_in (2.8) |
| scale | moco_in | – | moco_in (3.1) |
| Gaussian noise | moco_in | moco_in 64.3 (붕괴는 아니지만 시작 85.6보다 낮음) | – |

- 가장 뚜렷한 예: SatMAE, motion blur. tent는 29.6 → 28.5 → 23.6 → 6.5 → … → 4.0% (0, 10, 20, 50, …, 2,000 step), 예측 클래스 25 → 3개(50 step) → 1개.
  dsbr은 29.6 → 63.4(50 step) → 82.2%, 예측 클래스 30개. stream 한 바퀴가 31 step이므로 tent의 붕괴는 두 바퀴 안에 일어난다.
- DINO 계열(dino_in_s16, satdino_s16)에서는 tent도 붕괴하지 않고 대체로 오른다. 예외: dino_in_s16의 data gaps는 58.1 → 38.3으로 내려간다(클래스 수는 30 유지, dsbr은 85.5).
- **말할 수 있는 것**: 원래 Tent 설정에서도 "엔트로피 최소화 → 예측 클래스 합쳐짐 → 붕괴"가 일어나고, 일어나는 곳은 confidence가 가장 틀리는 백본(MAE, MoCo 계열)과 손상(motion blur, data gaps)이다.
  DSBR은 그 대부분을 막는다. 이것이 EO 백본에서 DSBR을 쓰는 근거다.
- **말하면 안 되는 것**: "DSBR은 붕괴하지 않는다". ScaleMAE의 motion blur와 moco_in에서는 DSBR도 한 클래스로 붕괴한다. DSBR은 엔트로피에 가중만 줄 뿐 다양성을 강제하는 항이 없다.
- feature 단계 fig4와 다른 점: fig4는 17개 백본 모두에서 tent가 내려가고 DSBR은 모두 오르는 그림이었다. Tent protocol에서는 백본에 따라 갈린다(DINO 계열은 tent도 오르고, moco_in은 DSBR도 무너진다).
- 한계: 같은 2,000장을 반복해 보여 주고 그 2,000장에서 잰 정확도다. seed 1개. 3단계만. 끝난 백본이 7개다.

## 5. 볼 것 (순서와 시간)

| 순서 | 무엇 | 어디 | 시간 |
|---|---|---|---|
| 1 | 이 문서 전체 | – | 20분 |
| 2 | 표준화와 정렬 (전처리 표준화·입력 정렬·head 입력 표준화의 구분, 식과 두 단계, 필요한 source 통계) | `docs/STUDY_alignment.md` §1, §2.4 | 20분 |
| 3 | 백본 표, augmentation 표, 손상 배치 표 | `docs/SETTINGS.md` §1, §2.5 | 15분 |
| 4 | feature 그림 두 장 | `docs/figs/2026-11-24_feature_dist/fig1_clean_geometry.png`, `fig2_shift_align_motion_blur_s3.png` | 10분 |
| 5 | Geometry 세 가지 (T8, Silhouette 상관, T9)와 k-means가 낮은 이유 | 이 문서 4절, `docs/figs/2026-11-24_feature_dist/README.md`의 fig3 절 | 20분 |
| 5 | 백본 사전학습 방식 (MoCo, GASSL, DINO, SatDINO는 원문 확인 필요) | `docs/STUDY_backbones.md` | 30분 + 논문 |
| 5 | 새로 찾은 RS TTA 논문 GRAM | `docs/reading/gram_slum_tta.md` | 10분 |
| 5 | Geometry의 설명 (k-means와 선형 head가 왜 다른가, EM 두 단계) | `docs/reading/entropy_enigma.md` §5 | 15분 |
| 5 | confidence 분석의 설명 (ECE, 과신, ATC) | `docs/STUDY_stage1_metrics.md` §3, §5.3, 수치는 `experiments/2026-11-16_backbone_feats/ANALYSIS.md` §1 | 15분 |
| 5 | feature 이동 지표 (rel_drift, frac_common, CKA). 이 문서에는 표가 없고 결론 한 문장만 있다 | `docs/STUDY_stage1_metrics.md` §4, §7, 수치는 `ANALYSIS.md` §3.5 | 필요할 때 |
| 6 | 10월 결과가 필요하면 | `docs/WRAPUP_progress_2026-11-23.md` §1 | 10분 |
| 7 | F2 선행 연구 조사 | `docs/PRIOR_ART_rs_tta_corruption.md` §1, §3 | 10분 |
| 7 | 백본 선택 근거의 선행 연구 (F5) | `docs/reading/matched_pair_study.md`의 "원문 직접 확인" 절 | 10분 |
| 8 | 질문이 나올 만한 논문 | `docs/reading/dsbr.md`, `entropy_enigma.md` §5, `sensebench.md` | 필요할 때 |

Geometry에서 이해해야 할 것은 세 문장이다.
- 선형 head는 95%를 맞히는데 k-means는 59%다. 클래스마다 덩어리는 하나지만 둥글지 않다. 클래스를 가르지 않는 방향으로 넓게 퍼져 있어서, 모든 방향을 똑같이 재는 k-means는 클래스를 못 찾는다(T8, fig3).
- 그래서 "잘 뭉쳐 있으면 정확도가 높다"(Entropy Enigma의 전제)는 우리 백본에서 일부에만 맞는다.
- 엔트로피 최소화가 실패할 때 일어나는 일은 예측 클래스가 합쳐지는 것이다(30개 → 23.5개).

10월 그림은 `docs/figs/meeting_2026-10/`에 있다(왜 TTA가 실패하나, 깊이별 정렬, 손상 예시 등).

## 6. 방법 후보 (D2)

| 순위 | 후보 | 한 줄 설명 | 근거 | 약점 |
|---|---|---|---|---|
| 1 | 촬영 조건 되돌리기 | stream이 공유하는 촬영 조건(밝기, 배율, 무효 영역)을 source 통계에 맞춰 추정하고 입력에서 되돌림. 학습 없음 | 밝기 쪽(입력 정렬)은 이미 거의 완전히 회복. scale과 data gaps는 여지가 크고 원인이 분명 | 조각 각각은 기존 기법과 닮음 |
| 2 | 라벨 없는 head 적응 | 클래스 비율을 강제하고 flip으로 고른 표본으로 head만 갱신 | 여지가 가장 크게 측정된 곳(자연 shift, motion blur) | 선행 연구가 많음(T3A, LAME, SHOT) |
| 3 | head가 읽는 부분공간 안의 정렬 (GOLD) | 29–44차원 안에서 공분산까지 정렬 | 이동의 83–94%는 head가 읽지 않는 방향 | 부품 수준 |
| 4 | 붕괴 감지로 적응 제어 | 예측 클래스 수와 flip으로 멈추거나 되돌림 | T9 | 이득이 작음 |

1번의 외부 근거: matched-pair 연구가 "위성 사전학습의 효과는 겉보기 크기가 GSD에 달린 작은 구조물에 몰린다"고 보고하고 배율을 통제한 실험을 future work로 남겼다.

첫 실험: 배율 탐색의 상한(scale 손상, RESISC45 ↔ UCM), data gaps에서 token 제외, 2번의 가능성 확인.
2번의 가능성 확인은 시작했다가 중단했다(결과 없음).

1번의 두 실험은 2026-11-25에 예비로 실행했다. 슬라이드에는 넣지 않고(사용자 결정: 면담에는 T5까지만), 질문이 나오면 답할 재료로만 둔다.
둘 다 **상한**이다(data gaps는 무효 영역 mask를 안다고 가정, scale은 후보 배율 안에 정답이 들어 있음).

| 17개 백본 평균 정확도 (%) | 그대로 | head 입력 표준화 | 입력에서 되돌림 | clean |
|---|---:|---:|---:|---:|
| AID-C data gaps (양쪽 픽셀로 보간) | 66.0 | 80.6 | 91.1 | 94.8 |
| AID-C scale (라벨 없는 통계로 배율 선택) | 73.1 | 82.7 | 93.4 | 94.8 |
| RESISC45-C scale (같은 방법) | 69.3 | 77.1 | 87.0 | 90.6 |

자연 shift에서는 배율 탐색이 듣지 않았다(R45 → UCM 79.0 → 76.9, UCM → R45 65.6 → 66.0). 출처: `experiments/2026-11-25_undo_acquisition/RESULTS_gaps.md`, `RESULTS_scale.md`.

## 7. 한계와 주의 (D2, 질문 대비)

**한계**
- seed는 대부분 1개다. 자연 shift는 데이터 쌍 하나, 분할은 모델 2개다.
- head 입력 표준화와 flip 결과는 배치 64, 한 종류의 손상만 있는 stream, **클래스가 섞여 들어오는 stream** 기준이다. 배치 16에서는 표준화 이득이 줄고 clean 비용이 생긴다.
  클래스 비율이 100배 치우치면 이득이 +5.5 → +2.1pt(AID-C)로 줄고, 클래스 순서로 들어오면 −54pt로 무너진다(T14). 한 지역을 훑는 RS stream은 클래스가 몰려 들어올 수 있으므로 실제 한계다.
- 방법은 아직 없다. 지금까지는 분석과 기존 방법의 적용이다.

**말하면 안 되는 것**
- "flip이 가장 좋은 정확도 추정기다": COT와 평균이 같고 큰 오차는 COT가 적다.
- "head 입력 표준화가 우리 방법이다": head 입력에 AdaBN을 적용한 것이고 FOA와도 겹친다. baseline이다.
- "DSBR Table 3을 재현했다": 이득의 크기가 비슷한 것까지만 확인했다.
- "SenseBench가 REOBench 손상의 고유성을 비판했다": 그 문장은 열화를 평가 대상으로 삼지 않았다는 뜻이다.
- "위성 사전학습이 낫다 / 못하다": 방식마다 반대이고 원인은 모른다.
- "손상된 RS 분류에 TTA를 적용한 선행 연구는 없다": 조사 범위 안에서 찾지 못했다는 것까지만 말한다.
- "분할에서도 flip이 confidence보다 낫다": 분할에서는 둘 다 정확도를 따른다(−0.94 / −0.96 대 +0.90 / +0.96).
- "한 클래스가 여러 덩어리로 쪼개져 있다": 측정으로는 덩어리 하나이고 둥글지 않을 뿐이다(T8).
- "RS feature는 모양이 다르다": 모양은 사전학습 목적함수를 따른다. 위성 데이터의 효과는 그 안의 작은 차이다.
- "tent는 모든 백본에서 무너지고 DSBR은 항상 안전하다": Tent protocol(T15)에서 tent의 붕괴는 MAE·MoCo 계열에 몰려 있고 DINO 계열은 오른다. DSBR도 ScaleMAE motion blur와 moco_in에서는 붕괴한다. fig4와 T9(head 앞 affine만 학습)는 본문에 쓰지 않는다.
- "flip 필터는 우리가 처음이다": confidence 대신 두 예측의 일치로 표본을 고르는 선례가 있다(GRAM). 우리 것은 일치를 만드는 방법(head 입력 표준화 전후)이 다를 뿐이다.

**예상 질문과 답**

| 질문 | 답 |
|---|---|
| 표준화·정렬은 source 통계를 쓰는데 반칙 아닌가 | 표준화해서 학습한 head에서는 test 식에 source 통계가 들어가지 않는다. 그대로 학습한 head에서 쓰는 통계도 BatchNorm의 running 통계와 같은 종류다. 다만 새롭지는 않다 |
| RS만의 문제가 무엇인가 | 이미지 도메인이 아니라 손상의 종류(data gaps, scale, clouds)와, shift가 촬영 단위로 stream 전체에 공유된다는 점이다 |
| 왜 DINOv3 web이 가장 강한가 | 같은 방법의 위성 모델은 14pt 낮으므로 방법이 아니라 데이터나 규모다. 어느 쪽인지는 가르지 못했다. matched-pair 연구와 DINOv3 저자들의 GEO-Bench 평가도 web이 위성을 이긴다고 보고했다 |
| DINOv3-SAT이 낮은 것은 feature를 잘못 뽑아서가 아닌가 | 우리는 CLS와 patch 평균을 이어 붙여 쓴다. matched-pair 연구가 세 가지 feature를 모두 시험했고 가장 좋은 것으로도 web에 뒤졌다 |
| 분할에서도 되나 | 현상(엔트로피 방법은 손해, 쏠림을 누르면 이득, 통계 정렬(BN 통계 교체)이 가장 큼, flip이 정확도를 따름)은 12종 전체에서 같다. 다른 점은 기하 손상이 분할에서는 거의 문제가 안 되고, confidence도 정확도를 따른다는 것이다(T5). feature geometry 설명은 분류에서만 확인했다 |
| k-means가 낮으면 feature가 나쁜 것 아닌가 | 아니다. 선형 head 95%, 5-NN 89%다. 클래스마다 덩어리는 하나인데 클래스와 무관한 방향으로 넓게 퍼져 있어 유클리드 거리만 쓰는 k-means가 못 찾는 것이다. 공통으로 퍼진 모양을 펴면 중심 하나로 92%다(T8) |
| 왜 DINO 쌍이 셋인가 | 공개된 SatDINO 가중치가 S/16, S/8, B/16 셋이고 DINO에 같은 구조가 다 있다. 세 쌍의 방향이 같다는 것이 크기와 무관하다는 근거다. ViT-L은 양쪽 다 없다 (`docs/STUDY_backbones.md` §4) |
| 다음은 무엇인가 | 6절의 1번과 2번. 여지가 어디 있는지는 T12로 쟀다 |

## 8. 로컬로 가져갈 것
- 이 파일과 그림은 모두 저장소에 commit돼 있다. 로컬에서는 `main`을 pull하면 된다(서버의 `repro`와 같은 내용).
- 그림: `docs/figs/midterm_2026-11-26/` 12장(발표용), `docs/figs/2026-11-24_feature_dist/` 3장(fig0 정사영 설명, fig1, fig2)과 `README.md`, `docs/figs/meeting_2026-10/` 7장.
- PPT skill에 줄 입력: 3절(슬라이드 구성) + 4절의 표 중 해당 번호 + 7절의 한계. 슬라이드마다 "한 줄 메시지"를 제목 아래 문장으로 쓰면 된다.
