# "Satellite Pretraining Is Not One Thing" (arXiv 2611.13849) — 우리 백본 쌍 설계와 겹치는 선행 연구

Agrawal, Sathish, Nambiar (SatSure Analytics). 2026-11-10 제출. GeoCV workshop @ WACV 2027 채택. 코드 링크 없음.
출처: 요약 agent가 arXiv 초록과 HTML 본문(1–6절)을 읽고 보고한 내용(2026-11-17). 부록은 못 읽음. Claude가 원문을 직접 읽지는 않음.

## 무엇을 했나
- 질문: 위성 사전학습 encoder가 강한 일반 encoder보다 나은가(라벨이 적은 장면 분류에서).
- frozen encoder 14개, 그중 대응 쌍 5개:
  DINO ViT-B/16 ↔ SatDINO, DINOv3 ViT-L/16 web ↔ SAT, MAE ViT-L/16 ↔ SatMAE, MAE ViT-L/16 ↔ Scale-MAE, CLIP ViT-L/14 ↔ RemoteCLIP.
- 데이터: PatternNet, RSICD, EuroSAT, UC Merced, RESISC45 (AID, Potsdam 없음). train 10% / test 90%, seed 6개.
- 평가: k-NN, 선형 probe, attentive probe, retrieval. fine-tuning 없음.

## 결론 (그쪽 수치, 위성 − 일반, k-NN / 선형 probe pt)
- SatDINO: RESISC45 +5.2 / +1.7, UC Merced +2.1 / +0.8 → 도움.
- DINOv3-SAT: RESISC45 −27.4 / −12.4, UC Merced −41.8 / −22.5 → 크게 해로움. CLS만 쓰면 UCM k-NN 84.9 vs 43.2.
- SatMAE: RESISC45 +0.0 / −6.5, UC Merced −5.9 / −8.8 → 해로움.
- Scale-MAE: RESISC45 +14.0 / +1.9 → 도움. 같은 fMoW-RGB MAE인 SatMAE와 반대.
- RemoteCLIP: RSICD·UC Merced는 사전학습에서 봤으므로 판정에서 제외. 나머지에서 +0.4~+1.7 선형 probe.
- 조언: "위성으로 사전학습했다"는 선택 기준이 못 된다.

## 다루지 않은 것 (agent가 절 단위로 확인)
손상 강건성, 데이터셋 간 shift, calibration / ECE, CKA, feature drift, test-time adaptation, 통계 정렬, 라벨 없는 정확도 추정: 전부 없음.

## 우리에게 주는 것
- **clean 정확도에서의 대응 쌍 비교는 더 이상 새롭지 않다.** 인용하고, 우리 기여는 shift·손상·confidence 쪽으로 한정할 것.
- **우리 clean 결과와 방향이 일치한다.** 우리(선형 head, RS − ImageNet): SatMAE AID −1.6 / RESISC45 −5.2, DINOv3 sat −2.3 / −4.3,
  SatDINO +0.2~+1.8.
- **DINOv3-SAT의 낮은 CLS k-NN은 우리 로딩 문제가 아니다.** 우리 AID k-NN 49.7(CLS)과 같은 현상을 그쪽도 보고.
- 그쪽 한계: 쌍마다 corpus·학습 길이·recipe가 달라 어떤 요인 때문인지는 못 가림. SatDINO·Scale-MAE를 GSD 입력 없이 돌림(우리도 같음).
- 그쪽에 없는 쌍: MoCo(ResNet50) ImageNet ↔ fMoW. 그쪽에 있고 우리에 없는 쌍: CLIP ↔ RemoteCLIP.
