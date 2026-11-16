# LSCD-TTA 발표 피드백 (2026-11 첫째 주, 지지난주 면담; 정확한 날짜 미기록)

## 원문 (사용자 전달 그대로)

> 일단 LSCD-TTA 발표후 아래처럼 피드백 왔다. 정리하면 REOBench같은 관련 논문(Beyond Accuracy_ Assessing Calibration of
> Geospatial Foundation Models and Their Sensitivity to Distribution Shifts, arXiv 2608) 찾고읽어보고, ResNet, ViT fmow로
> pretrain된거 찾기?(백본, head는 어떻게 pretrain 되었는지), 그리고 TTA 적용할 Backbone 선택 어떻게 할지 고민 등.인듯
>
> 논문 한계: 총 클래스 분포를 보게 아니라. 샘플별 prediction으로 분포를 조정한다는 앞에 2개의 항이 약간 한계로 볼만하다.
> Loss 덕지덕지 붙이는건 요즘 학회스타일은 아님.
> RS -> corrupted RS sample -> noise 에 대한 고찰 ex) Latent feature 분포를 한번 봐보기 . 즉 RS에 corruption이 가해졌을 때
> 영향이 고유하다. (LSCD-TTA의 RS 문제점 빌드업 처럼 쌓아나가기) Ex. Natural image Corruption과 Reobench의 Corruption의
> 영향이 약간 다를 수도 잇다.
> 즉 여기서 제시한 RS 영상 성질로 인한 TTA 문제가 Corruption에서도 이어지는지 선행 연구 및 분석해보기, Beyond Accuracy 같은
> REObench 계열의 논문 읽어보기 왜냐면 2개 이상으로 하는게 안전함. (ex. 특정데이터셋의 Reward Hacking 우려)
> ResNet, ViT fmow로 pretrain된거 찾기. Self supervised learning(백본), head는 어떻게?
> Backbone 선택 고민. (기존 저자들이 왜 이 모델을 선택했는지. Ex. GeoFM의 ability robustness 보완 해결하려는 문제에 대한 근본 고민),

## 항목별 정리

### F1. LSCD-TTA의 한계 (논문 평가)
- 앞의 두 항(WCSE, BCSE)은 **전체 클래스 분포가 아니라 샘플별 예측으로** 분포를 조정함 → 한계로 볼 만함.
- 손실 항을 여러 개 붙이는 방식은 요즘 학회 스타일이 아님.
- 관련 기록: 공식 코드에는 배치 평균 예측을 쓰는 diversity 항(논문에 없음)이 있고, 빼면 corruption에서 붕괴함
  (NOTES_tta §1.2 LSCD 행, 2026-11-03 실험). → "전체 분포를 보는 항이 실제로 안정성의 핵심"이라는 교수님 지적과 맞는 증거.

### F2. RS corruption의 고유성을 문제 제기로 쌓기
- RS 영상 → corrupted RS → noise에 대한 고찰. 예: **latent feature 분포를 직접 보기**.
- LSCD-TTA가 "RS 영상 성질(클래스 불균형, 클래스 간 유사성 등) 때문에 기존 TTA가 안 된다"고 빌드업한 것처럼,
  **그 RS 성질로 인한 TTA 문제가 corruption 상황에서도 이어지는지** 선행 연구 조사 + 분석.
- 자연 영상 corruption(ImageNet-C)과 REOBench corruption의 영향이 다를 수 있음.
- 관련 기록: feature shift 진단(NOTES_tta §5.7, degradation이 중간 블록 채널 평균 이동으로 나타남), pixel affine R²(§4.2),
  예측 붕괴 클래스(§4.3: motion blur→Beach 등).

### F3. REOBench 계열 논문을 더 읽고, 벤치마크를 2개 이상으로
- Beyond Accuracy (Assessing Calibration of Geospatial Foundation Models and Their Sensitivity to Distribution Shifts,
  arXiv 2608) 같은 REOBench 계열 논문 찾고 읽기.
- 이유: 데이터셋/벤치마크 하나에만 맞추면 **reward hacking** 우려 → 2개 이상이 안전.
- 관련 기록: DATASETS_2026-11.md (AID-C + RESISC45-C 생성 완료, RESISC45↔UCM split, Potsdam→Vaihingen).

### F4. fMoW로 사전학습된 ResNet / ViT 찾기
- 백본은 self-supervised learning으로 사전학습된 것. **head는 어떻게 학습되었는지**도 확인.
- 관련 기록: 현재 SatMAE(fMoW-RGB, ViT-L), ScaleMAE(fMoW, ViT-L). head는 우리가 AID train으로 linear probing
  (P / PN, `experiments/2026-10-12_cls_ablation`).

### F5. TTA 적용 백본 선택 근거
- 기존 저자들이 왜 그 모델을 골랐는지 확인.
- GeoFM의 능력·강건성을 보완한다는, 해결하려는 문제에 대한 근본적인 고민 필요.

## 할 일
- [x] F3: Beyond Accuracy 원문 확보·정독 (2026-11-16, reading/beyond_accuracy.md)
- [x] F3: REOBench 계열 논문 목록 (EarthShift 등) 정리 (2026-11-16, reading/earthshift.md)
- [ ] F2: RS corruption vs 자연 영상 corruption의 latent feature 분포 비교 분석 설계
- [ ] F2: "RS 성질로 인한 TTA 문제"가 corruption에서도 이어지는지 선행 연구 조사
- [ ] F4: fMoW 사전학습 ResNet / ViT 체크포인트 목록과 각 head 학습 방식
- [ ] F5: 백본 선택 근거 정리 (선행 논문들의 선택 이유)
- [ ] F1: LSCD-TTA 한계를 우리 방법 설계 원칙으로 반영 (분포 수준 신호, 손실 항 최소화)

## 진행 기록
(사용자의 논문 독해 메모와 결정 사항을 날짜별로 추가)

- 2026-11-16 F3: Beyond Accuracy 독해 메모 → [reading/beyond_accuracy.md](../reading/beyond_accuracy.md).
  가져갈 것: (a) 서로 다른 물리 메커니즘의 corruption으로 효과가 하나에 특이하지 않음을 보이는 선정 논리, (b) "정규화 전 손상 →
  정규화 불일치도 shift의 일부" 문장, (c) CKA·코사인·participation ratio로 표현 drift를 보는 분석 도구(F2의 latent 분석에 사용),
  (d) EO 인코더의 "rigid + 과신" = 엔트로피 TTA가 신호를 잃는 이유와 연결. 다음: EarthShift 메모 → 최종 생각 → 방향 결정.
- 2026-11-16 F2/F3: Beyond Accuracy Results 이후 분석 추가 (reading/beyond_accuracy.md "Claude 분석"). 재료 M1~M8.
  실험 후보(결정은 EarthShift 메모와 사용자 최종 생각 이후): (a) drift를 공통 이동과 샘플별 잔차로 분해 + CKA·cosine·PR,
  (b) TTA 방법별 ECE·NLL, (c) MAE ViT-L(ImageNet) vs SatMAE(fMoW) 통제 비교.
- 2026-11-16 F3/F5: EarthShift 독해 메모 → [reading/earthshift.md](../reading/earthshift.md). 가져갈 것: TTA 공백 문장, fine-tuning이
  강건성을 못 올림, scale shift(RESISC45↔UCM)에서 ID의 43% 손실, 합성+실제 shift 둘 다 필요. 주의: ResNet 우위는 프로토콜(ViT ID 77.5%)과
  Sentinel 계열 GFM의 입력 영역 문제가 섞여 있고, 우리 백본은 이 논문에 없음. 다음: 사용자 브레인스토밍 → 방향 결정.
- 2026-11-16: 사용자 브레인스토밍 수령, 방향 제안 작성 → [2026-11-16_direction.md](2026-11-16_direction.md) (사용자 확정 전).
