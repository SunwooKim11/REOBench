# 면담 PPT 생성 프롬프트 (2026-10-29/30 지도교수 개인면담, 약 20분)

아래 "프롬프트" 블록을 Claude에 그대로 붙여 넣고, "첨부 파일" 목록의 파일과 **PPT 템플릿 파일**을 같이 올린다.

## 첨부 파일 (repro/ 기준)

필수
1. `docs/NOTES_tta.md`: 실험 전체 기록. 숫자의 원본 (§3 분류 결과, §4.2 왜곡 그룹·affine R², §4.3 TTA 실패 근거, §5.5 sweep, §5.6 조합, §5.7 feature shift, §6 할 일)
2. `docs/PAPERS_tta_2026-10.md`: Panda / FOA 정리와 우리 실험과의 연결 (§1, §2, §5)
3. `docs/STUDY_alignment.md`: 각 선택의 이유·구현·선행연구 (발표 논리와 예상 질문)
4. `docs/NOTES_concepts.md`: 정규화 세 종류 설명 (§6)
5. `docs/figs/meeting_2026-10/fig1_why_tta_fails.png` ~ `fig7_norm_axes.png` (7장)
6. `experiments/2026-10-13_cls_tta/summary_src_ablation.txt`: 소스 통계 의존도 실험 (반칙 질문 대응)

선택 (논문 그림을 슬라이드에 넣고 싶을 때)
7. `~/REOBench/papers/TTA/Panda_ Test-Time Adaptation with Negative Data Augmentation.pdf`
8. `~/REOBench/papers/TTA/[FOA] Test-Time Model Adaptation with Only Forward Passes.pdf`
9. `experiments/2026-10-13_cls_tta/summary_tta.txt`, `experiments/2026-10-20_seg_tta/summary_depth_sweep.txt`,
   `experiments/2026-10-28_entropy_bias/summary.txt` (표 원본 숫자)

## 프롬프트

```
지도교수 개인면담용 발표 자료(약 20분)를 만들어 줘. 청중은 지도교수 1명이고, TTA(test-time adaptation)와
원격탐사(RS) 기초모델은 알지만 내 실험의 세부는 모른다. 발표 언어는 한국어, 그림 안의 영어는 그대로 둔다.

[내 연구 맥락]
- REOBench(RS 기초모델 강건성 벤치마크)를 재현하고, 그 위에서 TTA가 corruption 강건성을 얼마나 회복하는지 연구 중.
- 범위: MIM 기초모델 SatMAE, ScaleMAE(ViT-L) / AID 장면 분류(30클래스), Potsdam 분할(UperNet) /
  mild 프로토콜(각 corruption × severity를 독립 스트림으로 평가, 매번 모델 리셋). CLIP 계열(RemoteCLIP, GeoRSCLIP)은 아직 미실행.
- 첨부한 NOTES_tta.md, STUDY_alignment.md, PAPERS_tta_2026-10.md, NOTES_concepts.md가 모든 내용과 숫자의 출처다.
  숫자는 반드시 첨부 파일에서만 가져오고, 없는 숫자는 만들지 말고 "TBD"로 둬. 표의 행을 임의로 빼지 마.

[구성: 2부]
Part 1. 실험 정리 (약 12분)
 1. 표지
 2. 지난 목표와 이번 진행 요약 (한 장: 목표 → 한 일 → 핵심 발견 3줄)
 3. 설정: 모델·데이터·corruption 12종(radiometric / degradation / occlusion / geometry 4그룹), fig4 사용
 4. 기존 TTA 결과: Tent, EATA, SAR, DeYO, CoTTA 등이 none 대비 거의 이득이 없거나 손해 (NOTES_tta §3.1 표)
 5. 왜 기존 TTA가 안 듣는가: fig1 + NOTES_tta §4.3 근거 (확신하지만 틀림, entropy AUROC 저하,
    corruption과 닮은 클래스로 예측 쏠림, 이득은 엔트로피가 아니라 통계에서 나옴, mild 스트림이 짧음)
 6. 관찰 1: corruption은 두 종류다. radiometric은 채널별 affine(R² 0.97~0.99), degradation은 비-affine
    (R² 0.56~0.85, 픽셀 평균·표준편차 변화 없음). NOTES_tta §4.2 표 + fig6
 7. 왜 정규화가 이걸 못 지우나: fig7 (LayerNorm은 토큰마다 채널 평균을 빼므로 모든 토큰에 공통인 채널별 이동이 남음)
 8. 실험: 정렬 위치 sweep (입력 ISA / 블록 / head 입력 / final norm), fig2 + §5.5 표
 9. 해석: degradation 이동은 중간 블록에서 채널 평균 이동으로 나타남, fig5 (§5.7)
10. 조합: ISA + 블록 7이 두 그룹 효과를 더함, fig3 + §5.6 표
11. "소스 통계를 쓰는 건 반칙 아닌가?": 쓰는 정보는 이미지가 아니라 통계(ISA 6개 숫자), BN running stat·EATA·FOA와
    같은 수준. 32장 통계 / 모델 Normalize 상수만 쓴 source-free 결과 (summary_src_ablation.txt)
12. 어려움과 한계: CLIP 미실행, 분할 corruption 4종 없음(data gaps, rotate, scale, translate), occlusion·geometry
    미해결(ISA가 occlusion에서 소폭 손해), 정렬 위치를 test에서 고름(검증 세트 선택 필요), seed 0만
13. 향후 방향과 Target Research Question:
    "RS 기초모델에서 corruption 종류마다 분포 이동이 네트워크의 어느 깊이에 나타나는가, 그리고 그 위치에 맞춘
     학습 없는 통계 정렬이 엔트로피 기반 TTA를 대체/보완할 수 있는가?"
    다음 단계: CLIP 2종 같은 프로토콜로 baseline+sweep (MIM vs CLIP 비교), 위치 자동 선택(검증 세트 또는 왜곡 판별),
    occlusion용 MIM 마스킹·구름 제외 통계, geometry용 D4 회전 TTA, 분할 corruption 4종 생성, seed 추가

Part 2. 논문 2편 (약 8분, 각 3장 내외)
14. Panda (AAAI 2026): 문제(예측 편향 → entropy TTA 붕괴), 방법(패치 셔플 negative의 평균 특징을 β배 빼기),
    결과(noise에서 크고 brightness/contrast에서 작음), 한계(β를 test에서 grid search, "배치 평균 빼기" baseline 없음)
15. FOA (ICML 2024): 문제(양자화·엣지 모델은 역전파 불가 → forward만으로 적응), 방법 2개를 분리해서 설명:
    (a) 프롬프트 탐색: 입력 앞 프롬프트를 CMA-ES로 찾음. fitness = 엔트로피 항 + λ × 층별 CLS 평균·표준편차를 소스에 맞추는 항
    (b) activation shifting: 마지막 층 CLS를 e ← e + γ(μ_src − μ_test)로 소스 평균 쪽으로 이동 (학습 없음)
    소스 통계는 라벨 없는 32장. 결과: ImageNet-C 66.3 (NoAdapt 55.5).
    ablation(PAPERS §2 표): 엔트로피만 44.9(손해) / 통계 항만 63.4 / 통계 항을 SGD로 직접 최적화하면 70.5
    → 이득은 프롬프트나 엔트로피가 아니라 "통계를 소스에 맞추기"에서 나온다.
16. 왜 FOA를 골랐나 (메커니즘이 닮아서가 아니라 목표와 근거가 같아서):
    - 프롬프트 + CMA-ES는 "역전파 없이 통계를 맞추는 수단"일 뿐, 우리와 닮은 부분이 아님을 먼저 인정.
    - 같은 것: 목표(특징 통계를 소스에 맞추기), 근거(엔트로피는 해롭고 통계가 이득), 소스 정보 가정(라벨 없는 소량 통계),
      (b) activation shifting = 우리 "head 입력 정렬"의 평균만 버전.
    - 다른 것: FOA는 통계를 프롬프트로 "간접적으로" 맞추고 12층을 한꺼번에 봄 / 우리는 한 위치에서 "직접" 덮어쓰고,
      corruption 종류마다 맞출 깊이가 다르다는 것(입력 vs 중간 블록)을 보임 = FOA가 하지 않은 질문.
    - FOA의 corruption별 이득도 정황상 맞음: noise·JPEG(degradation) 이득이 작음(+4.7, +5.4) = 입력·마지막 층만으로는 부족.
    - Panda와의 연결: Panda = 최종 특징에서 평균 하나 빼기. 두 논문 모두 "마지막 층 평균 보정"이고, 우리는 깊이를 넓힌 것.
17. 정리 + 논의하고 싶은 것 (CLIP 방향 우선순위, 분할 corruption을 이번에 만들지 다음으로 미룰지)
Backup: 방법별 구현 출처(NOTES_tta §1.2), 정규화 세 종류(NOTES_concepts §6), 전체 sweep 표

[디자인: 전달력이 최우선]
- 템플릿: 내가 첨부하는 PPT 템플릿을 따르고, 템플릿이 없는 부분은 흰 배경 + 검정 글씨의 백지 스타일로.
  "AI가 만든 티"가 나는 장식은 쓰지 마: 그라데이션 배경, 카드/박스 남발, 아이콘·이모지, 장식용 도형, 모든 줄에 붙는 굵은 글씨.
- 글자는 장당 3~5줄. 한 줄은 짧게(한 문장 이하). 세부 설명과 숫자 근거는 발표자 노트로 보낸다.
- 장당 메시지 하나. 제목은 결론 문장으로 (예: "엔트로피는 corruption 아래에서 정답 신호를 잃는다").
- 말로 하기 어려운 건 그림으로: 논문 방법은 간단한 도형(사각형·화살표·블록)으로 직접 다이어그램을 그려 흐름을 보여 줘.
  예) Panda: 원본 이미지 → 패치 셔플 negative → 두 특징 → "원본 특징 − β·negative 평균" 흐름도
      FOA: 프롬프트 후보 여러 개 → forward → fitness 계산 → CMA-ES 갱신 루프, 그리고 CLS를 소스 평균 쪽으로 옮기는 화살표
      우리: 네트워크 깊이 축 위에 "radiometric = 입력에서, degradation = 중간 블록에서" 위치를 표시한 그림
- 수식·loss를 보여 줄 때는 수식만 두지 말고 각 항 아래에 "이 항이 의도하는 것"을 한 줄씩 달아 줘.
  예) FOA fitness = 엔트로피 항(예측을 확신 있게) + λ × 통계 항(각 층 CLS 평균·표준편차를 소스와 같게 = 분포 이동 되돌리기)
  같은 방식으로 Panda의 β·negative 빼기, ISA 식 x' = (x − μ_batch)/σ_batch · σ_src + μ_src 도 항별로 설명.
- 발표 흐름 참고: 내가 지금까지 들은 최고의 발표는 서울대 연구실의 Swin Transformer 논문 리뷰 영상
  (https://www.youtube.com/watch?v=2lZvuU_IIMA). 그 발표처럼 "기존 방법의 문제 → 핵심 아이디어 한 줄 → 그림으로 동작 설명
  → 결과 → 의미" 순서로, 청중이 각 장에서 '왜 이걸 보는지'를 먼저 알게 해 줘.
- 그림은 첨부 PNG를 그대로 쓰고, 표는 핵심 열만 크게(나머지는 backup).
- 각 슬라이드에 발표자 노트(한국어, 30~60초 분량 대본)를 달아 줘. 예상 질문과 답은 STUDY_alignment.md를 참고해 노트에 추가.
- 강조색이 필요하면 그림과 같은 파랑 #2a78d6, 주황 #eb6834, 초록 #1baf7a만 쓴다 (템플릿 색이 우선).
```
