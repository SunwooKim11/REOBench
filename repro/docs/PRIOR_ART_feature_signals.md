# Prior art: feature-statistics signals for mitigation and accuracy estimation (survey by agent, 2026-11-17)

질문: stage-1에서 잰 feature 지표(공통 이동, source 통계와의 거리, CKA, 유효 차원)와 flip을 **방법으로** 쓰는 논문이 있는가.
조사: 웹 검색 agent 1회. 각 논문은 agent가 검색 결과나 초록 페이지로 존재를 확인했다고 보고한 것이고, Claude가 원문을 직접 읽지는 않았다.
인용 전에 원문 확인 필요(특히 표 수치). 이전 조사 `PRIOR_ART_stat_alignment.md`에 있는 논문은 생략.

## 결론

| 우리 결과 | 가장 가까운 선행 연구 | 차이 |
|---|---|---|
| head 입력을 source 통계로 정렬하면 정확도 회복 | **NEO** (ICLR 2026): ViT의 penultimate embedding에서 test 평균을 빼고 head에 넣음 | 평균만(표준편차·source 통계 없음). fine-tune한 ViT. frozen SSL backbone + linear probe, RS 없음 |
| 정렬하면 confidence 기반 정확도 추정(ATC)이 다시 맞음 | **Kim et al.** (NeurIPS 2024): BN 적응 후 ATC·AC 표가 있으나 결과가 엇갈리고 주장하지 않음 | backbone BN 적응. agreement 기반 추정이 주제. frozen FM 없음 |
| 정렬 전후 예측 flip이 정확도를 따라감 | **Weighted Flips** (Entropy Enigma, ICML 2024): 사전학습 모델과 엔트로피 최소화 적응 모델 사이의 flip 수 | 적응에 gradient 학습 필요. flip→정확도 변환식을 따로 학습. BN CNN 중심 |
| CKA가 공통 이동을 못 봄 | 평균을 빼는 정의에서 나오는 알려진 불변성 | 분포 shift에서의 실패 사례로 보고한 논문은 못 찾음 |
| RS / EO foundation model | 해당 없음 | A–C 중 어느 것도 frozen EO FM에서 수행된 적 없음 |

**NEO의 평균 재중심화는 우리 align m과 사실상 같은 연산이다**(표준화 공간에서 source 평균은 0이므로 "배치 평균 빼기"가 곧 source 평균으로 옮기기).
정확도 회복 방법 자체의 새로움은 주장할 수 없다.

## A. feature 통계 shift를 쓰는 완화

| 논문 | 신호 | 방법 | 모델 |
|---|---|---|---|
| NEO: No-Optimization TTA through Latent Re-Centering — Murphy et al., ICLR 2026, arXiv 2510.05635 | test penultimate embedding의 running mean | shift를 전역 중심 이동 / 클래스별 이동 / 샘플별 잔차로 분해, 전역 항만 라벨 없이 계산 가능하다고 보고 그것만 뺌. ECE 개선도 보고(12개 중 9개 설정) | ImageNet·CIFAR-10 fine-tune ViT-S/B/L |
| PEA: Architecture-Agnostic TTA via Backprop-Free Embedding Alignment — Ma et al., ICLR 2026 | 블록별 source 평균·공분산 | shift를 이동(평균)·스케일(분산)·회전(공분산)으로 모델링, 블록마다 정렬 (검색 요약으로만 확인) | CNN, ViT |
| TCA / LinearTCA: Test-time Correlation Alignment — You et al., ICML 2025, arXiv 2505.00533 | confidence 높은 test 샘플로 만든 "pseudo-source" 상관 | embedding의 닫힌 형태 선형 변환, 모델 갱신 없음 | DG backbone |
| DN: Test-Time Distribution Normalization — Zhou et al., NeurIPS 2023, arXiv 2302.11084 | test 배치 평균 embedding | 내적 전에 평균을 뺌 | CLIP, ALBEF |
| SSA: TTA for Regression by Subspace Alignment — Adachi et al., ICLR 2025, arXiv 2410.03263 | feature가 몰려 있는 PCA 부분공간의 source 평균·분산 | 그 부분공간에서만, head에 중요한 방향에 가중치를 두고 정렬 | 회귀 CNN |
| GOLD: The Golden Subspace — Lai et al., CVPR 2026, arXiv 2603.21928 | 분류기의 row space | feature를 그 공간에 사영하고 작은 스케일 벡터 학습 | continual TTA |
| TTN — Lim et al., ICLR 2023, arXiv 2302.05155 | 채널별 shift 민감도 | source와 test BN 통계를 섞는 비율을 채널별로 설정 | BN CNN |
| Effective Restoration of Source Knowledge in CTTA — Niloy et al., WACV 2024, arXiv 2311.04991 | source BN 통계와 배치 BN 통계의 KL | 급증하면 도메인 변화로 보고 source 가중치로 리셋(언제 적응할지) | BN CNN |
| PeTTA — Hoang et al., NeurIPS 2024, arXiv 2311.18193 | 클래스별 feature 평균이 source 통계에서 벗어난 거리 | 그 값으로 정규화 세기와 EMA 속도 조절(얼마나 세게) | CNN |
| Distribution Alignment for Fully TTA with Dynamic Online Streams — Wang et al., ECCV 2024, arXiv 2407.12128 | feature 분포 | source 통계로 당기는 loss + 분포 기반 shift 감지 | non-i.i.d. stream |
| Channel Importance Matters in Few-Shot — Luo et al., ICML 2022 | 채널별 평균·분산 불일치 | frozen feature에 test-time 채널 변환 | frozen backbone, few-shot |
| ADAPT (preprint, arXiv 2508.15568) | 온라인 클래스 평균 + 공유 공분산 | 닫힌 형태 Gaussian 분류기 | CLIP |

빈 곳: 측정한 공통 이동 비중으로 "평균만 / 평균+표준편차 / 아무것도 안 함"을 고르는 방법은 없다.
head 입력 통계 방법을 frozen SSL foundation model + linear probe에서 보인 논문도 없다.

## B. 라벨 없는 정확도 추정 / 실패 감지

| 논문 | 신호 | 비고 |
|---|---|---|
| TTA Induces Stronger Accuracy and Agreement-on-the-Line — Kim et al., NeurIPS 2024, arXiv 2310.04941 | BN 적응 후 ID·OOD 클래스 평균·공분산이 거의 일직선(cos 0.97–0.99, 적응 전 약 0.7) | agreement 기반 추정이 정확해짐. Table 2에 ATC(vanilla→BN_Adapt): CIFAR-10-C 8.31→3.69, CIFAR-100-C 5.05→2.89, ImageNet-C 3.95→7.37 (agent가 HTML에서 추출, 원문 확인 필요). head만 고치는 T3A는 효과 약함 |
| The Entropy Enigma — Press et al., ICML 2024, arXiv 2405.05012 | Weighted Flips: 적응 전후 예측 flip 수(초기 confidence로 가중) | 23개 데이터셋 MAE 5.75% (ATC 11.81, COT 8.17). ResNet-50 중심, ViT는 변환식 재학습 필요 |
| SF-DAP — Lee et al., ICCV 2023, arXiv 2307.10062 | source 모델과 source-free 적응 모델의 불일치율 | 적응 모델 학습 필요 |
| Chuang, Torralba, Jegelka — ICML 2020, arXiv 2007.03511 | domain-invariant 모델과의 최대 불일치 | "정렬된 모델과의 불일치 = 오차"의 개념적 조상 |
| Self-training Ensembles — Chen et al., NeurIPS 2021, arXiv 2106.15728 | target에서 self-train한 ensemble과의 불일치 | 정확도 추정 + 오류 감지 |
| Detectron — Ginsberg et al., ICLR 2023, arXiv 2212.02742 | source에서 일치하고 test에서 불일치하도록 학습한 분류기의 불일치율 | 해로운 shift에 대한 검정 통계량 |
| Feature Separability — Xie et al., NeurIPS 2023, arXiv 2303.15488 | pseudo-class 중심들의 분산 | 학습 불필요, feature 공간만 |

- 통계 정렬이 calibration을 개선한다는 보고: Nado et al. 2020(prediction-time BN), NEO(ECE).
- "통계 정렬이 ATC를 다시 정확하게 만든다"는 명시적 주장: 못 찾음.
- "적응 안 한 모델과 통계 정렬 모델의 불일치"를 추정기·감지기로 쓰는 논문: 못 찾음.

## C. 표현 유사도·기하 지표

| 논문 | 내용 |
|---|---|
| Ding, Denain, Steinhardt — NeurIPS 2021, arXiv 2108.01661 | CKA는 상위 약 10개 주성분 밖의 변화를 못 잡음. Procrustes가 기능 변화를 더 잘 따라감 |
| Davari et al. — ICLR 2023, arXiv 2210.16156 | 일부 점을 옮기거나 outlier만으로 CKA를 크게 바꿀 수 있음 |
| Nguyen, Raghu, Kornblith — TMLR 2022, arXiv 2202.07184 | 주성분 하나가 지배하면 linear CKA는 첫 주성분 간 cos²로 수렴 |
| Cloos et al. — ICLR 2025, arXiv 2407.07059 | CKA 민감도는 주성분 분산의 제곱에 비례. 저분산 과제 관련 차원을 놓쳐도 CKA는 높을 수 있음 |
| RankMe — Garrido et al., ICML 2023, arXiv 2210.02885 | embedding의 effective rank로 라벨 없이 사전학습 모델 선택(shift 아님) |
| Sójka et al. — CoLLAs 2026, arXiv 2609.02507 | TTA 붕괴 시 effective rank가 약 절반으로 줄어듦(진단, 검색 요약으로만 확인) |
| Gondhalekar et al. — NeurIPS 2023 workshop, arXiv 2311.18007 | ID·OOD의 CKA 구조가 OOD 강건성과 상관 |

빈 곳: CKA나 participation ratio를 적응 여부의 스위치나 정확도 예측에 쓰는 설득력 있는 논문은 없다. 진단으로만 쓰인다.

## D. Remote sensing

A–C를 frozen EO foundation model에 적용한 것은 없다. 관련 있는 것:
- Niloy, Bhaumik, Woo, "Source-Free Online Domain Adaptive Semantic Segmentation of Satellite Images under Image Degradation",
  ICASSP 2024, arXiv 2401.02113 — target BN 통계 + 전역 클래스 중심. RS에서 통계 정렬에 가장 가깝지만 BN CNN 분할.
- TARDIS (Ekim et al., CVPR 2025 EarthVision workshop, arXiv 2412.13394) — 내부 activation으로 shift 감지기 학습. 감지만.
- HyperTTA (preprint 2509.08436), Murugesan et al. (Frontiers in AI 2026, MEMO/TENT/SAR on AID·UCM) — 엔트로피 계열.
- RS에서 ATC류 정확도 추정: 못 찾음.

## 방법 설계에 주는 것

1. **비교 기준선 3단계**: 평균만(NEO) / 평균+표준편차(우리) / 공분산까지(LinearTCA, PEA). 우리 stage-1에서 평균만과 평균+표준편차의 차이는
   ViT 0.4pt 이내, ResNet 1.1–5.2pt였다.
2. **공통 이동 비중으로 보정을 켜고 끄기**: NEO의 분해가 "전역 항만 고칠 수 있다"고 말하고, TTN·PeTTA·Niloy가 통계량으로 적응 세기를
   정한다. frozen FM에서 공통 이동(또는 source 평균 거리)으로 정하는 규칙은 비어 있다.
3. **head가 쓰는 방향에서만 정렬**: SSA·GOLD는 분류기가 쓰는 부분공간에서 정렬하는 것이 전체 공간 정렬보다 낫다고 보고.
4. **flip을 "학습 없는 Weighted Flips"로 위치시키기**: Weighted Flips·SF-DAP와 직접 비교. confidence 가중을 빌려와 시험.
5. **"정렬이 ATC를 고친다"를 Kim et al. Table 2와 대비**: 그쪽은 BN 적응 후 ATC가 엇갈리고 head-only(T3A)는 부족하다고 함.
6. **CKA 옆에 평균에 민감한 지표를 같이 보고**: Procrustes 거리 또는 평균 이동 항.

## 미확인

- IMSE (ICLR 2026, arXiv 2603.07926) — 검색 요약으로만.
- "Representation Mismatch in Remote Sensing Foundation Models" (ICLR 2026 ML4RS workshop) — 초록이 모호, test-time인지 불명.
- LSCD-TTA 저널판의 저널명.
