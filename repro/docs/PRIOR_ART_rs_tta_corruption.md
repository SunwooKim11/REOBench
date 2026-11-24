# 선행 연구 조사: RS 성질로 인한 TTA 문제가 손상(corruption)에서도 이어지는가 (F2, 2026-11-24)

질문(교수님 F2): LSCD-TTA는 "RS 영상의 성질(클래스 간 유사성, 불균형, 정보 밀도 등) 때문에 기존 TTA가 안 된다"고 쌓아 올렸다.
그 문제가 손상 상황에서도 이어지는지 선행 연구가 있는가.

출처 구분: LSCD-TTA는 Claude가 PDF 본문을 직접 확인. REOBench, Beyond Accuracy, EarthShift, RSPDBench는 기존 노트(직접 읽음).
나머지는 **조사 agent가 웹에서 연 페이지 기준**이고 Claude가 원문을 확인하지 않았다(표의 "확인" 열). 출판사 페이지(IEEE, MDPI, ScienceDirect)는
열리지 않아 TGRS·JSTARS·GRSL 논문은 빠졌을 수 있다. "없다"는 찾지 못했다는 뜻이다.

## 1. 결론

1. **RS 성질이 TTA를 깬다는 주장은 논증 수준이고, 측정된 적이 없다.** LSCD-TTA는 서론과 방법 절에서 주장할 뿐 클래스별 정확도, 혼동, confidence 분석이 없다.
   그 논문의 표에서도 Tent는 source보다 높다(ResNet-50 평균 source 74.19, Tent 76.08, SHOT 76.63, LSCD-TTA 81.62). 뒷받침되는 것은 "이득이 작다"이지 "실패한다"가 아니다.
2. **그 주장을 손상에서 시험한 연구는 찾지 못했다.** LSCD-TTA는 오히려 기존 TTA가 "인공 손상 데이터에서만 검증됐다"며 손상을 RS가 아닌 쪽의 설정으로 놓고, 데이터셋 간 shift만 실험했다.
3. **손상된 RS 장면 분류에 Tent, EATA, SAR, CoTTA를 돌린 연구를 찾지 못했다.** EO 기초모델에 TTA를 적용한 손상 연구도 없다.
   REOBench, Beyond Accuracy, RSPDBench, EarthShift는 TTA를 돌리지 않았고, 뒤의 둘은 "필요하다 / 탐구되지 않았다"고 적었다.
4. **RS에서 손상 + TTA 결과는 좁은 설정 몇 개뿐이고 결론이 엇갈린다.** 분할(DeepGlobe, BN CNN)에서는 Tent가 돕고, 검출(DIOR-C)에서는 Tent·BN·SHOT이 무적응보다 나쁘다.
   어느 쪽도 실패를 측정된 RS 성질과 연결하지 않았다.

## 2. 선행 연구 표

| 논문 | 설정 / task | RS 성질에 대한 주장 | 근거 | 손상 | TTA | 확인 |
|---|---|---|---|---|---|---|
| LSCD-TTA (Liang 외, arXiv 2408.16265) | AID·NWPU·UCM 사이 6개 전이, 분류, ResNet-50/101, ViT-B/16 | 클래스 간 유사성, 클래스 내 변이, 불균형, 정보 밀도, 기상 조건 | 논증. 손실 ablation은 방법을 시험할 뿐 성질을 시험하지 않음 | 없음 | Tent, AdaBN, SHOT, LAME, TRIBE | Claude (PDF) |
| Niloy 외, ICASSP 2024 (arXiv 2401.02113) | DeepGlobe test에 합성 손상 5종, 분할, DeepLabv3-R50 | 없음. 작은 배치에서 BN 통계 추정이 어렵다는 점만 | 실험. mIoU source / Tent / 제안: noise 35.2 / 48.2 / 53.9, fog 12.6 / 41.6 / 48.6 | 있음 | Tent가 5종 모두에서 이득 | agent |
| CLIP-Guided SFOD (Liu 외, arXiv 2401.05168) | DIOR → DIOR-C(19종), 회전 검출 | pseudo-label 오류 누적, 작은 배치 | 실패는 실험, 원인은 논증. mAP 무적응 30.5, Tent 21.3, BN 24.3, SHOT 23.3 | 있음 | Tent·BN·SHOT이 무적응보다 나쁨 | agent |
| CTAOD-RS (Yong 외, TCSVT 2026) | DIOR → DIOR-C, continual, 검출 | 센서·지역·조명·날씨·객체 크기의 shift | 논문 본문 미확인 | 있음 | continual TTA | agent (README만) |
| HyperTTA (arXiv 2509.08436) | 초분광 픽셀 분류, 합성 열화 9종 | 높은 분광 차원, 반사율 변이 | 실험. 평균 OA Tent 58.5 / 72.8, SAR 45.4 / 51.8, 제안 82.5 / 87.4. source 행이 없고 백본이 다름 | 있음 | 있음 | agent |
| APCoTTA (ISPRS J. 2026) | 항공 LiDAR 분할, 합성 손상 7종, continual | 일반적 원인(오류 누적, 망각). 불균형은 한계로만 언급 | 실험. pseudo-label만 source보다 낮음 | 있음 | Tent·CoTTA는 source보다 높음 | agent |
| Du 외 (arXiv 2309.02150) | Sentinel-2 ↔ Landsat 9, 구름 분류 | 없음. "EO에 TTA를 적용한 보고가 부족" | 실험. Tent·DUA는 엇갈리고 일부는 source보다 낮음 | 없음 | 있음 | agent |
| MMEarth-Bench (arXiv 2602.06285) | 지역·라벨 shift, EO 모델 7개 | 엔트로피는 회귀에 맞지 않고 많은 EO 모델에 BN이 없음 | 논증. Tent는 돌리지 않음 | 없음 | test-time training | agent |
| LoveDA (NeurIPS 2021 D&B) | 도시 → 농촌 UDA, 분할 | 다중 크기 객체, 복잡한 배경, 클래스 분포 불일치 | 성질별 실험이 있는 유일한 사례. 단 UDA | 없음 | 없음 | agent (초록) |
| RS DA survey (arXiv 2510.15615) | survey | GSD, 촬영 방식, 지역, 날씨, 조명 | 논증 | 손상은 과제로 언급되지 않음 | TTA 한 번 언급 | agent |
| DOTA-C (He 외, arXiv 2308.15378) | 19종 손상 + 구름, 검출 11개 | 작은 객체, 복잡한 배경, 크기 변화 (강건성에 대해) | 정성·크기별 관찰 | 있음 | 없음 | agent |
| Mei 외 (J. Remote Sensing 2024) | AID-NN, DOTA-NN 노이즈 7종 | 없음 | – | 있음 | 없음 | agent |
| REOBench | 12종 손상, EO 기초모델 | 없음 (모델 계열 수준의 설명) | 클래스별 분석 없음 | 있음 | 없음 | Claude |
| Beyond Accuracy | 구름, 노이즈, motion blur, encoder 16개 | 없음 | 손상에서 EO encoder가 과신, temperature scaling이 듣지 않음 | 있음 | 없음 | Claude |
| RSPDBench | 물리 사양 기반 열화 | 없음 | – | 있음 | §4.7 "label-free TTA가 필요하다", 실행 없음 | Claude |
| EarthShift | 자연 shift 5종 | – | – | 없음 | "TTA의 효과는 탐구되지 않았다" | Claude |

기존 조사(`PRIOR_ART_feature_signals.md` §D)에 적어 둔 Murugesan 외(Frontiers in AI 2026, AID·UCM에 MEMO/TENT/SAR)는 이번 조사에서 다시 나오지 않았다.
손상을 다루는지 확인하지 못했으므로 "손상된 RS 분류에 TTA를 돌린 연구가 없다"고 말하기 전에 이 논문을 확인해야 한다.

## 3. LSCD-TTA의 주장과 우리 손상 결과의 대조

| LSCD-TTA의 주장 (RS 성질) | 손상에서 우리가 본 것 | 판정 |
|---|---|---|
| 클래스 간 유사성이 높아 hard pseudo-label이 확신 있는 클래스만 강조한다 | 손상에서 예측이 몇 클래스로 쏠린다. 최다 예측 클래스 비율 clean 4% → 손상 12–25%, SatMAE motion blur 60%. tent를 오래 돌리면 예측 클래스가 30개에서 23.5개로 줄어든다(17개 백본) | 손상에서도 이어진다. 형태는 "클래스 합쳐짐" |
| 확신이 커지면 엔트로피 gradient가 사라지고 저신뢰 표본이 손실을 지배한다 | 손상에서는 반대다. 정확도 15–60%인 손상에서 91–99%의 이미지가 확신으로 분류된다. 문제는 저신뢰가 아니라 틀린 확신이다 | 이어지지 않는다. 손상에서는 과신이 문제 |
| 클래스 불균형이 분포 중심에서의 shift를 키운다 | 직접 실험 없음. 클래스 비율을 균등하게 강제하는 DSBR은 이득, 분할에서 source 비율을 목표로 하면 균등보다 나쁨 | 미검증 |
| 정보 밀도가 높고 장면이 복잡하다 | DeYO의 패치 섞기 신호가 질감형 클래스에서 거의 0이다(0.01–0.04, 선택 2–5%, 형태형 클래스는 0.45–0.48) | clean에서 확인, 손상에서는 미검증 |
| 기상 조건이 변한다 | haze와 밝기 변화는 입력 통계 정렬로 거의 회복된다. 실제 haze(RRSHID)는 합성과 달리 선형 변환이 아니다 | 합성에서는 가장 쉬운 shift |
| (LSCD-TTA 자체) | AID-C에 적용하면 손상 평균은 오르지만 clean이 크게 떨어진다(ScaleMAE 95.2 → 77.1). 논문에 없는 diversity 항을 빼면 손상에서 붕괴(−8~−12pt) | 손상에서 그대로 쓸 수 없다 |

출처: `docs/NOTES_tta.md` §1.2, §4.1, §4.3, `experiments/2026-11-24_cluster_cot/RESULTS_em_phases.md`, `2026-11-19_dsbr/`.

## 4. 선행 결과와 우리 결과가 만나는 곳

- **분할에서 Tent가 돕는다는 Niloy 외의 결과와 우리 결과는 모순이 아니다.** 그쪽 모델은 BN이 있는 CNN이고 Tent가 BN을 test 배치 통계로 바꾼다.
  우리 Potsdam에서도 BN을 배치 통계로 둔 Tent는 +2.7 / +3.8 mIoU이고, 그 대부분은 통계만 바꿔도 나온다(+1.9 / +3.1). BN을 고정한 Tent는 −2.0 / −1.2다.
- **검출에서 Tent·BN·SHOT이 무적응보다 나쁘다는 결과는 우리 분류 결과와 같은 방향이다**(17개 백본 평균 tent 81.4 대 none 81.8, RESISC45-C 71.2 대 75.2).
- **Beyond Accuracy의 "손상에서 EO encoder가 과신한다"는 엔트로피 최소화가 실패하는 조건과 같다.** 그 논문은 TTA를 돌리지 않았고, 연결은 우리가 실험으로 보였다.

## 5. 우리가 채우는 빈 곳 (주장할 수 있는 범위)

- 손상된 RS 장면 분류에서 기존 TTA(Tent, SAR, EATA, DSBR)를 EO 기초모델과 일반 모델 17개에 평가한 것.
- 실패를 클래스 수준에서 설명한 것(예측 쏠림, 클래스 합쳐짐)과 그것이 몰리는 손상(motion blur, data gaps, scale)을 특정한 것.
- 단, "선행 연구가 없다"는 조사 범위 안에서의 말이다. 위의 Murugesan 외와 출판사 뒤에 있는 논문을 확인한 뒤에 단정해야 한다.
