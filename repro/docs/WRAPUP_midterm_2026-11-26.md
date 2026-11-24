# 중간 발표 준비용 통합 정리 (2026-11-26 발표, 2026-11-24 작성)

발표 자료를 만들 때 이 파일 하나만 보면 되도록 모았다. 수치의 원본 위치는 각 표 아래에 적었다.
PPT skill에는 **3절(슬라이드 구성)과 4절(표)** 을 그대로 주면 된다. 그림 파일 경로는 5절에 있다.

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
| F2. RS corruption의 고유성 | 분석은 완료, 선행 연구 조사는 부분 | "RS 이미지라서 다르다"는 증거는 약하다. 고유성은 **손상의 종류**와 **촬영 단위로 공유되는 shift**에 있다 | 같은 손상을 ImageNet과 RS에 적용한 비교, 12종의 단계별 배치 표 (S11) |
| F3. 벤치마크 2개 이상, 논문 읽기 | 완료 | AID-C, RESISC45-C, Potsdam-C, RESISC45 ↔ UC Merced. 네 논문 모두 노트 있음 | S4 |
| F4. fMoW 사전학습 ResNet / ViT, head 학습 방식 | 완료 | fMoW 백본 7개와 ImageNet 짝. head는 frozen feature 위의 선형 head이고 학습 방식의 영향도 통제함 | S3, 4절 표 T10 |
| F5. 백본 선택 근거 | 부분 완료 | 구조와 방법이 같고 데이터만 다른 쌍으로 구성해 "위성 사전학습 때문인가"를 가른다 | S3. 같은 설계의 선행 연구(arXiv 2611.13849)가 이유를 명시. REOBench, Beyond Accuracy, EarthShift 저자들의 선택 이유는 미정리 |

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
- **실제 haze(RRSHID)는 합성 haze와 다르다**: 채널별 선형 변환이 아니다(R² 0.35–0.43, 합성은 1.0).
- **사전학습 augmentation이 일부를 설명한다**: Gaussian blur augmentation이 있는 11개 백본은 Gaussian blur 손상 하락이 평균 1.3 / 2.6pt,
  없는 6개는 14.1 / 16.3pt. scale 손상은 fMoW로 학습한 쪽이 5쌍 모두 강하다.
- 아직 안 한 것: "RS 성질(클래스 유사성, 불균형) 때문에 생기는 TTA 문제가 손상에서도 이어지는가"에 대한 선행 연구 조사.
  우리 쪽 관찰은 있다(손상에서 예측이 질감 없는 클래스로 쏠림, 비슷한 클래스가 feature에서 맞닿아 있음).

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
- 남은 것: REOBench, Beyond Accuracy, EarthShift 저자들이 각 모델을 고른 이유의 정리.

## 3. 슬라이드 구성 (안)

| 번호 | 제목 | 넣을 것 | 한 줄 메시지 |
|---|---|---|---|
| S1 | 지난 피드백과 대응 | 2절의 표 | 다섯 항목 중 둘은 완료, 셋은 부분 |
| S2 | 연구 질문 | 질문 한 문장, 10월 관찰(엔트로피 필터가 91–99%를 확신으로 분류) | confidence를 못 믿으면 무엇을 믿을 것인가 |
| S3 | 백본 17개 (F4, F5) | T10, 쌍 6개 그림 | 데이터만 다른 쌍으로 구성 |
| S4 | 데이터와 벤치마크 (F3) | T11 | 합성 손상 3개, 자연 shift 1개 |
| S5 | 기존 TTA는 평균적으로 손해 | T1 | 엔트로피 기반은 손해, 정렬이 가장 큼 |
| S6 | 왜 실패하나: 과신과 예측 쏠림 | T2, T9 | 과신은 3종 손상에 몰려 있고 예측 클래스가 합쳐진다 |
| S7 | 통계 정렬이 하는 일 | 식 한 줄, head 통제 결과 | head 입력을 test 배치 통계로 다시 표준화 |
| S8 | flip 신호와 정확도 추정 | T3 | flip은 정확도를 따르고, 추정 오차는 COT와 같은 수준 |
| S9 | 자연 shift | T4 | 신호는 유지, 정렬은 약함 |
| S10 | 분할 (Potsdam) | T5 | 분류와 같은 현상 |
| S11 | RS 손상의 고유성 (F2) | T6, T7 | 고유성은 손상 종류에 있다 |
| S12 | feature의 모양 (선택) | fig1, T8 | 갈라지지만 덩어리는 아니다 |
| S13 | 남은 여지 | T12 | head 쪽에 5–6pt, 자연 shift는 거의 전부 head |
| S14 | 방법 후보와 계획 | 6절 | 촬영 조건 되돌리기 + 라벨 없는 head 적응 |
| S15 | 한계 | 7절 | seed 1개, 자연 shift 쌍 1개 |

시간이 부족하면 S12를 빼고 S9와 S10을 한 장으로 합친다.

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

### T5. 분할 (Potsdam, 손상 8종 × 5단계 평균 mIoU)

| | SatMAE | ScaleMAE |
|---|---:|---:|
| none | 50.6 | 51.3 |
| BN 정렬 | 52.5 | 54.4 |
| 입력 정렬 | 53.8 | 53.9 |
| 입력 + BN 정렬 | 55.1 | 56.3 |
| tent (none 대비) | −1.96 | −1.19 |
| dsbr (none 대비) | +0.85 | +0.56 |
| tent + BN 정렬 (none 대비) | +2.73 | +3.79 |
| dsbr + BN 정렬 (none 대비) | +2.84 | +3.88 |
| flip과 픽셀 정확도의 순위상관 (stream 40개) | −0.94 | −0.97 |

clean은 61.9 / 65.0. 빠진 손상 4종(rotate, scale, translate, data gaps)은 생성했고 실행 중이라 발표에는 8종 기준으로 쓴다.
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
| Potsdam-C | 분할 5클래스 | 합성 손상 8종 + 4종(우리가 생성) | test 2,016 패치 |
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
| 2 | 정렬이 하는 일 (식과 두 단계) | `docs/STUDY_alignment.md` §2.4 | 15분 |
| 3 | 백본 표, augmentation 표, 손상 배치 표 | `docs/SETTINGS.md` §1, §2.5 | 15분 |
| 4 | feature 그림 두 장 | `docs/figs/2026-11-24_feature_dist/fig1_clean_geometry.png`, `fig2_shift_align_motion_blur_s3.png` | 10분 |
| 5 | Geometry 세 가지 (T8, Silhouette 상관, T9) | 이 문서 4절 | 15분 |
| 6 | 10월 결과가 필요하면 | `docs/WRAPUP_progress_2026-11-23.md` §1 | 10분 |
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

**예상 질문과 답**

| 질문 | 답 |
|---|---|
| 정렬은 source 통계를 쓰는데 반칙 아닌가 | 표준화해서 학습한 head에서는 test 식에 source 통계가 들어가지 않는다. 그대로 학습한 head에서 쓰는 통계도 BatchNorm의 running 통계와 같은 종류다. 다만 새롭지는 않다 |
| RS만의 문제가 무엇인가 | 이미지 도메인이 아니라 손상의 종류(data gaps, scale, clouds)와, shift가 촬영 단위로 stream 전체에 공유된다는 점이다 |
| 왜 DINOv3 web이 가장 강한가 | 같은 방법의 위성 모델은 14pt 낮으므로 방법이 아니라 데이터나 규모다. 어느 쪽인지는 가르지 못했다. matched-pair 연구와 DINOv3 저자들의 GEO-Bench 평가도 web이 위성을 이긴다고 보고했다 |
| DINOv3-SAT이 낮은 것은 feature를 잘못 뽑아서가 아닌가 | 우리는 CLS와 patch 평균을 이어 붙여 쓴다. matched-pair 연구가 세 가지 feature를 모두 시험했고 가장 좋은 것으로도 web에 뒤졌다 |
| 분할에서도 되나 | 현상(엔트로피 방법은 손해, 쏠림을 누르면 이득, flip이 정확도를 따름)은 같다. feature geometry 설명은 분류에서만 확인했다 |
| 다음은 무엇인가 | 6절의 1번과 2번. 여지가 어디 있는지는 T12로 쟀다 |

## 8. 로컬로 가져갈 것

- 이 파일과 그림은 모두 저장소에 commit돼 있다. `repro` 브랜치를 push한 뒤 로컬에서 받으면 된다.
- 그림: `docs/figs/2026-11-24_feature_dist/` 2장, `docs/figs/meeting_2026-10/` 7장.
- PPT skill에 줄 입력: 3절(슬라이드 구성) + 4절의 표 중 해당 번호 + 7절의 한계. 슬라이드마다 "한 줄 메시지"를 제목 아래 문장으로 쓰면 된다.
