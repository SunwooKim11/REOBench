# REOBench TTA 노트 (주제별 정리, 2026-10-16 기준)

> 시간순 원본은 `docs/archive/NOTES_tta_chrono_backup.md`. 개념 설명(스케줄러, GSD, drop_path, 정밀도)은 `docs/NOTES_concepts.md`.
> 코드 리뷰 방법은 `review/REVIEW_GUIDE.md`.

## 0. 한눈에 보기

**결론 요약**
- 구현 검증: 분류(ImageNet-C)와 분할(Cityscapes→ACDC) 모두 논문/공식 코드와 일치 (§2).
- RS 분류(AID)에서 기존 TTA 효과는 ±1~2%p 수준, 입력 정규화(P→PN, +7)보다 훨씬 작음 (§3).
- **입력 통계 정렬(ISA, batch)** 이 유일하게 꾸준히 이득: radiometric 왜곡(brightness·haze)에서 +10~29 (§3, §4.2).
- 파라미터를 많이 바꾸는 방법은 위험: CoTTA(ViT-L 전체)는 모든 설정에서 붕괴, ScaleMAE P는 SAR·DeYO·Tent가 붕괴 (§3).
- DeYO의 PLPD는 RS 장면 분류에서 신호가 약함(질감형 클래스는 거의 선택 안 됨) (§4.1).
- 정렬 위치 sweep: radiometric은 입력, degradation은 중간 블록(7~11) 정렬이 최선 — 분류·분할, SatMAE·ScaleMAE 모두 (§5.5).

**열린 결정 사항**
1. RS 분할 Tent 정의: 원문(픽셀 엔트로피 + BN test 통계) vs CoTTA 코드식 PL "TENT" (§5.2) — 제안: 원문을 기본, PL은 별도 표기.
2. SAR 복구 임계값 표기: 0.005(DeYO 구현, 주 결과) vs 0.2(원 논문, 보충) (§3.3).
3. CoTTA의 RS 보고 방식: "원 설정은 ViT-L에서 불안정"으로 보고할지, lr을 크게 낮춘 버전을 추가할지 (§3.3).
4. ViDA 분할 구현: CoTTA 분할 코드에 ViDA adapter 주입 (저자 안내 방식) (§5.1).

## 1. 실험 설정

### 1.1 소스 모델과 프로토콜 (AID, `scripts/tta/cls_tta.py`)
- 백본: SatMAE(fMoW-RGB `fmow_pretrain.pth`), Scale-MAE(`scalemae-vitlarge-800.pth`), ViT-L/16, 256px, 고정. drop_path=0.
- head: `scripts/cls/ablation.py`로 학습한 Linear head (`experiments/2026-10-12_cls_ablation/outputs/*/{P,PN}_s0_head.pth`). **P** = 논문 프로토콜(ToTensor만, A0ep), **PN** = P + 입력 정규화(SatMAE fMoW / Scale-MAE ImageNet 통계).
- 평가 단위: (왜곡 12종 × 강도 5) + clean = 61개 묶음, 각 2,000장, shuffle, batch 64. **묶음마다 source로 리셋**(mild = (왜곡, 강도)별 독립 평가, continual 아님). 코드·로그의 'stream'은 이 묶음을 뜻함.
- 정확도: 업데이트 직전 예측(online). `out = adapt(x)`의 반환값이 업데이트 전에 계산한 출력이라(`outputs = model(x)` → step → `return outputs`), 바깥에서 채점해도 업데이트 전 예측임. lr = 0.001/64 × bs (ViT, DeYO/SAR 규칙), CoTTA 0.01.
- 임계값은 × ln C (ln C = 최대 엔트로피, 즉 '최대 불확실도의 k배'; DeYO repo는 C=2 데이터셋에도 동일 적용) (C = 30): SAR E0 = 0.4 ln C, DeYO τEnt = 0.5 ln C / Ent0 = 0.4 ln C / τPLPD = 0.3, EATA E0 = 0.4 ln C.
- 업데이트 파라미터(Tent/SAR/DeYO): LayerNorm affine, **상위 25% 블록과 마지막 norm 제외** (ViT-L: blocks 18~23). EATA는 원본대로 모든 norm.
- 정밀도: fp32 + TF32 (DeYO/SAR 환경인 PyTorch 1.10 기본값과 같음).

### 1.2 방법 구현 출처 (`src/reobench_tta/tta/`, 차이는 `review/diffs/`)

| 방법 | 출처 | 변경 |
|---|---|---|
| Tent / SAR / DeYO / EATA / SAM | DeYO repo (Jhyun17/DeYO @3d7d289) | `collect_params`만 일반화(ViT-B 하드코딩 → 상위 25%). EATA·SAM은 0줄 변경 |
| CoTTA | qinenergy/cotta `imagenet/` | 이미지 크기 변수화, torchvision API, 증강 임계값 `ap` 인자화 |
| ViDA | Yangsenqiao/vida `imagenet/` | 두 파일 병합, HKA 버그는 기본 재현(`fix_hka`), 명세 `docs/verify/VIDA_SPEC.md` |
| LSCD | AEAQA/LSCD-TTA (loss만) | 공식 루프는 SyntaxError → Tent식 루프 |
| ISA | 자체 baseline | 입력 채널 평균·표준편차를 source(AID train) 통계로 재정렬. batch 단위 / 이미지 단위 |

### 1.3 방법별 핵심 동작 (요약)
- **Tent:** 예측 엔트로피 평균 최소화. 틀린 예측도 확신하게 만들 수 있어 심한 shift에서 붕괴 위험.
- **SAR:** ① 엔트로피 < E0인 샘플만 사용 ② SAM(ρ=0.05)으로 평평한 최솟값 ③ 엔트로피 EMA가 임계값 아래면 리셋. 비용 약 2배.
- **DeYO:** 엔트로피 필터 + 4×4 patch shuffle 전후 같은 클래스 확률 차이(PLPD) > τ 필터 + 재가중. "모양 정보로 판단한 샘플만 쓰자"는 가정. 비용 약 1.5배.
- **EATA:** 엔트로피 필터 + 중복 샘플 필터(ε=0.05) + Fisher 정규화(β=2000, source 2,000장).
- **CoTTA:** EMA teacher(0.999) + 저확신 batch에서 32개 증강 평균 pseudo-label + 확률적 복원(0.001), **모든 파라미터** 학습.
- **ViDA:** attention의 qkv/proj에 저랭크(1)·고랭크(128) adapter, 불확실도로 두 branch 비중 조절(HKA). 코드상 백본까지 전부 학습.

## 2. 구현 검증

### 2.1 ImageNet-C (`scripts/verify/imagenetc.py`, 결과 `experiments/2026-10-14_verify_imagenetc/summary.txt`, 기준 `docs/verify/REFERENCES.md`)

| 검증 | 우리 / 기준 | 판정 |
|---|---|---|
| DeYO T1 (R50-BN, mild, 정확도) | none 18.15/18.0, Tent 43.29/43.1, EATA 47.88/47.8, SAR 44.22/43.9, DeYO 48.73/48.6 | ✅ ≤0.35 |
| ViDA T1 (ViT-B, continual, 오류율) | source 56.15/55.8, ViDA 43.29/43.4 (repo 로그 43.37) | ✅ |
| ViT-B label shift (DeYO T5) | SAR 54.6/54.8, DeYO 64.7/62.3, EATA 51.4/49.2, Tent 53.3/47.7 | 🟡 차이는 붕괴형 열(std ±25)에 집중 |
| CoTTA (R50, continual) | 우리 66.73 / **공식 코드를 우리 환경에서 66.56** / ViDA T1 62.7 | ✅ 이식 정확 |
- 공식 CoTTA(`third_party/cotta_official/`): source가 우리 harness와 소수점까지 같음. 순서 0은 59.50(논문 10개 순서 평균 63.0±1.8) → 순서 의존성이 7%p.
- noise 계열 열만 원본 모델부터 논문보다 쉬움(ViT gauss 16.6 vs 9.4). Zenodo 공식 파일 → JPEG 디코딩 차이로 추정.
- 세 논문의 ViT-B source가 다름: SAR/DeYO 29.9(augreg + ImageNet 정규화), ViDA 44.2(augreg + 정규화 없음), EATA-C 39.84(미확인 → 보류).

### 2.2 분할 Cityscapes→ACDC (`third_party/cotta_acdc_svdp/`, 패치 `README_REOBench.md`)
- 코드: CoTTA ACDC 코드 사본(SVDP repo; 원본 OneDrive는 브라우저 필요). env `segtta`(torch 2.0.1 cu118, mmcv-full 1.7.2; H200은 CUDA ≥ 11.8 필요).
- SegFormer-B5 Cityscapes, 960×540, bs 1, 4조건 연속.

| 항목 (mIoU) | 우리 | 기준 |
|---|---|---|
| source fog / night / rain / snow | 69.08 / 40.28 / 59.72 / 57.75 | CoTTA T5 69.1 / 40.3 / 59.7 / 57.8 ✅ |
| PL "TENT" 10라운드 평균 | 52.74 (R1 69.07/40.19/60.02/57.43, R10 62.48/30.26/52.54/48.62) | 52.3 ✅ |
| 엔트로피 Tent, BN 고정, 10라운드 평균 | 52.28 | (PL과 같은 궤적) |
| 엔트로피 Tent, BN test 통계(원문), 1라운드 Δ | −4.5 / −0.5 / −2.3 / −2.4 (평균 −2.5) | Yi et al. −4.9/−0.3/−2.1/−2.7 ✅, ATASeg −6.1/−2.1/−3.3/−3.6 |
| CoTTA 1라운드 fog | 70.93 | 70.9 ✅ |
- CoTTA 분할 증강 = scale 7개 × flip = 14 forward, anchor 확률 > 0.69 픽셀은 1.0배 단일 예측. 이미지당 10~20초 → fog 400장만 확인.

## 3. RS 분류 TTA 결과 (AID, seed 0, `experiments/2026-10-13_cls_tta/summary_tta.txt`)

### 3.1 전체 결과 (none 대비 Δ clean / Δ corruption 평균)
> P = 논문 원형(ToTensor만), PN = P + 입력 정규화. 두 버전 모두 실행됨. **본문 결과는 PN 기준**, P는 논문 프로토콜 재현용 보조. (PN head는 seed 0만 → 본문용은 seed 3개 필요)

| 설정 | none (clean/avg) | Tent | SAR | DeYO | EATA | CoTTA | LSCD | ISA batch | ISA image |
|---|---|---|---|---|---|---|---|---|---|
| SatMAE P | 90.5/64.9 | −0.5/**+1.6** | −0.5/−1.5 | −4.1/−10.1 | −0.1/−5.0 | −44/−31 | −5.0/−13.7 | −0.1/**+4.8** | −10.9/−3.1 |
| SatMAE PN | 92.8/72.4 | 0/0 | 0/+0.8 | −1.2/+0.2 | 0/+1.0 | −43/−47 | −9.7/−11.0 | −0.2/**+1.7** | −11.3/−10.9 |
| ScaleMAE P | 91.5/68.0 | +0.7/−14.1 | −70/−47 | −81/−39 | 0/−12.5 | −33/−32 | −5.0/−15.5 | −0.6/**+3.9** | −9.8/−6.7 |
| ScaleMAE PN | 95.2/74.8 | 0/+1.1 | 0/+0.1 | −2.1/−7.7 | 0/**+2.6** | −20/−27 | −18.2/+2.3 | +0.1/+1.3 | −12.2/−13.4 |

- 입력 정규화(P→PN)의 효과(+7.5 / +6.8)가 어떤 TTA보다 큼.
- PN에서 이득: EATA(+1.0/+2.6), ISA batch(+1.7/+1.3), Tent(ScaleMAE +1.1). LSCD는 avg가 올라도 clean −18이라 사용 불가.
- ISA image(이미지별 정렬)는 clean −10~12 → 이미지 자체의 색 정보를 지워 해로움. batch 단위만 의미 있음.

### 3.2 lr 민감도 (`experiments/2026-10-13_cls_tta/summary_lr.txt`, Tent/SAR/DeYO × lr 1e-3 / 3e-4 / 1e-4)
- SatMAE: DeYO는 lr을 낮추면 개선(P +4.1, PN +2.9 @3e-4). Tent·SAR은 ±1 내외.
- ScaleMAE PN: Tent만 꾸준히 +0.4~1.1. ScaleMAE P: lr을 낮춰도 대부분 붕괴(Tent 1e-4에서도 clean −75).

### 3.3 설정 변경 실험
- **CoTTA 재실행** (EATA-C ViT 설정 lr 0.005, p_th 0.1; SatMAE PN만): clean 82.6 / avg 55.2 (첫 실행 49.5 / 25.0, none 92.8 / 72.4).
  증강은 한 번도 안 켜짐 → 원인은 ViT-L 전체 파라미터(~3억) 매 배치 업데이트. 스트림마다 붕괴 여부가 갈림(clouds s1 40.6 / s2 91.8). 나머지 3개 설정은 취소.
- **SAR 복구 임계값** (DeYO repo 0.005 vs 공식 SAR 0.2, `sar.py` L102 하드코딩):

  | clean/avg | none | SAR 0.005 | SAR 0.2 | 리셋 횟수(0.2) |
  |---|---|---|---|---|
  | SatMAE PN | 92.8/72.4 | 92.8/73.3 | 92.8/72.7 | 895 |
  | SatMAE P | 90.5/64.9 | 90.0/63.4 | 90.5/64.7 | 193 |
  | ScaleMAE PN | 95.2/74.8 | 95.2/74.9 | 95.2/75.0 | 312 |
  | ScaleMAE P | 91.5/68.0 | 21.4/20.9 | 17.6/18.8 | 25 |

  AID(C=30)는 정상 상태 엔트로피 EMA도 0.2 아래 → 전체 ~1,950 배치 중 최대 895번 리셋 → 사실상 source. 0.2는 ImageNet 기준 절대값.

### 3.3b CoTTA 붕괴 진단 (SatMAE PN, p_th 0.1, 7 스트림: clean + haze/gauss noise/motion blur × s3,s5; `run_cotta_diag.sh`)
| 설정 | 업데이트 | clean | haze s3/s5 | noise s3/s5 | mblur s3/s5 | 7개 평균 |
|---|---|---|---|---|---|---|
| none | – | 92.8 | 84.8 / 60.0 | 66.9 / 55.8 | 25.8 / 15.2 | 57.3 |
| Tent | LN 0.07M | 92.8 | 86.8 / 65.6 | 72.2 / 59.0 | 12.2 / 8.9 | 56.8 |
| CoTTA 전체 lr 0.01 | 303M (head 포함) | 73.5 | 24.5 / 29.1 | 44.0 / 31.9 | 8.9 / 12.5 | 32.0 |
| CoTTA 전체 lr 1e-5 | 303M | 92.75 | 84.8 / 60.1 | 66.9 / 55.7 | 25.8 / 15.3 | = none |
| CoTTA LN lr 0.01 | 0.07M | 92.75 | 77.1 / 68.4 | 63.2 / 55.7 | 28.5 / 14.7 | 57.2 |
| CoTTA LN lr 0.001 | 0.07M | 92.75 | 84.8 / 60.1 | 66.9 / 55.8 | 25.8 / 15.3 | = none |
- 붕괴는 "전체 파라미터 × 큰 lr"에서만. 작게 업데이트하면 CoTTA는 **source와 똑같아짐**(소수점까지).
- 이유: CoTTA의 출력은 EMA teacher(0.999). 스트림당 31 step이면 teacher에 섞이는 student 비중이 ~3%라 teacher ≈ source.
  student가 크게 발산할 때만(전체 lr 0.01) 그 3%로도 teacher가 망가짐. → **mild(스트림마다 리셋, 2,000장) 프로토콜에서는 CoTTA가 효과를 낼 구간이 없음.**
  CoTTA는 긴 연속 스트림(continual)용 설계. 공식 코드는 linear head까지 업데이트.

### 3.4 ScaleMAE P 붕괴 진단 (`scripts/analysis/diag.py`, clean 스트림)
- 첫 step 뒤 파라미터 이동량: SAR 1.14 (PN 0.0006, SatMAE P 0.002의 수백 배), 둘째 batch 정확도 95 → 2%.
- DeYO: 첫 step drift 0.028, 직후 `blocks.0.norm1.weight` gradient 2.8 → 1,549 폭발. Tent는 step당 0.003~0.009로 안정.
- 첫 블록 LayerNorm 근처의 가파른 방향으로 SAM 섭동·큰 step이 들어가면 발산. 저엔트로피형 붕괴가 아니라 SAR 복구(0.2)도 못 막음(리셋 25번).
- 정규화가 빠지면 첫 블록 입력의 이미지 정보 비중이 두 모델 모두 ~35% 줄지만 SatMAE는 안정 → ScaleMAE만의 원인은 미해결.

## 4. 분석

### 4.1 DeYO PLPD는 RS 장면 분류에서 약한 신호 (`plpd_analysis.py`, `trap_analysis.py`)
- 클래스 그룹(결과 보기 전 고정): texture 6(BareLand, Desert, Farmland, Forest, Meadow, Mountain), shape 8(Airport, BaseballField, Bridge, Church, Playground, Stadium, StorageTanks, Viaduct), other 16.
- clean, patch_len 4: texture PLPD 0.01~0.04 / 선택 2~5% / 예측 유지 94~98%; shape(SatMAE) PLPD 0.45~0.48 / 선택 50~55%.
  ScaleMAE는 모든 그룹에서 섞기에 둔감(shape도 선택 10~19%) → bag-of-patches식 판단 가능성.
- 정답 판별력(AUROC): PLPD 0.43~0.66 vs 엔트로피 0.67~0.95. PLPD 필터 추가 시 고른 샘플 정확도가 대부분 하락.
  예외: SatMAE의 강한 radiometric 왜곡(haze s5 +6, brightness s5 +10, gauss blur s5 +8).

### 4.2 왜곡 그룹과 처방 가설

| 그룹 | 왜곡 | PN none (SatMAE/ScaleMAE) | 기존 TTA | ISA batch | 가설 처방 |
|---|---|---|---|---|---|
| geometry | rotate, scale, translate | 86.3 / 91.3 | ±0.2 | ±0.5 | 회전·뒤집기 TTA 평균 |
| occlusion | clouds, data_gaps | 79.7 / 80.0 | +1 내외 | −0.5 ~ −1 | MAE 마스킹(가려진 패치 토큰 제외) |
| radiometric | brightness_contrast, haze | 80.3 / 85.9 | 불안정(S5 +12 ~ −23) | **BC 80→91 / 85→94, haze 81→92 / 87→95** (P는 61→90) | **입력 통계 정렬 — 확인됨** |
| degradation | noise×2, blur×2, JPEG | 58.1 / 58.4 | ±1, S5 손해 | ±1 | 뚜렷한 답 없음 |
- 그룹별 처방을 쓰려면 왜곡 종류를 테스트 때 자동 판별해야 함(채널 통계 이동, 고주파 에너지, 단색 패치 비율). 판별기 정확도도 보고.

## 5. 분할 확장

### 5.1 방법별 분할 코드 출처

| 방법 | 분할 코드 | 비고 |
|---|---|---|
| Tent | CoTTA ACDC 코드 `single_gpu_tent` (PL 변형) + 우리 `single_gpu_tent_entropy` (원문) | §5.2 |
| CoTTA | CoTTA ACDC 코드 `single_gpu_cotta` (SVDP 사본, EMA 인자 버그 수정) | 검증 완료 |
| ViDA | **공식 비공개**. 저자 공지: CoTTA 분할 프레임워크에 ViDA를 SegFormer에 주입 | issue #4: lr 7.5e-6, CoTTA 숫자 복사 → 공정성 논란 |
| ISA | 입력 변환이라 그대로 적용 | |
| SAR | 공식 없음. 유일한 이식본 PASS(의료 binary 분할, 이미지 단위 필터, /200 스케일)는 다중 클래스에 부적합 | 논문대로 직접 이식 필요 |
| EATA | 공식 없음. EATA-C T4(ACDC): 모든 norm, lr 7.5e-5, Fisher 20장, β 500, 분할에선 sample selection 끔 | |
- 우선순위(10-13 결정): 1순위 Tent, CoTTA, ViDA, ISA / 2순위 EATA, LSCD, SVDP·Hybrid-TTA / SAR·DeYO는 분류 결과까지만.

### 5.2 "TENT"의 정의 문제
- 원문 Tent: Shannon 엔트로피 최소화(논문 §3, 공식 `softmax_entropy(outputs).mean(0)`), BN affine 학습 + **BN을 test batch 통계로**. 분할 실험(HRNet-W18 GTA→Cityscapes 28.8 → 35.8)은 코드 미공개.
- CoTTA 분할 코드의 "TENT": 자기 argmax 예측을 정답으로 CE (임계값 없는 hard PL), `model.eval()`이라 BN 고정. Tent 논문은 PL을 별도 baseline으로 비교함.
- 이 PL 변형 숫자를 그대로 쓰는 곳: SVDP·BeCoTTA·Hybrid-TTA 코드, DAT·Continual-MAE·OCL(CoTTA 수치 복사), EATA-C(같은 궤적).
- 엔트로피 Tent 기준 수치(SegFormer-B5 + ACDC + 10라운드 + 코드 공개)는 없음. 가장 가까운 것: ATASeg(1라운드), Yi et al.(ACM MM'24, 도메인별 리셋, "BN-fixed" 행 ≈ source).
- 우리 결과: 성능 하락의 주원인은 엔트로피 loss가 아니라 **decode head BN을 이미지 1장 통계로 바꾸는 것**.

### 5.3 REOBench 분할 코드 확인 사항 (`Segmentation/configs/vit_base_win/`)
- UperNet(+FCN aux), 512 crop, AdamW 6e-5, batch 2, **백본 전체 fine-tune**(분류는 고정). decoder는 BatchNorm → §5.2의 BN 선택이 중요.
- SatMAE 분할 정규화 단위 오류 의심: 0~1 단위 mean/std를 0~255 픽셀에 적용 → 입력 최대 ≈ 900. (재현으로 확인 필요)
- SatMAE 분할도 `pretrain-vit-large-e199.pth`(Sentinel)를 가리킴 — 분류와 같은 문제.
- (10-16 추가 분석) 원 논문 Table 2 (mIoU): SatMAE clean 59.51 / avg 49.59, ScaleMAE 68.92 / 60.02, SatMAE++ 62.68 / 54.04.
- **학습 설정 불일치**: 논문 부록 = 12 epoch, batch 8, AdamW 6e-5 wd 0.05 (≈5,184 iter) / 코드 = `schedule_160k`(160k iter) + `samples_per_gpu=2`.
- **feature pyramid**: 백본이 blocks [3,5,7,11]만 사용(0-based; 논문 "blocks 4,6,8,12", RVSA ViT-B 기준) → ViT-L 24블록 중 상위 12블록은 계산만 하고 미사용.
  layer decay `num_layers=12`도 ViT-B 값.
- 사전학습 경로가 백본 `init_weights` 안에 하드코딩(config 무시), `strict=False`이고 missing key를 출력하지 않음.
- **공개 데이터 누락**: Potsdam test 이미지 zip에 12개 왜곡 중 9개(clean 포함)만 있음. data_gaps / rotate / scale / translate 이미지는 없음
  (라벨 zip은 있음). → repo의 `Corrupt Image Generation` 코드로 생성해야 할 수도 있음.
- 데이터 크기: train 3,456장(1.4 GB), test 2,016장 × 조건, test 이미지 zip 28 GB.
- **공개 코드가 공개 데이터로 그대로는 안 돌아감** (`third_party/reobench_seg`, 패치는 `[REOBench repro]` 표시):
  1. `build_backbone`이 항상 `.model`을 반환(CLIP 래퍼용) → MAE 백본은 빌드 실패. `getattr(b, 'model', b)`로 수정.
  2. 백본 `init_weights` 경로 하드코딩 → 환경변수(기본값 = 코드가 지정한 체크포인트), missing key 출력 추가. timm≥0.9 `needs_reset` 인자 허용.
  3. 공개 라벨에 255가 3~4% 있는데 config는 5(clutter)만 ignore → CE가 범위 밖 클래스로 GPU 크래시. 255 → 5(ignore)로 매핑.
     (논문은 non-eroded 라벨이라 했지만 255 경계가 있는 것은 eroded 라벨일 가능성)
  4. 환경: torch 2.0.1 fused SDPA backward가 H200에서 illegal instruction → `TIMM_FUSED_ATTN=0`(원래 timm 0.4와 같은 계산).
- 실행: `run_train.sh` (A = 공개 config + 논문 12 epoch / batch 8, seed 0), ScaleMAE 약 0.7 s/iter → 약 1시간.
- **A 결과 (seed 0, 8개 공개 왜곡, 강도 1~5 평균 mIoU; `experiments/2026-10-16_seg_baseline/summary.txt`)**:
  ScaleMAE clean 65.88 (논문 68.92), 왜곡 평균 54.09 (논문 같은 8개 평균 60.00). 차이가 왜곡에서 더 큼:
  haze 52.3 vs 63.9, S&P 42.0 vs 52.1, BC 55.3 vs 64.4, 나머지는 −2~−5.
  SatMAE(Sentinel 체크포인트, patch_embed 랜덤 → 전체 fine-tune 중 학습됨) clean 53.46 (논문 59.51), 왜곡 평균 41.75 (논문 46.76).
  SatMAE는 cloud만 논문보다 높고(47.1 vs 37.4) 나머지는 −1~−13.
  클래스별: impervious_surface IoU 26.9로 비정상적으로 낮음 → 공개 라벨의 클래스 번호와 config 클래스 이름 순서가 다를 가능성.
- **B (수정판, 2026-10-19 실행)** `configs/repro/{scalemae,satmae}_B_12ep.py`: A + ViT-L feature 블록 [7,11,15,23], layer decay 24층.
  SatMAE는 추가로 RGB 체크포인트(`fmow_pretrain.pth`) + 0~255 단위 정규화(fMoW mean/std × 255).
  백본에 `out_indices`, `ckpt` 인자 추가(기본값 = A). 실행 `run_B.sh` (모델당 학습 ~1.25 h + 평가 ~2.7 h).
- **A vs B 결과 (mIoU, clean / 8개 왜곡 평균; 논문 같은 8개 평균)**:

  | | A | B | 논문 |
  |---|---|---|---|
  | SatMAE | 53.46 / 41.75 | **61.90 / 50.60** | 59.51 / 46.76 |
  | ScaleMAE | 65.88 / 54.09 | 64.97 / 51.35 | 68.92 / 60.00 |

  SatMAE는 B에서 +8.4 / +8.9 → 논문보다 높아짐(세 수정의 합; 요인별 분리는 안 함, RGB 체크포인트가 주효과로 추정).
  ScaleMAE는 블록만 바꿨는데 −0.9 / −2.7, 특히 noise(GNoise 50.7→39.4, S&P 42.0→35.6) 악화 → 상위 블록 feature가 noise에 덜 강건하거나 12 epoch로는 부족.
- 공부 자료: `docs/seg/STUDY_GUIDE.md` (코드 읽는 순서, UperNet 설명), `trace_shapes.py` (이미지 1장 텐서 모양 추적).

### 5.4 Potsdam 분할 TTA 구현 (`scripts/seg/seg_tta.py`, 2026-10-20)
- 모델: 분할 B(SatMAE B, ScaleMAE B). 프로토콜 **mild = (왜곡, 강도)별 독립 평가**(continual 아님): (왜곡, 강도)마다 2,016장, shuffle, batch 8, 매번 리셋,
  업데이트 전 예측으로 mIoU(5클래스, 5·255 무시). 의료 분할 TTA 기본 평가(VPTTA, GraTa: 사이트별 적응 후 리셋)와 같은 구조,
  차이는 batch 크기(의료는 주로 1). continual(리셋 없이 도메인 연속; CoTTA·ACDC, SPEGC의 long-term 실험)은 범위 밖.
- `none`이 mmseg 공식 평가와 일치(clean 64.97, haze s5 32.23) → 데이터·평가 경로 확인.
- 방법: none / bnadapt(head BN을 test 배치 통계로, 학습 없음) / isa(입력 통계 정렬) / tent(픽셀 엔트로피, LN+BN affine,
  Adam 6e-5/8, `--bn fixed|batch`) / sar(픽셀 필터 0.4 ln 5, SAM + Adam, 복구 0.2) / cotta(14 view teacher, anchor 임계값 =
  train 5% 분위수 − 0.05 = 0.41, 전체 파라미터, 복원 0.01; 비용 때문에 clean + 강도 3·5만).
- 백본 수정: 입력 크기가 512가 아니면 위치 임베딩을 bicubic 보간(CoTTA 다중 스케일용). 512에서는 동작 동일.
- 스모크(ScaleMAE B, haze s5): none 32.2, bnadapt 46.6, isa 61.8(200장), tent 29.9, sar 30.6, cotta 34.8(48장).
- 실행: `scripts/archive/seg/run_seg_tta.sh` (2개 병렬). 요약: `scripts/seg/summarize_seg_tta.py` → `experiments/2026-10-20_seg_tta/summary.txt`.
- **결과 (none 대비 Δ clean / Δ 왜곡 평균 8종×강도 1~5)**:

  | | ScaleMAE B | SatMAE B |
  |---|---|---|
  | none (절대값) | 64.97 / 51.34 | 61.90 / 50.60 |
  | tent + BN test 통계 | −0.70 / **+3.79** | −0.69 / **+2.73** |
  | bnadapt | −0.77 / +3.08 | −0.79 / +1.87 |
  | isa | −0.18 / +2.60 | −0.40 / **+3.20** |
  | tent (BN 고정) | −0.61 / −1.19 | −0.92 / −1.96 |
  | sar (BN 고정) | −0.49 / −0.71 | −0.83 / −1.14 |
  | cotta (강도 3·5, 같은 범위 비교) | +0.11 / +1.02 | +0.20 / +0.14 |

  - 이득은 "통계 맞추기"(BN test 통계, 입력 통계)에서 나오고, 엔트로피만 최소화(BN 고정 Tent, SAR)는 −1~−2 (ACDC 검증과 같은 결론).
  - 왜곡별로 보완적: ISA는 BC·haze(ScaleMAE 55.5→64.6, 52.1→64.6), BN 통계는 noise·JPEG(GNoise 39.4→49.1, S&P 35.6→43.4).
    → ISA + BN-adapt 조합이 다음 후보.
  - CoTTA ≈ none: mild에서 teacher가 거의 갱신되지 않음(분류 §3.3b와 같은 현상).

### 5.5 정렬 위치 sweep (MIM, 2026-10-22, `scripts/run_depth_sweep.sh`)
- 방법 `fsa`(`src/reobench_tta/tta/stat_align.py`): 한 위치의 활성값을 test 배치 통계로 채널별 표준화 후 소스(clean train) 통계로 되돌림. 학습 없음.
  위치: 입력(=ISA), 블록 0/3/7/11/15/19/23 출력, 분류는 최종 norm·head 입력, 분할은 분류기(conv_seg) 입력·디코더 BN(=bnadapt).
- 분류: AID PN, 강도 1~5. 분할: Potsdam B, 강도 3·5. 결과 `experiments/2026-10-20_seg_tta/summary_depth_sweep.txt`.
- **none 대비 Δ (평균 / radiometric / degradation)**:

  | 위치 | 분류 SatMAE | 분류 ScaleMAE | 분할 ScaleMAE B | 분할 SatMAE B |
  |---|---|---|---|---|
  | 입력 (ISA) | +1.7 / **+11.1** / +0.1 | +1.3 / **+8.4** / −0.3 | +4.9 / **+19.1** / +0.5 | **+5.8** / **+25.4** / −0.7 |
  | 블록 0 | −0.3 / −7.5 / +2.5 | −6.0 / −10.8 / −9.9 | +3.9 / +7.5 / +3.4 | +3.6 / +13.0 / +0.6 |
  | 블록 3 | +0.2 / −7.3 / +3.4 | −0.8 / −6.3 / +0.2 | +7.8 / +9.4 / +8.9 | +4.9 / +14.4 / +2.2 |
  | 블록 7 | +4.9 / −2.1 / **+12.0** | +1.0 / −4.5 / +3.9 | **+8.0** / +9.3 / **+9.3** | +4.5 / +13.3 / +2.1 |
  | 블록 11 | +5.0 / −0.0 / +11.0 | −0.1 / −5.2 / +1.2 | +6.1 / +7.8 / +6.7 | +5.3 / +11.9 / **+3.8** |
  | 블록 15 | +0.7 / −1.4 / +2.3 | −1.0 / −5.0 / −0.9 | +4.6 / +7.1 / +4.5 | +3.5 / +7.5 / +2.8 |
  | 블록 19 | −0.9 / −2.1 / −0.7 | −2.4 / −4.8 / −3.9 | +2.2 / +4.2 / +1.9 | +1.6 / +3.0 / +1.3 |
  | 블록 23 | −0.5 / −2.4 / −0.3 | −2.9 / −5.2 / −4.3 | +2.7 / +3.5 / +3.0 | +1.6 / +2.8 / +1.5 |
  | 최종 norm | −0.1 / −1.7 / +0.3 | −4.7 / −8.7 / −6.6 | | |
  | head / 분류기 입력 | **+5.2** / +6.0 / +8.5 | **+8.0** / +5.6 / **+13.7** | +4.2 / +4.8 / +5.0 | +2.6 / +4.4 / +2.5 |
  | 디코더 BN | | | +4.5 / +5.8 / +5.0 | +3.0 / +5.5 / +2.7 |

  clean 변화는 모든 위치에서 −1.1 ~ +0.2. occlusion은 분류 head 입력(+3.6 / +7.4) 외에는 ±1.5 이내.
- 해석:
  - **radiometric은 입력 정렬이 최선** (4개 설정 모두). 분류에서는 블록 정렬이 오히려 radiometric을 해침(−2 ~ −11).
  - **degradation은 중간 블록(3~11) 정렬이 최선**. 분할 ScaleMAE 블록 7 +9.3은 디코더 BN(+5.0)보다 큼.
    분류에서는 SatMAE 블록 7·11(+12/+11), ScaleMAE는 head 입력(+13.7).
  - 깊은 블록(19~23)·최종 norm은 효과 작거나 손해. 분류 head 입력은 예외로 평균 최선(+5.2/+8.0): 선형 head 앞 정렬 = "분류기용 BN".
  - → §4.2 가설(affine radiometric = 입력, 비 affine degradation = 특징 수준)을 위치 sweep으로 확인. 최적 깊이는 블록 7~11(24층 중 중간).
  - 다음 후보: 입력 + 중간 블록 동시 정렬(ISA + fsa 블록 7). 위치는 검증 세트로 고정해야 함(지금 표는 test에서 고른 것이라 상한).

## 6. 할 일
- [x] MIM 정렬 위치 sweep (§5.5) → 다음: 입력 + 중간 블록 조합, CLIP 2종 sweep
- [ ] 결정 1~4 (§0)
- [ ] REOBench 분할 baseline 재현(Potsdam) 및 정규화·체크포인트 점검
- [ ] ViDA 분할 구현 (CoTTA 분할 코드 + adapter)
- [ ] CLIP 2종 분류 baseline (MIM과 같은 프로토콜)
- [ ] seed 추가 (현재 TTA는 seed 0만)
- [ ] 왜곡 자동 판별 + 그룹별 처방, occlusion용 MAE 마스킹 실험

## 부록 A. 재현성 이슈 (REOBench issue #4)
- 원 논문 AID: SatMAE 72.05 / ScaleMAE 75.75 clean. 공개 코드 기준 재현: 90.7±0.2 / 91.4±0.3 (A0ep, 3 seeds).
- 코드의 SatMAE 체크포인트 `pretrain-vit-large-e199.pth` = Sentinel(`args.dataset_type='sentinel'`, grouped patch_embed) → RGB ViT에서 patch_embed 랜덤 → clean 18.2.
- Scale-MAE는 체크포인트 정상 로드(L52 assert 통과)인데도 15%p 차이. 원인 후보 `data_type/*.pth`는 저자 측에서 분실(10-15 답변).
- 두 원 repo 모두 사전학습·평가에서 입력 정규화 → 정규화 추가 시 corruption 평균 +7.
- 저자 답변(10-15): "data.pth는 서버 이전 중 분실, 공개 AID로 재현 가능. baseline으로 쓸 거면 당신 결과를 쓰라." → 우리 재현 수치를 baseline으로 사용.
- 근거: fork `SunwooKim11/REOBench` 브랜치 `repro_aid` (`repro_aid/README.md`, `issue_check.py`).

## 부록 B. 코드 수정 이력 (영향 있는 것만)
- `params.py`: ViT-B 하드코딩 → 상위 25% 블록 제외 일반화; ResNet은 layer4 제외(원본 규칙).
- `tta.py`: CoTTA 메모리 누수 수정(스트림마다 ViT-L 3개 복사본 누적 → OOM), 저장 경로를 저장 시점에 재계산(폴더 rename 대비),
  옵션 `--lr`(`_lr`), `--cotta_ap`(`_ap`), `--sar_reset`(`_rst`).
- 로그 rc 버그: `$(date)`가 `$?`를 덮어써 실패가 rc=0으로 찍히던 문제 수정. `temp→repro` 이름 변경으로 저장 실패한 4개 결과는 `recover_from_log.py`로 복구.
