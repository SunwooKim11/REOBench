# 데이터셋 준비 현황 (2026-11-16)

선정 기준: (i) 선행 RS 강건성 연구(REOBench, Beyond Accuracy, EarthShift)에서 쓴 데이터, (ii) 평가 백본과 같은 고해상도 RGB,
(iii) 합성 손상과 실제 shift를 둘 다 포함, (iv) 사전학습 데이터(fMoW)는 테스트에서 제외.

## 1. 우선순위별 상태

| 단계 | 데이터 | 용도 | 상태 | 위치 |
|---|---|---|---|---|
| 1 | AID-C | 분류 합성 손상 | 있음 (REOBench) | `/data/rs/reobench/AID` |
| 1 | RESISC45-C | 분류 합성 손상 | **생성** (test 6,300장 × 12종 × 5강도) | `/data/rs/resisc45/RESISC45-C` |
| 1 | Potsdam-C | 분할 합성 손상 | 8종 있음, 4종 이미지 없음 (§4) | `/data/rs/reobench/Potsdam_extracted` |
| 2 | RESISC45 ↔ UCM | 분류 scale shift | **split 생성** (EarthShift 19클래스) | `/data/rs/{resisc45,ucmerced}/earthshift19` |
| 2 | Potsdam → Vaihingen | 분할 센서·도시 shift | 있음 (mmseg 패치) | `/data/rs/seg/isprs/Vaihingen/derived/mmseg` |
| 3 | LoveDA Urban → Rural | 분할 UDA | 있음 | `/data/rs/loveda_uda`, `/data/rs/seg/loveda` |
| 3 | DeepGlobe → DFC2022 | 분할 data source shift | DeepGlobe만 (비공식 HF parquet, 803장) | `/data/rs/seg/deepglobe_landcover` |

다운로드: `scripts/tools/download_extra_rs.sh` (torchgeo HF 미러).

## 2. RESISC45-C (`scripts/tools/make_corrupted_cls.py`)
- REOBench 손상 코드(`~/REOBench/Corrupt Image Generation/`)를 그대로 실행하되, **AID-C 실제 파일과 맞춰 본 결과 공개 코드와 다른
  파라미터 4개를 AID-C 쪽으로 고침** (AID 테스트 이미지 8~10장, JPEG 재저장 후 MAE 비교):

| 손상 | 공개 코드 | AID-C에서 역추정 (= 사용값) | MAE (공개값 → 사용값) |
|---|---|---|---|
| motion_blur 커널 길이 | 2, 4, 6, 8, 10 | 10, 15, 20, 25, 30 | 7.4~8.4 → 1.4~1.9 |
| compression JPEG quality | 30, 25, 20, 15, 10 | 50, 40, 30, 20, 10 | 5.2 → 1.1 (s1) |
| data_gaps 줄무늬 폭 | 3~7 | 5~9 (줄 수 2~6은 같음) | 검은 면적 0.117 → 0.150 (AID-C 0.157, s5) |
| scale_image | 검은 캔버스 | 축소 후 reflect 패딩 (AID-C엔 검은 픽셀 없음) | 7.7 → 4.5 (s5) |

- 공개값과 일치: gaussian blur, haze, brightness/contrast, rotate, translate(4방향 중 최적), gaussian noise σ
  (AID-C noise std 0.028/0.056은 JPEG 저장 효과로 재현됨).
- **clouds는 미일치**: AID-C 구름 면적(추정 0.013~0.129)이 공개 density(추정 0.04~0.22)보다 작음. 원인 못 찾아 공개값 유지.
- 모든 출력은 AID-C처럼 PIL 기본 JPEG(q75). (이미지, 손상, 강도)마다 고정 seed → 재현 가능 (공개 코드는 seed 없음).
- 출력 구조는 AID-C와 같음: `<Class>/<corruption>/severity_<s>/<class idx>/<file>.jpg`.
- 환경: `~/REOBench/.venv-corrupt` (segtta + scikit-image 0.21 + conda-forge `noise`; 서버에 gcc 없음).

## 3. RESISC45 ↔ UCM (`scripts/tools/make_rs_splits.py`)
- 클래스 매핑 = EarthShift 공식 코드 `configs/scale-shift-exp.json` (github.com/kerner-lab/earthshift). 19개 1:1, 병합 없음.
  논문은 "highway"라고 썼지만 코드는 양쪽 모두 freeway. 라벨 = 알파벳 순서(양쪽 같음).
- RESISC45 (torchgeo split을 19클래스로 거름): train 7,915 / val 2,726 / test 2,659 = **EarthShift 표 1과 정확히 일치**.
- UCM: train 1,142 / val 379 / test 379 / 전체 1,900. 논문 표의 UCM 숫자(420/420/1,260)는 어떤 필터 split과도 안 맞음
  → OOD 평가는 전체 1,900장(또는 test 379장)으로 두 가지 다 제공.
- 해상도: RESISC45 0.2~30m, UCM 0.3m. 둘 다 256px (UCM 일부 tif는 크기 확인 필요).

## 4. Potsdam-C 누락 4종 (rotate, scale, translate, data gaps)
- REOBench 배포본에 **라벨은 있음**(`anns/{rotate,scale_image,translate_image,create_data_gaps}`, 2,016장 × 5강도),
  **이미지만 없음** (`Potsdam_Images_test.zip`에 9종만).
- 그런데 기하 손상 라벨이 **선형 보간됨**: rotate/scale 라벨에 0~4가 아닌 값(39, 127 등)이 섞여 있고 3채널 PNG.
  → 배포 라벨을 그대로 쓰면 평가가 오염됨. 이미지와 라벨을 우리가 nearest 보간으로 직접 만들어야 함 (translate는 방향이 랜덤이라
  배포 라벨에서 방향을 역추정하거나 새로 생성).
- data gaps 라벨은 clean과 99.6% 같음(가림은 라벨을 안 바꿈) → 이미지만 만들면 됨.

## 5. Potsdam → Vaihingen
- Vaihingen은 IRRG(근적외-적-녹)만 있음. 우리 Potsdam 모델은 RGB로 학습 → RGB→IRRG 센서 shift가 같이 들어감
  (RS UDA 문헌의 P(RGB)→V(IRRG) 설정). Potsdam 5cm vs Vaihingen 9cm.
- mmseg 패치: train 344 / val 398 (512px). 라벨 1~6 (0 = 경계 무시, reduce_zero_label 형식) → Potsdam REOBench 라벨
  (0~4, 5 = clutter 무시, 255 무시)과 매핑 필요.
