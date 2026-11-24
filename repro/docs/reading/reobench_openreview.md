# REOBench OpenReview 리뷰 정리 (forum NfeA0waFaE)

- 출처: https://openreview.net/forum?id=NfeA0waFaE. 사용자가 Claude in Chrome으로 노트 28개를 읽고 정리한 것을 2026-11-16 전달받아 그대로 저장
  (서버에서는 OpenReview가 자동 접근을 막아 직접 확인하지 못함). 아래 "원문 정리"는 전달된 텍스트 그대로.

---

## 원문 정리 (사용자 전달)

REOBench는 **Accept (poster)**였어요. 리뷰어들이 가장 많이 짚은 건 **"합성 손상이 현실적이냐, 실제 shift는 왜 없냐"**였고, 이건 끝까지 해결되지 않았어요. TTA나 적응 기법은 리뷰어와 저자 모두 한 번도 언급하지 않았습니다.

Chrome에서 포럼을 열고 리뷰, rebuttal, 후속 코멘트, decision까지 노트 28개를 전부 읽었어요. 각 리뷰어의 초기 점수는 남아 있지 않아서 최종 점수만 확인했습니다.

### 결정과 점수

| 리뷰어 | 최종 rating | confidence | 점수 변화 |
|---|---|---|---|
| h8bd | 4 | 4 | 올림 |
| yhui | 5 | 4 | 올림 |
| FNsA | 5 | 4 | 언급 없음 |
| RXmf | 4 | 2 | 3 → 4 |

**decision에서 최종판에 요구한 것:**
- *"strengthen ecological validity by adding real-world and compound corruptions"*
- 실패 사례와 해석 분석
- 데이터셋 선택 근거
- 손상 생성 스크립트 공개

### 지적과 답변

**1. 합성 손상의 현실성, 실제 shift 부재** (h8bd, yhui, decision)
- **답변:** 라벨이 달린 실제 데이터를 만들기 어렵고, ImageNet-C도 같은 관례라고 했어요.
- 대신 FID만 제시했어요.
  - clean 대비 손상: severity 1 → 5에서 DIOR 2.36 → 19.81, AID 14.07 → 49.03
  - 실제 데이터 대비: 구름은 SEN12MS-CR 기준 평균 119.86, haze는 RRSHID 기준 90.56
- **h8bd가 끝까지 미해결로 남김:** *"direct evidence of cross-domain generalization (e.g., training on one domain and testing on another) is still missing"*

**2. 구름과 haze가 너무 단순하다** (h8bd: *"resembling a gray overlay"*)
- **답변:**
  - 구름: Perlin noise로 생성. severity는 구름 덮인 비율로 조절
  - haze: 균일한 흰 레이어를 선형으로 섞음 (α 0.2–0.6)
  - 실제 구름 마스크 대신 이렇게 한 이유로 재현성과 제어 가능성을 들었어요.

**3. 여러 손상이 겹친 경우가 없다** (yhui, decision)
- **답변:** DIOR-R 검출에서만 일부 조합을 실험했어요. RemoteCLIP 기준 clean 60.40, 단일 손상 약 56, 세 가지 동시 적용 40.58입니다.

**4. 고해상도 RGB만 다뤄서 RS 고유성이 부족하다** (FNsA, RXmf)
- FNsA: ImageNet-C와 결론이 같을 수 있다. *"the conclusions could be the same"*
- **답변:** "EO에서 첫 체계적 평가"라는 한 줄뿐이었어요. 다중분광 예비 실험 2건을 추가했습니다.
  - SatMAE (fMoW-S2): 하락폭 16.32
  - EarthDial (BigEarthNet): 하락폭 10.02

**5. 왜 실패하는지 분석이 없다** (yhui, decision)
- **답변:** 개정판에 attention drift 같은 분석을 넣겠다고 약속만 했어요.

**6. MIM이 왜 더 약하냐** (h8bd)
- **답변:** 사전학습 규모 차이로 설명했어요. MIM은 fMoW 36만 장, CLIP 계열은 수십억 장입니다.
- AID 상대 하락폭: MIM 17–35%, CL 5–9%, VLM 5–7%

**7. 재현성** (h8bd)
- **답변:** 공식 split을 쓰고, 증강은 flip과 crop뿐이며, 손상 생성 코드는 공개하겠다고 했어요.
- 실제로 공개됐는지 저장소에서 확인해 보세요. 재현하며 겪은 어려움과 관련이 있을 수 있어요.

### 토론에 없었던 것
- **TTA, 적응, 방어 기법:** 아무도 언급하지 않았어요.
- **frozen vs fine-tune 프로토콜:** 논의되지 않았어요. 저자가 말한 *"zero-shot evaluation protocol"*은 "해당 데이터셋으로 사전학습하지 않았다"는 뜻으로 읽혀요.

### 선우 연구에 주는 의미 (전달 원문)
1. **실제 shift 평가:** h8bd가 끝까지 요구한 cross-domain 실험. RESISC45 ↔ UCM, Potsdam → Vaihingen 구성이 여기에 대한 답.
2. **RS 고유성 논증:** FNsA의 "ImageNet-C와 같은 결론일 수 있다"는 지적 = 교수님 피드백 F2. latent 분석으로 답하면 강점.
3. **MIM 열세 원인 검증:** 저자는 "사전학습 규모" 하나로만 설명. 구조는 같고 사전학습만 다른 쌍으로 직접 검증 가능.
4. **실패 원인 분석:** 약속만 하고 넘어간 부분 → TTA 전후 feature 변화 분석으로 채울 수 있음.

---

## Claude 보충 (2026-11-16)
- 7번 재현성: 손상 생성 코드는 공개됐지만 **공개 코드와 AID-C 데이터의 파라미터가 4개 다름**(motion blur, JPEG, data gaps, scale;
  SETTINGS.md §2.5), Potsdam 기하 손상 4종은 이미지가 빠지고 라벨은 선형 보간됨(§4), issue #4 data.pth 누락. 재현 어려움의 직접 원인.
- 6번 "MIM 열세 = 사전학습 규모": 우리 백본 짝(MAE vs SatMAE, 둘 다 ViT-L, 데이터 규모 ImageNet 1.28M vs fMoW 0.36M; DINOv3 web vs sat는
  둘 다 수억 장)으로 규모·도메인·목적함수를 나눠 볼 수 있음.
- 1번 실제 haze: 리뷰 답변에 쓰인 **RRSHID(실제 haze/clear 짝 영상)가 서버에 있음** (`/data/rs/RRSHID`, thin/moderate/thick fog, train/val/test,
  hazy/gt, 256px, 총 6,108장). 클래스 라벨은 없지만 같은 장면의 clear/hazy 짝이라 **합성 haze와 실제 haze의 feature drift를 직접 비교** 가능
  (공통 이동 비율, CKA, 예측 일관성). "합성 손상이 현실적인가"에 FID보다 직접적인 답.
- 다중분광(4번): 우리 백본이 RGB라 범위 밖. 한계로 명시하거나, SatMAE fMoW-Sentinel 체크포인트(보유)로 소규모 확인 가능.
