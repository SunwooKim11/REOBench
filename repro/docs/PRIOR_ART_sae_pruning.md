# Prior art: SAE 분석 + pruning으로 shift 완화 (survey by agent, 2026-11-17)

아이디어: encoder 내부 activation에 sparse autoencoder(SAE)를 학습 → 데이터셋 1·2·3(clean / 손상 / 다른 데이터셋)에서 켜지는
feature를 비교 → shift에 민감하거나 쓸모없는 단위를 지워(pruning) 정확도 하락을 줄인다.
출발점: Deletion Tuning (Lu et al., "The Surprising Effectiveness of Deleting Weights in LLM Post-Training", NeurIPS 2026, arXiv 2611.10866).
조사: 웹 검색 agent 1회 + 논문 페이지 요약 agent 1회. 논문 존재는 agent가 초록·본문 페이지로 확인했다고 보고한 것이고 Claude가 원문을
직접 읽지는 않았다. 인용 전 원문 확인 필요.

## 결론

- **LLM은 필요 없다.** 순수 이미지 encoder에 SAE를 학습하고 해석하는 연구가 이미 많다. 언어 모델은 feature에 이름을 붙이는 편의 단계에만 쓰인다.
- **이 아이디어 그대로는 선행 연구가 없다.** "clean과 손상 사이의 SAE feature 차이를 보고 pruning해서 손상 정확도를 회복"은 EO든 일반 vision이든 못 찾음.
- **다만 효과가 약할 것이라는 근거가 여럿 있다.** 가장 가까운 성공 사례는 spurious correlation(최악 그룹 정확도) 교정이지 손상이 아니다.
  test 시점에 단위를 자르거나 눌러서 ViT·frozen FM의 손상 정확도를 올린 연구는 못 찾음.
- **LLM이 붙은 EO 모델로 바꾸는 것은 답이 아니다.** 거의 전부 일반 CLIP ViT-L 위에 얹은 것이라, 분석 대상이 우리 백본이 아니게 된다.

## Deletion Tuning이 실제로 하는 것

- LLM(Llama 1B–8B, Qwen 0.5B–30B)의 MLP gate projection 가중치마다 학습 가능한 값 하나를 두고, 과제 loss(RL 보상, SFT, distillation)로
  학습해 0.05% 미만의 가중치를 0으로 만든다. 가중치 자체는 고정. 초기값은 Wanda 점수.
- SAE 없음. 해석으로 고르는 것이 아니라 gradient로 고른다. 라벨(또는 보상)이 필요하다.
- 결과: 전체 fine-tuning과 비슷한 성능, 마스크 크기는 수백–수천 배 작음, 이전 과제 망각이 훨씬 적음.
- vision·VLM·분포 shift 실험은 없음(요약 기준).
- 우리 쪽으로 옮기면 "마스크만 라벨 없이 학습하는 TTA"가 된다. 라벨 없는 loss를 무엇으로 할지가 문제.

## A. 순수 vision encoder의 SAE

| 연구 | encoder, 층, token | 데이터 / dictionary | feature 해석 방법 |
|---|---|---|---|
| Prisma (arXiv 2504.19475, github Prisma-Multimodal/ViT-Prisma) | CLIP ViT-B/32 전 층, DINO-B | ImageNet-1k / 49,152 | 최상위 활성 이미지. SAE 80개 이상 공개 |
| Stevens et al., saev (arXiv 2502.06755) | CLIP ViT-B/16, DINOv2 ViT-B/14, 마지막 근처 층, patch | ImageNet 120만 장에서 patch activation 약 1억 개 / 24,576 | 사람이 최상위 patch 확인 |
| PatchSAE (ICLR 2025, arXiv 2412.05276) | CLIP ViT-B/16 layer 11, CLS + patch | ImageNet / 49,152 | 최상위 이미지, 라벨 엔트로피 |
| Steering CLIP's ViT (arXiv 2504.08729) | CLIP ViT-B/32 전 층 | ImageNet-1k / 49,152 | CLIP text 정렬 |
| Discover-then-Name (ECCV 2024, arXiv 2407.14499) | CLIP 최종 image embedding | CC3M | 가장 가까운 CLIP text embedding |
| Universal SAE (ICML 2025, arXiv 2502.03714) | DINOv2, SigLIP, supervised ViT 마지막 층 patch | ImageNet / 6,144 | activation maximisation |
| Archetypal SAE (ICML 2025, arXiv 2502.12892) | DINOv2, ConvNeXt, ViT, SigLIP, ResNet | — | 표준 SAE가 seed마다 불안정함을 보임 |
| Into the Rabbit Hull (ICLR 2026, arXiv 2510.08638) | DINOv2-B, patch·CLS·register | ImageNet 140만 장 / 32,000 | 시각적. 선형 head에 대한 개념 중요도 = dictionary × head 가중치 |
| From Activation to Intervention (arXiv 2611.05522) | DINOv3, SigLIP 2 ViT-L/16 | token 1억 개(ImageNet), 2,500만 개(CUB, Cars), CLS 200만 개 / 16,384·4,096 | 픽셀로 되돌려 확인 |
| Olson et al. (ICCV 2025 Findings, arXiv 2508.11277) | DINOv2, CLIP, ResNet, CLS 마지막 층 | ImageNet-1k | — |

- MAE 계열 + SAE는 Euclid 은하 이미지 한 건뿐(arXiv 2510.23749). SatMAE, Scale-MAE: 없음.

## B. SAE feature를 강건성에 쓴 연구

- **손상 시 어떤 feature가 변하는지 분석**: 못 찾음.
- **spurious feature 제거로 정확도 개선**:
  - Sparse Feature Circuits / SHIFT (arXiv 2403.19647, LLM): Bias in Bios 정확도 61.9 → 88.5, 최악 그룹 24.4 → 76.0.
  - Steering CLIP's ViT: 최악 그룹 CelebA 77.8 → 81.1, Waterbirds 22.4 → 24.6.
  - DIAL (ICLR 2026), SEM (arXiv 2603.19028): CLIP에서 SAE로 spurious 방향을 찾아 사영으로 제거.
  - SAE 없이: Gandelsman et al. (arXiv 2310.05916) — CLIP의 위치 관련 head를 지워 Waterbirds 최악 그룹 47.7 → 72.9.
- **TTA / OOD 감지**: SAID (arXiv 2607.12094) — covariate shift 감지 AUROC 83.5(최고 baseline 81.0). 감지만.
  Olson et al. — SAE feature의 OOD 정확도는 엇갈림(DINOv2: ImageNet-A +3.6, ImageNet-R −8.8).
  **SAE 기반 TTA로 ImageNet-C 정확도를 올린 연구: 못 찾음.**
- **분류기 steering**: VS2 (arXiv 2506.01247) — zero-shot 정확도 최대 +4.12%, oracle 상한 +21.44%. SAE-FT (arXiv 2605.15961) — WiSE-FT와 비슷(64.6 vs 64.4).

## C. SAE 없이 "단위를 자르기 / 누르기 / 가리기"

- activation shaping (ASH, arXiv 2209.09858): OOD 감지용. SCALE (ICLR 2024)는 ASH의 pruning 부분이 해롭고 scaling만 도움이 된다고 보임.
- TTA에서 채널 pruning: sensitivity-guided pruning (CVPR 2025, arXiv 2506.02462) 하나뿐, 객체 검출 + gradient 적응. 분류·ViT 대응은 없음.
- pruning이 손상 강건성에 도움이 되는가:
  - 도움: Diffenderfer et al. (arXiv 2106.09129, CIFAR-10-C, CNN).
  - 해로움: Hooker et al. (arXiv 1911.05248, ImageNet-C에서 크게 감소).
  - 작음: dynamic sparse training (ICLR 2025, arXiv 2410.03030) — DeiT-B ImageNet-C +0.05–0.47pt, 학습 시점 sparsity.
- 주의: Crabbé et al. (TMLR, arXiv 2310.13040) — 튀는 feature 차원은 강건한 CLIP encoder에서만 나타남. 이상해 보이는 차원을 지우면 강건성의 원인을 지울 수 있음.
- **학습 없는 단위 pruning·clipping·masking으로 ViT·frozen FM의 손상 정확도를 올린 연구: 못 찾음.**

## D. 데이터셋 간 / 모델 간 SAE 비교

- PatchSAE: CLIP과 prompt 적응 모델을 11개 데이터셋에서 비교. 적응은 대체로 기존 개념을 재배치. EuroSAT만 feature가 많이 생기고 사라짐.
- Universal SAE: 공유 개념과 모델 고유 개념을 구분. 독립 학습한 SAE와 일치하는 개념은 23–38%.
- 일반 vs 도메인 사전학습: 병리(NeurIPS 2025) — 생물학적 feature는 병리 사전학습 모델에만 있음. 의료(MICCAI 2026, arXiv 2603.23794).
- vision crosscoder로 일반 vs 도메인 사전학습 ViT 비교: 못 찾음(Prisma가 crosscoder를 지원).

## E. Remote sensing

- **Decomposing Earth Embeddings with SAEs** (Benson et al., NeurIPS 2026, arXiv 2611.11398, github EarthyScience/geosae):
  AlphaEarth·Tessera의 픽셀 embedding에 SAE(패치 10만 개, dictionary 1,024). 참조 지도로 평가.
  **경고**: 고정된 무작위 decoder의 SAE가 학습한 SAE와 해석 가능성이 같고, 무작위 sparse code도 OOD 오차 순위에서 비슷한 이득. 저자들이 "아직 신뢰할 수 없다"고 함.
- Prithvi, Clay, SatMAE, Scale-MAE, DOFA, DINOv3-sat, RemoteCLIP, GeoRSCLIP, TerraMind에 SAE·내부 해석: 못 찾음. 강건성 활용: 못 찾음.
- encoder + LLM인 EO 모델:

| 모델 | vision encoder | LLM | REOBench 포함 |
|---|---|---|---|
| GeoChat | CLIP ViT-L/14 (frozen) | Vicuna-1.5 | O |
| LHRS-Bot | CLIP ViT-L/14 | LLaMA2-7B | O |
| RS-LLaVA | CLIP ViT-L | Vicuna-1.5 | O |
| SkySenseGPT | CLIP ViT-L/14-336 (frozen) | Vicuna-1.5 | O |
| SkyEyeGPT | EVA-CLIP ViT-g (frozen) | LLaMA2-chat | X |
| EarthGPT | DINOv2 ViT-L/14 + CLIP ConvNeXt-L (frozen) | LLaMA-2 | X |
| TEOChat | CLIP ViT-L/14 (frozen) | Llama 2 | X |
| EarthDial | InternViT-300M | Phi-3-mini | 다중분광만 |
| GeoPixel | CLIP ViT-L/14 + SAM-2 encoder (frozen) | InternLM2-7B | X |

  독립된 EO foundation model의 encoder를 쓰는 것은 없다. 이름 붙이기가 필요하면 RemoteCLIP·GeoRSCLIP의 text encoder로 충분(LLM 불필요).

## 실현 가능성

- **된 것**: CLIP·DINOv2/v3·SigLIP의 SAE와 공개 도구(Prisma, saev, geosae), 라벨 없는 해석, spurious 교정, OOD 감지.
- **안 된 것**: clean vs 손상 SAE feature 비교, pruning으로 손상 정확도 회복, 고해상도 광학 EO encoder의 SAE, MAE·ResNet 계열.
- **가장 큰 위험**: 손상은 "몇 개의 sparse한 feature가 더해진 것"이 아닐 가능성이 크다. SCALE, Hooker, sparse training의 1pt 미만 이득,
  VS2의 oracle 격차, geosae의 무작위 baseline이 모두 효과가 작다는 쪽을 가리킨다. REOBench의 하락은 25pt 이상이다.
- **데이터 규모**: 공개 SAE는 이미지 약 100만 장 또는 patch token 약 1억 개로 학습. AID의 CLS 벡터 1만 개로는 부족 →
  patch token, 작은 dictionary(4–8배), 또는 라벨 없는 RS 이미지를 모아야 함. CUB·Cars의 2,500만 token 사례는 작은 데이터도 가능함을 보임.
- **재구성 오차**: SAE를 끼우는 것만으로 정확도가 변함(Olson: ImageNet-R −8.8). decoder 방향만 빼고 잔차는 유지하는 방식으로 지워야 함.
- **백본 수**: 백본·층마다 SAE 하나, seed마다 dictionary가 불안정 → 모델 간 feature 비교가 취약. ResNet은 별도 설계 필요.
- **CLS vs patch**: 선형 head는 마지막 벡터 하나만 읽는다. 거기서 SAE feature를 지우는 것은 선형 사영과 같다. 그래서 SAE 없는 baseline
  (clean−손상 차이의 PCA, 차원별 민감도, clipping)이 필수이고, 그것으로 충분하면 SAE가 필요 없다.

## 가장 싼 검증 (SAE 없이, 저장된 feature로)

1. AID에서 SatMAE와 강건한 백본 하나를 고른다.
2. 일부 손상에서 clean−손상 쌍 차이로 "shift 방향"을 추정한다(PCA 상위 k개, 또는 민감한 차원 k개).
3. 그 방향을 사영으로 빼거나 가린 뒤, **추정에 쓰지 않은 손상**에서 고정된 head의 정확도를 잰다.
4. baseline: head 입력 재표준화(align ms), 무작위 방향 k개.
5. 이 oracle(clean 쌍을 쓰는 가장 유리한 조건)이 몇 pt라도 회복하면 SAE를 학습해 PCA·원래 뉴런·무작위 sparse code보다 나은지 본다.
   회복하지 못하면 SAE로 가도 얻을 것이 없다.

## 미확인

- DIAL 수치(Waterbirds 31.9 → 69.2)는 2차 요약에서만.
- CD-Buffer (CVPR 2026, TTA에서 "빼는" 채널 제거)는 2차 메모에서만.
- GeoChat·RS-LLaVA의 세부(encoder, frozen 여부, LLM)는 검색 snippet 기준.
- ReAct, DICE 원문은 열지 않음(SCALE 논문의 기술로 대신).
