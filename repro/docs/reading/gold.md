# GOLD (CVPR 2026) — "head가 읽는 방향"을 먼저 정식화한 선행 연구

Lai, Zhou, Li, Ye. "The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation",
CVPR 2026 (arXiv 2603.21928). 코드: github.com/AIGNLAI/GOLD.
출처: Claude가 PDF(`papers/TTA/`) 본문 8쪽의 **텍스트**를 직접 읽음(2026-11-18). 그림은 캡션만, 표 1·3의 개별 수치와 부록은 읽지 않음.

## 1. 논문이 말하는 것

**문제**: continual TTA는 파라미터를 많이 고칠수록 적응은 잘 되지만 느려지고, pseudo-label 잡음과 파라미터 표류가 쌓인다.
"출력을 원하는 만큼 바꾸면서 feature는 최소로 고치는 부분공간"이 있는가.

**(1) golden subspace = 분류기 W의 row space (§3.2)**
- 설정: feature `F`(배치 × L차원) 위에 고정된 선형 분류기 `W`(C × L). 출력을 `ΔY`만큼 바꾸고 싶을 때 feature를 가장 적게
  움직이는 해는 `ΔF* = ΔY (Wᵀ)†` (pseudo-inverse, 최소 노름 해).
- 따라서 `rank(ΔF*) ≤ rank(WᵀW) ≤ 클래스 수`. 출력을 바꾸는 데 필요한 feature 이동은 `span(Wᵀ)` 안에 있다.
- 원문: "The classifier implicitly defines a set of directions in the feature space to which the output is most sensitive."
- **우리가 "head가 읽는 방향"이라고 부른 것과 같은 공간이다.** (우리는 클래스 간 차이만 남겨 C−1차원, 이 논문은 C차원.)

**(2) 그 부분공간을 test 때 갱신: AGOP (§3.3, §4.3)**
- `G_0 = WᵀW`로 시작. confidence가 임계값 τ 이상인 test 샘플에 대해 "가장 큰 logit의 feature에 대한 gradient" `g_i`를 구해
  `G_t = (1−α) G_{t−1} + α · mean(g_i g_iᵀ)`로 갱신(AGOP = Average Gradient Outer Product).
- 주기적으로 `G_t`를 고유분해해 상위 r개 고유벡터 `V_t`를 쓴다. 모든 실험에서 r = 64.
- Claude의 주석: 분류기가 선형이면 `g_i`는 예측된 클래스의 `w_c`이므로 `G_t`는 "confident하게 예측된 클래스의 w_c를 더 무겁게 센 WᵀW"가 된다.
  클래스가 10개인 CIFAR-10에서 r = 64가 어떻게 채워지는지는 본문만으로는 알 수 없다(부록·코드 확인 필요).

**(3) 방법 GOLD (§4.2, §4.4)**
- 백본은 고정. feature를 부분공간에 사영하고 좌표마다 scale만 학습: `f' = f + V_t (S_t ⊙ V_tᵀ f)`, 학습 파라미터는 `S_t`(r개).
- 손실: EMA teacher와의 self-training(SCE, 원본 + 증강 view) + **source 클래스 prototype**(클래스별 평균 feature)과의 contrastive loss.
  source 샘플은 안 쓰지만 prototype은 배포 전에 계산해 보관한다.
- 실험: CIFAR10-C(WideResNet-28), CIFAR100-C(ResNeXt-29), ImageNet-C(ResNet-50), severity 5, 15종을 이어 붙인 continual 설정.
  분할은 CarlaTTA. source 데이터로 end-to-end 학습한 모델이다.
- ablation (오류율 %, CIFAR10-C / CIFAR100-C / ImageNet-C): 부분공간 없음 16.24 / 29.87 / 62.45 → WᵀW 부분공간 15.11 / 29.15 / 60.24
  → AGOP + contrastive 14.12 / 28.56 / 59.32. 부분공간 제한 자체의 이득은 0.7–2.2pt.

## 2. 우리와의 관계

| | GOLD | 우리 |
|---|---|---|
| 부분공간 | span(Wᵀ), AGOP로 갱신, r = 64 | W의 클래스 차이 방향, C−1차원, 고정 |
| 부분공간의 쓰임 | **적응을 그 안으로 제한**(scale 학습) | **이동을 그 안팎으로 분해해 측정**(공통 / 표본별 × 안 / 밖) |
| 무엇을 고치나 | 좌표별 scale (곱) | 평균 이동(빼기)과 채널별 표준편차 |
| confidence 의존 | 있음(고신뢰 샘플로 AGOP, pseudo-label self-training) | 없음(통계 정렬, flip) |
| source 정보 | 클래스 prototype | feature 평균·표준편차 |
| 모델 | source로 end-to-end 학습한 CNN | frozen foundation model + 선형 head |
| 측정 | 정확도, 속도 | 정확도, ECE, 정확도 추정, 이동 에너지 분해 |

**주장할 수 없게 된 것**
- "분류기가 정하는 저차원 부분공간만 출력에 영향을 준다 / 그 안에서 적응하면 된다"는 생각. GOLD가 정식화했다.

**남는 것**
- 손상이 만든 이동의 **몇 %가 그 부분공간 안에 있는지**의 측정(우리: 6–17%). GOLD는 재지 않았다.
- 그 안의 이동을 공통(= bias 변화)과 표본별(= logit 잡음)로 나누고 과신·정확도 하락과 연결한 것.
- 과신 백본에서 confidence 기반 선택이 실패한다는 것. GOLD는 고신뢰 샘플에 의존하므로 mae_in·MoCo 같은 백본에서 약점이 될 수 있다(미검증).
- frozen EO foundation model.

**해야 할 것**
1. GOLD를 baseline으로 넣기(코드 공개). 특히 과신 백본에서.
2. "부분공간 안에서만 표준편차 정렬"을 시험할 때는 GOLD의 scale 벡터와의 관계를 명시: GOLD는 학습으로, 우리는 source 통계에서 닫힌 식으로.
3. GOLD는 평균 이동(bias)을 직접 고치지 않는다. 우리 결과에서는 평균 정렬이 이득의 대부분이므로, GOLD에 평균 정렬을 더했을 때의 변화도 볼 만하다.
