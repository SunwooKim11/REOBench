# feature 분포 그림 읽는 법 (2026-11-24 그림, 2026-11-26 설명 작성)

스크립트: `scripts/analysis/plot_feature_dist.py`. 수치: `experiments/2026-11-24_cluster_cot/RESULTS.md`.

공통 사항
- 점 하나 = AID clean test 이미지 한 장(2,000장)의 head 입력 z = (h − μ_src) / σ_src. h는 encoder 출력(CLS 등), μ_src, σ_src는 clean train 통계.
- 주거지 세 클래스(Dense / Medium / SparseResidential)만 색을 칠하고 나머지 27개 클래스는 회색이다. 서로 헷갈리기 쉬운 클래스를 골랐다.
- 행 = 백본 4개. satmae와 moco_in은 k-means 정확도가 가장 낮은 쪽(44), satdino_s8과 dinov3_web_l16은 가장 높은 쪽(80–81)이다.
- z는 384–2,048차원이라 2차원으로 내려서 그린다. 어느 2차원을 고르느냐에 따라 보이는 것이 달라서, 같은 점을 세 가지 방법으로 내렸다.

## fig1_clean_geometry.png: clean feature의 모양

행 이름 아래 숫자 세 개는 같은 clean test feature를 세 가지 방법으로 분류한 정확도다.

| 숫자 | 어떻게 분류하나 | 무엇을 재나 |
|---|---|---|
| linear | 학습한 선형 head (z W + b) | 클래스 사이에 평면 경계를 그을 수 있나 (선형 분리) |
| 5-NN | 가장 가까운 train 이미지 5장의 다수결 라벨 | 가까운 이웃이 같은 클래스인가 (국소 일관성) |
| k-means | 라벨 없이 30개 군집으로 나눈 뒤, 군집과 클래스를 가장 잘 맞게 짝지어 센 정확도 | 클래스마다 한 덩어리로 뭉쳐 있나 |

세 열(같은 점을 다르게 내린 그림):

| 열 | 축을 어떻게 골랐나 | 보는 것 |
|---|---|---|
| PCA | 2,000개 점 전체에서 분산이 가장 큰 두 방향. 라벨을 안 씀 | "feature가 가장 크게 퍼진 방향"에서는 세 클래스가 겹친다. 큰 분산은 클래스와 상관없는 변화(밝기, 장면 구성 등)가 차지할 수 있다 |
| plane through the 3 class means | 세 클래스의 train 평균 세 점을 지나는 평면. 라벨을 씀 | 세 클래스를 가장 잘 떼어 보이는 2차원. 여기서는 세 색이 갈라진다 = 이 세 클래스를 가르는 방향이 feature 안에 있다 |
| t-SNE | 각 점의 가까운 이웃 관계를 최대한 유지하도록 2차원에 배치(PCA 50차원 위에서, perplexity 30). 거리와 축의 값에는 뜻이 없다 | 같은 클래스의 이미지끼리 이웃인가. 같은 색이 작은 섬으로 모이면 이웃이 같은 클래스라는 뜻 |

- PCA의 "주성분"은 채널 하나가 아니라 채널들의 가중합인 방향이다.
- t-SNE의 "neighbourhood"는 한 점 주변의 가까운 점들이고, 그 점들이 같은 클래스인지가 보는 대상이다. 다른 클래스가 가까이 붙어 있으면 섬이 붙거나 섞여 보인다.

제목 "linearly separable, locally consistent, not one blob per class"의 뜻:
- **linearly separable**: 선형 head가 93–98%. 평면 경계로 클래스를 거의 다 가를 수 있다(가운데 열).
- **locally consistent**: 5-NN이 81–96%. 가까운 이웃은 대개 같은 클래스다(t-SNE의 같은 색 섬).
- **not one blob per class**: k-means는 44–81%. 한 클래스가 한 덩어리로 뭉쳐 있지는 않다. satmae는 한 클래스가 여러 섬으로 쪼개져 있고(t-SNE의 파랑이 여러 곳),
  PCA에서는 셋이 겹친다. 그래서 "군집이 뚜렷하다"를 가정하는 방법(Entropy Enigma의 Silhouette 해석, k-means 기반 방법)은 이 feature에서 잘 맞지 않는다.

## fig2_shift_align_motion_blur_s3.png: 손상이 feature를 어떻게 옮기고, head 입력 표준화가 무엇을 되돌리나

열: clean / motion blur 3단계 / 같은 손상 stream을 head 입력 표준화(배치 64)한 뒤. 제목의 accuracy는 그 상태의 분류 정확도다.

축(한 행의 세 그림이 같은 축을 쓴다):
- **x축 "direction of the common shift"**: 손상 stream의 평균 z에서 clean의 평균 z를 뺀 벡터 방향. 손상이 모든 이미지를 함께 민 방향이다.
  값은 clean 평균에서 그 방향으로 얼마나 떨어졌나이고, 단위는 z의 단위(source 표준편차)다. 0 = clean 평균(점선).
- **y축 "top clean direction"**: clean feature에서 분산이 가장 큰 방향(PCA 1번)을 x축과 직교하게 만든 것. 클래스들이 퍼져 있는 모양을 보려고 고른 두 번째 축이다.
- 삼각형 = 그 stream의 평균 x 위치.

읽는 법:
- 가운데 열: 전체가 오른쪽으로 통째로 이동한다(satmae 약 19, moco_in 약 150, satdino_s8 약 15, dinov3_web 약 26 단위). 클래스 사이의 배치는 어느 정도 남지만
  head의 경계는 clean 위치에 있으므로 정확도가 떨어진다(satmae 92.7 → 29.6, moco_in 93.3 → 10.7).
- 오른쪽 열: 표준화가 평균을 0으로 되돌린다(삼각형이 점선 위). 정확도가 오르지만(29.6 → 50.7, 10.7 → 60.9) clean까지는 못 간다.
  평균은 돌아와도 점들이 퍼진 모양과 클래스 사이의 섞임은 남기 때문이다. dinov3_web은 손상에도 클래스 배치가 유지되어 원래 덜 떨어진다(83.0).
- 이 그림은 2차원 단면이다. 표준화는 모든 채널의 평균과 표준편차를 맞추고, 그림은 그중 이동 방향 하나만 보여 준다.
