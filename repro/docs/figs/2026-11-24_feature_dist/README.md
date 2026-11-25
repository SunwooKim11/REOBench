# feature 분포 그림 읽는 법 (2026-11-24 그림, 2026-11-26 설명 작성, 같은 날 fig3 추가와 해석 정정)

스크립트: `scripts/analysis/plot_feature_dist.py`. 수치: `experiments/2026-11-24_cluster_cot/RESULTS.md`.

공통 사항
- 점 하나 = AID clean test 이미지 한 장(2,000장)의 head 입력 z = (h − μ_src) / σ_src. h는 encoder 출력(CLS 등), μ_src, σ_src는 clean train 통계.
- 주거지 세 클래스(Dense / Medium / SparseResidential)만 색을 칠하고 나머지 27개 클래스는 회색이다. 서로 헷갈리기 쉬운 클래스를 골랐다.
- 행 = 백본 4개. satmae와 moco_in은 k-means 정확도가 가장 낮은 쪽(44), satdino_s8과 dinov3_web_l16은 가장 높은 쪽(80–81)이다.
- z는 384–2,048차원이라 2차원으로 내려서 그린다. 어느 2차원을 고르느냐에 따라 보이는 것이 달라서, 같은 점을 세 가지 방법으로 내렸다.

## fig0_projection.png: 2차원 그림은 정사영이다 (2026-11-26 추가)

장난감 3차원 데이터로 그린 설명 그림이다. 각 점에서 평면으로 수직선을 내려 떨어진 자리를 찍는 것이 정사영(orthographic projection)이고,
fig1의 첫 두 열은 같은 점을 서로 다른 평면에 정사영한 것이다.
- PCA 평면: 점들이 가장 넓게 퍼진 두 방향. 라벨을 안 쓴다. 넓게 퍼진 방향이 모든 클래스에 공통인 변화(장난감에서는 세로)이면 클래스가 겹쳐 보인다.
- 세 클래스 평균을 지나는 평면: 별 세 개(클래스 평균)를 지나는 평면. 라벨을 쓴다. 세 클래스를 떼어 보기에 가장 좋은 평면이다.

선행 사례 (2026-11-26 확인): Müller, Kornblith, Hinton, *When Does Label Smoothing Help?* (NeurIPS 2019) §2, Fig. 1이 같은 방식을 쓴다.
"Pick three classes", "Find an orthonormal basis of the plane crossing the templates of these three classes", "Project the penultimate layer
activations of examples from these three classes onto this plane." 차이: 그 논문의 평면은 세 클래스의 template(마지막 층 가중치 w_k)을 지나고,
우리는 세 클래스의 clean train feature 평균을 지난다. 비슷한 목적의 다른 그림: Neural Collapse(Papyan et al. PNAS 2020) Fig. 1은 세 클래스의
평균·classifier·feature를 클래스 평균의 SVD로 3차원에 그린다. 라벨을 써서 축을 고른다는 점에서 LDA(Fisher 판별) 투영과도 같은 종류다.
읽을 때 주의: 평면을 세 클래스가 갈라지도록 골랐으므로 평균끼리 떨어져 보이는 것은 당연하다. 보는 것은 평균 사이 거리에 비해 각 클래스가 얼마나 퍼졌나이다.

PPT에 넣을 한글 이름 (서버에 한글 글꼴이 없어 그림 안은 영어):

| 그림 안 제목 | 한글 |
|---|---|
| projection on the 2 widest directions (PCA, no labels) | PCA 평면 정사영 (가장 넓게 퍼진 두 방향, 라벨 없음) |
| projection on the plane through the 3 class means (uses labels) | 세 클래스 평균을 지나는 평면 정사영 (라벨 사용) |
| t-SNE: nearest neighbours kept (distances and axes have no meaning) | t-SNE (가까운 이웃 관계만 유지, 축과 거리는 의미 없음. 섬의 개수도 믿지 말 것) |
| x: along the common shift (source std units) | x: 공통 이동 방향 (clean 평균 → 손상 stream 평균), 0 = clean 평균, 단위 = source 표준편차 |
| widest clean direction ⊥ x | y: clean feature가 가장 넓게 퍼진 방향 (x와 직교) |

## fig1_clean_geometry.png: clean feature의 모양

행 이름 아래 숫자 세 개는 같은 clean test feature를 세 가지 방법으로 분류한 정확도다.

| 숫자 | 어떻게 분류하나 | 무엇을 재나 |
|---|---|---|
| linear | 학습한 선형 head (z W + b) | 클래스 사이에 평면 경계를 그을 수 있나 (선형 분리) |
| 5-NN | 가장 가까운 train 이미지 5장의 다수결 라벨 | 가까운 이웃이 같은 클래스인가 (국소 일관성) |
| k-means | 라벨 없이 30개 군집으로 나눈 뒤, 군집과 클래스를 가장 잘 맞게 짝지어 센 정확도 | 라벨 없이 유클리드 거리만으로 클래스를 찾을 수 있나 (클래스가 둥글고 서로 떨어진 덩어리인가) |

세 열(같은 점을 다르게 내린 그림):

| 열 | 축을 어떻게 골랐나 | 보는 것 |
|---|---|---|
| PCA | 2,000개 점 전체에서 분산이 가장 큰 두 방향. 라벨을 안 씀 | "feature가 가장 크게 퍼진 방향"에서는 세 클래스가 겹친다. 큰 분산은 클래스와 상관없는 변화(밝기, 장면 구성 등)가 차지할 수 있다 |
| plane through the 3 class means | 세 클래스의 train 평균 세 점을 지나는 평면. 라벨을 씀 | 세 클래스를 가장 잘 떼어 보이는 2차원. 여기서는 세 색이 갈라진다 = 이 세 클래스를 가르는 방향이 feature 안에 있다 |
| t-SNE | 각 점의 가까운 이웃 관계를 최대한 유지하도록 2차원에 배치(PCA 50차원 위에서, perplexity 30). 거리와 축의 값에는 뜻이 없다 | 같은 클래스의 이미지끼리 이웃인가. 같은 색이 작은 섬으로 모이면 이웃이 같은 클래스라는 뜻 |

- PCA의 "주성분"은 채널 하나가 아니라 채널들의 가중합인 방향이다.
- t-SNE의 "neighbourhood"는 한 점 주변의 가까운 점들이고, 그 점들이 같은 클래스인지가 보는 대상이다. 다른 클래스가 가까이 붙어 있으면 섬이 붙거나 섞여 보인다.

제목 "linearly separable, locally consistent, but k-means does not find the classes"의 뜻 (2026-11-26 제목과 설명을 고침):
- **linearly separable**: 선형 head가 93–98%. 평면 경계로 클래스를 거의 다 가를 수 있다(가운데 열).
- **locally consistent**: 5-NN이 81–96%. 가까운 이웃은 대개 같은 클래스다(t-SNE의 같은 색 섬).
- **k-means does not find the classes**: k-means는 44–81%. 라벨 없이 유클리드 거리로 30개 덩어리를 찾으면 클래스와 맞지 않는다.
  이유는 fig3에서 따로 쟀다. 클래스가 여러 덩어리로 쪼개져 있어서가 아니라, 클래스마다 덩어리는 하나인데 클래스를 가르지 않는 방향으로 넓게 퍼져 있어서다.
- 정정: 이전 제목은 "not one blob per class"였고 "satmae는 한 클래스가 여러 섬으로 쪼개져 있다"고 적었다. t-SNE에서 그렇게 보이지만
  t-SNE는 이어진 덩어리도 섬으로 끊어 그리는 경우가 있고, fig3의 2분할 측정에서는 satmae의 30개 클래스 중 다봉으로 판정된 것이 없다. 그 설명은 틀렸다.
- 그래도 "군집이 뚜렷하다"를 가정하는 방법(Entropy Enigma의 Silhouette 해석, k-means 기반 방법)이 이 feature에서 잘 맞지 않는다는 결론은 같다.

## fig3_class_shape.png: k-means는 왜 선형 head보다 훨씬 낮은가 (2026-11-26 추가)

수치: `experiments/2026-11-24_cluster_cot/RESULTS_shape.md` (`scripts/analysis/cluster_shape.py`). 그림: `plot_feature_dist.py --shape_only`.

위 줄은 장난감 2차원 데이터(3클래스)로 그린 세 가지 가능성이다. 색 = 실제 클래스, 검은 십자와 선 = k-means가 라벨 없이 찾은 중심 3개와 경계.
아래 숫자는 네 가지 방법의 정확도다.

| 장난감 | 모양 | 선형 | 5-NN | 가장 가까운 클래스 평균 | k-means |
|---|---|---:|---:|---:|---:|
| (a) | 클래스마다 둥근 덩어리 하나 | 100 | 100 | 100 | 100 |
| (b) | 클래스마다 덩어리 하나, 모두 같은 방향으로 길게 퍼짐 | 98 | 95 | 86 | 38 |
| (c) | 클래스마다 떨어진 섬 여러 개 | 42 | 100 | 42 | 47 |

- (b)에서 k-means가 틀리는 이유: k-means는 "퍼진 정도"가 큰 방향을 따라 자른다. 길게 퍼진 방향이 클래스와 무관하면 경계가 클래스를 가로지른다(그림의 가로선).
  선형 head는 그 방향을 무시하고 클래스를 가르는 방향만 본다.
- (c)라면 선형 head와 클래스 평균이 둘 다 무너지고 k-NN만 산다.

아래 줄(2026-11-26 단순하게 다시 그림)은 위 장난감들과 **같은 네 방법**을 실제 feature에 적용한 값(파랑)을 장난감 값(회색)과 나란히 놓은 것이다.

| 방법 | 장난감 (a) | 장난감 (b) | 장난감 (c) | 실제 AID | 실제 RESISC45 |
|---|---:|---:|---:|---:|---:|
| 선형 head | 100 | 98 | 42 | 94.8 | 90.6 |
| 5-NN | 100 | 95 | 100 | 88.5 | 83.3 |
| 가장 가까운 클래스 평균 (클래스당 중심 1개, 중심은 train 라벨로 계산) | 100 | 86 | 42 | 85.1 | 75.8 |
| k-means (라벨 없음) | 100 | 38 | 47 | 59.1 | 51.3 |

- 실제 값의 모양은 (b)와 같다: 높음, 높음, 꽤 높음, 낮음.
- (c)가 아니라는 근거: 섬 여러 개라면 선형 head와 클래스 평균이 둘 다 무너져야 하는데(42) 실제로는 95와 85다.
- (a)가 아니라는 근거: 둥근 덩어리라면 k-means도 100이어야 하는데 59다.

### fig3b_class_shape_tests.png: 확인 측정 두 가지 (예비 장)

**test 1 (왼쪽): 퍼진 모양을 눌러 펴면 중심 하나로 거의 충분하다**
- "클래스 내 공분산으로 whitening"을 쉬운 말로: (b)에서 세 클래스는 모두 세로로 길게 퍼져 있다. 세로축을 눌러서 덩어리를 둥글게 만들면 (b)가 (a)처럼 된다.
  어느 방향으로 얼마나 눌러야 하는지는 "점들이 자기 클래스 중심에서 그 방향으로 얼마나 흩어져 있나"를 모든 클래스에서 모아 계산한다(= 클래스 내 공분산).
  각 방향을 그 흩어짐의 크기로 나누는 것이 whitening이다. head 입력 표준화(채널마다 표준편차로 나눔)를 채널 사이의 상관까지 포함해 모든 방향으로 한 것이다.
- 무엇을 확인하나: 클래스가 "덩어리 하나 + 클래스들에 공통인 퍼진 모양"이라면, 눌러 편 뒤에는 클래스 평균 분류가 선형 head만큼 좋아져야 한다.
  섬 여러 개라면 눌러도 섬은 그대로라 좋아지지 않는다.
- 결과: 85.1 → 92.3 (AID), 75.8 → 83.9 (RESISC45). 선형 head는 94.8 / 90.6. 대부분 따라잡는다.
- 이 계산은 라벨을 쓴다(클래스 중심과 흩어짐을 라벨로 구함). 방법이 아니라 모양을 확인하는 진단이다. 통계학에서는 LDA(선형 판별 분석)와 같은 모형이다.

**test 2 (오른쪽): 클래스 하나를 둘로 잘라 본다**
- 재는 법: 한 클래스의 점만 모아, 가장 잘 나뉘도록 두 묶음으로 자른다(2-means). 자르기 전과 후에 "점들이 자기 묶음 중심에서 흩어진 정도(분산)"가 몇 % 줄었는지 본다.
- 왜 이것이 덩어리 수를 알려 주나: 떨어진 섬 둘로 된 집단은 섬 사이를 자르면 각 묶음이 훨씬 촘촘해져서 분산이 크게 준다.
  이어진 덩어리 하나는 어디를 잘라도 덩어리를 반으로 가를 뿐이라 조금만 준다.
- "조금"과 "크게"의 기준을 두 개 만들었다.
  - 덩어리 **하나**의 기준: 그 클래스와 퍼짐이 똑같은 Gaussian 점들을 새로 뽑아 같은 방법으로 자른다. 만들 때부터 덩어리 하나다. → 8.1%
  - 덩어리 **둘**의 기준: 서로 다른 실제 클래스 둘을 합쳐 한 클래스인 척하고 자른다. 만들 때부터 덩어리 둘이다. → 17.6%
- 결과: 실제 클래스는 8.8%로 "하나" 기준(8.1%) 옆에 있고 "둘" 기준(17.6%)과 멀다. 17개 백본 모두 그렇다(점). RESISC45는 9.6 / 8.6 / 18.0%.
- Gaussian 기준보다 뚜렷이 큰(+3pt 초과) 클래스의 비율은 AID 1%, RESISC45 6%다. 예외는 dinov3_web(AID 10%, RESISC45 22%)이다.

같은 표의 나머지 수치.
- 클래스 내 분산의 76%(AID) / 69%(RESISC45)가 클래스 평균들이 만드는 부분공간 밖에 있다. 퍼짐의 대부분이 클래스를 가르지 않는 방향이다.
- 한 방향으로 가늘고 긴 타원은 아니다. 클래스 분산 중 첫 주성분의 몫은 평균 12%뿐이다. 여러 방향으로 넓게 퍼진 덩어리다.
- k-means 목적함수는 클래스에 맞는 분할을 가려내지 못한다. 클래스 평균에서 시작한 k-means(정확도 77.8 / 62.4)와 무작위로 시작한 k-means(59.1 / 51.3)의
  목적함수 값이 1% 이내로 같다. 시작점을 40번 바꾸면 정확도가 한 백본 안에서 9–19pt 흔들린다(mae_in 35–45%, satdino_s8 67–86%; 4개 백본에서 확인). 즉 59%라는 숫자에는 "k-means가 이 고차원에서 불안정하다"가 섞여 있다.
- 중심을 클래스당 8개로 늘려도(train에 240개 중심, 다수결 이름) 80.7 / 70.8로, 라벨로 구한 중심 1개(85.1 / 75.8)보다 낮다.

정리: "한 클래스가 구 여러 개"는 측정과 맞지 않고, "한 덩어리인데 둥글지 않다(클래스와 무관한 여러 방향으로 넓게 퍼짐)"가 맞다.
주의: 2분할 측정은 PCA 256차원에서 했고, 아주 작은 하위 군집까지 없다고 말하는 것은 아니다. whitening과 클래스 평균은 라벨을 쓰므로 방법이 아니라 진단이다.

PPT용 한글 캡션 (fig3):

| 그림 안 글자 | 한글 |
|---|---|
| (a) one round cloud per class | (a) 클래스마다 둥근 덩어리 하나 |
| (b) one cloud per class, all stretched the same way | (b) 클래스마다 덩어리 하나, 모두 같은 방향으로 퍼짐 |
| (c) several separate islands per class | (c) 클래스마다 떨어진 섬 여러 개 |
| nearest class mean | 가장 가까운 클래스 평균 (클래스당 중심 1개) |
| nearest class mean after squashing the spread the classes share (fig3b) | 클래스들에 공통인 퍼짐을 눌러 편 뒤의 클래스 평균 분류 |
| how much tighter the points get when the group is cut in two (fig3b) | 집단을 둘로 잘랐을 때 줄어드는 분산 |
| reference: ONE cloud (Gaussian, same spread) | 기준: 덩어리 하나 (같은 퍼짐의 Gaussian) |
| reference: TWO clouds (two real classes pooled) | 기준: 덩어리 둘 (실제 클래스 둘을 합친 것) |

## fig2_shift_align_motion_blur_s3.png: 손상이 feature를 어떻게 옮기고, head 입력 표준화가 무엇을 되돌리나

열: clean / motion blur 3단계 / 같은 손상 stream을 head 입력 표준화(배치 64)한 뒤. 그림 아래에 축 설명을 영어로 넣었다(2026-11-26). 제목의 accuracy는 그 상태의 분류 정확도다.

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
