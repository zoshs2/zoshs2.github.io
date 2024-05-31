---
title: "Graph Neural Networks' Brief History"
date: 2024-04-17 16:18:43 +0900
categories: [DL/ML, Story]
tags: [GNNs, Graph, Recursive, Neural, Network, RNN, RvNN, tanh, nltk, tree-structured, graph-structured, dataset, Deep Learning, Machine Learning, NLP, Language, DNN, MLP, Python]
math: true
toc: false
---

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Intro.](#intro)
- [Graph Neural Networks](#graph-neural-networks)
- [1990s: Recursive Neural Network](#1990s-recursive-neural-network)
  - [RvNN References](#rvnn-references)
- [In 1998, CNN: New era of Deep Learning](#in-1998-cnn-new-era-of-deep-learning)
- [2010s: Golden era for the advancement of GNNs](#2010s-golden-era-for-the-advancement-of-gnns)
- [In 2015: Graph Convolutional Network](#in-2015-graph-convolutional-network)
- [Conclusion](#conclusion)

# Intro.

이번 포스트에선 Graph Neural Networks(GNNs) 분야의 히스토리에 대해 살짝 정리하는 글을 남기고자 한다.

사실 Graph Convolutional Network(GCN)을 구현해보려 하다가, 문득 이것이 어떤 과정을 거쳐 발전되어 왔는지가 궁금해졌기 때문이다.

어떤 방법론을 공부하기 전에 그것이 발전되어 온 계보-전체적인 히스토리를 함께 이해하는 것은 도움도 되고, 창작자들이 어떤 문제를 돌파하기 위해 또는 어떤 생각으로 이 분야를 발전시켜 왔는지 답습해보는 일이 꽤 재미있기도 하다.

이 글의 작성 의도도 이러한 맥락에서 기획했고, 이론적인 부분은 생략하고 ...

# Graph Neural Networks

그래프 신경망(Graph Neural Networks; 이하 GNNs)은 RNN, LSTM, GRU, CNN 같이 특정 모델 및 아키텍쳐만을 지칭하는 term이 아니라, 그래프(또는 네트워크) 구조를 지닌 graph-structured data에 적용하는 것을 목적에 두고 디자인된 모든 신경망 모델들을 총칭해 부르는 용어이자, 하나의 커다란 분야다.

GNNs는 지금도 활발한 연구와 관련 모델들이 쏟아져 나오고 있으며, [Zhou J. et al.(2020)](https://www.sciencedirect.com/science/article/pii/S2666651021000012){:target="_blank"}의 GNNs에 대한 review paper에 따르면, 수년간의 세월동안 이뤄진 GNNs 발전형태는 이렇게나 다양하다.

![png](/assets/img/post/gnns_history/GNNs.png)*Zhou J. et al. Source: Graph Neural Networks: A Review of Methods and Applications.*

지금 여기서 이 모든 아키텍쳐들을 다 살펴볼 것은 물론 아니다. 대신 굵직한 것들만 대략적으로 살펴보자.

# 1990s: Recursive Neural Network

GNNs의 계보를 이야기할 때, 종종 1990년대 말의 Recursive Neural Network(RvNN; 재귀 신경망)에 대해 이야기하며 운을 땐다.

> Recursive Neural Network는 RNN 이라고도 부르기도 하는데, 주로 언어 모델에서 사용하는 Recurrent Neural Network의 약어와 혼동이 있을 수 있어서 RvNN으로 부르는게 (개인적으로) 더 적절해 보인다. - 두 모델은 서로 다른 모델이다.
{: .prompt-info }

RvNN은 주로 트리 구조 데이터(tree-structured data)를 처리하는 데 사용되며, 각 노드는 자식노드들의 정보($h_c$)를 재귀적으로 결합(aggregate)하여 상위 부모노드를 어떻게 표현할지($h_p$) 결정하는 데 활용하게 된다. 이러한 과정을 수식으로 표현한 예를 들자면, 아래와 같은 것이다.

$$
\begin{equation}
  h_p = \tanh{\left(W\left[h_{c1}, h_{c_2}\right]+b\right)}
\end{equation}
$$

예컨대, RvNN은 자연어처리(NLP) 분야에서 사용할 수 있는 아주 기초적인 모델 중 하나인데, 한 문장(여기서, "That movie was cool")의 parse tree(구문 트리)를 아래와 같이 만들고,

![png](/assets/img/post/gnns_history/ReNNs.png)*Example of sentence's parse tree. Source: [GeeksforGeeks](https://www.geeksforgeeks.org/difference-between-recursive-and-recurrent-neural-network/){:target="_blank"}*

Leaf Nodes(자식이 없는 노드; 가장 말단의 노드들)에서부터 Root Node까지 앞서 언급한 *결합 함수식*을 활용하여 상향식(bottom-up)으로, 그리고 Root Node에 도달할 때까지 재귀적(recursive)으로, 부모노드들의 (수학적) 표현들을 도출하는 것이다.

> 문장의 parse tree (구문 트리)는 파이썬 [NLTK](https://www.nltk.org/){:target="_blank"} 등과 자연어처리 라이브러리를 써서 구현할 수 있다.
{: .prompt-tip }

이후, 만약 문장의 감정 상태를 분석하는 모델을 만들고 싶다면, 특정 감정 상태(기쁨, 화남, 중립)에 대응하는 출력 레이블 값과 모델의 출력값 사이의 loss를 줄이는 방향으로 가중치 $W$ 값(앞선 식의 그 $W$ 맞음)을 **업데이트함으로써 모델을 학습**시키는 것이다.

이렇듯 RvNN은 트리 구조 및 상향-단방향식으로 정보들을 결합시켜 나간다는 측면에서 일종의 Directed acyclic(유향 비순환; 비순환: 연결구조에서 동일한 노드로 다시 돌아올 수 있는 싸이클 구조가 없다는 의미)한 Graph model 이고, 그렇기 때문에 GNNs 분야의 시작으로 언급되기도 하는 것이다.

## RvNN References

1. [Sperduti, Alessandro, and Antonina Starita. "Supervised neural networks for the classification of structures." IEEE transactions on neural networks 8.3 (1997): 714-735.](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=3e33eca03933caaec671e20692e79d1acc9527e1){:target="_blank"}
2. [Frasconi, Paolo, Marco Gori, and Alessandro Sperduti. "A general framework for adaptive processing of data structures." IEEE transactions on Neural Networks 9.5 (1998): 768-786.](https://www.math.unipd.it/~sperduti/PAPERI/general-framework.pdf){:target="_blank"}

# In 1998, CNN: New era of Deep Learning

GNNs 발전의 중요한 지렛대가 된 계기로서, Convolutional Neural Network (CNN; 합성곱 신경망) 이야기를 안하고 넘어갈 수 없다. CNN 모델과 그의 방법론은 GNNs 분야의 발전 뿐 아니라 DL/ML 전체 판도에 가속을 불어넣었기 때문이다.

합성곱 신경망에 대한 자세한 설명은 다른 포스트에서 다룰 예정이기에, 간단하게만 설명하자면, CNN 모델은 주로 이미지 데이터(like raster data, pixel data, grid data)를 입력받아 출력 레이블 값과의 복잡한 비선형적 관계를 표현하는데 아주 효과적인 모델이다. 'AI의 대부' 또는 '딥러닝의 대부'라 불리는 얀 르쿤(Yann LeCun)의 [LeCun et al. 1998](https://ieeexplore.ieee.org/document/726791){:target="_blank"}에서 합성곱 신경망(CNN) 구조 학습법이 처음 소개되었다. 

> 'AI의 대부' 또는 '딥러닝의 대부'라 불리는 대표적 3인은 얀 르쿤(Yann LeCun; 현 Meta 수석 AI Scientist 및 부사장) 요슈아 벤지오(Yoshua Bengio; 현 몬트리올 대학 컴퓨터과학과 교수), 제프리 힌튼(Geoffrey Hinton; 현 토론토 대학 컴퓨터과학과 교수)이다. 
{: .prompt-info }

> 제프리 힌튼 교수는 13년 3월부터 23년 5월까지 10년간 근무하던 구글(Google)을 퇴사했다. 앞으로 대중과 학계에게 **AI의 위험성**에 대해 자유롭게 말하고 싶다는 연유로. 심지어 그는 딥러닝 연구에 대한 그의 지난 업적들을 후회한다고 말했다고 한다. [참고: 뉴욕타임즈 기사](https://www.nytimes.com/2023/05/05/podcasts/ai-google-banks.html){:target="_blank"}
{: .prompt-info }

여기서 CNN이 GNNs 분야 발전에 영향을 준 아이디어 요소는 대상(이미지)의 국소적 영역을 처리하는 필터를 사용하여 지역적인 주요 특징을 추출한다는 것과 이들을 종합하여 더욱 함축된 표현으로 높은 수준의 해석이 가능하다는 것이다.

하지만 앞서 이야기했듯이 CNN 모델은 기본적으로 이미지(2D grids)나 텍스트(1D sequences)와 같은 Euclidean data에 특화되어 있다. Euclidean data란 것은 쉽게 말하자면, 우리가 직관적으로 이해할 수 있는 1/2 차원 공간 상에 정의된 일련의 데이터들을 의미하고, 더 자세히는 각 데이터 포인트가 **기술 가능한** 고정된 위치와 규칙적인 간격을 가지고 있는 일련의 데이터들을 말한다.

그렇지만, 노드 & 링크, 버텍스 & 엣지라는 구성 요소들이 서로 간에 맺고 있는 연결 관계에 중점을 두고 대상을 표현하는 그래프(Graph)라는 것은 정해진 위치나 순서, 간격이 없는 추상적인 non-euclidean 개체이기 때문에 CNN의 필터 스캐닝같은 개념이 처음 소개되었을 때, 바로 적용하기 어려운 실정이었다. 

![png](/assets/img/post/gnns_history/EuclideanData.png)*Source: Zhou, Jie, et al. (2020) "Graph neural networks: A review of methods and applications."*

결국 1998년 CNN 모델의 등장 이후 딥러닝의 대부라 불리는 "[얀 르쿤, 요슈아 벤지오, 제프리 힌튼(2015)](https://www.nature.com/articles/nature14539){:target="_blank"}"이 딥러닝이라는 새로운 시대의 서막을 알린 2010년 초 무렵까지, 비유클리드 범주(Non-euclidean domain)에 속하는 Graph를 어떻게 표현해야 하고, 이러한 데이터들을 입력받아야 하는 신경망 아키텍쳐는 또 어떻게 설계해야하는지 같은 문제들의 해결 방법이 제대로 정립되지 않은 채 GNNs 분야의 긴 공백기가 이어졌다. 

# 2010s: Golden era for the advancement of GNNs

그래프(Graph) 데이터와 신경망 딥러닝 학습 사이에 존재하는 본질적인 문제는 결국 "비유클리드 데이터를 어떻게 표현해야 하는지"에 있었다. **비유클리드**라함은 1/2차원과 같은 직관적인 이해가 가능한 유클리드 공간을 넘어서 보다 복잡하고 고차원의 (그래서 표현이 어려운, 그래서 이해가 어렵기에 '추상적이다'라고 퉁쳐서 말하는) 공간을 일컫는다. 다시 말해, "비유클리드 데이터를 어떻게 표현해야 하는지"의 문제는 곧 "우리가 기술가능한 표현 형태 및 차원으로 어떻게 전환시킬지"에 대한 문제이며, 이러한 시도와 적용 가능한 다양한 방법들을 "임베딩(Embedding)"이라고 하는 것이다.

2010년도 초에 들어서며 그래프(Graph)를 저차원 임베딩 공간으로 변환할 수 있는 획기적인 아이디어들이 제시되기 시작했다. 이 아이디어들은 큰 맥락에서 보면 그래프 구조 데이터를 처리하고 학습하는 방법이지만, **목적과 초점 그리고 범위**에 따라 크게 두 가지 분야로 나뉜다. 바로 **Graph Representation Learning** 분야와 **Geometric Deep Learning** 분야이다. 이들은 특정 딥러닝 모델 아키텍쳐가 아니라 GNNs과 같이 또 다른 한 분야이자, GNNs의 하위 범주이다.

**Geometric Deep Learning** 분야는 유클리드 및 비유클리드 공간 포함하는 모든 기하학적 구조의 데이터를 이해하고 학습하는 데 그 목적이 있다. 다시 말해 그래프(Graph) 뿐 아니라, 포인트 클라우드, 리만 다양체(Riemannian manifold)와 같은 다양한 기하학적 데이터 구조를 처리하는 데 그 초점이 맞춰 있다고 볼 수 있다. 관련한 일련의 모델 및 처리 방법론들에는 PointNet[(Qi, Charles R. et al., 2017)](https://arxiv.org/abs/1612.00593){:target="_blank"}, PointNet++[(Qi, Charles R et al., 2017)](https://arxiv.org/abs/1706.02413){:target="_blank"}, UMAP[(McInnes et al., 2018)](https://arxiv.org/abs/1802.03426){:target="_blank"}, MeshCNN[(Hanocka et al., 2019)](https://arxiv.org/abs/1809.05910){:target="_blank"} 들이 있었다. 이렇듯 그래프 뿐 아니라 다양한 기하학적 구조를 지닌 데이터들까지, 모든 비유클리드 데이터에 대한 Geometric Deep learning 분야의 학술적 노고들은 GNNs 분야의 다양성과 응용 범위를 넓히는 데 중요한 역할을 했다.

> 포인트 클라우드(Point Cloud) 데이터는 LiDAR(Light Detection and Ranging) 센서, 3D 스캐닝, Stereo Vision 기법 (두 개의 카메라로 찍은 이미지를 비교하여 3D 정보를 추출)에서 얻어지며, 3차원 공간 상의 객체의 형상을 표현하기에 3D 좌표 (x,y,z)를 갖는다. 일반적인 2D 시각적 이미지 정보는 픽셀(또는 raster/grid)이라는 고정된 위치의 정규 격자 구조 속에 저장되는 것에 반해, 포인트 클라우드 데이터 정보는 데이터 포인트들 간의 연결성이나 정규 격자에 담겨 있지 않기 때문에 이들을 모델에 다루기 위해서는 특별한 전처리나 알고리즘이 필요하다.
{: .prompt-info }

반면, **Graph Representation Learning** 분야의 목적은 **그래프에서 의미있는 표현**을 추출하여 임베딩 벡터로 변환하는 일에 초점이 맞추어져 있다. 그래프에 의미있는 표현은 노드(버텍스), 링크(엣지), 그들의 연결관계에 담겨 있다고 말했었다. 이들을 **저차원 벡터로 임베딩**하는 일련의 노력들은 노드 임베딩(word embedding[(Mikolov et al., 2013)](https://arxiv.org/pdf/1301.3781){:target="_blank"}, DeepWalk[(Perozzi et al., 2014)](https://dl.acm.org/doi/10.1145/2623330.2623732){:target="_blank"}, node2vec[(Grover and Leskovec, 2016)](https://dl.acm.org/doi/10.1145/2939672.2939754){:target="_blank"}, LINE[(Tang et al., 2015)](https://arxiv.org/abs/1503.03578){:target="_blank"}, TADW[(Yang et al., 2015)](https://www.ijcai.org/Proceedings/15/Papers/299.pdf){:target="_blank"}), 그래프 임베딩(graph2vec[(Narayanan, Annamalai, et al., 2017)](https://arxiv.org/abs/1707.05005){:target="_blank"}, SDNE[(Wang, D. et al., 2016)](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf){:target="_blank"})로 (굳이 나누자면) 나눠져서 2010년도 전반에 걸쳐 그래프 구조의 데이터를 처리하는 방법론 정립과 GNNs 분야 발전의 직접적인 동력을 제공했다.

# In 2015: Graph Convolutional Network

[딥러닝 시대의 시작을 알렸던 CNN](https://www.nature.com/articles/nature14539){:target="_blank"} 모델의 아이디어를 드디어 Graph 구조 데이터에도 적용할 수 있는 획기적인 진보가 2016년에 소개되었다.

* [Kipf, Thomas N., and Max Welling. (2016) "Semi-supervised classification with graph convolutional networks."](https://arxiv.org/abs/1609.02907){:target="_blank"}
* [Kipf, Thomas N., and Max Welling. (2016) "Variational graph auto-encoders."](https://arxiv.org/abs/1611.07308){:target="_blank"}

* * *

CNN 모델에서, 고정된 크기의 필터를 통해 이미지를 스캐닝하면서 이미지의 작은 영역에서의 **지역적 정보**, 또는 **국소적 스케일에서의 주요 특징**들을 추출하여 고도화된 데이터 해석과 학습을 가능케 했던 방법이 그래프 구조 데이터에 대한 적용 방식으로서 재정립된 것이다.

$$
\begin{equation}
  \text{H} = \sigma\left(\tilde{\text{D}}^{-\frac{1}{2}}\tilde{\text{A}}\tilde{\text{D}}^{-\frac{1}{2}}\text{X}\Theta\right)
\end{equation}
$$

![png](/assets/img/post/gnns_history/GCN_Scheme.png)*Difference between Neural Network(NN) and GCN. Source: [Graph Convolutional Networks (GCN) & Pooling](https://jonathan-hui.medium.com/graph-convolutional-networks-gcn-pooling-839184205692){:target="_blank"}*

$$
\begin{equation}
A - \lambda I = \begin{pmatrix} a & b \\ c & d \end{pmatrix} - \lambda \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} a - \lambda & b \\ c & d - \lambda \end{pmatrix}
\end{equation}
$$

$$
\begin{equation}
\det(A - \lambda I) = (a - \lambda)(d - \lambda) - bc
\end{equation}
$$

$$
\begin{equation}
\lambda^2 - (a + d)\lambda + (ad - bc) = 0
\end{equation}
$$



# Conclusion