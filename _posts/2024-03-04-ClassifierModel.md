---
title: "[PyTorch] Classifier: Logistic & Softmax Regression Model"
date: 2024-03-04 14:22:14 +0900
categories: [DL/ML, Study]
tags: [Logisitic, Classifier, Cross-Entropy, Likelihood, Log-Likelihood, NLL, Softmax, LogSoftmax, PyTorch, Torch, Deep Learning, Machine Learning, Python]
math: true
toc: false
---

# Table of Contents
- [Intro.](#intro)
- [Cost Function for Classification Model](#cost-function-for-classification-model)
- [Logistic Regression Model](#logistic-regression-model)
  - [Logistic Regression Class](#logistic-regression-class)
- [Softmax Regression Model](#softmax-regression-model)
  - [Cross Entropy Implementation](#cross-entropy-implementation)
    - [Data Preparation](#data-preparation)
    - [Bonus: Making of one-hot vectors](#bonus-making-of-one-hot-vectors)
    - [Low-level](#low-level)
    - [High-level](#high-level)
    - [High High-level](#high-high-level)
  - [Softmax Regression Class](#softmax-regression-class)
- [이어지는 글](#이어지는-글)

# Intro.

**본 공부 내용은 유원준 님, 안상준 님이 저술하신 [PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/book/2788) 위키독스 내용들을 배경으로 하고 있습니다.**
<br><br>

오늘은 두 가지 기본적인 분류 모델에 대해서 알아보자. 바로 **Logistic Regression Model**과 **Softmax Regression Model**이다.

Logistic Regression 모델은 합격/불합격, 정상메일/스팸메일 같이 이진 분류(Binary Classification) 문제를 학습하기 위한 모델이다.

Softmax Regression Model은 **이진 분류뿐 아니라** 다수의 종류에 대한 분류 문제를 풀기 위한 모델이다.

# Cost Function for Classification Model

우선, 로지스틱/소프트맥스 회귀 모델과 같이 분류 모델 학습에서의 손실 함수(cost/loss/error function)는, 앞서 단순/다중 선형 모델에서 사용했던 Mean Squared Error(F.mse_loss())가 아닌, Cross-Entropy Function을 사용한다.

$$
\begin{equation}
  \text{Cross Entropy Func.} = -\sum_{x\in X}{p\left(x\right)}\log{q\left(x\right)}
\end{equation}
$$

이는 곧 $\text{E}_p\left[-\log{q}\right]$로써, 모델이 $p$ 분포(실제/정답 분포)를 따른다고 가정했을 때 얻게 되는 정보량(놀라움, $-\log{q}$)의 기댓값을 의미한다. 만약 모델이 $p$라는 실제 확률 분포를 잘 반영하고 있다면, $\text{E}_p\left[-\log{q}\right]$값은 그리 높지 않을 거란 기대가 가능하다. 직관적으로 쉽게 이해하자면, 결국 Cross-Entropy 함수는 두 확률 분포(실제 정답 & 모델의 예측) 사이의 괴리를 설명해주며, 이러한 의미는 모델 학습 결정의 중요한 근거가 된다. 예컨대 $\text{E}_p\left[-\log{q}\right]$가 너무 크면, "현재 내 모델은 실제 모습이랑 거리가 너무 머네. **이것을 줄이는 방식**으로 다시 학습시켜야 겠어"라는 판단을 내릴 수 있는 손실 함수로서 활용할 수 있는 것이다.


```python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
```

# Logistic Regression Model

**로지스틱 회귀 모델**은 Intro.에서 말한대로, 이진분류(Binary Classification) 문제를 풀기 위한 대표적인 알고리즘이다. 일반적으로 **Sigmoid 함수**와 **특정 임계값**을 사용하여 확률에 의한 클래스 분류를 수행하는 것이 기본 매커니즘이다.

$$
\begin{equation}
  \text{Sigmoid}\left(z\right) = \frac{1}{1+\exp\left(-z\right)}
\end{equation}
$$

> 보통 시그모이드 함수는 $\sigma(z)$로 표기함. 지금 얘기할 부분은 아니지만, LSTM 모델에 대한 시각화 설명에서도 이 $\sigma$ 시그모이드 활성화 함수가 등장한다.
{: .prompt-info }

시그모이드 함수는 0~1 사이의 값을 반환하는 함수인데, 이를 확률로 여겨 분류 문제에 활용할 수 있다.

간단한 식이라 직접 구현하는 건 일도 아니지만, torch에도 이미 sigmoid가 구현되어 있으니 이걸 사용하자. torch.sigmoid는 $exp$ 계산에서 발생할 수 있는 Overflow 문제에 대해 보다 안정적이다.


```python
x_train = torch.FloatTensor([[1, 2, 2],
                             [2, 3, 1],
                             [3, 1, 1],
                             [4, 3, 5],
                             [5, 3, 6],
                             [6, 2, 4]])
y_train = torch.FloatTensor([[0], [0], [0], [1], [1], [1]])
```

예를 들어, 위와 같은 데이터를 가지고 이진분류 모델을 학습시킨다고 해보자.

조금 더 직관적 이해를 위해, 3과목의 점수를 입력받고 합격/불합격을 판정하는 모델을 만드는 상황이라고 봐도 좋다. 이 경우 x_train은 (총 6명에 대한)세 과목에 대한 각각의 점수이고, y_train의 0과 1은 각각 합격과 불합격을 의미하는 것이다.

즉, 3개의 입력값을 가중치 $W$ 행렬과의 행렬곱을 수행하는 선형 모듈 nn.Linear(): $XW+b$, 이후 0과 1사이 출력값을 반환하는 시그모이드 모듈 nn.sigmoid(): $\sigma\left(XW+b\right)$을 nn.Sequential()로 엮어서 구현할 수 있다. (아래 그림 참고)

![png](/assets/img/post/basic_classification/logistic_scheme.png)*Figure 1*


```python
logistic_model = nn.Sequential(
    nn.Linear(3, 1),
    nn.Sigmoid()
)
```


```python
logistic_model(x_train) # Forwarding results before training
```




    tensor([[0.1056],
            [0.0524],
            [0.0860],
            [0.0058],
            [0.0025],
            [0.0046]], grad_fn=<SigmoidBackward0>)




```python
list(logistic_model.parameters()) # Training Parameters before training
```




    [Parameter containing:
     tensor([[-0.5276, -0.5293, -0.2994]], requires_grad=True),
     Parameter containing:
     tensor([0.0481], requires_grad=True)]



로지스틱 회귀 모델 학습을 구현하기 전에 **이진분류**에서 다루는 손실함수, 즉 Cross Entropy Function 형태를 한번 짚고 넘어가자. Cross Entropy 함수의 원형은

$$
\begin{equation}
  \text{Cross Entropy Func.} = -\sum_{x\in X}{p\left(x\right)}\log{q\left(x\right)}
\end{equation}
$$

라고 했는데, 이진분류에서 실제 p분포는 이항분포(binomial distribution)일 것이다. 이 경우 발생할 수 있는 사건($X$)은 '합격 또는 불합격'와 같이 두 가지 사건뿐이다. 이를 다르게 보면, '합격이거나 또는 합격이 아님'으로 볼 수도 있다. 이를 확률로서 1(합격)과 0(합격이 아님; 즉 불합격)으로 표현할 수 있고, 결과적으로 이진분류 문제에서의 Cross Entropy 함수는 아래와 같은 형태일 것이다.

$$
\begin{equation}
  \text{Binary_Cross_Entropy Func.} = -\left[p_{pass}\log{q_{pass}} + \left(1-p_{pass}\right)\log\left(1-q_{pass}\right)\right]
\end{equation}
$$

이 함수를 통해 도출된 값 자체는 현재 모델과 실제 분포와의 괴리 (오차 크기)를 의미하며, training 과정동안 현 파라미터 값 위치에서의 오차 함수(또는 손실함수) 기울기를 역전파시켜 앞서 도출한 괴리(오차)를 줄여 나가는 방향으로 학습이 진행되게 된다.

$$
\begin{equation}
  W^+ = W - \eta\frac{\partial}{\partial W}Cost
\end{equation}
$$

$$
\begin{equation}
  b^+ = b - \eta\frac{\partial}{\partial b}Cost
\end{equation}
$$

> 1회 학습에 다수의 데이터 샘플을 사용하는 Mini Batch 또는 Batch 방식에서는 각 샘플마다의 오차를 더해서 샘플 갯수($n$)만큼 나누는 평균 오차 함수를 사용한다. 다시 말해서, 각 샘플마다 기울기 $∇{Cost}$를 계산하고 더해서 $1/n$을 곱한 값을 파라미터 학습에 활용할 최종 Gradient로 쓴다는 말임.
{: .prompt-info }


```python
optimizer = optim.SGD(logistic_model.parameters(), lr=.05)

W, b = logistic_model.parameters()
print(f"Initialized W: {W}, b: {b}", end='\n\n')

num_epochs = 100
for epoch in range(num_epochs + 1):

  # 모델 예측값 계산
  predict = logistic_model(x_train)

  # Cost 계산
  cost = F.binary_cross_entropy(predict, y_train)

  optimizer.zero_grad() # 1. 이전 기울기값 초기화 (이거 안하면 기울기값들이 계속 누적됨)
  cost.backward() # 2. aE/aW, aE/ab 계산
  optimizer.step() # 3. 새로운 w, b값으로 갱신

  if epoch % 10 == 0:
    # 특정 임계 확률값을 정해서 클래스 분류의 기준으로 삼는다.
    # 여기선 0.5 이상이 되면 True(합격)이라고 보겠다는 의미
    predicted_label = predict >= torch.FloatTensor([0.5])

    # 이렇게 모델이 수행한 분류가 실제 정답과 얼마나 일치하는지 비교를 통해 Accuracy를 도출
    accuracy = predicted_label == y_train
    accuracy = (accuracy.sum() / len(accuracy)).item() # item() 은 pandas DataFrame에서 values[0]과 같은거. (데이터 안의 값만 뽑아내기)

    print(f"Epoch: {epoch}/{num_epochs}, Cost: {cost.item():.3f}, Accuracy: {accuracy*100:.2f}%")
    if epoch == 0:
      print(f"aE/aW: {W.grad}, aE/ab: {b.grad}", end='\n\n')
```

    Initialized W: Parameter containing:
    tensor([[-0.5276, -0.5293, -0.2994]], requires_grad=True), b: Parameter containing:
    tensor([0.0481], requires_grad=True)
    
    Epoch: 0/100, Cost: 2.794, Accuracy: 50.00%
    aE/aW: tensor([[-2.4113, -1.2519, -2.4313]]), aE/ab: tensor([-0.4572])
    
    Epoch: 10/100, Cost: 0.498, Accuracy: 66.67%
    Epoch: 20/100, Cost: 0.452, Accuracy: 66.67%
    Epoch: 30/100, Cost: 0.416, Accuracy: 66.67%
    Epoch: 40/100, Cost: 0.385, Accuracy: 83.33%
    Epoch: 50/100, Cost: 0.359, Accuracy: 83.33%
    Epoch: 60/100, Cost: 0.338, Accuracy: 100.00%
    Epoch: 70/100, Cost: 0.319, Accuracy: 100.00%
    Epoch: 80/100, Cost: 0.302, Accuracy: 100.00%
    Epoch: 90/100, Cost: 0.287, Accuracy: 100.00%
    Epoch: 100/100, Cost: 0.274, Accuracy: 100.00%


## Logistic Regression Class

[선형 회귀 모델 포스팅](https://zoshs2.github.io/posts/LinearRegressionModel/#pytorch-%EB%AA%A8%EB%8D%B8%EC%9D%84-%ED%81%B4%EB%9E%98%EC%8A%A4%ED%99%94-%EC%8B%9C%ED%82%A4%EA%B8%B0:~:text=0.1914-,PyTorch%20%EB%AA%A8%EB%8D%B8%EC%9D%84%20%ED%81%B4%EB%9E%98%EC%8A%A4%ED%99%94%20%EC%8B%9C%ED%82%A4%EA%B8%B0,-%EB%94%A5%EB%9F%AC%EB%8B%9D%20%EB%AA%A8%EB%8D%B8%EC%9D%80%20%EA%B1%B0%EC%9D%98){:target="_blank"}에서도 언급했지만, DL/ML 모델은 클래스로 구현하여 사용하는 것이 일반적이다. 따라서, 앞서 nn.Sequential()로 구현했던 로지스틱 회귀 모델을 클래스로도 한번 만들어보자.


```python
class BinaryClassifier(nn.Module): # DL/ML 모델의 기본 구조를 정의해둔 PyTorch 라이브러리의 nn.Module를 상속받는 것에서부터 시작.
  def __init__(self):
    super().__init__() # Parent Class 초기화
    self.linear = nn.Linear(3, 1) # BinaryClassifier 클래스의 인스턴스를 생성하면, 알아서 그 객체에 대한 w, b값이 초기화된다.
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    return self.sigmoid(self.linear(x))
```


```python
model = BinaryClassifier()
```


```python
list(model.parameters())
```




    [Parameter containing:
     tensor([[ 0.1236,  0.1227, -0.2454]], requires_grad=True),
     Parameter containing:
     tensor([-0.4398], requires_grad=True)]




```python
optimizer = optim.SGD(model.parameters(), lr=0.05)
num_epochs = 100

for epoch in range(num_epochs+1):
  predict = model(x_train)

  cost = F.binary_cross_entropy(predict, y_train)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 10 == 0:
    predicted_label = predict >= torch.FloatTensor([0.5])
    acc = predicted_label == y_train
    acc = (acc.sum() / len(acc)).item()
    print(f"Epoch: {epoch}/{num_epochs}, Cost: {cost.item():.3f}, Accuracy: {acc*100:.2f}%")
```

    Epoch: 0/100, Cost: 0.847, Accuracy: 50.00%
    Epoch: 10/100, Cost: 0.550, Accuracy: 66.67%
    Epoch: 20/100, Cost: 0.488, Accuracy: 66.67%
    Epoch: 30/100, Cost: 0.439, Accuracy: 83.33%
    Epoch: 40/100, Cost: 0.399, Accuracy: 83.33%
    Epoch: 50/100, Cost: 0.367, Accuracy: 83.33%
    Epoch: 60/100, Cost: 0.339, Accuracy: 100.00%
    Epoch: 70/100, Cost: 0.316, Accuracy: 100.00%
    Epoch: 80/100, Cost: 0.297, Accuracy: 100.00%
    Epoch: 90/100, Cost: 0.280, Accuracy: 100.00%
    Epoch: 100/100, Cost: 0.265, Accuracy: 100.00%


# Softmax Regression Model

앞서 이야기했듯이 소프트맥스 회귀 모델은 **다중분류** 문제를 풀기 위한 모델이다. 소프트맥스 함수는 **입력으로 들어오는 데이터 크기**만큼 0부터 1사이의 확률값들을 반환하고 이들의 총 합은 1이 되는 특징이 있기 때문에 다중 분류 추론에 대한 확률적 접근이 가능한 것이다. 예컨대 데이터 $z=(z_1, z_2, z_3)$가 소프트맥스의 입력으로 주어지면, 이 데이터에 대해 소프트맥스 함수는 아래와 같은 출력값들을 반환한다.

$$
\begin{equation}
  \text{Softmax}\left(z\right) = \left[\frac{e^{z_1}}{\sum^{3}{e^{z_i}}}, \frac{e^{z_2}}{\sum^{3}{e^{z_i}}}, \frac{e^{z_3}}{\sum^{3}{e^{z_i}}}\right] = \left[p_1, p_2, p_3\right]
\end{equation}
$$

$$
\begin{equation}
  \sum_{i=1}^{3}{p_i} = 1
\end{equation}
$$

이 예시는 3가지 클래스를 확률적으로 분류해내는 상황이라 할 수 있다. 이런 식으로 3가지, 4가지, 5가지... 등의 다중 분류가 가능하다.

이를 신경망 그림으로 표현하자면 아래와 같다.

![png](/assets/img/post/basic_classification/softmax_scheme.png)*Figure 2*

입력층과 출력층으로 구성된 모델은 4개의 입력 요소를 받는다. 이를 바탕으로 3가지 클래스에 대한 분류를 수행하는 모델을 제작하려 한다. 4개의 입력 요소와 3개의 출력 요소를 포괄하기 위해선, nn.Linear(4, 3)을 사용한다. 이후 3개의 출력 요소에 대해서 Softmax 함수를 적용하여 총 합이 1인 $p_1, p_2, p_3$ 확률값을 반환한다.

이를 행렬 수식으로 표현하면 아래와 같다.

$$
\begin{equation}
  X =
  \begin{pmatrix}
    x_{11} & x_{12} & x_{13} & x_{14} \\
    x_{21} & x_{22} & x_{23} & x_{24} \\
    x_{31} & x_{32} & x_{33} & x_{34} \\
    x_{41} & x_{42} & x_{43} & x_{44} \\
    x_{51} & x_{52} & x_{53} & x_{54} \\
  \end{pmatrix}
\end{equation}
$$

$X$가 모델을 1회 학습시킬 때 사용하는 훈련 데이터라고 하자. 이들의 요소를 일반적으로 표현하자면 $x_{hm}$ 으로 나타낼 수 있다. $h$는 학습에 활용한 배치 크기이자 샘플 데이터의 인덱스다. $m$은 (입력층의 뉴런들 중) 몇 번째 뉴런인지를 나타내는 인덱스다. 예컨대, $x_{32}$는 입력층의 2번째 뉴런에 입력되는 3번째 샘플 데이터의 값이라는 의미다.

출력층에는 3개 분류를 수행하기 위해 3개의 뉴런이 있다. 그리고 앞서 입력 행렬의 크기가 5X4 이니, 가중치 행렬 $W$는 아래와 같은 형태임을 알 수 있다.

$$
\begin{equation}
  W =
  \begin{pmatrix}
    w_{11} & w_{12} & w_{13} \\
    w_{21} & w_{22} & w_{23} \\
    w_{31} & w_{32} & w_{33} \\
    w_{41} & w_{42} & w_{43} \\
  \end{pmatrix}
\end{equation}
$$

가중치 행렬도 표현하자면 $w_{mn}$ 으로 나타낼 수 있다. $m$은 입력층에서 몇번째 뉴런인지에 대한 인덱스고, $n$은 출력층의 뉴런 인덱스로 여길 수 있다. 예컨대, $w_{23}$은 **입력층의 2번째 뉴런에서 출력층의 3번째 뉴런 사이의 가중치 값**을 의미한다.

종합하자면, 행렬곱 연산인 nn.Linear(4, 3)을 통해 이뤄지는 연산은 아래와 같다. (Bias는 고려하지 않겠다.)

$$
\begin{equation}
  Z = XW
\end{equation}
$$

$$
\begin{equation}
  =
  \begin{pmatrix}
    x_{11} & x_{12} & x_{13} & x_{14} \\
    x_{21} & x_{22} & x_{23} & x_{24} \\
    x_{31} & x_{32} & x_{33} & x_{34} \\
    x_{41} & x_{42} & x_{43} & x_{44} \\
    x_{51} & x_{52} & x_{53} & x_{54} \\
  \end{pmatrix}
  \times
  \begin{pmatrix}
    w_{11} & w_{12} & w_{13} \\
    w_{21} & w_{22} & w_{23} \\
    w_{31} & w_{32} & w_{33} \\
    w_{41} & w_{42} & w_{43} \\
  \end{pmatrix}
  =
  \begin{pmatrix}
    z_{11} & z_{12} & z_{13} \\
    z_{21} & z_{22} & z_{23} \\
    z_{31} & z_{32} & z_{33} \\
    z_{41} & z_{42} & z_{43} \\
    z_{51} & z_{52} & z_{53} \\
  \end{pmatrix}
\end{equation}
$$

이렇게 계산된 $z_{i1}, z_{i2}, z_{i3}$에 Softmax 함수를 적용한 게 앞서 본 Figure 2 상황인 것이고, 최종 수식의 형태는 아래와 같다. ($i$는 몇 번째 샘플 데이터에 대한 결과인지를 의미)

$$
\begin{equation}
  \text{Softmax}
  \left(
  \begin{pmatrix}
    z_{11} & z_{12} & z_{13} \\
    z_{21} & z_{22} & z_{23} \\
    z_{31} & z_{32} & z_{33} \\
    z_{41} & z_{42} & z_{43} \\
    z_{51} & z_{52} & z_{53} \\
  \end{pmatrix}
  \right)
  =
  \begin{pmatrix}
    p_{11} & p_{12} & p_{13} \\
    p_{21} & p_{22} & p_{23} \\
    p_{31} & p_{32} & p_{33} \\
    p_{41} & p_{42} & p_{43} \\
    p_{51} & p_{52} & p_{53} \\
  \end{pmatrix}
\end{equation}
$$

이 결과를 실제 '정답 레이블에 해당하는 확률 데이터'와 함께 Cross Entropy 손실함수 상에서 비교하며 손실이 줄이는 방향으로 weight(& bias) 와 같은 모델의 학습 매개변수들이 학습되게 되는 것이다.

> **정답 레이블에 해당하는 확률 데이터**: 정해진 특정 클래스에 해당하는 확률이 당연히 1 (100%)인 정답, 예컨대 3가지 분류에 대한 정답 데이터를 [1, 0, 0], [0, 1, 0], [0, 0, 1] 같은 벡터 데이터 형태로 구성해 사용해볼 수 있다. 이렇게 선택지의 개수만큼 차원을 갖도록 하고, 각 인덱스마다 선택지에 대응시켜서, 해당 선택지에 해당할 시 1 그리고 그 외는 전부 0 으로 다중분류 정답 데이터를 확률적으로 표현할 수 있게 된다. 이러한 표현기법을 원-핫 인코딩(One-hot encoding), 이 표현기법을 활용한 벡터를 원-핫 벡터(One-hot vector)라고 한다.
{: .prompt-info }

로지스틱 회귀 모델에서 다룬 이진분류든, 소프트맥스 회귀 모델의 다중분류든, 분류 문제에서 **손실함수**는 아래와 같은 Cross Entropy 함수를 사용한다고 했다.

$$
\begin{equation}
  \text{Cost} = -\sum_{j=1}^{k}y_j\log\left({p_j}\right)
\end{equation}
$$

$y_j$는 정답 확률을 나타내고, 통상 정답 데이터는 원-핫 벡터를 쓰므로 1 또는 0값이 된다. $p_j$는 모델 내 Softmax 함수를 통해 반환된 확률값, 다시 말하자면 주어진 입력값 세트에 대해 모델이 추론한 $j$번째 클래스일 확률을 말한다.

위 식의 표현은 사실 전체 훈련 데이터 중 샘플 한 개에 대해서 학습할 경우의 일반적인 손실함수 형태이고, 학습 주기를 h (batch size)개의 훈련 샘플 데이터 단위씩 학습시키는 보통의 Mini-batch 학습 방식에서는 Cross Entropy 손실함수가 아래와 같은 형태라고 볼 수 있다.

$$
\begin{equation}
  \text{Cost} = -\frac{1}{h}\sum_{i=1}^{h}\sum_{j=1}^{k}{y_{j}^{\left(i\right)}\log{\left(p_{j}^{\left(i\right)}\right)}}
\end{equation}
$$

> Mini-batch 학습 방식에서는 각각의 샘플 데이터 값(x_train, y_train)에 대해서 w, b에 대한 손실함수(Cost) 기울기를 구하고, 1/h를 곱한 평균 Gradient를 w, b 업데이트에 활용한다.
{: .prompt-info }

이 Cross Entropy 손실함수는 당연히 PyTorch 라이브러리에도 구현이 되어 있기 때문에, pytorch.nn.functional.cross_entropy()를 그냥 가져다 쓰면 되지만, 더욱 deep한 이해를 위해 원초적 수준에서부터 손실함수 구현을 직접 해보자.


## Cross Entropy Implementation

### Data Preparation


```python
torch.manual_seed(1)
z = torch.rand(5, 3, requires_grad=True) # 소프트맥스를 거치기 전, 출력층 값이라고 보면 된다. 5 X 3 크기로서 5개의 훈련 데이터 샘플과 3가지 분류 문제라고 보면 된다.
print(f"- 출력층 값들 (아직 소프트맥스-손실함수 과정 진입 X):")
print(z)
print()

prob_of_z = F.softmax(z, dim=1)
print("- 소프트맥스 함수를 거친 후 결과:")
print(prob_of_z)
print()

target_classes = ['고양이', '강아지', '앵무새'] # 아무튼 이런 3가지 분류 문제라는 상황 가정
print(f"- 분류 대상: {target_classes}")
print()


# 랜덤하게 y_train 레이블 데이터 5개 생산 (앞서 사용할 훈련 데이터 샘플이 5개라고 했으니, 이를 일치시키기 위해)
y_labels = torch.randint(0, len(target_classes), size=(5, )) # [0,3) 사이 정수 '5개를 나란히 size=(5,)' 랜덤하게 추출
print(f"- y_train 레이블: {y_labels}")
print()

# 이러한 클래스에 대해 y_train 데이터를 원-핫 벡터로 표현하는 방식
y_one_hots = F.one_hot(y_labels, num_classes=len(target_classes)).float()
print(f"- y_train 레이블을 원-핫 벡터로 표현했을 시:")
print(y_one_hots)
```

    - 출력층 값들 (아직 소프트맥스-손실함수 과정 진입 X):
    tensor([[0.7576, 0.2793, 0.4031],
            [0.7347, 0.0293, 0.7999],
            [0.3971, 0.7544, 0.5695],
            [0.4388, 0.6387, 0.5247],
            [0.6826, 0.3051, 0.4635]], requires_grad=True)
    
    - 소프트맥스 함수를 거친 후 결과:
    tensor([[0.4308, 0.2670, 0.3022],
            [0.3904, 0.1928, 0.4167],
            [0.2764, 0.3951, 0.3284],
            [0.3020, 0.3689, 0.3291],
            [0.4018, 0.2755, 0.3227]], grad_fn=<SoftmaxBackward0>)
    
    - 분류 대상: ['고양이', '강아지', '앵무새']
    
    - y_train 레이블: tensor([0, 0, 0, 2, 1])
    
    - y_train 레이블을 원-핫 벡터로 표현했을 시:
    tensor([[1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.]])


여기까지 기본적인 상황을 구성해봤다. 보다시피 소프트맥스 함수를 거친 데이터셋은 총합이 (행별로 연산시) 1인, 0부터 1사이의 값들이 반환되었다. 분류할 클래스는 3가지 이고, 각 클래스마다 인덱스로 표현된 **고유 레이블** 세트를 랜덤하게 추출했다. 이 때, 인덱스 0이 '고양이'인지, '강아지'인지, 또는 '앵무새'인지는 사전에 본인이 처리하기 나름이다. 그리고 각 클래스를 의미하는 원-핫 벡터로 구성된 y_train 데이터셋도 마련해뒀다.

### Bonus: Making of one-hot vectors

원-핫 벡터 변환에 쓸만한 방법을 잠시 소개한다. scikit-learn 라이브러리 LabelBinarizer 클래스의 fit_transform 메서드라는 편리한 기능이 존재한다.


```python
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()

classes = ['고양이', '강아지', '앵무새', '강아지', '강아지', '고양이', '앵무새']
print(classes)
print()

one_hot_labels = lb.fit_transform(classes)
print(one_hot_labels)
```

    ['고양이', '강아지', '앵무새', '강아지', '강아지', '고양이', '앵무새']
    
    [[0 1 0]
     [1 0 0]
     [0 0 1]
     [1 0 0]
     [1 0 0]
     [0 1 0]
     [0 0 1]]


### Low-level

원초적인 수준부터 손실값(Cost, or Loss)을 도출해보자.

$$
\begin{equation}
  \text{y_one_hots} =
  \begin{pmatrix}
      1 & 0 & 0 \\
      1 & 0 & 0 \\
      1 & 0 & 0 \\
      0 & 0 & 1 \\
      0 & 1 & 0 \\
  \end{pmatrix}
  ,
  \text{Softmax(z)} =
  \begin{pmatrix}
      0.4308 & 0.2670 & 0.3022 \\
      0.3904 & 0.1928 & 0.4167 \\
      0.2764 & 0.3951 & 0.3284 \\
      0.3020 & 0.3689 & 0.3291 \\
      0.4018 & 0.2755 & 0.3227 \\
  \end{pmatrix}
\end{equation}
$$

위에 **데이터 준비** 단계에서 뽑은 데이터를 그대로 옮겨온 것이다.

$$
\begin{equation}
  \text{Cost} = -\frac{1}{h}\sum_{i=1}^{h}\sum_{j=1}^{k}{y_{j}^{\left(i\right)}\log{\left(p_{j}^{\left(i\right)}\right)}}
\end{equation}
$$

Cross Entropy 함수가 이렇다고 했으니, $\text{y_one_hots}$과 $\log{\left(\text{Softmax(z)}\right)}$를 element-wise 곱(또는 아다마르 곱; 같은 크기의 두 행렬에서 같은 위치 성분끼리 곱하는 것)을 하고, row별로 sum한 결과들을 평균내면 손실값을 얻을 수 있을 것 같다.



```python
log_softmax_z = torch.log(prob_of_z)
print("Log Softmax of z:")
print(log_softmax_z)
print()

cost = (-y_one_hots * log_softmax_z).sum(dim=1).mean()
print(cost)
```

    Log Softmax of z:
    tensor([[-0.8421, -1.3204, -1.1967],
            [-0.9405, -1.6459, -0.8753],
            [-1.2858, -0.9285, -1.1134],
            [-1.1972, -0.9973, -1.1114],
            [-0.9118, -1.2893, -1.1309]], grad_fn=<LogBackward0>)
    
    tensor(1.0938, grad_fn=<MeanBackward0>)


### High-level

출력층값, $z$에 소프트맥스 함수를 씌우고 log를 씌우는 작업을 F.log_softmax()로 한 큐에 끝낼 수 있다.

> F 는 torch.nn.functional의 alias였다.
{: .prompt-tip }


```python
print("F.log_softmax(z):")
print(F.log_softmax(z, dim=1))
print()

cost = (-y_one_hots * F.log_softmax(z, dim=1)).sum(dim=1).mean()
print(cost)
```

    F.log_softmax(z):
    tensor([[-0.8421, -1.3204, -1.1967],
            [-0.9405, -1.6459, -0.8753],
            [-1.2858, -0.9285, -1.1134],
            [-1.1972, -0.9973, -1.1114],
            [-0.9118, -1.2893, -1.1309]], grad_fn=<LogSoftmaxBackward0>)
    
    tensor(1.0938, grad_fn=<MeanBackward0>)


다음 과정으로 넘어가기 전에, 앞에서 Cross-entropy를 low-level에서 구현하여 계산할 때 한 가지 특징(?)이 있었다는 점을 짚고 넘어가자.

$$
\begin{equation}
  \text{y_one_hots} =
  \begin{pmatrix}
      1 & 0 & 0 \\
      1 & 0 & 0 \\
      1 & 0 & 0 \\
      0 & 0 & 1 \\
      0 & 1 & 0 \\
  \end{pmatrix}
  ,
  \text{Softmax(z)} =
  \begin{pmatrix}
      0.4308 & 0.2670 & 0.3022 \\
      0.3904 & 0.1928 & 0.4167 \\
      0.2764 & 0.3951 & 0.3284 \\
      0.3020 & 0.3689 & 0.3291 \\
      0.4018 & 0.2755 & 0.3227 \\
  \end{pmatrix}
\end{equation}
$$

두 행렬(y_one_hots 행렬과 log_softmax(z) 행렬)의 element-wise 곱을 하는 과정에서, 결국 제대로 하는 계산은 **정답 레이블 인덱스에 해당하는 log_softmax 값**임을 알 수 있다 즉, 0에 대응하는 항들은 어차피 사라지고 정답 1에 해당하는 항은 log_softmax(z)만 고려하면 그것이 곧 우리가 얻고자 했던 손실 결과값이란 의미다.

이러한 의미를 담아 적용할 수 있는 함수가 Pytorch의 F.nll_loss() 함수다.
> nll은 Negative Log-Likelihood를 의미한다. log-likelihood 함수에 대한 상세한 설명과 관계는 건너 띄고, [이곳 블로그 링크](https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81){:target="_blank"}로 대신하겠다.
{: .prompt-tip }

![png](/assets/img/post/basic_classification/nll_loss_torch.png)*torch.nn.NLLLoss, Source: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#nllloss*

처리되는 방식만 대략적으로 설명하면, negative log probability를 의미하는 $l_n$들을 **정답 레이블 인덱스**(문서 내에서 target이라고 표현하는 부분)에 해당하는 값들만 도출해서 위 계산과정을 수행하는 것이다. 옵션 reduction은 디폴트가 mean인데, 이 의미는 클래스마다의 weight 합을 나눠서 평균내라는 의미인데, weight 디폴트는 1이라 따로 지정하지 않으면 그냥 negative log prob.들의 평균이 된다. 직접 결과들을 뽑아 보는게 더 이해가 잘 될 것이다. 아래 일련의 결과들을 보자.



```python
F.log_softmax(z, dim=1) # log probability 값들
```




    tensor([[-0.8421, -1.3204, -1.1967],
            [-0.9405, -1.6459, -0.8753],
            [-1.2858, -0.9285, -1.1134],
            [-1.1972, -0.9973, -1.1114],
            [-0.9118, -1.2893, -1.1309]], grad_fn=<LogSoftmaxBackward0>)




```python
y_labels # 정답 레이블 인덱스; 이 인덱스 위치에 원-핫 벡터 내 1값 존재했었다.
```




    tensor([0, 0, 0, 2, 1])



다시 말해 nll_loss()를 적용하면 정답 레이블 인덱스들에 해당하는 log probability 값들인, '0.8421, 0.9405, 1.2858, 1.1114, 1.2893' 들만 sum+mean한다는 의미다. 이는 low level로 구현했던 과정과 똑같다.

> nll_loss()에서는 입력받은 로그 확률값에 마이너스를 기본적으로 씌운다.
{: .prompt-tip }


```python
# 레이블 인덱스에 해당하는 log prob.들의 mean
(0.8421 + 0.9405 + 1.2858 + 1.1114 + 1.2893) / 5
```




    1.09382




```python
# nll_loss()에서도 동일한 결과를 냄을 알 수 있음.
F.nll_loss(F.log_softmax(z, dim=1), y_labels, reduction='mean')
```




    tensor(1.0938, grad_fn=<NllLossBackward0>)




```python
# 추가: reduction 옵션을 'sum'으로 하면 샘플 크기만큼 나눠 평균내지 않고, 더하기만 한다.
print(0.8421 + 0.9405 + 1.2858 + 1.1114 + 1.2893)
print(F.nll_loss(F.log_softmax(z, dim=1), y_labels, reduction='sum'))
```

    5.4691
    tensor(5.4691, grad_fn=<NllLossBackward0>)


### High High-level
nll_loss(log_softmax(z, dim=1), y_labels)를 F.cross_entropy()로 대체할 수 있다.

> 손실 함수로 쓰일 F.cross_entropy()에 log_softmax 함수가 이미 포함되어 있음에 주목하자. 이러한 이유로 softmax regression model에서 nn.Softmax() 층을 따로 추가해넣지 않는다. (굳이 모델에 포함시키려면, F.cross_entropy 손실함수 대신 F.nll_loss를 써야겠지.)
{: .prompt-warning }


```python
F.cross_entropy(z, y_labels)
```




    tensor(1.0938, grad_fn=<NllLossBackward0>)



## Softmax Regression Class

'8개의 훈련 데이터 샘플, 입력층에서 받는 4개의 입력값, 0/1/2로 레이블로 구분된 3가지 클래스 분류' 이러한 조건으로 다중 분류 모델을 학습시킨다고 해보자.


```python
torch.manual_seed(1)
```




    <torch._C.Generator at 0x7c842c14d6d0>




```python
x_train = torch.FloatTensor([[1, 2, 1, 1],
                             [2, 1, 3, 2],
                             [3, 1, 3, 4],
                             [4, 1, 5, 5],
                             [1, 7, 5, 5],
                             [1, 2, 5, 6],
                             [1, 6, 6, 6],
                             [1, 7, 7, 7]])

y_labels = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])
```


```python
class SoftmaxClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(4, 3)

  def forward(self, x):
    return self.linear(x)
```


```python
model = SoftmaxClassifier()
```


```python
list(model.parameters())
```




    [Parameter containing:
     tensor([[ 0.2576, -0.2207, -0.0969,  0.2347],
             [-0.4707,  0.2999, -0.1029,  0.2544],
             [ 0.0695, -0.0612,  0.1387,  0.0247]], requires_grad=True),
     Parameter containing:
     tensor([ 0.1826, -0.1949, -0.0365], requires_grad=True)]



> 아니 근데 입력 데이터의 shape은 8X4인데, model.parameters()에서 나타난 weight 행렬의 shape은 3X4 지?? 당황하지말라 (사실 내가 당황했었다). Pytorch에서는 입력받은 데이터와 weight 파라미터 행렬을 matmul하기 전에, weight 행렬을 먼저 transpose한다고 한다고 한다. 다시 말해, output = input.matmul(weight.T())로 계산한다는 것이다.
{: .prompt-tip }


```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
num_epochs = 1000
for epoch in range(num_epochs + 1):

  prediction = model(x_train)

  cost = F.cross_entropy(prediction, y_labels)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 100 == 0:
    acc = (torch.argmax(prediction, dim=1) == y_labels).float().mean()
    print(f"Epoch: {epoch}/{num_epochs}, Cost: {cost:.4f}, Accuracy: {acc*100:.2f}%")

```

    Epoch: 0/1000, Cost: 1.6168, Accuracy: 25.00%
    Epoch: 100/1000, Cost: 0.6589, Accuracy: 62.50%
    Epoch: 200/1000, Cost: 0.5734, Accuracy: 62.50%
    Epoch: 300/1000, Cost: 0.5182, Accuracy: 62.50%
    Epoch: 400/1000, Cost: 0.4733, Accuracy: 75.00%
    Epoch: 500/1000, Cost: 0.4335, Accuracy: 75.00%
    Epoch: 600/1000, Cost: 0.3966, Accuracy: 75.00%
    Epoch: 700/1000, Cost: 0.3609, Accuracy: 87.50%
    Epoch: 800/1000, Cost: 0.3254, Accuracy: 87.50%
    Epoch: 900/1000, Cost: 0.2892, Accuracy: 87.50%
    Epoch: 1000/1000, Cost: 0.2541, Accuracy: 100.00%


> 아니 model의 추론 결과 값들은 (로그-)소프트맥스를 거치지 않은 선형 계산값들인데, 이들을 가지고 소프트맥스 회귀 모델의 정확도는 어떻게 따지고, 학습 이후 test는 어떻게 해? 너무 어렵게 생각하지말라 (내가 그랬다). 학습과정에서 모델의 학습 파라미터들(w, b)은 자연스럽게 정답 레이블에 높은 확률을 부여하도록 학습된다. 그리고 Softmax 함수든, Log Softmax 함수든 monotonic & strictly increasing function 들이다. 즉, 높은 입력값이 주어질 때마다 항상 높은 값을 반환하는 특징을 지녔다는 의미다. 그러니까 $x_1<x_2$일 때, 항상 $f\left(x_1\right)<f\left(x_2\right)$다. 다시 말해서, nn.Linear() 층만 거치고 출력된 값들, 이들의 상대적인 비교만으로 지금 모델이 어떤 레이블을 추론하고 있는지 이야기할 수 있다는 말이다. 그래서 torch.argmax(prediction)로 높은 값이 출력된 인덱스와 정답 레이블 간 비교로 모델을 평가할 수 있는 것이다.
{: .prompt-info }

쓰다보니 이내용저내용 추가하면서 너무 심각하게 글이 길어진 것 같다. 여기서 한번 끊고, 다음 포스팅은 손글씨 이미지 데이터(MNIST)를 학습한 Softmax Regression Model 구현 내용을 올리도록 하겠다.

* * *

# 이어지는 글
1. [Softmax의 안정성, Log Softmax, LogSumExp Function](https://zoshs2.github.io/posts/StableSoftmax/){:target="_blank"}
2. Softmax Regression Model로 MNIST 데이터 분류하기 (업데이트 예정)
