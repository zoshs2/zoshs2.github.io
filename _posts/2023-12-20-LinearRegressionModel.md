---
title: "[PyTorch] Linear Regression Model"
date: 2023-12-20 17:04:50 +0900
categories: [DL/ML, Study]
tags: [PyTorch, Torch, Deep Learning, Linear Regression, Python]
math: true
toc: false
---

# Intro.

**본 공부 내용은 유원준 님, 안상준 님이 저술하신 [PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/book/2788) 위키독스 내용들을 배경으로 하고 있습니다.**
<br><br>

본래 내가 사용하던 DL/ML 프레임워크는 ***Keras***였는데, 최근 프레임워크의 추세가 PyTorch로 빠르게 넘어 가고 있다고 한다. (인공지능 관련 대학원에 재학 중인 내 지인들도 연구실에서 PyTorch를 주력으로 쓴다더라..) 커뮤니티 등에 의하면, PyTorch가 Tensorflow보다 모델 구현이 더 직관적이고 유연성(fine-tuning 작업 등의)이 뛰어나며 전체적으로 쉽다고 하는데... 근데 그건 사람마다의 성향과 익숙함의 차이로 견해가 다를 수도 있을거 같다. 

그보다, 큰 맥락적 부분에서 PyTorch가 주목을 끄는 이유는 최근 NLP/transformer-based model의 급진적인 발전이 큰 역할을 했다고 본다. 오늘날 일상의 커다란 패러다임을 견인한 GPT 모델부터 BERT 모델까지, 전부 PyTorch 로 개발이 이루어졌고, 당연히 지금도 PyTorch 프레임워크로 차세대 모델 개발이 이뤄지고 있다고 한다. 그러다보니 자연스럽게 새로운 응용 연구 및 비즈니스 프로덕트 내 적용에서도 대부분 PyTorch 프레임워크 활용을 선호하는 결과를 낳은게 아닌지 생각된다.

아무튼, 나도 PyTorch를 공부하고자 한다. 새로 접한 프레임워크인 만큼 앞으로 가장 기본적인 선형모델 구현부터 차근차근 공부해 나가고자 한다. 공부를 하면서, 머릿 속에서 고민한 흔적들, 입으로 내뱉는 두서없는 말들까지 전부 여기에 기록으로 남기고자 한다. 
<br><br>

# Table of Contents <!-- omit in toc -->
- [Intro.](#intro)
- [Linear Regression Model](#linear-regression-model)
- [Multi-variable Linear Regression](#multi-variable-linear-regression)
- [PyTorch 제대로 활용하기](#pytorch-제대로-활용하기)
  - [PyTorch - 단순선형회귀](#pytorch---단순선형회귀)
  - [PyTorch - 다중선형회귀](#pytorch---다중선형회귀)
- [PyTorch 모델을 클래스화 시키기](#pytorch-모델을-클래스화-시키기)

<br>

# Linear Regression Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```

$H(x) = Wx + b$

W와 b를 학습하는 모형.
$H(x)$는 Hypothesis 가설 모형을 지칭하는 의미로 $H$라고 함.


```python
x_train = torch.FloatTensor([[1], [2], [3]]) # 학습 입력값
y_train = torch.FloatTensor([[2], [4], [6]]) # 학습 라벨
```


```python
type(x_train)
```
    torch.Tensor



```python
# 모델 초기화 (가중치, 편향)
w = torch.zeros(1, requires_grad=True) # requires_grad는 선언 변수를 앞으로 (기울기를 구해서) 학습시키겠다는 의미 (default: False임)
b = torch.zeros(1, requires_grad=True)
```

$H(x) = 0x + 0$

즉, 학습시킬 모델의 초기 상태는 위와 같은 식임. $W=0$, $b=0$인 상태


```python
SGD_optim = optim.SGD([w, b], lr=0.01) # w, b 는 SGD(Stochastic Gradient Descent) 방식으로 업데이트 시키겠다는 의미 (학습률lr은 1%)
```


```python
num_epochs = 2000 # 학습 데이터셋(x_train, y_train)을 총 2000번 학습시키겠다는 의미

for epoch in range(num_epochs):
  # 학습 데이터를 모두 사용했을 때를 1 epoch이라고 한다. 여기선 mini-batch 학습 방식(학습데이터를 나누어 여러 번 학습시키는 방식)을 사용하지 않았다.

  model = x_train * w + b # 학습시킬 내 모델

  # Cost Function (Error Function, Loss Function, Objective Function 이라고도 함) 정의
  # 선형회귀에서 자주 쓰이는 Mean Squred Error(평균제곱오차; MSE)를 오차 함수로 사용한다.
  # MSE = ( (모델의 예측값 - 실제값) ** 2 ) / n  << n은 데이터 샘플수
  # 즉, mini-batch에서는 batch-size가 곧 n이고, 모든 데이터를 한번에 사용하는 batch 방식에서는 train dataset 의 total size가 n이 된다.
  cost = torch.mean((model - y_train) ** 2)

  # 중요!!! 학습되는 과정!!
  ## 먼저 optimizer 업데이트 룰에서 등장하는 aE/aw(가중치에 대한 기울기) 또는 aE/ab(편향에 대한 기울기)를 0으로 초기화시키겠다는 의미다.
  ## 파이토치(PyTorch)에서는 기울기를 구할때, 현재 상태에서 구한 기울기를 이전에 얻었던 기울기에다가 계속 '누적'시키는 특징이 있기 때문이다. (왜 이렇게 설계했을까? 이유가 있겠지?)
  ## 그래서 (지금 모델에서는) 현재 모델을 어느 방향으로 학습시킬 지는 현재 기준으로 학습시켜야 할 필요가 있으므로, 계속 새롭게 기울기를 구해야할 필요가 있다. 그래서 zero_grad() 초기화를 사용.
  SGD_optim.zero_grad()

  ## cost function 변수의 backward()는 이제 오차를 역전파 시키는 것이다.
  ## 이 말이 뭐냐? 옵티마이저에서 w,b를 업데이트 할 수 있도록, 현재 상태에서의 Error에 대한 w,b 기울기 모델에 전파시켜두라는 것이다.
  ## 이 단계에서 aE/aw, aE/ab이 계산됨.
  ## >> 아래서 직접 계산해본다.
  cost.backward()
  if epoch % 100 == 0 or epoch+1 == num_epochs:
    print(f"w_grad: {w.grad}, b_grad: {b.grad}")

  ## 찐 학습(업데이트)단계.
  ## optimizer.step()은 cost.backward()를 통해 w,b 인스턴스 입력된 w.grad, b.grad를 기반으로
  ## SGD룰을 적용해 new w와 b로 갱신시킨다.
  SGD_optim.step()

  if epoch % 100 == 0 or epoch+1 == num_epochs:
    print(f"Epoch: {epoch if epoch+1 != num_epochs else epoch+1}/{num_epochs}, W: {w.item():.3f}, b: {b.item():.3f}, Cost: {cost.item():.3f}", end='\n\n') # item은 현재 값을 보여준다.
```

    w_grad: tensor([-18.6667]), b_grad: tensor([-8.])
    Epoch: 0/2000, W: 0.187, b: 0.080, Cost: 18.667
    
    w_grad: tensor([-0.0614]), b_grad: tensor([0.1392])
    Epoch: 100/2000, W: 1.746, b: 0.578, Cost: 0.048
    
    w_grad: tensor([-0.0482]), b_grad: tensor([0.1095])
    Epoch: 200/2000, W: 1.800, b: 0.454, Cost: 0.030
    
    w_grad: tensor([-0.0379]), b_grad: tensor([0.0861])
    Epoch: 300/2000, W: 1.843, b: 0.357, Cost: 0.018
    
    w_grad: tensor([-0.0298]), b_grad: tensor([0.0677])
    Epoch: 400/2000, W: 1.876, b: 0.281, Cost: 0.011
    
    w_grad: tensor([-0.0234]), b_grad: tensor([0.0532])
    Epoch: 500/2000, W: 1.903, b: 0.221, Cost: 0.007
    
    w_grad: tensor([-0.0184]), b_grad: tensor([0.0418])
    Epoch: 600/2000, W: 1.924, b: 0.174, Cost: 0.004
    
    w_grad: tensor([-0.0145]), b_grad: tensor([0.0329])
    Epoch: 700/2000, W: 1.940, b: 0.136, Cost: 0.003
    
    w_grad: tensor([-0.0114]), b_grad: tensor([0.0258])
    Epoch: 800/2000, W: 1.953, b: 0.107, Cost: 0.002
    
    w_grad: tensor([-0.0089]), b_grad: tensor([0.0203])
    Epoch: 900/2000, W: 1.963, b: 0.084, Cost: 0.001
    
    w_grad: tensor([-0.0070]), b_grad: tensor([0.0160])
    Epoch: 1000/2000, W: 1.971, b: 0.066, Cost: 0.001
    
    w_grad: tensor([-0.0055]), b_grad: tensor([0.0126])
    Epoch: 1100/2000, W: 1.977, b: 0.052, Cost: 0.000
    
    w_grad: tensor([-0.0043]), b_grad: tensor([0.0099])
    Epoch: 1200/2000, W: 1.982, b: 0.041, Cost: 0.000
    
    w_grad: tensor([-0.0034]), b_grad: tensor([0.0078])
    Epoch: 1300/2000, W: 1.986, b: 0.032, Cost: 0.000
    
    w_grad: tensor([-0.0027]), b_grad: tensor([0.0061])
    Epoch: 1400/2000, W: 1.989, b: 0.025, Cost: 0.000
    
    w_grad: tensor([-0.0021]), b_grad: tensor([0.0048])
    Epoch: 1500/2000, W: 1.991, b: 0.020, Cost: 0.000
    
    w_grad: tensor([-0.0017]), b_grad: tensor([0.0038])
    Epoch: 1600/2000, W: 1.993, b: 0.016, Cost: 0.000
    
    w_grad: tensor([-0.0013]), b_grad: tensor([0.0030])
    Epoch: 1700/2000, W: 1.995, b: 0.012, Cost: 0.000
    
    w_grad: tensor([-0.0010]), b_grad: tensor([0.0023])
    Epoch: 1800/2000, W: 1.996, b: 0.010, Cost: 0.000
    
    w_grad: tensor([-0.0008]), b_grad: tensor([0.0018])
    Epoch: 1900/2000, W: 1.997, b: 0.008, Cost: 0.000
    
    w_grad: tensor([-0.0006]), b_grad: tensor([0.0014])
    Epoch: 2000/2000, W: 1.997, b: 0.006, Cost: 0.000
    

처음 cost.backward()를 해서 w.grad와 b.grad에 현재 (loss에 대한) 기울기로 저장시킨 값, 즉 $\frac{\partial E}{\partial w}$와 $\frac{\partial E}{\partial b}$,들은 각각 -18.667, -8이었다. 수식적으로 이게 어떻게 유도된 값인지 확인해보자.

$$
\begin{equation}
  Cost = \frac{1}{n}\sum({y_{pred} - y_{true}})^2
\end{equation}
$$

우리가 정의했던 Cost Function은 위와 같은 수식의 Mean Squared Error(MSE)였다. $\frac{\partial E}{\partial W}$에 대한 값을 구한다는 의미는, 이렇게 사전에 정의된 Cost Function (Error Function이라고도 함)에 미분을 취해서 현재 W를 대입하여 현 위치의 기울기(gradient)를 뽑아 내겠다는 의미다.

> 딥러닝 공부 시 초반에 항상 등장하는 LOSS(ERROR) vs Weight 에 대한 곡선그래프를 떠올려라. 현재 weight기준에서 어느 방향(+/-)으로 얼만큼 이동할지를 기울기로 판단한다. 
{: .prompt-info }

우리가 3개의 학습 데이터 샘플을 가지고 학습을 했단 걸 반영한다면, $E$(또는 $Cost$) 함수는 아래와 같고, 이를 W(가중치)와 b(편향)에 대해 각각 미분해보자.

$$
\begin{equation}
  Cost = \frac{1}{3}\sum_{i=1}^{3}{Wx_{i}+b}
\end{equation}
$$

$$
\begin{equation}
  \frac{\partial E}{\partial W} = \frac{1}{3}\left(\sum{2x_i}(Wx_i+b-y_{i, true})\right)
\end{equation}
$$

$$
\begin{equation}
  \frac{\partial E}{\partial b} = \frac{1}{3}\left(\sum{2}(Wx_i+b-y_{i, true})\right)
\end{equation}
$$

<br>

> b에 대한 편미분 결과는 $2x_i$가 아님.
{: .prompt-warning }

이렇게 미분한 식에다가 현재의 W, B와 더불어 x_train 입력 벡터값들과 y_train 출력 벡터값(정답값)들을 넣자.

$$
\begin{equation}
  \frac{\partial E}{\partial W} = \frac{1}{3}\left((-4)+(-16)+(-36) \right) = -18.6666666666...
\end{equation}
$$

$$
\begin{equation}
  \frac{\partial E}{\partial b} = \frac{1}{3}\left((-4)+(-8)+(-12) \right) = -8
\end{equation}
$$

이렇게 구한 기울기 값들을 아래 SGD Update rule에 써서 새로 갱신된(학습된) w, b 값을 얻는다.

$$
\begin{equation}
  W_{new} = W - \eta\frac{\partial E}{\partial W} = 0 - (0.01)(-18.6666..) = 0.186666...
\end{equation}
$$

$$
\begin{equation}
  b_{new} = b - \eta\frac{\partial E}{\partial b} = 0 - (0.01)(-8) = 0.08
\end{equation}
$$

<br><br>

# Multi-variable Linear Regression

앞서 배운 것이 1차 선형(변수가 x 한 개)였다면, 이번엔 변수가 여러 개 존재하는 다중변수 선형회귀를 살펴보자. 우리가 학습시키고자 하는 다중선형회귀 모델은 아래와 같다.

$$
\begin{equation}
  H = w_1x_1+w_2x_2+w_3x_3+b
\end{equation}
$$

우리가 학습으로 활용할 데이터셋은 아래와 같다.

$$
\begin{equation}
  X =
  \begin{pmatrix}
    x_{11} & x_{12} & x_{13}  \\
    x_{21} & x_{22} & x_{23}  \\
    x_{31} & x_{32} & x_{33}  \\
  \end{pmatrix}
  ,
  Y =
  \begin{pmatrix}
    y_{1} \\
    y_{2} \\
    y_{3} \\
  \end{pmatrix}
\end{equation}
$$

$x_{23}$은 **두 번째** 데이터 샘플의 $x_3$란 의미다. 즉, 이 경우 총 세 개의 데이터셋을 1회 학습에 사용하는 것이다.

정리하자면, 우리가 학습시키고자 하는 모델 $H$는 이렇게 표기할 수 있다.

$$
\begin{equation}
  H =
  \begin{pmatrix}
    x_{11} & x_{12} & x_{13}  \\
    x_{21} & x_{22} & x_{23}  \\
    x_{31} & x_{32} & x_{33}  \\
  \end{pmatrix}
  \begin{pmatrix}
    w_{1} \\
    w_{2} \\
    w_{3} \\
  \end{pmatrix}
  +
  \begin{pmatrix}
    b \\
    b \\
    b \\
  \end{pmatrix}
\end{equation}
$$

$$
\begin{equation}
  =
  \begin{pmatrix}
  x_{11}w_1+x_{12}w_2+x_{13}w_3+b \\
  x_{21}w_1+x_{22}w_2+x_{23}w_3+b \\
  x_{31}w_1+x_{32}w_2+x_{33}w_3+b \\
  \end{pmatrix}
  =
  \begin{pmatrix}
    y_{1} \\
    y_{2} \\
    y_{3} \\
  \end{pmatrix}
\end{equation}
$$

이 경우를 PyTorch로 구현해보자.


```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88 ,93],
                             [89, 91, 80]])
y_train = torch.FloatTensor([[152],
                             [185],
                             [180]])
W = torch.zeros((3, 1), requires_grad=True) # 총 학습데이터셋 크기가 아니라, 당연히 입력변수 갯수에 차원을 맞춰야 한다. (이 경우 3X1)
b = torch.zeros(1, requires_grad=True) # 편향은 변수 1개이므로 브로드캐스팅하면 됨
```


```python
SGD_optim = optim.SGD([W, b], lr=1e-5) # 학습률 0.00001
```


```python
num_epochs = 100
for epoch in range(num_epochs):
  model = x_train.matmul(W) + b

  # Cost Function
  Cost = torch.mean((model - y_train)**2)

  # Learning
  SGD_optim.zero_grad() # 학습시킬 파라미터들(W, b)의 기울기는 항상 초기화
  Cost.backward() # 현재 W, b에 대한 기울기 계산 및 w.grad, b.grad에 저장
  if epoch % 10 == 0:
    print(f"W 기울기: {W.grad.squeeze()}, b 기울기: {b.grad.squeeze()}")
  SGD_optim.step() # W, b 갱신

  if epoch % 10 == 0:
    print(f"Epoch: {epoch}/{num_epochs}, Model: {model.squeeze().detach()}, Cost: {Cost.item()}", end='\n\n')

```

    W 기울기: tensor([-29547.3340, -29880.0000, -28670.0000]), b 기울기: -344.66668701171875
    Epoch: 0/100, Model: tensor([0., 0., 0.]), Cost: 29909.666015625
    
    W 기울기: tensor([-111.6118, -107.2457,  -84.7163]), b 기울기: -0.9918721914291382
    Epoch: 10/100, Model: tensor([154.2491, 185.2458, 176.0172]), Cost: 6.993833541870117
    
    W 기울기: tensor([-10.0125,  -4.4648,  13.8864]), b 기울기: 0.19413232803344727
    Epoch: 20/100, Model: tensor([154.7782, 185.8788, 176.6342]), Cost: 6.606343746185303
    
    W 기울기: tensor([-9.6506, -4.0808, 14.1765]), b 기울기: 0.19814038276672363
    Epoch: 30/100, Model: tensor([154.7767, 185.8747, 176.6458]), Cost: 6.57510232925415
    
    W 기울기: tensor([-9.6341, -4.0461, 14.1316]), b 기울기: 0.1981201171875
    Epoch: 40/100, Model: tensor([154.7734, 185.8685, 176.6553]), Cost: 6.5441718101501465
    
    W 기울기: tensor([-9.6207, -4.0147, 14.0837]), b 기울기: 0.19805908203125
    Epoch: 50/100, Model: tensor([154.7701, 185.8622, 176.6648]), Cost: 6.513326168060303
    
    W 기울기: tensor([-9.6055, -3.9818, 14.0380]), b 기울기: 0.19801855087280273
    Epoch: 60/100, Model: tensor([154.7667, 185.8560, 176.6743]), Cost: 6.482738494873047
    
    W 기울기: tensor([-9.5912, -3.9498, 13.9914]), b 기울기: 0.197967529296875
    Epoch: 70/100, Model: tensor([154.7634, 185.8499, 176.6837]), Cost: 6.45227575302124
    
    W 기울기: tensor([-9.5793, -3.9205, 13.9427]), b 기울기: 0.19788622856140137
    Epoch: 80/100, Model: tensor([154.7601, 185.8437, 176.6930]), Cost: 6.422029495239258
    
    W 기울기: tensor([-9.5630, -3.8868, 13.8985]), b 기울기: 0.19785547256469727
    Epoch: 90/100, Model: tensor([154.7568, 185.8376, 176.7024]), Cost: 6.3919358253479
    


가중치(W)에 대해 학습할 파라미터가 3개라고 해봤자, $\frac{\partial E}{\partial w_1}$, $\frac{\partial E}{\partial w_2}$, $\frac{\partial E}{\partial w_3}$같이 미분을 따로 따로 해서 현재 값들을 대입 후 기울기를 얻어내는 거밖에 없다.
<br><br>

# PyTorch 제대로 활용하기

앞서, 선형회귀 모델을 학습시킬 때 우리는 아래와 같은 걸 미리 정의해두고 사용했다.

$$
\begin{equation}
  Model = XW + B \\
  \\
  Cost_{MSE} = \frac{1}{n}\sum(y_{model} - y_{true})^2
\end{equation}
$$

그런데, PyTorch에는 사실 이런 것들이 이미 구현되어 있고, 이들을 활용하는게 당연히 효율적이겠다.

```python
model = nn.Linear(input_dim, output_dim)
Cost = F.mse_loss(prediction, y_train)
```

이제 앞서, 실습했던 선형회귀, 다중선형회귀 내용들을 PyTorch하게 다시 구현해보자.
<br><br>

## PyTorch - 단순선형회귀

아래와 같은, (편향 bias도 없는) 단순한 선형회귀 모델을 학습시켜보자.

$$
\begin{equation}
  y = 2x
\end{equation}
$$

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
```

```python
# 훈련 데이터 준비
x_train = torch.FloatTensor([[1],
                             [2],
                             [3]])
y_train = torch.FloatTensor([[2],
                             [4],
                             [6]])
```

```python
# nn.Linear(input_features_num, output_features_num)
# 입력값 1개가 들어와서 출력값 1개를 뱉는다는 걸 미리 정해주는 것이다.
model = nn.Linear(1, 1)
```


```python
# 이렇게 생성된 모델안에는 이미 w, b 가 내장되어 있다.
# 입력값 '1개', 출력값 '1개'인 '선형회귀모델'라는 정보만으로 적절한 차원을 가진 학습 파라미터들을 w, b 준비해둔다.
# model.paramters()로 확인해보면, 학습시킬 파라미터들을 생성시켜둔 걸 알 수 있다.
# 'requires_grad=True' 로 이미 설정되어 있음에 주목하자.
list(model.parameters())
```




    [Parameter containing:
     tensor([[0.9933]], requires_grad=True),
     Parameter containing:
     tensor([-0.7766], requires_grad=True)]




```python
# 옵티마이저 - 오차를 줄여나갈 전략을 설정 (여기선 SGD 방식 사용)
SGD_optimizer = optim.SGD(model.parameters(), lr=0.01)
```


```python
num_epochs = 2000 # 2000회 학습
for epoch in range(num_epochs+1):
  if epoch == 0:
    continue

  prediction = model(x_train) # model에 입력값을 넣고, 현 w,b에 대해 예측값을 얻어내는 모습

  # Cost (Error) 계산
  cost = F.mse_loss(prediction, y_train)

  # model.paramters 들의 gradient를 0으로 초기화 (중요: gradient 값은 항상 현 w,b 에 대해서 다시 구해야한다.)
  SGD_optimizer.zero_grad()

  # 오차를 역전파시켜서 모든 학습 파라미터들의 기울기들을 계산
  cost.backward()

  # 계산된 기울기들을 바탕으로, 학습 파라미터들 갱신
  SGD_optimizer.step()

  if epoch % 100 == 0:
    print(f"Epoch: {epoch}/{num_epochs}, Cost: {cost.item():.3f}")
```

    Epoch: 100/2000, Cost: 0.007
    Epoch: 200/2000, Cost: 0.004
    Epoch: 300/2000, Cost: 0.003
    Epoch: 400/2000, Cost: 0.002
    Epoch: 500/2000, Cost: 0.001
    Epoch: 600/2000, Cost: 0.001
    Epoch: 700/2000, Cost: 0.000
    Epoch: 800/2000, Cost: 0.000
    Epoch: 900/2000, Cost: 0.000
    Epoch: 1000/2000, Cost: 0.000
    Epoch: 1100/2000, Cost: 0.000
    Epoch: 1200/2000, Cost: 0.000
    Epoch: 1300/2000, Cost: 0.000
    Epoch: 1400/2000, Cost: 0.000
    Epoch: 1500/2000, Cost: 0.000
    Epoch: 1600/2000, Cost: 0.000
    Epoch: 1700/2000, Cost: 0.000
    Epoch: 1800/2000, Cost: 0.000
    Epoch: 1900/2000, Cost: 0.000
    Epoch: 2000/2000, Cost: 0.000



```python
# 원하는 만큼 학습을 시켰으니, 모델이 제대로 예측을 잘 수행하는지 확인해보자.
x_test = torch.FloatTensor([[4.0]])

pred_y = model(x_test) # 학습된 model에 그냥 집어넣으면 된다.

print(f"선형회귀모델 훈련 후, 입력값 4에 대한 예측값: \n{pred_y}")
```

    선형회귀모델 훈련 후, 입력값 4에 대한 예측값: 
    tensor([[8.0017]], grad_fn=<AddmmBackward0>)



```python
# 파라미터들(w,b)은 어떤 값으로 학습되었는지 확인해보자.
list(model.parameters())
```

    [Parameter containing:
     tensor([[2.0010]], requires_grad=True),
     Parameter containing:
     tensor([-0.0023], requires_grad=True)]



최종 학습된 단순 선형 회귀 모델
$$
\begin{equation}
  My Model = 2.0010x - 0.0023
\end{equation}
$$
<br><br>

## PyTorch - 다중선형회귀
<br>

```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88 ,93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152],
                             [185],
                             [180],
                             [196],
                             [142]])
```


```python
# model 선언
model = nn.Linear(3, 1) # 이번엔 입력되는 변수가 3개이고, 출력값은 1개이다
# (학습전) 학습 파라미터
list(model.parameters())
```




    [Parameter containing:
     tensor([[ 0.0839, -0.4786, -0.4261]], requires_grad=True),
     Parameter containing:
     tensor([-0.2034], requires_grad=True)]




```python
SGD_optim = optim.SGD(model.parameters(), lr=1e-5)
```


```python
num_epochs = 2000
for epoch in range(num_epochs+1):
  if epoch == 0:
    continue

  prediction = model(x_train)

  cost = F.mse_loss(prediction, y_train)

  SGD_optim.zero_grad()
  cost.backward()
  SGD_optim.step()

  if epoch % 100 == 0:
    print(f"Epoch: {epoch}/{num_epochs}, Cost: {cost.item():.4f}")
```

    Epoch: 100/2000, Cost: 0.2176
    Epoch: 200/2000, Cost: 0.2162
    Epoch: 300/2000, Cost: 0.2149
    Epoch: 400/2000, Cost: 0.2136
    Epoch: 500/2000, Cost: 0.2124
    Epoch: 600/2000, Cost: 0.2112
    Epoch: 700/2000, Cost: 0.2101
    Epoch: 800/2000, Cost: 0.2090
    Epoch: 900/2000, Cost: 0.2080
    Epoch: 1000/2000, Cost: 0.2071
    Epoch: 1100/2000, Cost: 0.2062
    Epoch: 1200/2000, Cost: 0.2053
    Epoch: 1300/2000, Cost: 0.2045
    Epoch: 1400/2000, Cost: 0.2037
    Epoch: 1500/2000, Cost: 0.2030
    Epoch: 1600/2000, Cost: 0.2023
    Epoch: 1700/2000, Cost: 0.2016
    Epoch: 1800/2000, Cost: 0.2009
    Epoch: 1900/2000, Cost: 0.2003
    Epoch: 2000/2000, Cost: 0.1997



```python
# 테스트 데이터 준비
x_test = torch.FloatTensor([[73, 80, 75]])
```

원래 model 학습은 테스트 데이터까지 잘 예측해야 학습이 완료되었다고 말한다. 그래서 일반적으론 보유한 데이터 전부를 오로지 학습에만 사용하는 것이 아니라 training(+validation), test 셋으로 split해서 사용하게 된다.


```python
y_pred = model(x_test)
print(f"다중선형회귀 모델 학습 후, (73, 80, 75) 값에 대한 y값의 예측: {y_pred.item():.5f}")
```

    다중선형회귀 모델 학습 후, (73, 80, 75) 값에 대한 y값의 예측: 151.26434



```python
# (학습후) 학습 파라미터
list(model.parameters())
```

    [Parameter containing:
     tensor([[1.0176, 0.4781, 0.5190]], requires_grad=True),
     Parameter containing:
     tensor([-0.1914], requires_grad=True)]



최종 학습된 **다중 선형 회귀 모델**
$$
\begin{equation}
  My Model = 1.0176x_1 + 0.4781x_2 + 0.519x_3 - 0.1914
\end{equation}
$$
<br><br>

# PyTorch 모델을 클래스화 시키기

딥러닝 모델은 거의 대부분 클래스화 시켜 구현한다. 유지/관리/확장/가독성 측면에서 이점이 많기 때문이다. 앞으로 모든 DL/ML 모델을 **파이썬 클래스**로 구현하는 버릇을 들이고, 이 파트에서 다루는 코드 구문 및 구조들은 그 토대가 되므로 반드시 머릿 속에 넣어두자.

앞서, (다중) 선형회귀 모델을 다음과 같이 구현했었다.
```python
model = nn.Linear(3, 1)
```


```python
# 모델을 클래스로 구현하기
class Multivariate_LinearRegression_Model(nn.Module): # nn.Module 클래스의 기본 속성들을 상속받음
  def __init__(self):
    # 모델의 기본 구조와 동작을 정의하는 Constructor(생성자)이다. 클래스 객체를 생성시 가장 먼저 호출되어 실행된다.

    # super()는 상속받는 부모클래스를 지칭하는 것이고, 부모클래스에서 초기화시킨 값들을 여기 클래스에서도 지니고 있으라는 의미다.
    super().__init__()

    # 모델의 기본 구조 정의
    # 입력값 3개를 받으면, 출력값 1개를 내뱉는 선형 모델
    self.linear = nn.Linear(3, 1)

  def forward(self, x):
    # 주어진 입력데이터를 모델에 흘려보내서 예측값을 return시키라는 의미
    # 순전파(Forwarding Propagation)
    return self.linear(x)
```


```python
model = Multivariate_LinearRegression_Model()
```


```python
list(model.parameters()) # 이게 되는 이유가 nn.Module의 모든 기능을 다 상속받았기 때문이다.
```




    [Parameter containing:
     tensor([[ 0.3678,  0.1006, -0.1698]], requires_grad=True),
     Parameter containing:
     tensor([-0.0518], requires_grad=True)]




```python
SGD_optim = optim.SGD(model.parameters(), lr=1e-5)
```


```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88 ,93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152],
                             [185],
                             [180],
                             [196],
                             [142]])

num_epochs = 2000
for epoch in range(num_epochs+1):
  if epoch == 0:
    continue

  prediction = model.forward(x_train) # 중요!! nn.Module의 이미 모든 것을 가지고 있기에 model(x_train)해도 된다.
  cost = F.mse_loss(prediction, y_train)

  SGD_optim.zero_grad() # 기울기 누적되지 않도록, 이전 위치 w,b에 대한 기울기 초기화
  cost.backward() # 현재 위치 w,b에 대한 기울기 계산
  SGD_optim.step() # w,b 갱신

  if epoch % 100 == 0:
    print(f"Epoch: {epoch}/{num_epochs}, Cost: {cost.item():.5f}")
```

    Epoch: 100/2000, Cost: 0.62474
    Epoch: 200/2000, Cost: 0.60068
    Epoch: 300/2000, Cost: 0.57787
    Epoch: 400/2000, Cost: 0.55627
    Epoch: 500/2000, Cost: 0.53581
    Epoch: 600/2000, Cost: 0.51644
    Epoch: 700/2000, Cost: 0.49808
    Epoch: 800/2000, Cost: 0.48069
    Epoch: 900/2000, Cost: 0.46423
    Epoch: 1000/2000, Cost: 0.44862
    Epoch: 1100/2000, Cost: 0.43383
    Epoch: 1200/2000, Cost: 0.41984
    Epoch: 1300/2000, Cost: 0.40657
    Epoch: 1400/2000, Cost: 0.39401
    Epoch: 1500/2000, Cost: 0.38211
    Epoch: 1600/2000, Cost: 0.37084
    Epoch: 1700/2000, Cost: 0.36016
    Epoch: 1800/2000, Cost: 0.35005
    Epoch: 1900/2000, Cost: 0.34047
    Epoch: 2000/2000, Cost: 0.33139



```python
list(model.parameters()) # 학습 시간 이후의 학습된 파라미터 값들
```

    [Parameter containing:
     tensor([[0.9865, 0.6091, 0.4186]], requires_grad=True),
     Parameter containing:
     tensor([-0.0442], requires_grad=True)]

마지막으로, 다중 선형 회귀 모델에서 다뤘던 아래 수식은
$$
\begin{equation}
  H(x) = w_1x_1 + w_2x_2 + w_3x_3 + b
\end{equation}
$$

퍼셉트론 신경망 형식으론 다음과 같이 표현할 수 있다. 

![png](/assets/img/post/pytorch_linear)