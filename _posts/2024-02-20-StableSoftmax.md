---
title: "Softmax의 안정성, Log Softmax, LogSumExp Function"
date: 2024-02-20 19:24:50 +0900
categories: [DL/ML, Study]
tags: [Softmax, LogSoftmax, Log, Information, LogSumExp, LSE, PyTorch, Torch, Deep Learning, Machine Learning, Python]
math: true
toc: false
---

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Softmax Function Stability](#softmax-function-stability)
- [Overflow and Underflow Problem](#overflow-and-underflow-problem)
- [Log Softmax Function](#log-softmax-function)
- [Conclusions: 이들의 의미](#conclusions-이들의-의미)


# Softmax Function Stability

Softmax 함수의 기본 형태는 아래와 같다.

$$
\begin{equation}
  p_j = \frac{e^{x_j}}{\sum_{i=1}^{k}e^{x_i}}
\end{equation}
$$

일반적으로 Softmax 함수는 $X=\{x_1, x_2, x_3, ..., x_k\}$같은 데이터 값 원소들을 담은 집합에 대해 적용되며, Softmax 함수를 거친 집합의 outputs은 0부터 1사이의 값을 갖고 이들을 모두 더하면 1이 되는 특징을 갖기 때문에 확률적 추론(클래스 분류 등)을 하는 모델에 주로 활용한다.


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Original Softmax
def softmax(x):
  exp_x = np.exp(x)
  return exp_x / exp_x.sum()

X = np.array([1,2,3,4])
softmax(X)
```


    array([0.0320586 , 0.08714432, 0.23688282, 0.64391426])



# Overflow and Underflow Problem

Softmax 함수에는 특징이 있다. 보다시피 분모/분자 모두에 지수함수(exponential)가 있다는 점이다. 지수함수는 변수인 지수 증가에 따라 그에 대한 값도 [매우 빠르게 증가](https://namu.wiki/w/%EC%A7%80%EC%88%98%ED%95%A8%EC%88%98#:~:text=%EC%96%B4%EB%96%A4%20%ED%98%84%EC%83%81%EC%9D%B4%EB%82%98%20%EC%88%98%EC%B9%98%EA%B0%80%20%EA%B0%91%EC%9E%90%EA%B8%B0%20%EB%8A%98%EC%96%B4%EB%82%98%EB%8A%94%20%EC%96%91%EC%83%81%5B14%5D%EC%9D%84%20%EC%98%81%EC%96%B4%EB%A1%9C%EB%8A%94%20exponential%20growth){:target="_blank"}한다. 이는 그냥 Softmax 함수가 그렇게 생겨 먹은거라 Softmax의 잘못은 없지만, 컴퓨터에서 Softmax 계산을 돌리고자 할 땐 골치아픈 문제가 발생한다.

크게 두 가지인데, 첫 번째는 $e^x$를 계산하면서 계산 값이 너무 커져서 컴퓨터가 저장 및 표현할 수 있는 정수 범위(대략 1.7964120280206387e+308)를 초과해버리는 **Overflow** 문제다. 자연상수에 대한 지수함수는 대략 $exp(709.782)$이 그 한계값이며, 지수값이 709.783 이상이면 모두 **inf**로 반환된다. 두 번째 문제는 반대로 계산 값이 0에 가깝도록 너무 작아져서 생기는 **부동소수점 언더플로우**(Floating Point Underflow) 문제다. 오버플로우와 비슷한 맥락으로 부동소수점 자릿수를 표현할 수 있는 컴퓨터 표현 범위를 넘어가서 생기는 문제다. 보통 이 경우 0 또는 NaN, 최소 표현값으로 처리해 반환되는 것이 일반적이다.


```python
# Overflow
print(f"Before Overflow: {np.exp(709.782)}")
print(f"After Overflow: {np.exp(709.783)}")
print()
```

    Before Overflow: 1.7964120280206387e+308
    After Overflow: inf
    


    <ipython-input-40-4b1f5d860aa1>:3: RuntimeWarning: overflow encountered in exp
      print(f"After Overflow: {np.exp(709.783)}")



```python
# Floating Point Underflow
print(f"Before Underflow: {np.exp(-745)}, Check if zero: {np.exp(-745)==0}")
print(f"After Underflow: {np.exp(-746)}, Check if zero: {np.exp(-746)==0}")
```

    Before Underflow: 5e-324, Check if zero: False
    After Underflow: 0.0, Check if zero: True


결과적으로 입력 데이터 값들 중 어떤 $exp(x_i)$ 요소 하나라도 Overflow 문제 (또는 sum of exp가 Overflow 되는 문제)에 직면한다면, Softmax 출력값들이 (분모가 inf라서) 전부 0이 되는 상황과, 모든 $exp(x_i)$ 요소가 전부 Underflow 문제에 직면해서 (분모가 0이 되어) 값을 정의할 수 없게 되는 상황(NaN)이 발생할 수 있는 것이다.

> NumPy는 np.array([0])으로 나누는 상황과 np.array([np.inf]) / np.array([np.inf]) 상황에서 **RuntimeWarning: invalid value encountered in divide** 에러 문구를 낸다.
{: .prompt-info }

이러한 문제 상황을 해결하기 위한 방법으로, Softmax에 적용되는 입력 데이터를 Max값으로 조정하는 방법이 있다. 예를 들어, $X = \{400,500,600,700,710\}$ 같은 오버플로우를 발생시키는 입력 데이터가 주어졌을 때, $Max(X)$인 710을 모든 값에서 빼주고 이를 Softmax 함수에 적용하는 것이다. 이렇게 해도 처음 $X$ 값들을 대입한 결과와 동일하기 때문이다.

$$
\begin{equation}
  \text{Softmax}(x_i) = \frac{exp(x_i)}{\sum_{j=1}^{k}{exp(x_j)}}
\end{equation}
$$

$$
\begin{equation}
  = \frac{Cexp(x_i)}{C\sum_{j=1}^{k}{exp(x_j)}} = \frac{exp(x_i+lnC)}{\sum_{j=1}^{k}{exp(x_j+lnC)}}
\end{equation}
$$

$$
\begin{equation}
  = \frac{exp(x_i+C^*)}{\sum_{j=1}^{k}{exp(x_j+C^*)}}
\end{equation}
$$

따라서, 이 방법을 이용하여 모든 요소들의 $exp(x)$가 inf되지 않도록 할 수 있고(Overflow 문제 방지), 또 최대값이었던 요소가 0이 되어 $exp(0)=1$로서 분모가 0인 나눗셈을 하게 되는 난처한 Underflow 문제를 모면할 수 있다.


```python
# Overflow 문제 발생 예시
X1 = np.array([600,700,800])
print(softmax(X1)) # np.inf / np.inf = NaN
```

    [ 0.  0. nan]

    <ipython-input-2-52341cec6f66>:3: RuntimeWarning: overflow encountered in exp
      exp_x = np.exp(x)
    <ipython-input-2-52341cec6f66>:4: RuntimeWarning: invalid value encountered in divide
      return exp_x / exp_x.sum()



```python
# Underflow 문제 발생 예시
X2 = np.array([-1000, -900, -800])
print(softmax(X2)) # np.array([0]) / np.array([0]) = NaN
```

    [nan nan nan]

    <ipython-input-2-52341cec6f66>:4: RuntimeWarning: invalid value encountered in divide
      return exp_x / exp_x.sum()



```python
# 개선된 Softmax Func: Max(X) 값으로 입력 데이터 조절
def adj_softmax(x):
  exp_x = np.exp(x - np.max(x))
  return exp_x / exp_x.sum()

# Overflow 문제 해소
print(adj_softmax(X1))

# Underflow 문제 해소
print(adj_softmax(X2))
```

    [1.38389653e-87 3.72007598e-44 1.00000000e+00]
    [1.38389653e-87 3.72007598e-44 1.00000000e+00]


0으로 나누는 상황과 np.inf/np.inf를 나누는 상황에서 마주했던 오류 문구가 더 이상 발생하지 않는 것을 확인할 수 있다. 이렇게 Softmax 함수를 stablize(안정화)할 수 있다.

# Log Softmax Function

다른 한 가지 방법이 더 있다. Softmax 함수에 log를 씌워 값의 스케일을 낮추는 방법이다. 큰 $exp(x)$값을 표현하기 힘든 컴퓨터의 한계를 $ln$ $e^x=x$로 작은 수로 표현하도록 하는 것이다. 이렇게 메모리 효율성 뿐만 아니라 계산 속도/안정성 측면의 이점도 취할 수 있다.

**invalid value encoutered in divide**문제가 있던 나눗셈 계산 상황을 log를 취함으로서 단순 산술(덧셈/뺄셈)으로 바꿀 수 있다.

$$
\begin{equation}
  \text{LogSoftmax}(x_i) = ln\left(\frac{exp(x_i)}{\sum_{j=1}^{k}{exp(x_j)}}\right)
\end{equation}
$$

$$
\begin{equation}
  = ln(exp(x_i)) - ln\left(\sum_{j=1}^{k}{exp(x_j)}\right) = x_i - ln\left(\sum_{j=1}^{k}{exp(x_j)}\right)
\end{equation}
$$

결과적으로 0과 np.inf로 나누는 문제를 미연에 방지함으로써 안정성을 갖게 되고, 단순 덧셈/뺄셈의 산술 계산으로 계산속도 측면의 이점도 있다.

> 최종 식을 보면 대충 짐작할 수 있는 부분이 있는데, LogSoftmax 함수값은 $x_i - max(X)$ 값에 근사할 거 같다는 점이다. 두번째 항인 $ln(\sum{exp})$에서 결국 $exp(max(X))$값이 dominant 할 것이기 때문이다.
{: .prompt-tip }

```python
def log_softmax(x):
  return x - np.log(np.sum(np.exp(x)))

X = np.random.uniform(-100, 100, 10)
print(f"Input: {X}", end='\n\n')
print(f"LogSoftMax: {log_softmax(X)}", end='\n\n')
print(f"x_i-max(X): {X-np.max(X)}")
```

    Input: [-70.09513588 -86.78997926  54.10601059  86.76733716  39.11617457
       8.60793719  89.87944752 -56.00961407 -82.64056151 -10.24505867]
    
    LogSoftMax: [-1.60018128e+02 -1.76712972e+02 -3.58169819e+01 -3.15565530e+00
     -5.08068179e+01 -8.13150553e+01 -4.35449371e-02 -1.45932607e+02
     -1.72563554e+02 -1.00168051e+02]
    
    x_i-max(X): [-159.9745834  -176.66942678  -35.77343694   -3.11211036  -50.76327295
      -81.27151034    0.         -145.8890616  -172.52000903 -100.12450619]


하지만 여전히 Overflow/Underflow 문제는 존재한다. LogSoftmax는 메모리 효율적이고, 계산속도가 빠르고, (0으로 나누고, np.inf로 나누는 문제가 없기에) 보다 안정적이지만, 어찌됐건 계산에 $exp(x)$가 들어가 있어 누구든 이를 마주한 순간 Overflow/Underflow의 늪에서 빠져나올 수 없다.

> 참고로, Floating Point Underflow로 sum of exp결과가 log(0)이 되면 NumPy에서는 **"zero encountered in log"**라는 RuntimeWarning 경고문구가 뜬다.
{: .prompt-warning}

그래서 궁극적으로 이 경우에도 입력 데이터 전체에 $Max(X)$를 빼주는 방식을 사용하게 된다. 그리고 **당연히 그 결과는 처음 X 데이터에 대한 LogSoftmax값과 동일하다**.

$$
\begin{equation}
  \text{LogSoftmax}\left(x_i - C^*\right) = ln\left(\frac{exp\left(x_i-C^*\right)}{\sum_{j}{exp\left(x_j-C^*\right)}} \right)
\end{equation}
$$

$$
\begin{equation}
  = ln\left(exp\left(x_i-C^*\right)\right) - ln\left[exp\left(x_1-C^* \right)+exp\left(x_2-C^*\right)+... \right]
\end{equation}
$$

$$
\begin{equation}
  = \left(x_i - C^*\right) - ln\left[exp\left(x_1-C^* \right)+exp\left(x_2-C^*\right)+... \right]
\end{equation}
$$


```python
def log_adj_softmax(x):
  adj_x = x - np.max(x)
  return adj_x - np.log(np.sum(np.exp(adj_x)))

X = np.array([700, 800, 900, 1000])
print(f"기본 log softmax: {log_softmax(X)}")
```

    기본 log softmax: [-inf -inf -inf -inf]

    <ipython-input-88-4c2262f2e634>:6: RuntimeWarning: overflow encountered in exp
      return x - np.log(np.sum(np.exp(x)))



```python
print(f"보정된 log softmax: {log_adj_softmax(X)}")
```

    보정된 log softmax: [-300. -200. -100.    0.]


이런 식으로 본래 softmax 함수가 지녔던 Overflow/Underflow에 대한 안정성 문제, 컴퓨터 비용 측면의 메모리 효율 문제, 연산 속도 문제들을 해결(또는 우회)할 수 있는 방법들이 존재한다.

# Conclusions: 이들의 의미
Log Softmax 함수에서 도출됐던 식을 다시 보자.

$$
\begin{equation}
  \text{LogSoftmax}(x_i) = x_i - \log\left(\exp\left({x_1}\right)+...+\exp\left({x_n}\right)\right)
\end{equation}
$$

사실 여기서 로그(log)로 씌워진 두 번째 항, $\log(\sum{exp})$는 LogSumExp Function (소위 LSE 함수)이라는 명칭이 있는 함수다. 즉, 따라서 Log Softmax 함수는 결국 아래의 형태인 것이다.

$$
\begin{equation}
  \text{LogSoftmax}\left(x_i\right) = x_i - \text{lse}(X)
\end{equation}
$$

이게 어떤 의미가 있을까?

LSE 함수는 **differentiable**(미분 가능)하며 **convex**하다는 특징이 있다.

> 참고로, LSE의 편미분은 Softmax 함수다.
{: .prompt-tip }

$$
\begin{equation}
  \nabla_{x_i}\text{LSE} = \frac{\partial}{\partial x_i}\left(\log\left(\sum_{j=1}^{n}{\exp\left(x_j\right)}\right)\right) = \frac{\exp\left(x_i\right)}{\sum_{j=1}^{n}{\exp\left(x_j\right)}} = \text{Softmax}\left(x_i\right)
\end{equation}
$$

이 특징들의 의미는 곧 DL/ML 모델 학습에서 Gradient를 계산하고 Global minimum으로 모델을 최적화할 수 있다는 가능성을 시사한다. Back propagation을 통해 손실 함수(Cost function)의 기울기, $\frac{\partial \text{Cost}}{\partial W}$, 를 계산할 때  $\text{Cost}\left(f\left(wx+b\right)\right)$의 $f$함수는 결국 미분 가능한 함수여야 하기 때문이다.

그렇기 때문에, LSE 함수는 DL/ML 모델의 활성화 함수(activation function)로 활용하기 좋은 함수가 되고, 이를 포함하는 Log Softmax 함수 또한 $\log\left(\text{softmax}\right)$ -> $\log(\text{Probability})$ - 즉 정보이론 분야의 Shannon Information(Information Content, Self-Information, Surprisal; 여러 명칭으로 불림)란 의미를 담고 있다는 점에서 Negative Log Likelihood Function과 결합해 활용하면, 실제 정답값 분포와 모델 예측값 분포 사이 오차를 줄여 나가는 방식의 지도학습을 수행할 수 있게 된다.

> Shannon Information의 수학적 정의는 사실 $-\log\left(P\right)$ 로 마이너스(-)가 붙는다.
{: .prompt-tip }

종합하자면 Log Softmax를 쓰는 이유는 앞서 살펴본 안정성, 메모리 효율, 계산속도 측면에서 이점이 있으면서도, 손실 함수와 결합하여 모델의 학습 방향을 결정하는 중요한 지표로 활용할 수 있기 때문에 그 활용 의의가 있다고 할 수 있다.

```python
# LSE 함수를 구현한 것.
# Overflow/Underflow 문제를 방지 위해 log-sum-exp trick을 적용
# https://en.wikipedia.org/wiki/LogSumExp
def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log() # 이 부분은 앞서 유도했던 Log(Adjusted_Softmax) 식의 뒤쪽 부분과 일치한다.
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
```
