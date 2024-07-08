---
title: "Markov Model and Hidden Markov Model"
date: 2024-06-14 20:02:20 +0900
categories: [DL/ML, Statistical Model]
tags: [Markov, Markov Model, Hidden, Hidden Markov Model, HMM, Sequence, Sequential, Transition, Transition Probability, Markov Property, Markov Assumption, Memoryless, Markov Chain, Markov Process, Observable, Language, Weather, NLP, Evaluation, Decoding, DP, Dynamic Programming, Viterbi, Forward, Algorithm, Forward Probability, Emission Probability, Emission, EM, Expectation-Maximization, Baum-Welch, Baum-Welch Algorithm, Complexity, BigO, Induction, Recursion, Inductively, Viterbi Probability, Backtracking, Backtracking Pointer, Reinforcement, Reinforcement Learning, RL, Deep Learning, Machine Learning]
math: true
toc: false
---

# Table of Contents

- [Introduction](#introduction)
- [Markov Model](#markov-model)
- [Hidden Markov Model](#hidden-markov-model)
  - [Solvable Problems](#solvable-problems)
- [Model Summary](#model-summary)
- [Evaluation Problems](#evaluation-problems)
  - [Forward Algorithm](#forward-algorithm)
    - [Initialization](#initialization)
    - [Induction/Recursion](#inductionrecursion)
    - [Termination](#termination)
  - [Benefit from Forward Algorithm](#benefit-from-forward-algorithm)
  - [Decoding Problems](#decoding-problems)
  - [Viterbi Algorithm](#viterbi-algorithm)
    - [Initialization](#initialization-1)
    - [Induction/Recursion](#inductionrecursion-1)
    - [Termination](#termination-1)
    - [Backtracking](#backtracking)
- [Conclusions](#conclusions)

# Introduction

Markov Model(마르코프 모델), Hidden Markov Model(은닉 마르코프 모델)에 대해 알아보자. 오늘 날 다루는 딥러닝 모델/아키텍쳐들에 Markov Model 모델의 성능이 미치진 못하지만, Markov Model 기초는 무수한 분야 및 모델들의 바탕으로 활용된다. 따라서, 그 바탕이 되는 Markov Model의 골자가 무엇인지 이해하는 것을 목표로 글을 남기고자 한다.

# Markov Model

Markov Model은 일련의 데이터의 시퀀스(Sequence)를 이해하는 하나의 방법론이자 확률형 모델이다. 그 의미를 하나씩 뜯어보자.

**데이터의 시퀀스를 이해하는 하나의 방법론**. 흔히 날씨의 변화 양상, '날씨 시퀀스'를 Markov model 예시로 활용한다. 여기서도 이 예시를 들어보자. 최근 일주일 간의 날씨를 조사해보니 아래와 같은 순서였다고 하자.

$$
\begin{equation}
  \text{비} \rightarrow \text{해} \rightarrow \text{해} \rightarrow \text{해} \rightarrow \text{비} \rightarrow \text{비} \rightarrow \text{해}\rightarrow \text{?}
\end{equation}
$$

우리는 이런 경험적인 데이터(Empirical data)를 가지고, $\text{?}$ 자리에 어떤 날씨가 올 지 예측하길 원한다. 간단히 떠올릴 수 있는 방법은, 먼저 빈도수를 직접 세어보는 것이다. 예컨대 $\text{비} \rightarrow \text{해}$, $\text{비} \rightarrow \text{비}$, $\text{해} \rightarrow \text{비}$, $\text{해} \rightarrow \text{해}$ 의 각 전이(transition) 상황에 대해서 빈도수를 경험적인 데이터를 통해 세어보는 것이다.

그렇다면, 위와 같은 예시의 경우, 각 전이 상황에 대한 빈도수는 아래 왼쪽 그림과 같을 것이다.

![png](/assets/img/post/markov_model/transition_prob_scheme.png)*Left panel: 각 transition case의 빈도 수를 표현. Right panel: 각 transition case의 빈도 수를 바탕으로 확률로 표현.*

이 때 $\text{비}$ 상태에서 시작한 전이상황, 또는 $\text{해}$ 상태에서 시작한 전이상황을 기준으로 확률로 표현할 수 있고, 이것을 표현한 그림이 위의 오른쪽 그림이고 이러한 행렬을 **State transition probability matrix, 상태전이확률 행렬**이라고 한다. 그러면 이제 우리는 경험적인 데이터 해석을 통해 꽤 타당한 근거(?)로 $\text{'?'}$ 자리에 올 날씨는 $\text{'해'}$ 일 **가능성이 높겠네**.라고 주장할 수 있다.

즉, Markov model의 확률적 추론에 따르면, $\text{'?'}$에 올 날씨는 $\text{'해'}$인 것이다. 이게 전부다. "엥? 이게 무슨 모델이야."라고 할 수 있다. 왜냐하면, 이 추론 방식은 결국, 경험적 데이터에 대한 통계로 얻어낸 State tranisiton probabilities를 바탕으로 **현재 (날씨)상태만 고려하여**, 다음에 올 (날씨)상태가 무엇일지 예상한 것에 불과하기 때문이다. 혹자는 '더 과거의 (날씨)상태에 의존하지 않는다고 생각할 근거가 어디에 있어?'라고 주장할 수도 있고, 그 주장은 다음에 올 (날씨)상태에 대한 확률은 아래와 같다는 말하는 것과 같다.

$$
\begin{equation}
  P(S_{t}|S_{t-1}, S_{t-2}, S_{t-3}, ...)
\end{equation}
$$

이 때, $S_{t}$는 시점 $t$에서의 어떤 상태($S$), 예시에서 '비'나 '해',를 의미한다. 이 조건부 확률의 내용을 요약하자면, **내일 비가 올 확률**을 $P(비|비비비비...)$, $P(비|비해비비비...)$, $P(비|해비비비해...), ...$ 뭐 이런 경우들을 전부 따져서 예상하는게 옳지 않냐는 것이다. 충분히 타당한 주장이지만, 여기서 Markov Model의 **핵심**이 그냥 그런 것일 뿐이다. 즉 **다음 미래 상태의 확률은 현재 상태에만 의존한다**고 가정하는 것. 이러한 과감한 가정 및 속성을 **Markov Assumption**(또는 **Markov Property**, 또는 **Memorylessness**, 또는 **absence of memory**)라고 표현하고, 이러한 가정을 지닌 일련의 확률형 모델 프로세스를 **Markov Process**라고 한다.

$$
\begin{equation}
  P(S_{t}|S_{t-1}, S_{t-2}, S_{t-3}, ...) = P(S_{t}|S_{t-1})
\end{equation}
$$

"대체 어떻게 그럴 수 있어."라고 계속 말해봤자(사실 처음에 내가 이런 반응이었다), 더 드릴 말씀이 없다. Markov process 기반의 stochastic model은 그냥 이런 식으로 complex system을 단순화시켜서 '세상은 대충 이렇게 돌아간다'라고 표현하는 확률적 방법론이다. 하지만 이제부터 소개할 내용에서 보듯이, 이 간단한 가정을 기반으로 출발한 확률형 모델은 굉장히 단순하면서도 강력한 예측력을 보이고, 여러 가지 문제들을 해결할 수 있다.

# Hidden Markov Model

타대학 대학원생 '성재'라는 친구가 있다. 당신은 어릴 적부터 성재와 매우 각별한 사이어서, 매일 같은 시간에 통화를 하며 '뭐하냐'는 말을 시작으로 안부를 묻는다. 그럴 때마다, 성재는 꼭 '**연구** 중이야', 또는 '**데이트** 중이야', 또는 '그냥 집에 있어(**결근**)'라고 답한다. 한 몇 개월간 통화하다 보니, 문득 당신은 성재의 일상 패턴이 **날씨 패턴**과 굉장히 연관이 있어 보인다고 생각했고, 날씨와 성재일상 사이의 관계를 모델링해볼까라는 생각까지 도달하게 된다.

이럴 때, 사용할 수 있는 모델이 Hidden Markov Model(은닉 마르코프 모델; 이하 HMM)이다. HMM은 이렇듯 동일한 시점에 발생한 서로 다른 두 종류(예시에서의 일상시퀀스와 날씨시퀀스)의 state sequences 사이의 관계를 모델링한다. 그리고 이 두 종류의 state sequence를 **Hidden state sequence**와 **Observable state sequence** 라고 말하고, 이 글에선 특별히 두 종류의 sequence를 각각 $\{S_{T}\}$, $\{O_T\}$로 표현하겠다. Subscript $T$는 특정 시점 $T$에서의 state를 의미한다.

그렇다면, Hidden과 Observable를 나누는 기준은 무엇인가? Hidden은 **우리가 알고싶은 숨겨진 정보나 상태이고, Observable state 설명의 근거로 삼고자 하는 대상**이다. 그리고 Observable은 **우리가 직접적으로 또는 손쉽게 관찰 가능한 상태**이다. 나는 이 글에서 Hidden state를 **날씨**로 사용할 것인데, 혹자는 '날씨가 숨겨져 있어? 직접 관찰 가능한 대상이잖아?'라고 의문을 제기할 수 있다. 물론 문제에 따라 '날씨'도 observable 로 정의할 수 있겠지만, 그렇다면 다음으로 이 날씨를 설명하고자 하는 hidden state의 정의가 꽤 까다로울 것이다. (예컨대 '성재가 오늘 연구(Hidden)를 해서, 오늘 날씨(Observable)가 영 좋지 못하다'라고 주장하긴 힘들지 않겠나.) 즉, Hidden state를 정할 때는 **Observable state보다는 직접적인 설명이 어려운 무언가, 또는 영향을 주는 매개요인들이 (헤아리기 힘들 정도로)많은 대상**을 기준으로 삼는 것도 적절해 보인다.

> 보다 직관적인 예시도 가능하다. HMM은 문장 데이터에도 적용할 수 있는데, 이 경우 Hidden state는 품사로 생각해볼 수 있다. 예컨대, ('I', 'am', 'a', 'student')라는 문장 sequence를 Observable state sequence로 보고, 이에 대응하는 품사 ('대명사', '동사', '한정사', '명사')를 Hidden state sequence로 여기는 것이다. 일반적으로 '문장'은 우리가 직접적으로 관찰하는 대상이고, '품사'는 숨겨져 있다고 볼 수 있기 때문이다. 이 밖에도, 주식시장 분석에서의 주가의 변동(매일의 주가 상승/하락; Observable)-시장상태(강세장/약세장; Hidden), 언어표현(Observable)-감정상태(Hidden) 등을 생각해볼 수 있다.
{: .prompt-tip }

## Solvable Problems

HMM 모델을 활용할 수 있는 방식은 크게 2가지로 보고, 각 유형에 대해 던질 수 있는 질문과 그에 대한 대답이 다르다.

1. Evaluation 방식: 다음 Observable sequence가 발생할 확률은? >> ex. "70 %"
2. Decoding 방식: 주어진 Observable sequence에 대응하는 최적의(또는 발생 확률이 높은) Hidden state sequence는? >> ex. "비>비>비>해 (날씨 패턴)"

이제부터 더 자세히 설명할테지만, 결과부터 얘기하자면 HMM은 위 문제들을 각각 Forward(Backward) 알고리즘과 비터비(Viterbi) 알고리즘이라는 Dynamic programming(DP)류 알고리즘들을 사용하여 문제에 대한 답을 내놓게 된다.

> Dynamic Programming (동적계획법; DP) 알고리즘은 쉽게 말해서 앞서 구했던 답을 현재의 답을 구할 때 사용하는 일종의 문제 풀이 방식이다. 예컨대, 함수 내부에서 함수 자기자신을 호출하면서 원하는 답에 도달하는 재귀함수도 DP 알고리즘이라 볼 수 있다.
{: .prompt-tip }

> 단순한 방식에 비해 다소 거창해보이는 Dynamic Programming 이란 이름은 미국의 응용수학자 [Bellman](https://en.wikipedia.org/wiki/Richard_E._Bellman){:target="_blank"}이 처음 제안했는데, 그 사연이 꽤 재밌다. 당시 벨만은 RAND 연구소에서 미국 국방부 산하의 연구과제를 수행하였는데, 그 가운데 그가 그토록 매진하고 연구하는 분야가 수학(Mathematical)이라는 사실을 은연 중에 감추고자 프로그래밍(Programming)이라는 단어를 골랐다고 한다. 당시 국방부 장관인 '찰스 어윈 윌슨(5대 국방장관)'는 research라는 단어를 병적으로 싫어 했고, 따라서 'Mathematical' 관련 용어들도 싫어할거 같았기 때문이다. 그리고 본인이 담고 싶었던 '여러 단계를 거쳐 결정에 도달하는 방법(multistage)', '시간의 변동(time-varying)' 등의 의미를 담기에 'Dynamic'이 아주 적절했으며, 관련 상위 관리자들이 dynamic이란 단어를 보기에도 불편해하지 않았을 거 같아서 최종적으로 'Dynamic Programming'이란 단어를 사용했다고 한다. 아무튼 눈치를 엄청 봐야했던 환경이었나보다ㅜ (본 내용은 벨만의 자서전인 <허리케인의 눈 (1984)> p.159에 기술되어 있다.)
{: .prompt-info }

그렇다면, HMM 모델이 어떻게 Evaluation 및 Decoding 문제를 해결하는지 하나하나 살펴보자.


# Model Summary

우선 전체적인 문제의 예시는 앞서 언급한 **날씨**와 **일상**을 Hidden state, Observable state로 상정하고 이야기해보도록 하자.

그리고 문제를 풀기 전에, HMM 모델에는 크게 3가지 파라미터(trainable parameters)가 필요하다. Hidden state들 사이의 전이확률에 대한 파라미터 State transition probability, **$A$**. 그리고 각 Hidden state에서 Observable state들로 전이할 확률에 대한 파라미터 Emission probability, **$B$**. 첫 Hidden state를 결정할 확률을 담은 파라미터 initial probability, **$\pi$**이다. 즉, Hidden Markov Model은 이 3가지 파라미터 $A$, $B$, $\pi$로 구성된 모델이다. 나는 앞으로의 설명을 아래 그림 소 Hidden Markov Model을 바탕으로 설명하고자 한다.

![png](/assets/img/post/markov_model/HMM_basic_setup.png)*A Hidden Markov Model with trainable parameters, $\theta = \left(A, B, \pi \right)$*

참고로 당연히 위 3가지 파라미터들($A, B, \pi $)의 값을 알기 위해서는 별도의 learning step이, 문제를 풀기전, 사전적으로 필요하다. 이 과정에서 초기 임의로 할당된 파라미터값들에서 시작해서(Initialization 단계) **forward/backward probability** 계산을 통해 주어진 training dataset인 observable state sequences에 대한 optimal한 emission/transition/initial probability로 도달하도록 학습이 이뤄진다. 이러한 일련의 과정을 HMM 모델에 적용하게끔 나온 알고리즘이 Expectation-Maximization(EM) 알고리즘을 모사한 Baum-Welch Algorithm인데, 이 글에선 이에 대해 더 자세히 설명하지 않도록 하겠다.

> Forward/Backward Probability 개념은 Evaluation 챕터의 Forward Algorithm 파트에서 소개한다.
{: .prompt-tip }

대신, HMM 모델을 공부하면서 둘러본 여러 블로그 및 사이트들 중에서 Baum-Welch algorithm을 매우 잘 설명해두신 분들의 글들을 링크로 남겨 둔다.

1. [ratsgo's SPEECH BOOK](https://ratsgo.github.io/speechbook/docs/am/baumwelch){:target="_blank"}
2. [볼록티님 블로그: 머신러닝 'Expectation Maximization'](https://data-science-hi.tistory.com/158){:target="_blank"}
3. [고려대 김성범 교수님의 핵심 머신러닝 강의: 'Hidden Markov Models - Part 2 (Decoding, Learning)'](https://youtu.be/P02Lws57gqM?t=1263){:target="_blank"}


# Evaluation Problems

Evaluation 파트에서 던질 수 있는 질문은 다음과 같다.

* (성재가) '연구-데이트-연구'라는 일상 시퀀스를 보일 확률은?

주의할 점은, '어떤 날씨 시퀀스(Hidden state sequence)에 대해서' 라는 조건이 붙은게 아니다. 즉, 이 질문은 취지는 '모든 hidden sequence의 가능성을 전부 고려했을 때, 이를 바탕으로 해당 observable state sequence가 발생할 확률이 얼마나 되는데?'라고 묻는 질문인 것이다.

이를 수학적으로 풀어 말하자면,

$\text{P}\left(o_{1}=연구, o_{2}=데이트, o_{3}=연구|비,비,비\right)$

$\text{P}\left(o_{1}=연구, o_{2}=데이트, o_{3}=연구|비,비,해\right)$

$\text{P}\left(o_{1}=연구, o_{2}=데이트, o_{3}=연구|비,해,비\right)$

$\text{P}\left(o_{1}=연구, o_{2}=데이트, o_{3}=연구|해,비,비\right)$

$\text{P}\left(o_{1}=연구, o_{2}=데이트, o_{3}=연구|해,해,비\right)$

$\text{P}\left(o_{1}=연구, o_{2}=데이트, o_{3}=연구|해,비,해\right)$

$\text{P}\left(o_{1}=연구, o_{2}=데이트, o_{3}=연구|비,해,해\right)$

$\text{P}\left(o_{1}=연구, o_{2}=데이트, o_{3}=연구|해,해,해\right)$

이 확률들을 모두 **Summation** 해서 나온 확률이 얼마냐는 것이다.

위 질문과 수학적 표현을 조금 더 일반화시켜 생각해보자. Hidden state sequence $S$의 가능한 경우의 수를 $Q$개 라고 한다면, 앞서 한 Evaluation 문제의 질문은 아래와 같다. ($O$는 $O=\{o_1, o_2, o_3, ..., o_T\}$같은 **특정 관찰시퀀스**라고 하자.)

$$
\begin{equation}
  P\left(O|\theta\right) = \sum^{Q}_{q=1} P\left(O, S_{q}|\theta\right) = \text{ ?}
\end{equation}
$$

Observable state sequence와 Hidden state sequence는 서로 대응되는, 동일한 시점에 발생하는 사건들이므로 이렇게 Joint probability로 표현이 가능하다. 이는 다시 Conditional probability로 표현이 가능하고, 여기에 state sequence의 길이 $T$까지 고려한 상황과 Markov Property 를 수식에 대입한다면, Evaluation 문제를 Naive하게 풀기 위해서는 아래와 같은 계산이 수행되어야 함을 의미한다.

$$
\begin{equation}
  P\left(O|\theta\right) = \sum^{Q}_{q=1} P\left(O, S_{q}|\theta\right) = \sum^{Q}_{q=1}P\left(O|S_q,\theta\right)P\left(S_q\right) = \sum^{Q}\prod^{T}_{t=1}P\left(o_t|s_t,\theta\right)\prod^{T}_{t=1}P\left(s_t|s_{t-1}\right)
\end{equation}
$$

여기서 첫 번째 $\prod$ 텀이 Emission probability(특정 hidden state에서 특정 observable state가 나올 확률)에 관한 연산이고, 두 번째 $\prod$ 텀이 (Hidden state) Transition probability에 관한 연산이다. 각각 $T$회, $T-1$회 연산을 수행하므로, 총 $2T-1$ operations이 수행된다. 여기까지가

$\text{P}\left(o_{1}=연구, o_{2}=데이트, o_{3}=연구|비,비,비\right)$

와 같은, 특정 hidden state sequence 하나에 대한 총 연산 횟수라 보면 된다.

* * *

하지만 우리는 가능한 모든 경우의 수 $Q$개에 대해서 위와 같은 연산을 다 해줘야 하고, 이는 곧 hidden state의 갯수 $N$과 시퀀스의 길이 $T$일 때 $Q=N^T$개 이므로, $\left(2T-1\right)N^{T}$ 회를 해줘야 한다는 의미이고, 마지막으로 이들을 모두 summation 하는 연산이 $N^{T}-1$회 이므로(참고: 1+2+3를 계산하기 위한 연산 횟수은 2회다), 결과적으로 Evaluation Problem 을 Naive하게 풀기 위해 수행해야 하는 전체 연산 횟수는  $\left(2T-1\right)N^{T}+\left(N^{T}-1\right)$회가 되는 것이다. 여기서 Dominant term만 고려해 computational complexit를 표현하자면 $O\left(TN^{T}\right)$로서, 이런 연산량은 실전 적용 문제에서 활용하는 hidden state 갯수 $N$과 sequence 길이 $T$를 고려한다면 굉장히 challenging 한 연산량이다.  

## Forward Algorithm

이러한 harsh-computational complexity, $O\left(TN^{T}\right)$를 극복할 수 있는 방법이 바로 Forward algorithm이고, 이를 통해 HMM Evaluation 문제를 풀게 된다.

앞서 본 방식이 각각의 경우에 대해서 잘게 쪼개서 각각의 확률을 따져서 summation 하는 방식이었다면, 이 forward algorithm 동적계획법 방식에서는 귀납적으로(inductively) 확률을 **집계**해 나가며 풀이하게 된다. 여기서 **aggregate** variable인 **Forward probability, $\alpha$** 개념이 나오게 된다.

Forward 알고리즘 단계는 크게 3가지 단계인 초기화(Initialization), 귀납단계(Induction, 또는 Recursion), 종결(Termination)단계로 구분된다.

### Initialization

**초기화단계**(Initialization step)에서는 각 hidden state, $s_j$에 대한 초기확률(위 Model summary 챕터에서 Initial Probability)과 각 $s_j$로부터 첫번째 observable state $o_1$가 발생할 확률(Model summary 챕터 그림에서의 Emission prob.)을 근거로 Forward probability, $\alpha_{t=1}(j)$를 계산한다. 이를 수식으로 표현하면 다음과 같다.

$$
\begin{equation}
  \alpha_{1}\left(j\right)=\pi_{j}b_{j}\left(o_1\right)
\end{equation}
$$

이 계산은 각 hidden state들에 대해서 모두 수행되며, 즉 $N$개의 hidden state를 지닌 문제일 시, 위와 같은 muliplication operation을 $N$번 한다는 의미다.

앞서 던졌던 질문인 '(성재가) '연구-데이트-연구'라는 일상 시퀀스를 보일 확률은?'의 경우에 대해서 이 수식적 상황을 상상해보면 아래 그림과 같다. (각각의 값은 ***Model summary 파트***를 참고)

![png](/assets/img/post/markov_model/InitializationStep.png)*Evaluation with forward algorithm: Initialization step*

### Induction/Recursion

이제 Forward algorithm, Dynamic Programming 방식의 진가가 나오는 부분이다. 앞서 초기화단계에서 얻은 각 hidden state $s_j$들에 대한 forward probability $\alpha_1$ 값들을 시작으로, $\alpha_2, \alpha_3, ..., \alpha_T$를 계산해 나간다. 이 과정을 일반화된 수식 표현으로 나타내면 다음과 같다.

$$
\begin{equation}
  \alpha_{t}\left(j\right) = \left[\sum^{N}_{i=1}\alpha_{t-1}\left(i\right)a_{ij}\right]b_j\left(o_t\right);
\end{equation}
$$

$$
\begin{equation}
 N \text{ is the number of hidden states}, 1 < t \leqq T
\end{equation}
$$

수식의 의미를 따져보면, 앞 단계에서 얻은 각 hidden state들에 대한 forward probability $\alpha_{t-1}$값들과 hidden states 사이의 전이확률(Model summary 그림에서 Transition probability), 그리고 다시 Emission probability 를 활용하여 다음 시점에서의 hidden state별 forward probability를 계산하는 것이다.

이를 다시 우리 성재의 상황에 대입해 생각해보면, 초기화단계 이후 다음 스텝에서 아래 그림과 같은 연산과정이 이뤄진 것이라 할 수 있다.

![png](/assets/img/post/markov_model/RecursionStep.png)

이러한 연산을 주어진 관측 시퀀스 길이만큼 계속해서 수행하면 되고, 여기서 주목할 점은, 앞선 시점에서 얻은 forward probability 값들을 다음 시점의 forward prob. 값의 계산에 활용하는 이런 일련의 과정이 반복적으로 이뤄진다는 것이다. 이런 맥락에서 귀납적인(inductively), 반복적인(recursively) operation step임을 알 수 있다.

이제 성재 상황에 대해 Induction/Recursion 단계가 마무리되었을 때, 어떤 결과일지 확인해보자. 결과는 아래 그림과 같을 것이다.

![png](/assets/img/post/markov_model/RecursionFinal.png)

이제 Induction/Recursion 단계에서 operations 수를 세어보자. 심플하게 $t$시점의 **특정** $j$ state로 가는 연산 수를 생각해보자. 예컨대,

$$
\begin{equation}
  \alpha_{2}(1) = \left[\sum^{N}_{i=1}\alpha_{1}(i)a_{i1}\right]b_{1}\left(o_k\right)
\end{equation}
$$

에 관한 수만 세어보자는 말이다. 우선 $t=1$시점의 hidden state 갯수 $N$개만큼 대괄호안의 multiplication 연산이 수행된다. 그리고 이들의 결과를 summation 하는 것까지 $N-1$회이다. 마지막으로 최종 결과에 $b_1\left(o_k\right)$ 곱에 대한 연산 1회까지 해야한다. 결과적으로 다음 시점 $t=2$의 특정 hidden state $s_j$에 대한 forward probability 를 계산하는데 수행되는 연산 수는 $N+\left(N-1\right)+1$로써, 즉 $2N$ operations이 수행된다.

이러한 연산이 다음 시점 $t=2$의 모든 hidden state들(총 $N개$)에 대해 수행되므로 $N \times 2N=2N^2$ 회 진행된다. 여기까지가 $t-1$과 $t$ 사이의 operations 크기다. 이러한 연산이 Initialization 단계 이후인 $t>1$부터 진행되니, 결과적으로 Induction/Recursion 단계에서 드는 총 operations 크기는 $2N^2\left(T-1\right)$회임을 알 수 있다.

### Termination

이제 최종적으로 Evaluation 문제에 대한 답이 나오는 부분이다. 주목할 점은, 지금까지 시점마다의 Forward Probability를 귀납적으로 계산해왔고, 이는 앞선 시점에서 나타난 모든 observable state $o_k$ 관측 확률까지 고려된 확률값들이란 점이다. 쉽게 말하자면, 마지막 시퀀스 시점인 $t=T$에 이르러서 얻게 되는 Forward Probability 안에는 이미 given observable state sequence, $O$에 대한 관측 확률이 다 담겨있다고 보는 것이다.

따라서, Termination 단계에서는 $t=T$ 시점에 최종적으로 얻은 Forward Probability 들을 단순히 Summation 하는 것만으로 given observable state sequence에 대한 관측 확률, 즉 우리가 처음에 질문을 던졌던 $O=\{o_1=\text{research}, o_2=\text{dating}, o_3=\text{research}\}$라는 (성재의) 일상 시퀀스를 보일 확률은? 에 대한 답이 되는 것이다. 이를 수식적으로 표현하면 다음과 같다.

$$
\begin{equation}
  P\left(O|\theta\right)=\sum^{N}_{i=1}\alpha_{T}\left(i\right)
\end{equation}
$$

아래 그림은 우리가 처음 던졌던 질문 "(성재가) '연구-데이트-연구'라는 일상 시퀀스를 보일 확률은?"에 대한 답을 구하는 Termination 단계의 과정이다.

![png](/assets/img/post/markov_model/TerminationStep.png)*성재는 '연구-데이트-연구' 시퀀스를 보일 확률이 3.47%다.*

Termination 단계에서도 역시 operations 수를 구해보면, $N-1$회 라는걸 쉽게 알 수 있다.

## Benefit from Forward Algorithm

처음 Evalution 도입부에서 봤던, 모든 경우의 수를 그저 나열해서 각각 따로 계산하는 Naive한 계산 방식은, 올바른 답을 내놓는다는 목적은 달성할 수 있지만, $O(TN^T)$ 복잡도를 보이는 만큼 실제 적용 문제에서는 계산적으로 힘들다는 문제가 있었다.

그래서 HMM 모델 Evalution 알고리즘으로 제시된 것이 Forward Algorithm이었다. 이 Forward 알고리즘의 3단계에서 본 operations 수를 모두 종합해보면,

$$
\begin{equation}
  N\left<\text{initialization step}\right> + 2N^2\left(T-1\right)\left<\text{induction/recursion step}\right> + \left(N-1\right)\left<\text{termination step}\right>
\end{equation}
$$

으로, 총 $2N^2T-2N^2+2N-1$회의 operations을 Evaluation 문제풀이에 수행하게 된다. 이는 다시 말해, Dominant term만 고려하여 complexity를 보면 $O(TN^2)$과 같다는 의미이고, Naive calculation의 스케일($O(TN^T)$)과 비교했을 때, Forward 알고리즘이 계산적으로 상당히 효율적인 방법임을 짐작할 수 있다.

## Decoding Problems

이제 Hidden Markov Model의 정수라 할 수 있는 Decoding 문제 풀이다. 이 질문의 유형은 다음과 같다고 했었다.

* 주어진 Observable sequence에 대응하는 최적의(또는 발생 확률이 높은) Hidden state sequence는? >> ex. "비>비>비>해 (날씨 패턴)

다시 성재를 데리고 와보자. 비록, 성재가 'research'-'dating'-'research' 시퀀스를 보일 확률이 3.47% 밖에 되지 않았지만, 아무튼 이런 일상 시퀀스를 보였을 때 **the most probable sequence of (weather; or hidden) states를 찾는 문제**인 것이다.

HMM 모델은 Decoding 문제에서도 역시, 앞서 Forward 알고리즘에서 보았던 귀납적 방식/반복적 계산(inductively/recursively)을 바탕으로 하고 있지만, 세부적으로 약간 다른 연산을 취한다. 이를 Viterbi algorithm(비터비 알고리즘)이라고 부르며, 이 풀이 방식을 통해 Decoding 질의에 대한 답을 구하게 된다.

Forward 알고리즘과 유사하게 크게 3단계, Initialization/Recursion/Termination 단계로 수행되고, 연산 방식 역시 '이전 시점의 forward prob. $\times$ transition prob.'을 $\text{sum()}$ 이 아닌 $\text{max()}$를 취한다는 것만 빼곤 핵심적으로 동일하다. 하지만 Viterbi 알고리즘에는 Forward 알고리즘에는 없는 component가 하나 더 있는데, 바로 백트래킹 포인터(Backtracking pointer)라는 요소다.

Backtracking pointer($bt_{t}$, 'b곱하기t'가 아니라 BackTracking의 약어임)는 모든 연산을 마치고 hidden state 'sequence'를 다시 재구성하는 과정에서 필요한 요소다. 그렇기 때문에 백트래킹 포인터는 개념적으로 **역추적**이며, **앞선 시점에** 어떤 state였는지를 가리키는 도구다. (이해를 돕기 위해 예를 들자면, $bt_{2}$는 $t=1$ 시점에 어떤 hidden state였는지를 말해준다.)

그럼 이제, 수식을 통해 단계별로 살펴보자.

## Viterbi Algorithm

두 가지 변수를 먼저 소개한다. Forward 알고리즘에서 보았던 Forward Probability와 유사하고, $\text{sum()}$이 아닌 $\text{max()}$를 취한다는 사실이 다른 Viterbi Probability $v_{t}(j)$와 the most probable sequence of hidden states를 재구성하기 위한 $bt_{t}(j)$. 이들의 값들이 각 Initialization/Recursion/Termination 단계에서 어떻게 결정되는지, 그리고 어떻게 최종적으로 Decoding 질문의 답에 도달하는지 살펴보자.

### Initialization

처음 시작은 다음과 같다. 백트래킹 포인터에 대한 요소가 있는 거 빼곤, Forward 알고리즘 때와 똑같다.

$$
\begin{equation}
  v_{1}\left(j\right) = \pi_{j}b_{j}\left(o_1\right)
\end{equation}
$$

$$
\begin{equation}
  bt_{1}\left(j\right) = 0
\end{equation}
$$

이 때, 백트래킹 포인터 $bt_{1}$가 0이라는 의미는, '$t=0$ 시점의 (hidden) state는 정의되지 않는다'는 의미다. 왜냐? 시퀀스는 $t=1$부터 시작이니까.

### Induction/Recursion

초기화단계에서 구한 viterbi prob. 값들을 근거로, $t=2$ 시점부터는 아래 수식처럼 계산이 진행된다.

$$
\begin{equation}
  v_{t}\left(j\right) = \left[\max^{N}_{i=1}v_{t-1}(i)a_{ij}\right]b_{j}\left(o_{t}\right)
\end{equation}
$$

$$
\begin{equation}
  bt_{t}\left(j\right) = \left[\text{arg}\max^{N}_{i=1}v_{t-1}(i)a_{ij}\right]
\end{equation}
$$

$$
\begin{equation}
 N \text{ is the number of hidden states}, 1 < t \leqq T
\end{equation}
$$

다시 성재를 데리고 오자. $t=1$까지의 viterbi prob., backtracking prob. 은 initial step에서 끝난 상태임을 유념하자. 아래 그림은 성재의 HMM 모델에 대한 $t=2$ 까지 Viterbi calculation이 수행된 결과다. 녹색 선은 백트래킹 포인터 $bt$가 앞선 시점 $t=1$의 most probable hidden state를 가리키고 있는 것을 시각화한 것이다.

![png](/assets/img/post/markov_model/ViterbiRecursion.png)

이러한 연산 과정을 반복적으로 남은 시퀀스까지 수행하면, 성재 예시의 경우 다음과 같은 결과가 된다.

![png](/assets/img/post/markov_model/ViterbiRecursionResult.png)

### Termination

시퀀스 길이 $T$까지 Induction/Recursion 과정을 수행하면, 이제 최종적으로 Decoding 문제에서 Viterbi 알고리즘은 다음과 같이 답을 도출한다.

$$
\begin{equation}
  \text{BestPathProbability}, P* = \max^{N}_{i=1}v_{T}(i)
\end{equation}
$$

$$
\begin{equation}
  \text{The Start of backtrace}, q* = \text{arg}\max^{N}_{i=1}v_{T}(i)
\end{equation}
$$

마지막 시점 $t=T$에 얻은 viterbi probability 값들을 바탕으로, 가장 큰 값을 지닌 state에서부터 역추적하여 the most probable sequence of hidden states를 재구성하게 된다(이게 우리의 Decoding 질문이었음에 유념하자).

### Backtracking

마지막 시점인 $t=T$의 most probable (hidden) state는 다음과 같이 초기화된다.

$$
\begin{equation}
  q_T = q*
\end{equation}
$$

이후 $T-1, T-2, T-3, ..., 1$까지 **역순으로** 다음과 같은 수식에 따라 most probable states들을 할당시킨다.

$$
\begin{equation}
  q_t = bt_{t+1}\left(q_{t+1}\right)
\end{equation}
$$

$$
\begin{equation}
  \text{for } t=T-1, T-2, T-3, ..., 1
\end{equation}
$$

예컨대, $t=T-1$에 대해서 $q_{T-1}=bt_{T}\left(q_{T}\right)$ 이고, 앞서 Induction/Recursion 단계에서 구해놓았던 각 state가 지닌 백트래킹 포인터를 대응시키는 것이다.

따라서, 성재 예시의 경우 다음 그림과 같은 결과로 귀결된다.

![png](/assets/img/post/markov_model/BackTrackingResult.png)

정리하자면, 주어진 관측시퀀스 '연구'-'데이트'-'연구'에 대응하는, 가장 optimal(most probable) sequence of (weather; hidden) states는 $S=\{s_2, s_2, s_2\}$인 것이고, 이를 해석하면 'Sunny'-'Sunny'-'Sunny' 인 것이다.

그리고 Termination 단계에서 언급을 안했던 요소, $\text{BestPathProbability } P*$는 이렇게 재구성한 optimal hidden sequence가 실제로 발생할 확률을 의미한다.

즉 성재의 경우, 비록 '연구'-'데이트'-'연구' 시퀀스를 수행할 확률은 (Evaluation에서 구했던) 3.47%에 불과하지만, 만약 그런 관측시퀀스를 보인다는 상황에서 가장 최적의 날씨(은닉) 시퀀스는 'Sunny'-'Sunny'-'Sunny'이고 그 확률은 $P*=$1.5552% 라는 의미다.

# Conclusions

본 글에서는 Markov Model의 성질과 Hidden Markov Model(이른바 HMM)에 대해 공부했다. HMM 모델은 크게 2가지 서로 다른 질문 유형인 Evaluation/Decoding 를 Forward/Viterbi 이라는 Dynamic Programming 알고리즘으로 풀 수 있었다.

이를 가능케 한 것은 바로 Markov Property라고 불리는 '특정 상태의 (발생) 확률은 바로 이전 시점의 상태에 의존한다'는 단순한 가정에서 출발한 것이었다.

$$
\begin{equation}
  P(S_{t}|S_{t-1}, S_{t-2}, S_{t-3}, ...) = P(S_{t}|S_{t-1})
\end{equation}
$$

이 가정은 Hidden Markov Model뿐 아니라, Reinforcement Learning(강화학습)의 기초인 [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process){:target="_blank"}에서도 활용된다.

그만큼 Markov Model이 DL/ML 분야 발전에 중요한 기여를 해왔고, 특히 sequential/temporal dataset 분석에 큰 역할을 했던 모델이자 개념이기에 한번 짚고 넘어가고자 이 글을 작성했다.

* * *

**본 글을 쓰면서 고려대학교 산업경영공학부 김성범 교수님의 [핵심 머신러닝] 강의 중 'Hidden Markov Models Part 1/2' 강의 영상들로부터 정말 큰 도움을 받았다 ("감사합니다, 교수님"). 정말 김성범 교수님의 DL/ML 분야에 관한 강의는, 하나도 빠짐없이, 머릿 속에 속속 박히는 명강들이라 생각한다. [김성범 교수님의 유튜브 채널](https://www.youtube.com/@user-yu5qs4ct2b){:target="_blank"}에 주옥같은 머신러닝 강의 영상들이 많으니 DL/ML 분야 공부에 관심있으시면 방문하셔서 '좋댓구알'하시길 추천드린다.**

**KAIST 기계공학과 & [Industrial AI 연구실](https://iai.postech.ac.kr/home){:target="_blank"}의 이승철 교수님의 [심층 강화학습] 강의 중 'Markov chain(마르코프 체인)' 파트 강의도, 마르코프 모델에 대한 이해에 큰 도움이 되었다. ("감사합니다")이승철 교수님의 강의도 강력 추천드린다.**

그 외, 공부에 많은 도움을 받은, 다른 레퍼런스 자료들도 여기에 기록해둔다.

* [An Introduction to the Hidden Markov Model](https://www.baeldung.com/cs/hidden-markov-model){:target="_blank"}

* [Speech and Language Processing. Daniel Jurafsky & James H. Martin.](https://web.stanford.edu/~jurafsky/slp3/){:target="_blank"}: 2024년 3월 draft 버전의 e-book이다. 스탠퍼드 대학교 Computer Science과 교수 [Dan Jurafsky](https://web.stanford.edu/~jurafsky/){:target="_blank"}분께서 이 책의 저자이시고, 교내 스탠퍼드 강의에서도 이 책의 내용을 중심으로 강의하시는 듯하다. 링크를 들어가면, 실제 강의에서 활용하신 듯한 Lecture Note / PPT 파일도 열람할 수 있다. 이 글을 쓰는 시점을 기준으로, 여전히 아직 Draft버전이라 모든 chapter 내용이 전부 올라와 있진 않고, 계속해서 업데이트 중이라고 안내가 적혀있다. 아무튼 책이 전반적으로 이해하기 쉽고 핵심이 잘 전달되게끔 잘 쓰여있다. 삽입된 그림들도 대부분 굉장히 직관적! 앞으로 다른 주제의 내용을 공부할 때도, 종종 활용해야겠다. Good!!


