---
title: "[PyTorch] Batch, Stochastic, Mini-Batch Gradient Descent"
date: 2024-01-15 20:15:50 +0900
categories: [DL/ML, Study]
tags: [PyTorch, Torch, Deep Learning, SGD, Mini-Batch, Stochastic Gradient Descent, Batch, Python]
math: true
---

# 모델의 훈련 방식

훈련 데이터(Training Dataset)를 전부 사용해서 (1회) 학습시키는 방식을 **Batch 훈련 방식**이라고 한다.
> ***In batch gradient descent, we use all our training data in a single iteration of the algorithm.***
{: .prompt-tip }

여기서 iteration의 의미는 학습 매개변수가 학습되는 일련의 과정을 의미한다. 즉, 10 times iterations은 결국 학습 매개변수들이 10회 학습과정을 거쳤다는 의미다.

이와 대조적으로, 훈련 데이터셋 전체를 일정 갯수만큼(batch-size) 묶어서 소그룹(Mini-batch groups)별로 학습시키는 방식을 **Mini Batch 훈련 방식** 이라고 한다. 즉, 학습 파라미터들은 mini-batch group마다 1회 학습한다.

> ***In Mini-batch gradient descent, we use a group of samples called mini-batch in a single iteration of the training algorithm.***
{: .prompt-tip }

<br>

# Batch & Stochastic & Mini-batch Gradient Descent

경사 하강법 (Gradient Descent)으로 모델을 최적화한다고 생각해보자. 이 경우 앞서 언급한 Batch/Mini-Batch 방식을 포함해 일반적으로 크게 3가지 훈련 방식이 존재하게 된다.

**Batch Gradient Descent, Stochastic Gradient Descent, Mini-batch Gradient Descent**

1. Batch Gradient Descent(BGD): 모든 훈련 데이터 샘플들을 한번에 사용해서 그들의 평균 오차 기울기를 모든 학습 파라미터에 반영시키는 것.

2. Stochastic Gradient Descent(SGD): 훈련 데이터 샘플 각각에 대해서 오차 기울기를 각각 계산해서 학습시키는 것. 즉 100개 훈련 샘플이 있다면, 100번의 파라미터 학습이 이뤄진다.

3. Mini-batch Gradient Descent(MBGD): 훈련 데이터 샘플을 batch-size 갯수만큼 묶어서 batch group을 구성하고, batch group 마다 평균 오차 기울기를 계산해 학습시키는 것.

![png](/assets/img/post/gradient_descent/BG_Types.png)*"Differences Between Epoch, Batch, and Mini-batch", Source: https://www.baeldung.com/cs/epoch-vs-batch-vs-mini-batch* 

그럼에도 불구하고, 우리는 BGD든 SGD든 MBGD든 딥러닝 프레임워크에서 optimizer를 선택할 때, torch.optim.SGD()을 사용한다.

