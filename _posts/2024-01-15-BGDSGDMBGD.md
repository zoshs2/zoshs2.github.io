---
title: "[PyTorch] Batch, Stochastic, Mini-Batch Gradient Descent"
date: 2024-01-15 20:15:50 +0900
categories: [DL/ML, Study]
tags: [PyTorch, Torch, Deep Learning, SGD, Mini-Batch, Stochastic Gradient Descent, Batch, Python]
math: true
---

# The way of training

훈련 데이터(Training Dataset)를 전부 사용해서 (1회) 학습시키는 방식을 **Batch 훈련 방식**이라고 한다.
> ***In batch gradient descent, we use all our training data in a single iteration of the algorithm.***
{: .prompt-tip }

여기서 iteration의 의미는 학습 매개변수가 학습되는 일련의 과정을 의미한다. 즉, 10 times iterations은 결국 학습 매개변수들이 10회 학습과정을 거쳤다는 의미다.

