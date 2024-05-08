---
title: "[PyTorch] MNIST Dataset - Softmax Regression Model"
date: 2024-03-17 18:15:22 +0900
categories: [DL/ML, Study]
tags: [Classifier, Cross-Entropy, Softmax, PyTorch, Torch, Deep Learning, Machine Learning, CUDA, GPU, Test, Train, Evaluation, MNIST, Torchvision, Mini-batch, Python]
math: true
toc: false
---

# Table of Contents
- [Intro.](#intro)
- [Pytorch Load MNIST Dataset](#pytorch-load-mnist-dataset)
- [Setup of Computational Configurations](#setup-of-computational-configurations)
- [Model Architecture](#model-architecture)
- [Model Test](#model-test)
  - [PyTorch for test procedure](#pytorch-for-test-procedure)

# Intro.

**본 공부 내용은 유원준 님, 안상준 님이 저술하신 [PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/book/2788) 위키독스 내용들을 배경으로 하고 있습니다.**
<br><br>

이번 포스트에선, 지난 [로지스틱 및 소프트맥스 회귀 분류 모델](https://zoshs2.github.io/posts/ClassifierModel/){:target="_blank"}글에 배운 내용을 MNIST 손글씨 이미지 데이터 분류 문제에 직접 적용해본 기록을 남기고자 한다.

MNIST 데이터는 내가 Keras & Tensorflow 프레임워크로 처음 DL/ML을 입문했을 때도 가장 먼저 다뤄봤던 데이터인데, 그만큼 입문용 데이터셋으로 많이 다루는 데이터이다.


```python
# tensorflow 버전 MNIST 로드 방식
# 반환되는 데이터 타입은 numpy.ndarray
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11490434/11490434 [==============================] - 2s 0us/step



```python
import numpy as np
import matplotlib.pyplot as plt

samples_idx = np.random.randint(x_train.shape[0], size=(10,))
mnist_x_samples = x_train[samples_idx, :, :]
mnist_y_samples = y_train[samples_idx]

fig, axs = plt.subplots(nrows=2, ncols=5, facecolor='w', figsize=(15, 6))
for i, ax in enumerate(axs.flatten()):
  ax.imshow(mnist_x_samples[i, :, :], cmap='gray')
  ax.set_title(f"label: {mnist_y_samples[i]}", fontsize=12)
  ax.axis('off')

plt.suptitle("MNIST DATASET LOADED BY TENSORFLOW", fontsize=25)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()
```
    
![png](/assets/img/post/softmax_mnist/SoftmaxClassifierForMNIST_3_0.png)
    
이렇게 MNIST 데이터셋은 28 X 28 픽셀 크기에 손글씨가 표현된 입력 데이터(x)와 그에 해당하는 정답 레이블(y)로 이루어진 데이터셋이고, 이 둘의 관계를 학습시키는 것이 문제의 핵심이다.

# Pytorch Load MNIST Dataset

나는 'PyTorch와 친해지기' 다짐을 했으니, MNIST 데이터 로드부터 분류 모델 설계 및 평가까지 PyTorch식 재현을 해보자.

```python
import torch
import torchvision # 아직 익숙하지 않으니, alias를 쓰지 않겠다.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
```

PyTorch식 MNIST 데이터 로드 방법은 [torchvision](https://pytorch.org/vision/stable/index.html){:target="_blank"} 이라는 라이브러리를 이용하는 것이다. 도큐멘트 홈페이지에 들어가보면 torchvision은 a part of the PyTorch project라고 설명하고 있다. 라이브러리 이름에서도 알 수 있듯이 torchvision는 이미지/영상 분야용으로 특화된 라이브러리로서, 관련 (MNIST 처럼) [유명 데이터셋](https://pytorch.org/vision/stable/datasets.html){:target="_blank"}과 DL/ML 모델 아키텍처, 이미지 데이터에 유용한 전후처리 기능들이 담겨 있다.

**torchvision**말고도 풀어야 문제의 큰 성격에 따라 [torchaudio](https://pytorch.org/audio/stable/index.html){:target="_blank"}, [torchtext](https://pytorch.org/text/stable/index.html){:target="_blank"} 라이브러리도 따로 존재한다.

그럼 이제 torchvision을 사용해서 MNIST 데이터를 불러와보자.


```python
mnist_train = torchvision.datasets.MNIST(
    root='MNIST_data/', # 다운받을 디렉터리 (없으면 새로 만듬)
    train=True, # = 학습할 때 사용할 데이터니?
    download=True,
    transform=torchvision.transforms.ToTensor() # 불러올 (MNIST 이미지) 데이터를 PyTorch Tensor로 변환시켜 가져온다.
)

mnist_test = torchvision.datasets.MNIST(
    root='MNIST_data/', # 다운받을 디렉터리 (없으면 새로 만듬)
    train=False, # train이 아니라 test용이라 False
    download=True,
    transform=torchvision.transforms.ToTensor() # 불러올 (MNIST 이미지) 데이터를 PyTorch Tensor로 변환시켜 가져온다.
)
```


```python
mnist_train # 60000개의 학습 입출력 데이터셋
```




    Dataset MNIST
        Number of datapoints: 60000
        Root location: MNIST_data/
        Split: Train
        StandardTransform
    Transform: ToTensor()



이렇게 PyTorch에 내장된 데이터를 불러오면, 학습에 활용할 입력 데이터와 출력 데이터에 접근하는 방식 구분되어 있다. 방금 사용한 데이터 객체 변수 mnist_train를 기준으로 설명한다면,

- mnist_train.data: 입력 데이터셋($X$)에 대한 접근
- mnist_train.targets: $X$에 대응하는 출력 데이터값(레이블)


```python
samples_idx = np.random.randint(mnist_train.data.shape[0], size=(10,))
mnist_x_samples = mnist_train.data[samples_idx, :, :] # 입력 데이터셋에 대한 접근 mnist_train.data
mnist_y_samples = mnist_train.targets[samples_idx] # 출력 데이터셋에 대한 접근 mnist_train.targets

fig, axs = plt.subplots(nrows=2, ncols=5, facecolor='w', figsize=(15, 6))
for i, ax in enumerate(axs.flatten()):
  ax.imshow(mnist_x_samples[i, :, :], cmap='gray')
  ax.set_title(f"label: {mnist_y_samples[i]}", fontsize=12)
  ax.axis('off')

plt.suptitle("MNIST DATASET LOADED BY TORCHVISION", fontsize=25)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()
```


    
![png](/assets/img/post/softmax_mnist/SoftmaxClassifierForMNIST_11_0.png)
    


이제 이 데이터셋을 가지고 Mini-batch 학습을 진행할 건데, 앞선 [Batch, Stochastic, Mini-Batch Gradient Descent](https://zoshs2.github.io/posts/BGDSGDMBGD/#pytorch-mini-batch-%EA%B5%AC%ED%98%84-%EB%B0%A9%EC%8B%9D:~:text=%EC%9D%BC%EB%B0%98%EC%A0%81%EC%9D%B8%20%EB%93%AF%20%ED%95%98%EB%8B%A4.-,DataLoader,-%EB%8A%94%20%EC%9B%90%ED%95%98%EB%8A%94%20batch_size){:target="_blank"}에서 사용한 torch.utils.data.DataLoader를 사용할 것이다. DataLoader는 원하는 batch_size, shuffle 유무, **drop_lasts** 같은 유용한 데이터 처리 기능을 제공하고, 데이터를 iterable 형태로 준비시켜주는 유용한 클래스이다.


```python
# batch_size=64 : 1회 학습에 샘플 64개에 대한 평균 기울기를 사용한다. 즉, 이 경우 1-epoch당 60000/64회 학습이 이뤄지는 셈.
# shuffle=True: 매 epoch마다 데이터셋 전체를 셔플하고 배치를 다시 구성함.
# drop_last=True: 전체 데이터셋 크기를 배치크기로 나눴을 때, 남는 '나머지' 크기만큼의 데이터셋을 학습에 활용할지 말지 정한다.
# >> drop_last: 이 경우 60000 % 64 = 32 개의 나머지 데이터샘플은 학습에 활용하지 않고 버린다.
# >> drop_last 옵션의 default는 False.
data_loader = DataLoader(dataset=mnist_train,
                          batch_size=64,
                          shuffle=True,
                          drop_last=True)
```

모델 학습에 필요한 데이터 처리 및 준비는 이 정도로 마치고, 이제 모델 설계 및 학습 단계에 들어가보자.

# Setup of Computational Configurations

학습 과정에서 수행하는 계산을 **GPU 연산**으로 수행할지, **CPU 연산**으로 수행할지 사전에 세팅하는 단계가 필요하다. 이러한 세팅 방식과 문법들은 실제로 자주 사용하게 될테니, 익숙해지도록 하자.


```python
USE_CUDA = torch.cuda.is_available() # GPU 사용이 가능하면 True, CPU 연산밖에 안되는 상황이면 False를 반환한다.
print(f"CUDA(GPU) IS AVAILABLE ?: {USE_CUDA}", end='\n\n')

device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"[INFO] {device} 환경에서 모델 학습을 진행합니다.")
```

    CUDA(GPU) IS AVAILABLE ?: True
    
    [INFO] cuda 환경에서 모델 학습을 진행합니다.


# Model Architecture

MNIST 입력 데이터는 가로-세로 각 28픽셀 크기안에 손글씨 이미지가 표현된 데이터라고 했다. 이미지 데이터를 인식하는 데 효과적인 딥러닝 모델에는 Convolutional Neural Network(CNN) 등과 같은 모델들이 있으나, 이 글에서는 앞서 배운 Softmax Regression Model을 활용한다.

28 X 28의 2차원 텐서 데이터를 입력층 nn.Linear()에 주입하기 위해, 1차원 텐서로 reshape(784,)하여 nn.Linear()의 입력 데이터로 활용할 것이다. 또한 분류해야할 레이블은 총 0부터 9까지 10개이므로, 출력 갯수는 10이 될 것이다.


```python
model = nn.Linear(784, 10).to(device)
```

to() 함수는 연산을 어디서 수행할지를 명시하는 메소드다. **CPU 연산은 디폴트**이기에 CPU 사용시 굳이 .to('cpu')라고 쓸 필욘없지만, 'cuda(GPU)'를 사용할 경우엔 이렇게 .to('cuda')로 직접 명시해줘야 한다.

아래 손실함수 정의에서도 연산 device를 .to()로 명시해줘야 한다.

> 앞서 [로지스틱 및 소프트맥스 회귀 분류 모델](https://zoshs2.github.io/posts/ClassifierModel/){:target="_blank"} 글에서는 손실함수를 구현할 때, torch.nn.functional.cross_entropy(prediction, labels)를 사용했는데, 이 글에서는 nn.CrossEntropyLoss()를 사용했다. 이 둘의 궁극적인 기능은 동일하지만, 전자는 함수처럼 후자는 클래스 인스턴스로 활용한다는 차이만 있다.
{: .prompt-tip }


```python
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.02)
```


```python
training_epoch = 10
for epoch in range(training_epoch):
  avg_cost = 0 # 한 에포크 내에서 발생한 배치 평균 오차를 구하기 위한 변수
  total_batch = len(data_loader) # Batch가 몇개 있는지 (Batch_size가 아님)

  for x_batch, y_batch in data_loader:
    x_batch = x_batch.reshape(-1, 28 * 28).to(device)
    y_batch = y_batch.to(device)

    predict = model(x_batch)

    optimizer.zero_grad()
    cost = criterion(predict, y_batch)
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch

  print(f"Epoch: {epoch+1}/{training_epoch}, Cost: {cost:.4f}")
```

    Epoch: 1/10, Cost: 0.4809
    Epoch: 2/10, Cost: 0.3895
    Epoch: 3/10, Cost: 0.4972
    Epoch: 4/10, Cost: 0.4733
    Epoch: 5/10, Cost: 0.3773
    Epoch: 6/10, Cost: 0.3910
    Epoch: 7/10, Cost: 0.2716
    Epoch: 8/10, Cost: 0.4915
    Epoch: 9/10, Cost: 0.3889
    Epoch: 10/10, Cost: 0.3041


# Model Test

모델을 학습시키고 나면, 당연히 제대로 학습이 되었는지 **평가**를 해줘야 한다. 평가 과정에선 학습에서 사용하지 않은 데이터를 활용하기 때문에, 앞서 mnist_train / mnist_test 데이터를 따로 분류해둔 것이다. 만약 학습 데이터셋에 대해 높은 정확도와 낮은 손실값을 보여 잘 학습된 것 같아도, 이는 이 데이터셋에만 한정된 성능을 보인 것일 수 있다. 따라서 모델의 설명력을 검증하기 위해선, 모델 평가(Model test) 과정이 꼭 필요하다.

평가 과정에선 학습 파라미터들(weight, bias)의 업데이트를 수행하지 않는다. 즉, 말 그대로 모델을 검증하기 위한 절차이기에, 학습에 관련된 기능들을 off 시키고 정답을 잘 맞추는지만 확인한다.

## PyTorch for test procedure

평가 단계에서는 **torch.no_grad**와 **model.eval** 을 통해 상황을 통제시킨다. torch.no_grad()는 PyTorch의 **Autograd engine을 비활성화**시키는 역할을 수행한다. 파이토치의 Autograd engine은 모델의 forward 및 backward 모든 과정에서 텐서의 연산 흐름을 추적하고 Gradient 계산에 필요한 계산 그래프를 따로 구축한다. 궁극적으로 이러한 일은 평가 단계에서 필요하지 않기에, torch.no_grad()를 통해 비활성화시켜 메모리를 절약하고 계산 속도를 높이는 이점을 얻을 수 있다.

model.eval()은, 이 글에서는 활용하진 않겠지만, Batch Normalization이나 Drop-out과 같이 모델을 구성하는 layer의 일부로서 학습에 활용되는 기능을 비활성화시키는 역할을 수행한다. 계속 언급했듯이, 평가 단계에서는 학습된 모델 상태에 대해 있는 그대로 평가하는 것이 목적이기 때문이다.


```python
mnist_test
```




    Dataset MNIST
        Number of datapoints: 10000
        Root location: MNIST_data/
        Split: Test
        StandardTransform
    Transform: ToTensor()




```python
mnist_test.data.shape
```




    torch.Size([10000, 28, 28])



> 매뉴얼에서는 **mnist_test.test_data**로 input 데이터에 접근하지만, 최근 PyTorch 버전에서는 **mnist_test.data**로 변경되었다. 다음과 같은 UserWarning이 출력됨. **UserWarning: test_data has been renamed data.**
{: .prompt-warning }

> 매뉴얼에서는 **mnist_test.test_labels**로 label 데이터에 접근하지만, 이는 **mnist_test.targets**로 변경되었다. 다음과 같은 UserWarning이 출력됨. **UserWarning: test_labels has been renamed targets.**
{: .prompt-warning }



```python
# with torch.no_grad()의 코드 블록 수행시, PyTorch의 Autograd engine을 비활성화 시킨다. (메모리 절약 및 계산속도 향상 효과)
with torch.no_grad():
  X_test = mnist_test.data.reshape(-1, 28*28).float().to(device)
  Y_test = mnist_test.targets.to(device)

  # test 데이터 샘플 10000개에 대한 출력층 10개 노드에 대한 linear 결과가 출력되므로, shape은 (10000, 10)이 출력됨.
  prediction = model(X_test)

  # 10000개에 대한 결과 중 최대값 레이블 인덱스만 추출, 이를 정답 레이블 인덱스와 비교.
  accuracy = (torch.argmax(prediction, dim=1) == Y_test).float().mean()
  print(f"Accuracy: {accuracy*100:.3f}%")

```

    Accuracy: 90.500%


테스트 데이터 샘플셋에서 랜덤으로 데이터를 뽑아 figure와 함께 추론결과를 확인해보자.

> **TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first**: 사전에 tensor가 to('cuda')로 되어 있는 경우, 데이터 타입 자리에 device='cuda:0'가 들어가 있다. 이는 텐서 안의 data 값만을 추출할 때 또는 np.array()로 변환할 때 문제가 되는데, 이를 해결하기 위해서는 tensor변수.cpu() 또는 tensor변수.cpu().numpy()를 사용하면 된다.
{: .prompt-tip }


```python
# 평가) 그림으로 추론 결과 확인
with torch.no_grad():
  test_indices = np.random.randint(0, mnist_test.targets.shape[0], size=(10, ))

  X_test = mnist_test.data[test_indices, :, :].reshape(-1, 28*28).float().to(device)
  Y_test = mnist_test.targets[test_indices].to(device)

  output = model(X_test)
  predicted_labels = torch.argmax(output, dim=1)

fig, axs = plt.subplots(nrows=2, ncols=5, facecolor='w', figsize=(15, 6))
for i, ax in enumerate(axs.flatten()):
  test_img = X_test[i].reshape(28, 28)
  ax.imshow(test_img.cpu(), cmap='gray')
  ax.set_title(f"Label: {Y_test[i]}, Predict: {predicted_labels[i].item()}", fontsize=12)
  ax.axis('off')

plt.suptitle("MODEL INFERENCE RESULTS", fontsize=25)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.show()

```


    
![png](/assets/img/post/softmax_mnist/SoftmaxClassifierForMNIST_28_0.png)
    

