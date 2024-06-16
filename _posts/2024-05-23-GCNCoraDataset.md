---
title: "GCN Model to Cora dataset [ft. Pytorch Lightning]"
date: 2024-05-23 17:23:54 +0900
categories: [DL/ML, Graph Neural Networks]
tags: [Convolution, Graph Convolution Network, GCN, GNNs, Graph, Network, Neural, relu, graph-structured, Deep Learning, Machine Learning, DNN, MLP, Python, Kipf, CNN, Convolutional Neural Network, PyTorch, Torch, PyTorch Geometric, PyG, Message, Message Passing, Filter, Kernel, localization, feature, node feature, propagation formula, pl, PyTorch Lightning, callbacks, EarlyStopping, GCNConv, GCN layer, Graph Neural Networks, Cora, real network, torchinfo, summary, LightningModule, BatchNorm, Batch Normalization, classification]
math: true
toc: false
---

# Table of Contents
- [Introduction](#introduction)
- [Cora Dataset: Citation network](#cora-dataset-citation-network)
- [GCN Model with torch_geometric.nn](#gcn-model-with-torch_geometricnn)
- [Training and Test](#training-and-test)
  - [PyTorch Lightning](#pytorch-lightning)
  - [pl.Trainer](#pltrainer)
- [Conclusions](#conclusions)

# Introduction

이번 포스트에서는 PyTorch 딥러닝 프레임워크를 적극 활용하여, Graph Convolutional Network(GCN) 모델을 설계하고 모델의 benchmark 데이터로서 많이 사용하는 graph-structured dataset 하나를 골라 직접 적용해보는 시간을 갖겠다.

# Cora Dataset: Citation network

원래 [앞선 포스트](https://zoshs2.github.io/posts/GNNsHistory/#conclusion:~:text=%EC%98%88%EC%8B%9C%EB%A5%BC%20%EA%B0%80%EC%A0%B8%EC%98%A8%20%EA%B2%83%EC%9D%B4%EB%8B%A4.-,Conclusion,-%EC%82%AC%EC%8B%A4%20%EC%9D%B4%EB%B2%88%20%EA%B8%80%EC%9D%84){:target="_blank"}에서 "그래프/네트워크 데이터의 MNIST라 할 수 있는 자카리 가라데 클럽 데이터(Zachary's karate club)를 가지고 뭔가 해보는 글이 될 것 같다."라고 말했었는데, 다른 benchmark 데이터셋으로 바꿔 사용하고자 한다.

바로 Cora dataset 이다. Cora dataset은 학술 논문들 사이 인용 관계에 대한 citation network다. 총 2708개의 노드로 구성되어 있고, 각 노드는 한 편의 학술 논문을 의미한다. 그리고 두 노드 사이의 링크는 (피)인용 관계에 있다는 것을 의미한다.

![png](/assets/img/post/gcn_cora/cora_visualization_better.png)*Visualization of the Cora dataset, Source: https://graphsandnetworks.com/the-cora-dataset/*

이 Cora dataset을 가지고 우리가 해야하는 일의 목표는 각 노드(논문)의 클래스(과학 세부 분야)를 분류해내는 모델을 만드는 것이다. 각 노드(논문)는 총 7가지 클래스(분야) 중 하나의 클래스가 할당되어 있다.

그럼 노드(논문)의 어떤 feature들을 활용하여, ***논문이 어느 분야에 속한 논문인지***, 판별하도록 할 것인가? Cora dataset은 주요 학술 키워드 단어들을 모아둔 (Pre-defined) dictionary란 것이 있다. 간단하게 **학술 단어 모음집**이라고 하자. 이 학술 단어 모음집은 총 1433개의 단어들이 수록되어 있는데, 이 단어들이 2708개의 각 논문들의 본문에서 등장하는지에 따라 0(미출현) 또는 1(출현)로 표기할 수 있고, 이를 Input node feature matrix로 활용한다. 이렇게 특정 단어의 출현 여부를 0과 1만으로 표기하는 방식을 Binary bag-of-words vector라고 한다.

![png](/assets/img/post/gcn_cora/cora_input_features.png)*Input node features matrix of Cora dataset*

이제 Cora dataset을 직접 한번 불러오자. Cora dataset은 [PyG(PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/en/latest/){:target="_blank"} 라이브러리에서 불러올 수 있다.


```python
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

- - -

2.3.0+cu121
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.9/10.9 MB[0m [31m19.6 MB/s[0m eta [36m0:00:00[0m
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.1/5.1 MB[0m [31m21.3 MB/s[0m eta [36m0:00:00[0m
[?25h  Installing build dependencies ... [?25l[?25hdone

Getting requirements to build wheel ... [?25l[?25hdone
Preparing metadata (pyproject.toml) ... [?25l[?25hdone
Building wheel for torch-geometric (pyproject.toml) ... [?25l[?25hdone
```


```python
from torch_geometric.datasets import Planetoid

cora_dataset = Planetoid(root="./cora_dataset", name='Cora')
```

    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.x
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.tx
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.allx
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.y
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ty
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ally
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.graph
    Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.test.index
    Processing...
    Done!



```python
>>> !apt-get install tree
>>> !tree ./cora_dataset

    Reading package lists... Done
    Building dependency tree... Done
    Reading state information... Done
    The following NEW packages will be installed:
      tree
    0 upgraded, 1 newly installed, 0 to remove and 45 not upgraded.
    Need to get 47.9 kB of archives.
    After this operation, 116 kB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu jammy/universe amd64 tree amd64 2.0.2-1 [47.9 kB]
    Fetched 47.9 kB in 1s (83.1 kB/s)
    Selecting previously unselected package tree.
    (Reading database ... 121913 files and directories currently installed.)
    Preparing to unpack .../tree_2.0.2-1_amd64.deb ...
    Unpacking tree (2.0.2-1) ...
    Setting up tree (2.0.2-1) ...
    Processing triggers for man-db (2.10.2-1) ...
    [01;34m./cora_dataset[0m
    └── [01;34mCora[0m
        ├── [01;34mprocessed[0m
        │   ├── [00mdata.pt[0m
        │   ├── [00mpre_filter.pt[0m
        │   └── [00mpre_transform.pt[0m
        └── [01;34mraw[0m
            ├── [00mind.cora.allx[0m
            ├── [00mind.cora.ally[0m
            ├── [00mind.cora.graph[0m
            ├── [00mind.cora.test.index[0m
            ├── [00mind.cora.tx[0m
            ├── [00mind.cora.ty[0m
            ├── [00mind.cora.x[0m
            └── [00mind.cora.y[0m
    
    3 directories, 11 files
```


```python
# cora_dataset 은 graph가 하나 뿐이다. 그래서 0 indexing 한게 cora_dataset의 전부임.
data = cora_dataset[0]
data
```




    Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])



input node features 인 x의 크기가 앞서 본 그림의 예시처럼 2708 by 1433 임을 알 수 있다. 그리고 edge_index는 GCN propagation 모델에서 adjacency matrix 연산 수행, 즉 네트워크 연결 구조를 반영한다. y는 Cora dataset 설명에서 얘기한 7가지 클래스에 대한 레이블 인덱스들이 담겨 있다.


```python
import pandas as pd

print("Y labels (Classes): ", pd.unique(data.y.numpy()))
```

    Y labels (Classes):  [3 4 0 2 1 5 6]


# GCN Model with torch_geometric.nn

GCN 모델을 구현해보자. GCNConv layer는 torch 라이브러리에 없고, torch_geometric 라이브러리에 있다. 정식 명칭은 **PyTorch Geometric(PyG) 라이브러리** 인데, 내가 다룰 GCN과 graph-structured Cora dataset과 같이 그래프 신경망(GNNs) 학습에 특화된 확장 라이브러리라 보면 된다.

아래 GCNModel 클래스에서 c_in은 input node features의 갯수(Cora 데이터는 1433개), c_hidden은 node feature들을 message로 변환시키는 역할을 하면서 각 노드의 (기존 1433개 였던 feature를) 새로운 feature 갯수를 결정하는 은닉층 내 가중치 행렬의 크기다. (아래 그림 참고)

![png](/assets/img/post/gcn_cora/io_channels_meaning.png)*GCN Propagation formula: c_in, c_hidden의 의미*

마지막으로 c_out은 출력층에 해당하는 GCN layer의 output feature 차원인데, 일반적으로 classification 문제에서 분류하고자 하는 클래스의 갯수로 지정한다. 따라서 Cora dataset의 경우 c_out=7이 되겠다. num_layers는 (입/출력층을 포함해) GCN layer를 총 몇개 넣을 것인지의 옵션이다.

> 은닉층의 입출력 feature 차원 수는 일정하게 유지되게 끔 했다. 예컨대 c_hidden=16, num_layers=10이면, 입출력 층(I/O layer)을 제외하고 가운데 8개 GCN hidden layer들은 모두 input feature 차원도 16, output feature 차원도 16이다.
{: .prompt-tip }


```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geom_nn
import torch_geometric.loader as geom_loader

class GCNModel(nn.Module):
  def __init__(self, c_in, c_hidden, c_out, num_layers=2, **kwargs):
    super().__init__()
    gcn_layer = geom_nn.GCNConv

    layers = []
    in_channels, out_channels = c_in, c_hidden

    # 중간 은닉층
    for l_idx in range(num_layers-1):
      layers += [
          gcn_layer(in_channels=in_channels,
                    out_channels=out_channels,
                    **kwargs),
          nn.ReLU(inplace=True),   # BatchNorm 을 ReLU 앞에 써야되나, 뒤에 써야되나의 문제는 여전히 논란이 분분하다.
          nn.BatchNorm1d(num_features=out_channels)
      ]
      in_channels = c_hidden

    # 마지막 출력층
    layers += [gcn_layer(in_channels=in_channels,
                          out_channels=c_out,
                          **kwargs)
    ]
    self.layers = nn.ModuleList(layers)

  def forward(self, x, edge_index):

    for l in self.layers:
      if isinstance(l, geom_nn.MessagePassing):
        # GNN 모듈인지 검사 - GNN 모듈일 시 edge_index가 꼭 필요하므로
        # GNN 모듈들은 모두 MessagePassing 클래스를 상속받고 있기에, 이런 condition을 사용한다.
        x = l(x, edge_index)

      else:
        x = l(x)

    return x
```


```python
my_gcn = GCNModel(c_in=cora_dataset.num_features, c_hidden=16, c_out=cora_dataset.num_classes, num_layers=3)
my_gcn

- - -

    GCNModel(
      (layers): ModuleList(
        (0): GCNConv(1433, 16)
        (1): ReLU(inplace=True)
        (2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): GCNConv(16, 16)
        (4): ReLU(inplace=True)
        (5): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): GCNConv(16, 7)
      )
    )
```


이렇게 만든 모델의 architectural profile를 보기 좋게 요약해주는 **torchinfo**라는 라이브러리가 있다. 이를 통해 trainable parameters 갯수가 얼마나 되는지, data dimension이 layer를 통과하면서 어떻게 변화하는지 한 눈에 보기 쉽게 정리해준다.


```python
!pip install torchinfo


Collecting torchinfo
Downloading torchinfo-1.8.0-py3-none-any.whl (23 kB)
Installing collected packages: torchinfo
Successfully installed torchinfo-1.8.0
```


```python
from torchinfo import summary
summary(my_gcn, input_data=(data.x, data.edge_index))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    GCNModel                                 [2708, 7]                 --
    ├─ModuleList: 1-1                        --                        --
    │    └─GCNConv: 2-1                      [2708, 16]                16
    │    │    └─Linear: 3-1                  [2708, 16]                22,928
    │    │    └─SumAggregation: 3-2          [2708, 16]                --
    │    └─ReLU: 2-2                         [2708, 16]                --
    │    └─BatchNorm1d: 2-3                  [2708, 16]                32
    │    └─GCNConv: 2-4                      [2708, 16]                16
    │    │    └─Linear: 3-3                  [2708, 16]                256
    │    │    └─SumAggregation: 3-4          [2708, 16]                --
    │    └─ReLU: 2-5                         [2708, 16]                --
    │    └─BatchNorm1d: 2-6                  [2708, 16]                32
    │    └─GCNConv: 2-7                      [2708, 7]                 7
    │    │    └─Linear: 3-5                  [2708, 7]                 112
    │    │    └─SumAggregation: 3-6          [2708, 7]                 --
    ==========================================================================================
    Total params: 23,399
    Trainable params: 23,399
    Non-trainable params: 0
    Total mult-adds (M): 63.26
    ==========================================================================================
    Input size (MB): 15.69
    Forward/backward pass size (MB): 1.54
    Params size (MB): 0.09
    Estimated Total Size (MB): 17.32
    ==========================================================================================



GCNConv은 우리가 c_hidden, c_out으로 output dimension이 결정되는데, 위 모델 summary에서 16, 16, 7로 되어 있는건 trainable bias parameters의 갯수를 의미한다. 그리고 GCNConv 에서 행렬곱 연산 과정에 관여하는 22928(1433 X 16), 256(16 X 16), 112(16 X 7)개의 trainablee weight parameter들이 존재한다는 의미다. (GCN Propagation formula를 다시 떠올려 보자.)

그리고 이전 [Graph Convolutional Network & Message Passing 포스트](https://zoshs2.github.io/posts/GCNHandbook/){:target="_blank"}에서도 언급했듯이 message aggregation 방식을 평균으로 취하면 node-specific information이 사라지는 이슈가 있기 때문에, Sum 집계방식을 택하거나 | 애초부터 $H^{(l)}W^{(l)}$ 연산을 수행할 시 노드 자신에 대한 부분과 이웃 노드에 대한 부분으로 구분하는 연산 방식을 일반적으로 사용한다고 했었다. 그래서 여기서도 각 노드에 모인 이웃 노드들의 Message들을 Sum 한 것이고, SumAggregation이라는 의미가 그런 의미인 것이다.

BatchNorm layer 부분은, **각 feature dimension에 대해** 평균($\mu_{B}$)과 분산($\sigma^{2}_{B}$)을 계산하고 정규화를 진행한다. (BatchNormalization에 대한 자세한 내용은 이 글에서 풀지 않겠다.) 이 과정에서 학습되는 파라미터가 각 feature dimension마다 2개씩 존재하는데, 정규화된 값의 스케일(scale) 조정을 위한 파라미터 $\gamma$ (감마)와 오프셋 조정을 위한 파라미터 $\beta$ (베타) 값이 그것이다.

![png](/assets/img/post/gcn_cora/BatchNorm_algorithm.png)*Batch Normalization Algorithm, Source: [S Ioffe, C Szegedy (2015)](https://arxiv.org/abs/1502.03167){:target="_blank"}*

이 2개의 Batch Normalization 학습 파라미터들은 feature dimension 마다 적용되므로, 16 X 2($\gamma$, $\beta$) = 32개의 BatchNorm 파라미터 갯수로 요약되는 것이다.

# Training and Test

이제 모델을 훈련시키고, 잘 훈련되었는지 평가 또한 수행해보자. 앞서 말했듯이, 이 모델 및 Cora dataset의 task type은 Node Classification이다. 모델 입력으로서, 논문들의 인용 관계가 표현된 network 구조(edge_index)와 binary bag-of-words vector들을 지닌 feature matrix가 모델의 입력 데이터로 주입되고, 최종적으로 모델이 각 논문들에 맞는 클래스(세부 분야)를 분류해내는 것이다.

한 가지 유념할 부분은, Cora dataset은 Graph가 한 개짜리 데이터셋이다. 즉, 각기 다른 연결 구조의(또는 각기 다른 feature vectors들을 지닌) 다양한 그래프들을 가지고 모델을 학습시키는 방식이 아니란 소리다. 오직 하나의 Cora network만 존재하기 때문에, 하나의 output 결과에서 학습(train)시킬 output part/ 검증(validation)할 output part/ 테스트(test)할 output part를 나눠서 진행한다. 이는 이미 boolean masking array로서, torch_geometric.datasets의 Cora dataset에 내장되어 있다.


```python
print(data.train_mask)
print(data.val_mask)
print(data.test_mask)
```

    tensor([ True,  True,  True,  ..., False, False, False])
    tensor([False, False, False,  ..., False, False, False])
    tensor([False, False, False,  ...,  True,  True,  True])


쉽게 말해서 Cora dataset을 통해 모델 학습시, forwarding을 통해 나온 output 결과에서 train masking된 **일부** 노드들의 결과들과 그에 대응하는 **일부** 정답들 사이의 loss 만을 가지고 모델을 학습시키는 것이다. 그리고 검증(Validation), 평가(Test) 단계 또한 이처럼 진행된다. (아래 그림 참고)

![png](/assets/img/post/gcn_cora/cora_dataset_usage.png)*Schematic of Cora Dataset Usage*


```python
my_gcn = GCNModel(c_in=cora_dataset.num_features, c_hidden=16, c_out=cora_dataset.num_classes, num_layers=3)
data_loader = geom_loader.DataLoader(cora_dataset, batch_size=1)

loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(params=my_gcn.parameters(), lr=0.01)
optimizer = optim.SGD(params=my_gcn.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)

def gcn_train(num_epochs):

  for epoch in range(num_epochs):
    my_gcn.train(mode=True)

    for batch_data in data_loader:
      train_mask = batch_data.train_mask
      input_x, edge_index = batch_data.x, batch_data.edge_index
      outputs = my_gcn(input_x, edge_index)

      optimizer.zero_grad()
      train_loss = loss_fn(outputs[train_mask], batch_data.y[train_mask])
      train_loss.backward()
      optimizer.step()

    train_acc = (outputs[train_mask].argmax(dim=-1) == batch_data.y[train_mask]).sum().float() / sum(train_mask)

    my_gcn.eval() # my_gcn.train(mode=False) 와 동일
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
      for batch_data in data_loader: # 이와 다르게 원래 일반적으론 train/val/test loader 따로 구성해놓는다.
        val_mask = batch_data.val_mask
        input_x, edge_index = batch_data.x, batch_data.edge_index
        outputs = my_gcn(input_x, edge_index)

        val_loss += loss_fn(outputs[val_mask], batch_data.y[val_mask])
        val_acc += ((outputs[val_mask].argmax(dim=-1) == batch_data.y[val_mask]).sum().float() / sum(val_mask))

      avg_val_loss = val_loss / len(data_loader)
      avg_val_acc = val_acc / len(data_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f} (Acc: {100.0*train_acc:4.2f}%), Validation Loss: {avg_val_loss:.4f} (Acc: {100.0*avg_val_acc:4.2f}%)')

def gcn_test():
  my_gcn.eval()
  test_size = 0
  batch_correct = 0

  with torch.no_grad():
    for batch_data in data_loader: # 이와 다르게 원래 일반적으론 train/val/test loader 따로 구성해놓는다.
      test_mask = batch_data.test_mask
      test_size += sum(test_mask)

      input_x, edge_index = batch_data.x, batch_data.edge_index
      outputs = my_gcn(input_x, edge_index)

      batch_correct += (outputs[test_mask].argmax(dim=-1) == batch_data.y[test_mask]).sum()

  test_acc = batch_correct / test_size
  print(f"Test Accuracy: {100.0 * test_acc:4.2f}%")

```


```python
gcn_train(num_epochs=50)
```

    Epoch [1/50], Training Loss: 2.5227 (Acc: 10.71%), Validation Loss: 1.9305 (Acc: 22.20%)
    Epoch [2/50], Training Loss: 1.2865 (Acc: 57.14%), Validation Loss: 1.9027 (Acc: 36.40%)
    Epoch [3/50], Training Loss: 0.8965 (Acc: 77.86%), Validation Loss: 1.8675 (Acc: 41.60%)
    Epoch [4/50], Training Loss: 0.7435 (Acc: 84.29%), Validation Loss: 1.8251 (Acc: 49.40%)
    Epoch [5/50], Training Loss: 0.6341 (Acc: 87.86%), Validation Loss: 1.7756 (Acc: 53.60%)
    Epoch [6/50], Training Loss: 0.5429 (Acc: 91.43%), Validation Loss: 1.7189 (Acc: 57.00%)
    Epoch [7/50], Training Loss: 0.4638 (Acc: 92.86%), Validation Loss: 1.6535 (Acc: 59.20%)
    Epoch [8/50], Training Loss: 0.3933 (Acc: 94.29%), Validation Loss: 1.5793 (Acc: 61.20%)
    Epoch [9/50], Training Loss: 0.3300 (Acc: 96.43%), Validation Loss: 1.4977 (Acc: 62.60%)
    Epoch [10/50], Training Loss: 0.2747 (Acc: 97.14%), Validation Loss: 1.4109 (Acc: 65.20%)
    Epoch [11/50], Training Loss: 0.2280 (Acc: 98.57%), Validation Loss: 1.3225 (Acc: 66.00%)
    Epoch [12/50], Training Loss: 0.1893 (Acc: 98.57%), Validation Loss: 1.2360 (Acc: 67.60%)
    Epoch [13/50], Training Loss: 0.1576 (Acc: 99.29%), Validation Loss: 1.1551 (Acc: 68.60%)
    Epoch [14/50], Training Loss: 0.1318 (Acc: 99.29%), Validation Loss: 1.0824 (Acc: 68.40%)
    Epoch [15/50], Training Loss: 0.1107 (Acc: 99.29%), Validation Loss: 1.0193 (Acc: 70.40%)
    Epoch [16/50], Training Loss: 0.0936 (Acc: 99.29%), Validation Loss: 0.9664 (Acc: 70.60%)
    Epoch [17/50], Training Loss: 0.0793 (Acc: 100.00%), Validation Loss: 0.9241 (Acc: 71.20%)
    Epoch [18/50], Training Loss: 0.0676 (Acc: 100.00%), Validation Loss: 0.8922 (Acc: 72.40%)
    Epoch [19/50], Training Loss: 0.0578 (Acc: 100.00%), Validation Loss: 0.8697 (Acc: 73.20%)
    Epoch [20/50], Training Loss: 0.0495 (Acc: 100.00%), Validation Loss: 0.8552 (Acc: 72.80%)
    Epoch [21/50], Training Loss: 0.0425 (Acc: 100.00%), Validation Loss: 0.8479 (Acc: 72.20%)
    Epoch [22/50], Training Loss: 0.0366 (Acc: 100.00%), Validation Loss: 0.8466 (Acc: 72.00%)
    Epoch [23/50], Training Loss: 0.0316 (Acc: 100.00%), Validation Loss: 0.8502 (Acc: 72.20%)
    Epoch [24/50], Training Loss: 0.0275 (Acc: 100.00%), Validation Loss: 0.8577 (Acc: 72.00%)
    Epoch [25/50], Training Loss: 0.0242 (Acc: 100.00%), Validation Loss: 0.8679 (Acc: 72.20%)
    Epoch [26/50], Training Loss: 0.0214 (Acc: 100.00%), Validation Loss: 0.8803 (Acc: 72.20%)
    Epoch [27/50], Training Loss: 0.0191 (Acc: 100.00%), Validation Loss: 0.8938 (Acc: 72.40%)
    Epoch [28/50], Training Loss: 0.0172 (Acc: 100.00%), Validation Loss: 0.9078 (Acc: 72.60%)
    Epoch [29/50], Training Loss: 0.0156 (Acc: 100.00%), Validation Loss: 0.9220 (Acc: 72.80%)
    Epoch [30/50], Training Loss: 0.0142 (Acc: 100.00%), Validation Loss: 0.9359 (Acc: 73.00%)
    Epoch [31/50], Training Loss: 0.0130 (Acc: 100.00%), Validation Loss: 0.9495 (Acc: 73.00%)
    Epoch [32/50], Training Loss: 0.0119 (Acc: 100.00%), Validation Loss: 0.9628 (Acc: 72.60%)
    Epoch [33/50], Training Loss: 0.0109 (Acc: 100.00%), Validation Loss: 0.9756 (Acc: 72.40%)
    Epoch [34/50], Training Loss: 0.0101 (Acc: 100.00%), Validation Loss: 0.9880 (Acc: 72.00%)
    Epoch [35/50], Training Loss: 0.0094 (Acc: 100.00%), Validation Loss: 0.9996 (Acc: 72.20%)
    Epoch [36/50], Training Loss: 0.0088 (Acc: 100.00%), Validation Loss: 1.0106 (Acc: 72.00%)
    Epoch [37/50], Training Loss: 0.0082 (Acc: 100.00%), Validation Loss: 1.0208 (Acc: 72.00%)
    Epoch [38/50], Training Loss: 0.0077 (Acc: 100.00%), Validation Loss: 1.0305 (Acc: 72.20%)
    Epoch [39/50], Training Loss: 0.0073 (Acc: 100.00%), Validation Loss: 1.0394 (Acc: 72.20%)
    Epoch [40/50], Training Loss: 0.0070 (Acc: 100.00%), Validation Loss: 1.0476 (Acc: 72.20%)
    Epoch [41/50], Training Loss: 0.0067 (Acc: 100.00%), Validation Loss: 1.0552 (Acc: 71.80%)
    Epoch [42/50], Training Loss: 0.0064 (Acc: 100.00%), Validation Loss: 1.0622 (Acc: 72.00%)
    Epoch [43/50], Training Loss: 0.0061 (Acc: 100.00%), Validation Loss: 1.0686 (Acc: 72.00%)
    Epoch [44/50], Training Loss: 0.0059 (Acc: 100.00%), Validation Loss: 1.0744 (Acc: 72.00%)
    Epoch [45/50], Training Loss: 0.0057 (Acc: 100.00%), Validation Loss: 1.0797 (Acc: 72.40%)
    Epoch [46/50], Training Loss: 0.0056 (Acc: 100.00%), Validation Loss: 1.0845 (Acc: 72.60%)
    Epoch [47/50], Training Loss: 0.0054 (Acc: 100.00%), Validation Loss: 1.0891 (Acc: 72.40%)
    Epoch [48/50], Training Loss: 0.0053 (Acc: 100.00%), Validation Loss: 1.0933 (Acc: 72.00%)
    Epoch [49/50], Training Loss: 0.0051 (Acc: 100.00%), Validation Loss: 1.0971 (Acc: 72.00%)
    Epoch [50/50], Training Loss: 0.0050 (Acc: 100.00%), Validation Loss: 1.1005 (Acc: 71.80%)



```python
gcn_test()
```

    Test Accuracy: 71.60%


## PyTorch Lightning

앞서 모델의 훈련과 검증, 테스트를 수행할 때 지금까지 해왔던 방식대로, 모델을 먼저 따로 클래스화하고 또 training/validation/test 함수를 따로 만들고, optimizer와 loss function을 따로 선언하는 방식으로 진행했다.

보다시피 DL/ML 모델은 모델(클래스) 그 자체만이 중요한 것이 아니라, 각 단계에 대해 종속적이며 loss function 및 optimizer에 대한 적절한 선택도 중요한 부분을 차지한다. 모델이라 함은 이들 모두를 총칭하는 것이다. 따라서 한 개별 DL/ML 모델에 대해서 이들을 종합적으로 관리, 즉 일종의 **구조적 관리**가 필요하다. 구조적 관리가 가능한 모델은 그만큼 특정 부분을 쉽게 변경하거나 추가할 수 있는 유연성(flexibility)을 갖게 된다.

> DL/ML 모델의 유연성이 중요한 이유를 한 가지 더 덧붙이자면, DL/ML 분야 자체가 **경험적이고 직접적인 파라미터 실험에 의해** 모델을 채택하기 때문이다. 즉, 절대적인 답이란 없다. 왜 서로 다른 layer의 순서를 바꿨더니 성능이 더 좋아지고, layer 수를 줄였더니 성능이 좋아지고, Optimizer을 Adam이 아닌 SGD로 바꿨더니 훨씬 더 좋아지고, BatchNorm layer를 ReLU 앞에 넣었더니 성능이 더 나빠지고 등에 대한 명확한 답이나 원인을 설명하기 힘들다. (실제로 BatchNorm layer를 ReLU 앞에 넣어야 되는지, 뒤에 넣어야 하는지에 대한 논쟁은 아직까지 분분하다.). 이에 대해 Reddit에서 누군가는 이렇게 말했다. "The field of machine learning has so much empirical experimenting. I suggest to just try whether it improves performance or not." - 즉, (그걸 했을 때) 성능이 좋아질지 안좋아질지 물어보지 말고 일단 한번 해봐. 성능이 좋으면 그게 좋은거야. - 그래서 하나의 모델을 구조화시키고 유연한 보수가 가능하게끔 설계하는 것이 tuning의 측면에서 좋은 것이다.
{: .prompt-info }

[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/){:target="_blank"}은 코드의 구조화를 돕고 복잡한 모델 훈련 과정을 간소화시켜주는 딥러닝 프레임워크다. 2019년 뉴욕대학교(NYU) 박사 학위 과정 - 당시 그의 지도교수: 조경현 교수(Kyunghyun Cho), 얀 르쿤(Yann LeCun) - 에 있던 [William Falcon](https://www.williamfalcon.com/){:target="_blank"}이 페이스북 인공지능 연구소(Facebook AI Research; FAIR)와 연계된 연구를 수행하는 기간 중, PyTorch Lightning 라는 독립된 딥러닝 프레임워크를 오픈소스로 공개했다고 한다. 현재는 Lightning AI의 CEO다.

PyTorch Lightning을 통해, 앞서 수행했던 학습/검증/평가 과정을 다시 재현해보고 그 차이를 실감해보자.




```python
!pip install --quiet pytorch-lightning>=1.4
```


```python
import pytorch_lightning as pl

class classificationGCN(pl.LightningModule):
  def __init__(self, **model_kwargs):
    super().__init__()

    self.save_hyperparameters()

    self.gcn_model = GCNModel(**model_kwargs)
    self.loss_fn = nn.CrossEntropyLoss()

  def forward(self, data, mode='train'):
    x, edge_index = data.x, data.edge_index
    outputs = self.gcn_model(x, edge_index)

    if mode == 'train':
      mask = data.train_mask
    elif mode == 'val':
      mask = data.val_mask
    elif mode == 'test':
      mask = data.test_mask
    else:
      raise ValueError("Choose one mode of 'train', 'val', and 'test'.")

    loss = self.loss_fn(outputs[mask], data.y[mask])
    acc = (outputs[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
    return loss, acc

  def configure_optimizers(self):
    optimizer = optim.SGD(params=self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
    return optimizer

  def training_step(self, batch, batch_idx):
    loss, acc = self.forward(batch, mode="train")
    self.log('train_loss', loss)
    self.log('train_acc', acc)
    return loss

  def validation_step(self, batch, batch_idx):
    loss, acc = self.forward(batch, mode='val')
    self.log('val_loss', loss)
    self.log('val_acc', acc)

  def test_step(self, batch, batch_idx):
    _, acc = self.forward(batch, mode='test')
    self.log('test_acc', acc)
```

**pl.LightningModule** 클래스를 상속받음으로써 PyTorch Lightning 식의 구조화가 가능하다. 보다시피, 이 하나의 클래스 안에 loss function, optimizer, train/validation/test 과정이 각 클래스의 메소드(method)들로 전부 구현할 수 있고, 이렇게 만들고 나니 해당 모델이 어떻게 설계되어 동작하는지 훨씬 더 한눈에 잘 들어온다.

참고로 pl.LightningModule 기반으로 모델 설게시, configure_optimizers(), training_step(), validation_step(), test_step() 등의 **메소드 이름들을 함부로 변경해서는 안된다**. LightningModule 에서 상속받은 기능들을 활용하기 위해선, 이처럼 정형화된 규칙들은 인지하도록 노력하고 미리 알아두어야 한다.


```python
model = classificationGCN(c_in=cora_dataset.num_features, c_hidden=16, c_out=cora_dataset.num_classes, num_layers=3)
model
```




    classificationGCN(
      (gcn_model): GCNModel(
        (layers): ModuleList(
          (0): GCNConv(1433, 16)
          (1): ReLU(inplace=True)
          (2): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): GCNConv(16, 16)
          (4): ReLU(inplace=True)
          (5): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): GCNConv(16, 7)
        )
      )
      (loss_fn): CrossEntropyLoss()
    )



## pl.Trainer

자 그럼 lightning으로 설계한 모델을 먼저 어떻게 학습시키고 사용할까? training_step() / validation_step() / test_step() 메서드들이 있으니 이걸 그냥 쓰면 될까? 그렇지 않다. 이들은 pytorch_lightning(pl) Trainer 클래스의 인스턴스로 동작되는 대상일 뿐이다.

앞으로 보겠지만, [Trainer 클래스](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer){:target="_blank"}의 fit() 메서드로 training_step()와 validation_step() 메서드가 동작하고, test() 메서드로 test_step() 메서드가 동작한다.


```python
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)
```

    cpu



```python
from pytorch_lightning.callbacks import EarlyStopping

SAVED_MODEL_PATH = './saved_model'
root_dir = os.path.join(SAVED_MODEL_PATH, 'classificationGCN')

trainer = pl.Trainer(default_root_dir=root_dir,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=4, mode='min')], # 콜백 리스트에 EarlyStopping(학습 조기 중단) 추가
                     accelerator="gpu" if str(device).startswith('cuda') else "cpu",
                     max_epochs=200, enable_progress_bar=False)
```

    INFO:pytorch_lightning.utilities.rank_zero:GPU available: False, used: False
    INFO:pytorch_lightning.utilities.rank_zero:TPU available: False, using: 0 TPU cores
    INFO:pytorch_lightning.utilities.rank_zero:HPU available: False, using: 0 HPUs



```python
# 원래 2번째 argument는 training data loader, 3번째는 validation data loader이지만,
# Cora dataset는 말했듯이 그래프 데이터셋 한 개로 수행하는 task라 이렇게 동일한 데이터셋 로더를 넣었다.
trainer.fit(model, data_loader, data_loader)
```

    INFO:pytorch_lightning.callbacks.model_summary:
      | Name      | Type             | Params | Mode 
    -------------------------------------------------------
    0 | gcn_model | GCNModel         | 23.4 K | train
    1 | loss_fn   | CrossEntropyLoss | 0      | train
    -------------------------------------------------------
    23.4 K    Trainable params
    0         Non-trainable params
    23.4 K    Total params
    0.094     Total estimated model params size (MB)
    /usr/local/lib/python3.10/dist-packages/pytorch_lightning/loops/fit_loop.py:298: The number of training batches (1) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.


고맙게도 학습할 모델의 요약도 출력해준다. 파라미터 수는 앞서 torchinfo.summary에서도 보았듯이 23,399개(~23.4K)로 잘 나타나고 있다.

trainer 인스턴스 생성 당시, 콜백 목록에 EarlyStopping()을 추가했는데 이는 학습의 조기 중단 조건을 기입해준 것이다. epoch을 무한히 돌려봤자 모델 성능 개선이 없다면 학습을 중단시키는 것이 당연히 합리적일 것이다.

내가 정의한 EarlyStopping()의 조건은 'val_loss' log를 추적하면서, 이 val_loss minimum(mode='min')의 개선(loss decreasing)이 4 epochs(patience=4)동안 없으면, training을 멈추라는 것이다.

> 반대로 monitor metric을 monitor='val_acc'(정확도)로 보고 싶으면, mode를 max로 바꿔야 한다. 이는 val_acc의 maximum의 개선이 patience epochs 동안 발생하지 않는다면 조기중단하라는 의미다.
{: .prompt-tip }

EarlyStopping 말고도 콜백 목록으로 추가할 수 있는 기능들이 ModelCheckpoint, LearningRateMonitor, GPUStatsMonitor, TensorBoardLogger 등 있는데, 이들의 내용은 다른 포스트에서 더 자세히 다뤄보도록 하자.

아무튼 이제 학습된 모델의 성능을 확인해보자.


```python
trainer.current_epoch # 몇 epoch까지 돌다가 멈췄는지 trainer에게 물어봐서 확인할 수 있다.
```




    33




```python
# model = classificationGCN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
test_result = trainer.test(model, data_loader, verbose=False)
one_batch = next(iter(data_loader))
train_loss, train_acc = model.forward(one_batch, mode='train')
val_loss, val_acc = model.forward(one_batch, mode='val')
print(f"train_loss: {train_loss:.2f}, train_acc: {100.0*train_acc:.2f}%")
print(f"val_loss: {val_loss:.3f}, val_acc: {100.0*val_acc:4.2f}%")
print(f"test_acc: {100.0 * test_result[0]['test_acc']:4.2f}%")
```

    train_loss: 0.01, train_acc: 100.00%
    val_loss: 0.979, val_acc: 74.80%
    test_acc: 76.20%


# Conclusions

오늘은 Cora dataset이라는 논문인용관계 네트워크를 가지고, 각 노드(논문)들의 클래스를 분류하는 GCN 모델을 구현해보았다.

Graph Convolutional Netwok(GCN)과 같은 GNNs 아키텍처들은 PyTorch Geometric(일명 PyG)라는 라이브러리에 불러다 사용할 수 있었다.

오늘은 지금까지 수행했던 모델 구현방식과는 다르게 PyTorch Lightning(파이토치 라이트닝)이라는 high-level 딥러닝 프레임워크 도움을 얻어 구현해봤는데, 이를 통해 상당히 짜임새있는, 구조적인 코드 작성이 가능했다.

이렇게 구조화시켜 구현한 모델은 pl.Trainer()의 인스턴스로 학습 및 평가를 진행시키는데, Trainer의 콜백 목록(callbacks=[ ])에 다양한 유용한 기능들을 추가할 수 있었다. 비록 여기서 나는 EarlyStopping이라는 조기학습중단 기능만을 추가해 사용했는데, 다른 콜백 기능이나 이에 대한 더 자세한 내용은 다른 포스트에 따로 정리해 기록하도록 하겠다.

PyTorch Lightning 모듈화 과정 중 configure_optimizers() 메서드 정의에서 learning rate **scheduler** 기능(에포크 스텝마다 learning rate를 동적으로 조절)도 추가하려다가 또 그에 대한 설명으로 글이 길어질까봐 그만뒀다. 하지만, 이 기능 또한 모델 성능의 최적화를 전략적으로 달성하는 데에 효과적인 기능이다. 이에 대한 내용 또한 다른 포스트에서 따로 다루도록 하겠다.