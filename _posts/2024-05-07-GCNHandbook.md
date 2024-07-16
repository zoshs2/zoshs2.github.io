---
title: "[Hands-on] Graph Convolutional Network & Message Passing"
date: 2024-05-07 20:04:26 +0900
categories: [DL/ML, Graph Neural Networks]
tags: [Convolution, Graph Convolution Network, GCN, GNNs, Graph, Network, Neural, relu, graph-structured, Deep Learning, Machine Learning, DNN, MLP, Python, Kipf, adjacency matrix, identity matrix, degree matrix, degree, linear algebra, CNN, Convolutional Neural Network, PyTorch, Torch, PyTorch Geometric, PyG, Message, Message Passing, Filter, Kernel, localization, feature, node feature, propagation formula, GCNConv, GraphConv, GCN layer, Graph Neural Networks]
math: true
toc: false
---

# Intro.

이번 포스트에서는 지난 포스트[(Graph Neural Networks' Brief History)](https://zoshs2.github.io/posts/GNNsHistory/#in-2015-graph-convolutional-network:~:text=%EC%A7%81%EC%A0%91%EC%A0%81%EC%9D%B8%20%EB%8F%99%EB%A0%A5%EC%9D%84%20%EC%A0%9C%EA%B3%B5%ED%96%88%EB%8B%A4.-,In%202015%3A%20Graph%20Convolutional%20Network,-%EB%94%A5%EB%9F%AC%EB%8B%9D%20%EC%8B%9C%EB%8C%80%EC%9D%98%20%EC%8B%9C%EC%9E%91%EC%9D%84){:target="_blank"}에서 공부했던 GCN을 조금더 자세하고 명확하게, 그리고 PyTorch GCN 모듈이 아닌 수식에 의한 구현 및 적용까지 한 내용을 남기도록 하겠다.


# Table of Contents
- [CNN and GCN](#cnn-and-gcn)
- [GCN Principle](#gcn-principle)
- [GCN Implementation](#gcn-implementation)
- [Conclusions](#conclusions)

# CNN and GCN

GCN(Graph Convolutional Network)는 앞선 글에서 기술했듯이 아래와 같은 Propagation formula를 지닌 모델을 의미하고, Convolutional Neural Network(CNN)의 아이디어가 이 속에 전부 담겨 있다.

$$
\begin{equation}
  H^{\left(l+1\right)} = \sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{\left(l\right)}W^{\left(l\right)}\right)
\end{equation}
$$

CNN의 핵심 아이디어는 **Filter** 라는 학습 가중치 매개변수들이 담긴 특정 크기의 window(또는 patch라고도 표현)를 이미지 전체에 대해 Scanning하여, 이미지의 각 국소적 영역들의 정보를 함축해 특징이라고 여길 만한 feature들을 뽑아내는 과정이다. 그래서 이 과정을 통해 나온 출력 대상을 **feature map**이라고 말한다. 그리고 여기서 한 가지 더 중요한 포인트는 **Filter라는 학습 매개변수 세트가 이미지의 모든 부분에서 동일하게 쓰인다**는 점이다.

![png](/assets/img/post/gcn_implement/KernelOperation.png)*Kernel(Filter) Operation, Source: [Understanding the Convolutional Filter Operation in CNN’s](https://medium.com/advanced-deep-learning/cnn-operation-with-2-kernels-resulting-in-2-feature-mapsunderstanding-the-convolutional-filter-c4aad26cf32){:target="_blank"}*

그래프 구조에서 국소적 영역은 특정 노드와 그의 이웃노드들(Neighbor nodes)까지의 영역으로 생각해볼 수 있다. 그리고 1) 각 노드들은 자기 자신을 어떤 **node feature** vector로 표현하고 있고, 이후 2) 어떤 **동일한 가중치 학습 파라미터 세트**의 연산으로 이웃노드들과 서로의 node feature vector들을 공유하고, 3) 이렇게 모아진 정보들을 어떤 **특정한 방법으로 결합 및 함축**시킨다면, 이러한 과정은 CNN 아이디어와 핵심적인 부분에서 일치한다고 볼 수 있다.

> GCN: 그래프 내 모든 노드들에 대해서, 이웃노드들의 정보(CNN: 국소적 영역)를 한 군데로 모아 동일한 로직(CNN: Filter 공유)으로 새롭게 함축시킨다.
{: .prompt-tip }

이런 과정이 GCN 모델에 내포된 내용이다. 그리고 커다란 측면에서 노드 간에 서로의 정보를 전달하는 과정으로 **'Message Passing'** 방법이라고도 한다.

# GCN Principle

그럼 GCN 모델의 알고리즘을 그림과 함께 이해해보자.

먼저 그래프 내에 존재하는 각 노드들에 대해 자기만의 feature 벡터를 보유하고 있다. 이게 Graph Convolution 레이어의 입력값(input)으로 쓰이게 되고, 이웃노드들에게 전달할 **message의 재료**가 된다. (아래 그림 왼쪽)

두번째로, 각 노드들은 자신의 feature vector를 이웃노드들에게 전달한다. 이것이 **message passing**이 이뤄진 것이다. (아래 그림 오른쪽)

![png](/assets/img/post/gcn_implement/GCN_message_passing.png)*Graph Neural Netwoks. Source: [UvA DL Notebooks](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html){:target="_blank"}*

이제 이 과정을 GCN의 propagation formula로 이해해보자.

$$
\begin{equation}
  H^{\left(l+1\right)} = \sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{\left(l\right)}W^{\left(l\right)}\right)
\end{equation}
$$

$H^{(l)}$은 $N \times F$ 크기의 node들의 feature vector들이 담긴 행렬이다. 위 그림을 예시로 들자면, 총 4개의 노드들이 모두 3개의 요소를 지닌 feature vector들로 표현되어 있으므로, $H^{(l)}$ 행렬의 크기는 $4 \times 3$인 것이다.

> 실제 적용 문제에서는 해당 Node만의 말그대로 특성을 나타낼 수 있는 Clustering Coeff. 나 각종 centrality 값들을 모델 입력 데이터의 node feature로 사용한다고 한다.
{: .prompt-tip }

그리고 $W^(l)$은 학습 파라미터를 담은 행렬이고, $H^{(l)}W^{(l)}$ 연산으로 input (node) features들을 실제 이웃노드들에게 전달할 Message로 변환시키는 역할을 수행한다. (앞서 각 노드가 지닌 feature vector들은 message의 **재료**라고 표현한 이유가 여기에 있다.)

이제 그럼 $H^{(l)}W^{(l)}$를 통해 완성된 Message를 이웃노드들에게 전달할 것인데, 이 때 그래프(네트워크)의 연결구조를 알 수 있는 Adjacency matrix($A$)를 사용한다. 하지만, 일반적으로 (셀프 루프가 없는) Adjacency matrix $A$는 diagonal elements들이 모두 0이고, 이럴 경우 자기 자신(노드)의 message는 유지할 수 없기 때문에, Adjancency matrix $A$에 단위행렬 Identity matrix $I$를 더한 $\hat{A}=A+I$를 GCN propagation model에서 사용하는 것이다. 이제 마지막으로, 이렇게 각 노드에 모인 message들(위 그림의 오른쪽 상태)을 어떻게 결합시켜서 활용할지에 대한 질문이 남았다. 단순히 **더해서** 활용해도 좋고, 이들의 **평균**만 활용해도 무관하다. 위 식에서 $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$는 **message들의 평균**을 활용하라는 의미와 같다. $D$는 diagonal elements만 값이 존재하는 diagonal matrix고, $D_{ii}$ 값에는 $i$번째 노드가 지닌 이웃노드의 수인  차수(degree)가 쓰인다. 그리고 마지막으로 $\sigma\left(-\right)$는 임의의 활성화 함수로서 수식처럼 꼭 sigmoid($\sigma$) 함수를 사용할 필요는 없고, 보통 $\text{ReLU}$-based activation function을 활용하는 것이 일반적이다.

이렇게 주변 노드들로부터 그들의 message들을 passing 받고, message 정보들을 평균(혹은 sum)낸 후 활성화 함수에 이르는 일련의 과정이 하나의(single) GCN layer에서 벌어지는 일들이고, 이렇게 나온 output이 다음 GCN layer의 새로운 message 재료이자 각 노드들의 새로운 feature vector들이 된다.

# GCN Implementation

이제 간단한 그래프 구조 데이터를 토대로 GCN layer의 연산 과정을 다시 답습해보자. 아래 내가 예시로 들 그래프 데이터를 그려 보았다.

![png](/assets/img/post/gcn_implement/simple_graph.png)*My Simple Graph Example*

4개의 노드로 구성된 그래프이고, 모든 각 노드는 2개씩 node feature (feature vector)를 지니고 있다.

> 앞선 Message Passing 그림 예시에서는, 각 노드가 3개의 node feature를 지녔다고 볼 수 있다.
{: .prompt-tip }

4개의 노드들이 지닌 각 node feature들(총 8개)은 단순하게 0부터 7까지 부여하도록 하고, 위 그래프 구조의 Adjacency-identity matrix 는 아래와 같이 표현할 수 있다.


```python
import torch

node_features = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
adj_matrix = torch.Tensor([[[1, 1, 0, 0],
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1]]])
print("Node features: \n", node_features, end='\n\n')
print("Adjacency-Identity Matrix: \n", adj_matrix)
```

    Node features: 
     tensor([[[0., 1.],
             [2., 3.],
             [4., 5.],
             [6., 7.]]])
    
    Adjacency-Identity Matrix: 
     tensor([[[1., 1., 0., 0.],
             [1., 1., 1., 1.],
             [0., 1., 1., 1.],
             [0., 1., 1., 1.]]])


두 행렬은 모두 3차원 텐서로 표현했는데, 이는 일반적으로 딥러닝 모델을 학습시킬 떄, Mini-batch 방식을 사용하기에 다소 일반적인 표현을 하고 싶어서 이렇게 표현했다.

이 상황을 시각적으로 떠올려 보면, 아래 그림처럼 각 배치의 adjacency matrix가 천장을 보고 누워 있는 3차원 텐서 행렬의 구조다.

![png](/assets/img/post/gcn_implement/three_dimensional_adjacency.png)*Schematic of adjacency matrix variable with three dimensional tensor*

이제 앞서 정의한 node features와 그래프 구조 데이터를 GCN layer에 forwarding 시키기 위해, GCN layer 클래스를 정의하자.


```python
import torch.nn as nn

class GCNLayer(nn.Module):
  def __init__(self,in_features, out_features):
    super().__init__()
    self.msg_projection = nn.Linear(in_features, out_features)

  def forward(self, node_feats, adj_matrix):
    num_neighbours = adj_matrix.sum(dim=2, keepdims=True)
    messages = self.msg_projection(node_feats)
    received_msg = torch.bmm(adj_matrix, messages) # batch matrix-matrix(bmm) product
    new_node_feats = nn.functional.relu(received_msg / num_neighbours)
    return new_node_feats

```

여기 GCNLayer에서 msg_projection 모듈은 앞서 본 GCN propagation 수식에서 $H^{\left(l\right)}W^{\left(l\right)}$ 연산에 대한 내용이다. 이는 단순 행렬곱으로서 nn.Linear()로 기술된다. 이 때, 몇 개의 node features들이 들어오는지(in_features)와 $W$의 column dimension 크기(out_features)가 쓰인다.

이후 이렇게 계산된 message 정보들, 즉 $H^{\left(l\right)}W^{\left(l\right)}$는 그래프의 연결 구조를 반영하는 Adjacency matrix $A$와의 행렬곱 연산(torch.bmm: batch matrix-matrix product)을 통해 이웃노드들로 전파된다. - 여기까지 한게 received_msg 라는 결과.

마지막으로, 자기 자신의 message와 이웃노드들로부터 받은 message들을 종합해서 **"평균"**을 내고, activation function ReLU를 통과시켜 해당 노드의 새로운 node feature matrix가 탄생한다. 즉, new_node_feats는 수식의 $H^{\left(l+1\right)}$에 해당한다.

> 다시 강조하지만, 종합한 messages들의 평균에 대한 건 선택의 문제다. 종합한 messages들의 mean/sum/min/max/mul 같이 다양한 집계 방식들이 존재할 수 있고 사용할 수 있다. torch_geometric 라이브러리에서 제공하는 GCNConv 모델의 클래스는 [MessagePassing](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html#torch-geometric-nn-conv-messagepassing){:target="_blank"}이라는 클래스를 상속받고 있는데(대부분의 GNNs 모델들의 base가 이 MessagePassing 클래스다), *aggr* 라는 argument로 이 집계 방식의 선택이 가능하다. Default는 add다.
{: .prompt-info }

* * *

이제 이렇게 만든 모델에 처음 정의한 adj_matrix와 node_features 데이터를 흘려 보내보자 - forwarding.

이 때, nn.Linear()에 활용하는 가중치, 즉 propagation formula 속 $W^{\left(l\right)}$는 단위 행렬(identity matrix)를 활용하도록 강제할 것이다. 이는 다른 말로, input node features 자체를 그냥 message로 활용하겠다는 의미다.

정리하자면, 앞서 예시로 든 그래프 상태로 출발해 GCN layer 연산을 수행하면서 아래 그림과 같은 상황이 이뤄지는 것이다.

![png](/assets/img/post/gcn_implement/gcn_receiving_msgs.png)*Receiving messages from neigbor nodes*

![png](/assets/img/post/gcn_implement/gcn_new_features_output.png)*Aggregate messages by averaging & ReLU, which its output is new node features*

정말 그렇게 될 지 확인해보자.


```python
single_gcn = GCNLayer(in_features=2, out_features=2)

# nn.Linear() 모듈의 weight와 bias를 직접 변경 및 설정
single_gcn.msg_projection.weight.data = torch.Tensor([[1., 0.],
                                                      [0., 1.]])
single_gcn.msg_projection.bias.data = torch.Tensor([0., 0.]) # feature 수 만큼

# Autograd engine off
with torch.no_grad():
  new_node_feats = single_gcn(node_features, adj_matrix)

print("Initial node features: \n", node_features, end='\n\n')
print("Node features after GCN layer: \n", new_node_feats)
```

    Initial node features: 
     tensor([[[0., 1.],
             [2., 3.],
             [4., 5.],
             [6., 7.]]])
    
    Node features after GCN layer: 
     tensor([[[1., 2.],
             [3., 4.],
             [4., 5.],
             [4., 5.]]])


# Conclusions

이렇게 Kipf의 Graph Neural Network 모델이 내포하고 있는 의미를 Message Passing 측면에서 이해해 보았고, torch의 [GCNConv 모듈](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch-geometric-nn-conv-gcnconv){:target="_blank"}을 사용하지 않고, 직접 GCN layer를 구현해보았다. (사실 그냥 GCN propagation formula인 $DADHW$ 행렬곱을 한 거라 특별한 것은 없다.)

하지만 이렇게 하나의 GCN layer 만을 사용하면, 위 message 교환 과정에서 봤다시피, 각 노드의 이웃노드들의 정보밖에 활용하지 못한다. 그래프 신경망(GNNs) 모델의 이상적인 학습 방식은 그래프(네트워크) 전체 정보를 활용하고-포괄하여 그래프 그 자체에 대한 고차원적 표현이 가능하도록 하는 데에 있다. 그렇기 때문에, GCN layer를 하나만 사용하는 것이 아니라, GCN layer를 여러겹 겹쳐서 각 노드가 가진 message들이 (함축되어) 그래프 전체에 퍼져나가도록 해야 한다.

마지막으로, Kipf의 전통적 GCN model에는 한 가지 문제가 존재한다는 걸 짧게 언급하고 마무리하려고 한다.

$$
\begin{equation}
  H^{\left(l+1\right)} = \sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{\left(l\right)}W^{\left(l\right)}\right)
\end{equation}
$$

위 식과 같은 로직대로 forward했을 때, 앞선 예시에서 본 output 결과에서 3번과 4번 노드의 새로운 node features vector 가 서로 동일하게 변한 것을 보았다.

3번과 4번 노드를 상징하던 feature vector는 각각 [4, 5], [6, 7]이었던 것을 상기해 볼 때, GCN layer를 거치고 나오니 두 노드가 이제는 완전히 동일한 feature vector로 표현되고 있다는 상황이 다소 마음에 들지 않는다. 이는 ***GCN layers can make the network forget node-specific information**, 즉 원래 각 노드가 지니고 있던 특색(information)들이 GCN layer를 지나고 모두 사라진다는 의미이다.

이는 두 노드가 '동일한 이웃노드 리스트를 보유하고 있다는 점'과, 더 본질적으론 **'수신된 Message들을 평균내는 집계 방식을 사용한다는 점'**에서 비롯된다.

> 앞서 살짝 언급했지만, [torch_geometric.nn.conv.GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch-geometric-nn-conv-gcnconv){:target="_blank"} 모델 클래스는 MessagePassing 클래스를 상속받고 있고, 이 MessagePassing 클래스의 *aggr*라는 message 집계방식에 대한 default는 "mean"이 아니라 "add"다. 아마도 이런 node-specific info.를 유지해야 하는 이슈가 있기 때문에 add가 디폴트 아닐까?란 생각이 든다 (나혼자만의 생각...).
{: .prompt-tip }

이렇게 GCN 모델 전파 과정에서 각 노드의 특색이 완전히 사라지지 않고, 어느 정도 유지되도록 하는 트릭(?)들이 제시되어 왔고 DL/ML 프레임워크 내 모듈들에서도 구현이 되어 있다. 예컨대,

conv.GCNConv()의 파라미터 중 improved라는 옵션(default=False)이 있다. 이 옵션을 True로 하면, Adjacecny-identity matrix를 $\hat{A}=A+2I$로 활용한다. 이를 통해 Message(들)을 수집하는 과정에서 자신의 message 비중을 더 높여 계산하게 된다. 쉽게 말하자면, 내 message는 가중치를 더 주겠다는 것.

또는 Message 변환 과정에서부터, 즉 $H^{\left(l\right)}W^{\left(l\right)}$를 계산하는 부분을 **노드 자신에 대한 부분**과 **이웃 노드들에 대한 부분**으로 아예 따로 나눠서 계산시키는 방법도 있다. 이 아이디어를 내포한 모델이 [torch_geometric.nn.conv.GraphConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GraphConv.html#torch-geometric-nn-conv-graphconv){:target="_blank"}이고, 이에 대한 새로운 GCN propagation formula는 아래와 같다.

$$
\begin{equation}
  x^{\left(l+1\right)}_i=W^{\left(l\right)}_{1}x^{\left(l\right)}_{i} + W^{\left(l\right)}_{2}\sum_{j\in N(i)}{e_{j,i}}\cdot x^{(l)}_j
\end{equation}
$$

다시 구현의 측면에서 쉽게 보자면, nn.Linear()를 2개 사용한다고 보면 된다. 실제 내부 소스코드를 보면 root에 대한 nn.Linear(), 이웃노드들에 대한 부분의 nn.Linear() 두 개가 따로 연산되어 최종적으로 sum 하는 것을 볼 수 있다. - 그렇다. 이 GraphConv 모델에서는 messages 집계 방식이 기본적으로 평균이 아닌 sum인 것이다.

다음 시간에는 Node classification/ Link prediction/ Graph classification 같은 문제들을 푸는 GCN 모델을 구현해보는 시간을 갖자.
