---
title: "중국 청두시 도로망 및 속도 데이터 분석"
date: 2023-04-05 17:32:23 +0900
categories: [Open-Data, EDA]
tags: [python, network, traffic, visualization, eda, china]
---

# 들어가며
'중국 청두시 도로망 통행속도 데이터(이하 **Megacity 데이터**)'를 살펴본 내용이다. 본 데이터를 알게 된 계기는 다음 논문을 읽게 되면서이다. - < Urban link travel speed dataset from a megacity road network > - Guo, Feng, et al., Scientific Data (May, 2019) - 공개 배포된 형태로 데이터셋을 전처리 및 가공하기까지 어떤 수순을 거쳤는지 논문 본문에 잘 설명이 되어 있다.

# Road Network & Travel Speed Dataset in Chengdu, China (aka, Megacity dataset)

```python
import os
import sys
import wget
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import networkx as nx
import networkit as nk
import matplotlib.colors as mcolors
import matplotlib as mpl
import pygraphviz as pgv
import cv2
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
```


## Data Acquisition
-  **Figshare**: 그림, 데이터셋, 이미지 및 비디오 등을 포함한 연구 결과를 보존하고 공유하는 온라인 오픈 액세스 레포지토리
-  터미널 상에선, 아래 command line을 통해 데이터셋을 다운받을 수 있다.
```bash
>>> wget https://figshare.com/ndownloader/articles/7140209/versions/4  # 4(.zip) file download
>>> unzip 4 -d <dir_name_after_unzip>
```

- 2015.06.01 ~ 2015.07.15 사이 총 45일간의 중국 청두시의 도로망 통행 속도 데이터셋이다.
- 해당 기간동안, '5개의 대표 시간대(5 representative time horizons)'만을 다루고 있다.
  - 03:00 ~ 05:00: Dawn
  - 08:00 ~ 10:00: Morning
  - 12:00 ~ 14:00: Noon
  - 17:00 ~ 19:00: Evening
  - 21:00 ~ 23:00: Night
<br><br>
- 데이터셋은 2분 단위의 기록이다. 결과적으로 하루마다 총 300개의 시간 기록(논문에선 '300 time periods'라 표현)이 존재한다.
- 각 날마다, 데이터셋은 2개의 csv 파일(_[0], _[1])로 분할하여 배포되고 있다.
  - _[0].csv: 1 ~ 150 time periods
  - _[1].csv: 151 ~ 300 time periods
<br><br>
- 본 도로망 통행 속도 데이터는, 해당 기간 동안, 택시 12,000대 이상으로부터 나온 30억개 이상의 GPS 궤적 데이터로부터 집계 및 추산한 결과이다.
- 따라서, 위 과정에서 < map matching >, < link travel speed estimation >, < data imputation > 에 대한 데이터 정제 및 전처리는 이미 되어 있다.

- Megacity Dataset
  - link.csv: 도로 네트워크 데이터
  - speed[date]_[i].csv: 도로망 내 통행속도 데이터

```python
dataset/
├── [430K]  link.csv
├── [ 23M]  speed[601]_[0].csv
├── [ 22M]  speed[601]_[1].csv
├── [ 22M]  speed[602]_[0].csv
├── ...
├── [ 23M]  speed[714]_[1].csv
├── [ 23M]  speed[715]_[0].csv
└── [ 22M]  speed[715]_[1].csv

2.0G used in 0 directories, 91 files
```

<br>

## Road Network
"link.csv"
- Attributee:
  - **Link**: 도로링크 아이디
  - **Node_Start**: 시작노드
  - **Longitude_Start, Latitude_Start**: 시작노드 경위도
  - **Node_End**: 종료노드
  - **Longitude_End, Latitude_End**: 종료노드 경위도
  - **Length**: 도로링크 길이(m)


```python
DataPath = '/home/ygkwon/megacity/dataset'
DataContents = os.listdir(DataPath)
print(DataContents[0])
```

    link.csv



```python
road_network = pd.read_csv(os.path.join(DataPath, DataContents[0]))
road_network.head(5)
```

<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Link</th>
      <th>Node_Start</th>
      <th>Longitude_Start</th>
      <th>Latitude_Start</th>
      <th>Node_End</th>
      <th>Longitude_End</th>
      <th>Latitude_End</th>
      <th>Length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>103.946006</td>
      <td>30.750660</td>
      <td>48</td>
      <td>103.956494</td>
      <td>30.745080</td>
      <td>1179.207157</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>103.946006</td>
      <td>30.750660</td>
      <td>64</td>
      <td>103.941276</td>
      <td>30.754493</td>
      <td>620.905375</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>103.946006</td>
      <td>30.750660</td>
      <td>16</td>
      <td>103.952551</td>
      <td>30.756752</td>
      <td>921.041014</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>104.062539</td>
      <td>30.739077</td>
      <td>1288</td>
      <td>104.062071</td>
      <td>30.732501</td>
      <td>730.287581</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>104.062539</td>
      <td>30.739077</td>
      <td>311</td>
      <td>104.060024</td>
      <td>30.742693</td>
      <td>467.552294</td>
    </tr>
  </tbody>
</table>
</div>
<br>

### Geographical visualization

```python
NodeList = pd.concat([road_network['Node_Start'], road_network['Node_End']], ignore_index=True).unique()
fig, ax = plt.subplots(facecolor='w', figsize=(8,8))
for row in tqdm(road_network.iloc):
    link_loc = row[['Longitude_Start', 'Longitude_End', 'Latitude_Start', 'Latitude_End']].values
    x, y = link_loc[:2], link_loc[2:]
    ax.plot(x, y, marker='o', markerfacecolor='w', ms=0.7, color='black', linewidth=.4)

title = f"< Road network in Chengdu, China >\n# of Nodes: {NodeList.shape[0]:,} / # of Links: {road_network.shape[0]:,}"
ax.set_title(title, fontsize=16)
ax.axis('off')
plt.show()
```

    5943it [00:05, 1048.30it/s]



<br>    
![png](/assets/img/post/megacity/megacity_7_1.png)
<br>

### Network visualization

```python
# Make a dictionary containing real-world coordinates of nodes
node_pos_dict = defaultdict(list)
for node in tqdm(road_network['Node_Start'].unique()):
    lon, lat = road_network[road_network.Node_Start==node][['Longitude_Start', 'Latitude_Start']].values[0]
    if len(node_pos_dict[node]) == 0:
        node_pos_dict[node] = [lon, lat]

for node in tqdm(road_network['Node_End'].unique()):
    lon, lat = road_network[road_network.Node_Start==node][['Longitude_End', 'Latitude_End']].values[0]
    if len(node_pos_dict[node]) == 0:
        node_pos_dict[node] = [lon, lat]

print(f"The total number of nodes: {len(node_pos_dict.keys()):,}")
```

    100%|██████████| 1902/1902 [00:01<00:00, 1559.17it/s]
    100%|██████████| 1902/1902 [00:01<00:00, 1583.15it/s]

    The total number of nodes: 1,902

```python
megaNetG = nx.MultiDiGraph()
megaNetG.add_edges_from(zip(road_network.Node_Start, road_network.Node_End))

# Create a graph object for Graphviz
agraph = nx.nx_agraph.to_agraph(megaNetG)

# For Graphviz, set the node positions, node_size, label as none.
# If wanting to adjust the size of node, 'fixedsize' attribute of node must be 'True' (** important !!)
# Note that setting the values of width / height / arrowsize / Graphviz scale is under some trial-error.
for node in agraph.nodes():
    node.attr['pos'] = '{},{}'.format(node_pos_dict[int(node)][0], node_pos_dict[int(node)][1])
    node.attr['fixedsize'] = True
    node.attr['width'] = '0.07'
    node.attr['height'] = '0.07'
    node.attr['label'] = ''

for edge in agraph.edges():
    edge.attr['arrowsize'] = '0.3'
    edge.attr['penwidth'] = '2.5'

# Draw the graph using Graphviz
agraph.graph_attr.update(scale='15000')

# forcing to my_position of nodes, rather than 'neato' layout.
agraph.draw('megacity_road_network.png', prog='neato', args='-n')

# Show the plot
fig, ax = plt.subplots(facecolor='w', figsize=(10,10))
img = plt.imread('megacity_road_network.png')
ax.imshow(img)
ax.axis('off')
plt.show()
```

<br>
![png](/assets/img/post/megacity/megacity_10_0.png)
<br>

## Link Travel Speed
"speed[date]_[i].csv"
- Attributee:
  - **Period**: 통행 속도 시점대
  - **Link**: 도로링크
  - **Speed**: 해당 도로링크의 통행 속도


```python
print(DataContents[1])
```

    speed[601]_[0].csv



```python
origin_mega = pd.read_csv(os.path.join(DataPath, DataContents[1]))

# 150개의 시점(time periods) * 5943개의 도로링크 = 891,450개의 rows
print(origin_mega)
```

                 Period  Link      Speed
    0       03:00-03:02     1  31.769048
    1       03:00-03:02     2  49.450000
    2       03:00-03:02     3  42.780000
    3       03:00-03:02     4  35.150000
    4       03:00-03:02     5  25.950000
    ...             ...   ...        ...
    891445  12:58-13:00  5939  22.030769
    891446  12:58-13:00  5940  25.875000
    891447  12:58-13:00  5941  20.572917
    891448  12:58-13:00  5942  43.980000
    891449  12:58-13:00  5943  16.480000
    
    [891450 rows x 3 columns]


### Personal renewal idea
데이터셋 하나로 전부 다루고 싶어서 개인적으로 해보는 전처리 (다시 말해서, 따라할 필요없음)
1) 날짜 컬럼 추가 (dtype: int64). 
   - ex) 1 June -> 601
2) 요일 컬럼 추가 (dtype: str). 
   - ex) 1 June -> 'Mon'
3) 기존의 'Period' 컬럼 이름을 'time'으로 변경, 그리고 시작 시점을 기준으로 단순화 (dtype: int64). 
   - ex) 03:00 ~ 03:02 >> 300
4) time 컬럼 기반, 시간대 컬럼 추가 (dtype: str). 
   - ex) 300 ~ 458 time: 'Dawn', 800 ~ 958: 'Morn', 1200 ~ 1358: 'Noon', 1700 ~ 1858: 'Even', 2100 ~ 2258: 'Night'
5) 하루 기준, 시간 순서 컬럼 추가 (dtype: int64). (논문 내 'time periods'에 해당하는 값)
   - ex) 'time' 컬럼의 {300, 302, 304, ..., 2354, 2356, 2358} 에 대응하는 {1,2,3, ..., 298, 299, 300}
6) 45일 간의 모든 데이터 하나의 .pkl로 저장 (~ 3.3 GB)

* Checkpoints during this renewal
  - Checkpoint_1: check if the size of each dataset is 891,450.
  - Checkpoint_2: check if the number of road links is 5,943.
  - Checkpoint_3: check if each of 5,943 road links has 150 time moments.


```python
assign_horizon = lambda t: 'Dawn' if 300 <= t < 500 else \
         'Morn' if 800 <= t < 1000 else \
         'Noon' if 1200 <= t < 1400 else \
         'Even' if 1700 <= t < 1900 else \
         'Night' if 2100 <= t < 2300 else \
         'wtf'
         
time_seq = [[],[]]
for i in range(2):
    for seq in range(1+(i*150), 151+(i*150)):
        for _ in range(5943):
            time_seq[i].append(seq)

mega_dataset = pd.DataFrame()
for filename in tqdm(DataContents[1:]):
    ZeroOne = int(filename.split(']')[1].split('[')[1])
    mon_date = [int(filename.split(']')[0].split('[')[1])]
    ymd = f"2015{mon_date[0]:04d}"
    day = [calendar.day_name[datetime.strptime(ymd, '%Y%m%d').weekday()][:3]]
    one_mega = pd.read_csv(os.path.join(DataPath, filename))
    if (one_mega.shape[0]!=891450) | (one_mega.groupby('Link').ngroups!=5943): # Checkpoint_1 + Checkpoint_2
        error_msg = f"Strange dataset detected: {filename}"
        print(error_msg)
        print(''.ljust(len(error_msg), '-'))
        continue

    one_mega['mon_date'] = pd.Series(mon_date*891450)
    one_mega['day'] = pd.Series(day*891450)
    one_mega['time'] = one_mega['Period'].apply(lambda x: int(''.join(x.split('-')[0].split(':'))))
    one_mega['Period'] = one_mega['time'].apply(assign_horizon)
    if not np.array_equal(one_mega.value_counts('time').values, [5943]*150): # Checkpoint_3
        error_msg = f"Not all links have 150 time_moments: {filename}"
        print(error_msg)
        print(''.ljust(len(error_msg), '-'))
        continue

    one_mega = one_mega.sort_values(by='time').reset_index(drop=True)
    one_mega['time_seq'] = pd.Series(time_seq[ZeroOne])
    mega_dataset = pd.concat([mega_dataset, one_mega], ignore_index=True)

mega_dataset.to_pickle("ygkwon_megacity_dataset_20150601_to_0715.pkl")
```