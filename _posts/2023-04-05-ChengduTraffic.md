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