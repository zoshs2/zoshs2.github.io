---
title: "Car traffic and parking density maps from Uber Movement travel times"
date: 2023-08-13 22:04:50 +0900
categories: [Open-Data, EDA]
tags: [python, traffic, parking, uber, uber movement, visualization, eda, LA, paris, london, melbourne, mumbai, sydney, gif, travel time]
math: true
---

# 들어가며
오늘 살펴볼 내용은 [Uber Movement Team](https://movement.uber.com/?lang=en-US){:target="_blank"}에서 공개 오픈 중인 (Traffic) <u>Travel Time 데이터셋들을 기반으로</u> 산출한 전 세계 **34개 도시**의 주차 밀집도 및 운행 활성도(?) 데이터를 들여다 볼 것이다. 
<br>

여담으로, Uber에선 공익 차원의 도시교통 연구 증진 목적으로 자사 측 기기의 운행 기록들을 <u>가공</u>하여 양질의 데이터셋을 정기적으로 공개 배포하고 있다. 가공 방식에 대해서도 Uber Movement Team에서 공유하고 있으니 궁금하다면 참조하길 바란다. (아래 References 참고)

![png](/assets/img/post/parking_density/uber_movement.png)
*Introduction by Uber Movement; Credit: https://movement.uber.com*
<br><br>

오늘 중점적으로 살펴볼 데이터는, [Aryandoust, A., van Vliet, O., & Patt, A. (2019)](https://www.nature.com/articles/s41597-019-0159-6){:target="_blank"}에서 Uber 데이터에 그들의 모델을 적용하여 새로 산출한 **Parking Density** 와 **Traffic Activity** 데이터이긴 하지만, 서두에서 그들이 이용한 Uber Travel Time 데이터셋이 어떻게 생겨먹은 것인지 (아주아주) 살짝 들여다 보기로 하자.
* * *
**References**
1. [Uber Movement: Travel Times Calculation Methodology](https://movement.uber.com/_static/c9bce307d99643c3.pdf){:target="_blank"}
2. [Uber Movement: Speeds Calculation Methodology](https://movement.uber.com/_static/97e6e916ed8e8176.pdf){:target="_blank"}

<br>

# Uber Movement dataset
현재 [Uber Movement 홈페이지](https://movement.uber.com/?lang=en-US){:target="_blank"}에서 공개 배포 중인 데이터는 아래 3가지 종류이다.
1. Travel Time: zone-to-zone average travel time across a city
2. Speed: street speed across a city
3. Mobility Heatmap: the volume of activity of mobility devices(Uber share, bikes, Scooters)
* * *
- 모두 도시별로 나눠져 있으며, 런던/맨체스터/마드리드/LA/워싱턴DC/암스테르담/보스턴/브뤼셀(벨기에)/카이로(이집트) 등등 굉장히 많은 글로벌 도시에 대한 데이터가 존재한다.
- 대체로 분기 단위로 집계하여 배포하는데, 어째서인지 모든 도시의 집계가 2020년 1분기 까지만 공개되어 있다.
- 본문의 Raw 데이터로 활용되는 <u>Travel Time 데이터</u>셋만 한번 들여다 보기로 하자.

<br>

## Travel Time for London, UK
- London의 경우, 2016-1분기 ~ 2020-1분기 까지 (17개의 분기) 데이터가 존재한다. ***아마 도시마다 이용 가능한 서비스 범위가 다를 것이다.***
- 각 분기 데이터마다, 집계하여 배포하는 데이터는 7가지 유형으로 나뉜다. (* Travel Time 데이터 기준)
    - (1) by Hour of Day (All Days): {city}-{scale}-{year}-{quarter}-All-HourlyAggregate.csv
    - (2) by Hour of Day (Weekdays Only): {city}-{scale}-{year}-{quarter}-OnlyWeekdays-HourlyAggregate.csv
    - (3) by Hour of Day (Weekends Only): {city}-{scale}-{year}-{quarter}-OnlyWeekends-HourlyAggregate.csv
    - (4) by Month (All Days): {city}-{scale}-{year}-{quarter}-All-MonthlyAggregate.csv
    - (5) by Month (Weekdays Only): {city}-{scale}-{year}-{quarter}-OnlyWeekdays-MonthlyAggregate.csv
    - (6) by Month (Weekends Only): {city}-{scale}-{year}-{quarter}-OnlyWeekends-MonthlyAggregate.csv
    - (7) by Day of Week: {city}-{scale}-{year}-{quarter}-WeeklyAggregate.csv
<br><br>

- 이들을 다시 크게 보면 3가지 유형인데,
    - Hour of Day: 시간 단위로 응용집계된 데이터; **HourlyAggregate.csv**
    - Month: 월 단위로 응용집계된 데이터; **MonthlyAggregate.csv**
    - Day of Week: 요일 단위로 응용집계된 데이터; **WeeklyAggregate.csv**
<br>

그리고, 요일 단위 집계(Day of Week)를 제외하고는, 집계를 할 때 평일만 모아서 집계했는지 / 주말만 모아서 집계했는지 / 평일주말 모두 합쳐서 집계했는지에 따라 각각 'OnlyWeekdays', 'OnlyWeekends', 'All' 이름이 붙어있다. <u>데이터 내부의 Data Structure에는 차이가 없으며</u>, 데이터 값이 어떤 집계 기준으로 산출된 것인지 다를 뿐이다. (찍먹 수준도 안되는) 여기 서두에선 (2), (5), (7) 유형의 데이터들만 한번 들여다 본다.
<br>

> **_NOTE:_** 가만보면 일별 단위는 없는 것 같은데, 2020년 1분기(3개월)에만 일별(hourly resolution) 데이터가 함께 있다. (* 데이터 꽤 큼. ~ 1.5 GB)


```python
import json, os
import sys
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
import dask.dataframe as dd
import zipfile
import shutil
import contextily as cx
from tqdm import tqdm
import imageio.v3 as iio
```


```python
>>> print(sys.version)

'3.10.10 | packaged by conda-forge | (main, Mar 24 2023, 20:08:06) [GCC 11.3.0]'
```



```python
DataPath = os.path.join(os.getcwd(), 'uber_dataset/')
DataContents = [file for file in os.listdir(DataPath)]
```

```python
>>> print(DataContents)

['london-lsoa-2020-1-OnlyWeekdays-HourlyAggregate.csv', 'london-lsoa-2018-2-OnlyWeekdays-MonthlyAggregate.csv', 'london-lsoa-2018-2-WeeklyAggregate.csv']
```

기본적으로 Travel Times 데이터들은 모두, 유형에 관계없이, OD(Origin to Destination)에 관한 컬럼정보로 구성되어 있다. 시작노드지역(sourceid)에서 종료노드지역(dstid) 방향으로 운행할 때의 평균 통행시간이 기록되어 있다. 노드 ID는 Uber Movement 팀에서 자체 넘버링한 Sequential ID(0부터 N까지)를 쓰는 듯 하다. ID에 대한 지리정보 데이터는 Uber Movement 홈페이지에서 내려받을 수 있다. (아래 [GeoJson Files](#regional-id-and-geojson-files) 내용 참고)
* * *
Origin(sourceid)과 Destination(dstid) 컬럼 이후의 또 다른 공통 컬럼들로는 **mean_travel_time/standard_deviation_travel_time**과 **geometric_mean_travel_time/geometric_standard_deviation_travel_time**이 있다. <br><br>
**mean_travel_time**은, 집계된 데이터 분포 상, 일반적인 산술평균(arithmetic mean)을 사용해 얻은 평균 통행시간값이고, <br>
**geometric_mean_travel_time**은 기하평균(geometric mean) 계산을 통해 얻은 평균 통행시간값이다. <br>
> **_NOTE:_** 오늘 살펴볼 데이터의 연구팀은 mean_travel_time 산술평균 통행시간을 기준으로 모델을 적용하여 parking density & traffic activity 데이터를 추출했다고 한다. 
<br>

* * *
Hourly / Monthly / Weekly Aggregation마다 컬럼 내용이 다른 한 가지가 있는데, 그 해석은 다음과 같다. 
* HourlyAggregate의 **'hod'** 컬럼: hour of a day의 약자; [0, 1, 2, ..., 23]까지 24개의 integer값이 들어있고, 시간을 의미한다.
* MonthlyAggregate의 **'month'** 컬럼: 월간 집계범위를 의미하는 integer 값이 들어있고, 분기에 따라 한 데이터 파일엔 [1,2,3] or [4,5,6] or [7,8,9] or [10,11,12] 값이 들어있다.
* WeeklyAggregate의 **'dow'** 컬럼: day of a week의 약자; [1, 2, 3, ..., 7]까지 7개의 interger값이 들어있고, 각각 월요일부터 일요일까지에 해당한다.


```python
# HourlyAggregate.csv; (Hour of Day: 시간 단위로 집계된 데이터)
HourlyAgg = pd.read_csv(os.path.join(DataPath, DataContents[0]))
HourlyAgg
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sourceid</th>
      <th>dstid</th>
      <th>hod</th>
      <th>mean_travel_time</th>
      <th>standard_deviation_travel_time</th>
      <th>geometric_mean_travel_time</th>
      <th>geometric_standard_deviation_travel_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>491</td>
      <td>497</td>
      <td>0</td>
      <td>1861.41</td>
      <td>537.96</td>
      <td>1807.55</td>
      <td>1.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>489</td>
      <td>517</td>
      <td>0</td>
      <td>1241.77</td>
      <td>406.64</td>
      <td>1176.96</td>
      <td>1.41</td>
    </tr>
    <tr>
      <th>2</th>
      <td>639</td>
      <td>875</td>
      <td>12</td>
      <td>2594.00</td>
      <td>892.39</td>
      <td>2482.40</td>
      <td>1.32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>487</td>
      <td>537</td>
      <td>0</td>
      <td>537.26</td>
      <td>188.19</td>
      <td>511.69</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>4</th>
      <td>475</td>
      <td>657</td>
      <td>0</td>
      <td>291.68</td>
      <td>206.60</td>
      <td>213.75</td>
      <td>2.37</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6329297</th>
      <td>489</td>
      <td>191</td>
      <td>13</td>
      <td>1508.00</td>
      <td>434.21</td>
      <td>1456.69</td>
      <td>1.29</td>
    </tr>
    <tr>
      <th>6329298</th>
      <td>441</td>
      <td>672</td>
      <td>9</td>
      <td>822.24</td>
      <td>270.83</td>
      <td>793.53</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>6329299</th>
      <td>56</td>
      <td>952</td>
      <td>22</td>
      <td>1156.18</td>
      <td>222.10</td>
      <td>1137.27</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>6329300</th>
      <td>869</td>
      <td>753</td>
      <td>19</td>
      <td>1815.90</td>
      <td>442.65</td>
      <td>1772.09</td>
      <td>1.24</td>
    </tr>
    <tr>
      <th>6329301</th>
      <td>121</td>
      <td>703</td>
      <td>11</td>
      <td>1559.75</td>
      <td>173.37</td>
      <td>1550.01</td>
      <td>1.12</td>
    </tr>
  </tbody>
</table>
<p>6329302 rows × 7 columns</p>
</div>
<br>

```python
# MonthlyAggregate.csv; (Month: 월 단위로 집계된 데이터)
MonthlyAgg = pd.read_csv(os.path.join(DataPath, DataContents[1]))
MonthlyAgg
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sourceid</th>
      <th>dstid</th>
      <th>month</th>
      <th>mean_travel_time</th>
      <th>standard_deviation_travel_time</th>
      <th>geometric_mean_travel_time</th>
      <th>geometric_standard_deviation_travel_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>245</td>
      <td>434</td>
      <td>4</td>
      <td>2365.92</td>
      <td>607.61</td>
      <td>2293.37</td>
      <td>1.28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>140</td>
      <td>433</td>
      <td>4</td>
      <td>1599.58</td>
      <td>426.72</td>
      <td>1549.97</td>
      <td>1.28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>92</td>
      <td>388</td>
      <td>5</td>
      <td>645.00</td>
      <td>226.34</td>
      <td>613.62</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151</td>
      <td>363</td>
      <td>6</td>
      <td>2727.62</td>
      <td>793.67</td>
      <td>2616.75</td>
      <td>1.33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>479</td>
      <td>979</td>
      <td>6</td>
      <td>2717.92</td>
      <td>717.78</td>
      <td>2630.25</td>
      <td>1.29</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1297711</th>
      <td>969</td>
      <td>161</td>
      <td>6</td>
      <td>718.51</td>
      <td>390.65</td>
      <td>647.55</td>
      <td>1.55</td>
    </tr>
    <tr>
      <th>1297712</th>
      <td>268</td>
      <td>683</td>
      <td>6</td>
      <td>2948.77</td>
      <td>1254.45</td>
      <td>2713.56</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>1297713</th>
      <td>908</td>
      <td>231</td>
      <td>6</td>
      <td>991.62</td>
      <td>354.11</td>
      <td>947.18</td>
      <td>1.34</td>
    </tr>
    <tr>
      <th>1297714</th>
      <td>596</td>
      <td>416</td>
      <td>5</td>
      <td>1635.86</td>
      <td>926.05</td>
      <td>1522.37</td>
      <td>1.39</td>
    </tr>
    <tr>
      <th>1297715</th>
      <td>362</td>
      <td>709</td>
      <td>4</td>
      <td>2671.00</td>
      <td>925.87</td>
      <td>2536.75</td>
      <td>1.36</td>
    </tr>
  </tbody>
</table>
<p>1297716 rows × 7 columns</p>
</div>
<br>


```python
# WeeklyAggregate.csv; (Day of Week: 요일 단위로 집계된 데이터)
WeeklyAgg = pd.read_csv(os.path.join(DataPath, DataContents[2]))
WeeklyAgg
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sourceid</th>
      <th>dstid</th>
      <th>dow</th>
      <th>mean_travel_time</th>
      <th>standard_deviation_travel_time</th>
      <th>geometric_mean_travel_time</th>
      <th>geometric_standard_deviation_travel_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>210</td>
      <td>724</td>
      <td>5</td>
      <td>1671.72</td>
      <td>709.38</td>
      <td>1573.56</td>
      <td>1.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>363</td>
      <td>745</td>
      <td>3</td>
      <td>1664.14</td>
      <td>455.72</td>
      <td>1606.91</td>
      <td>1.30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>374</td>
      <td>635</td>
      <td>3</td>
      <td>2089.46</td>
      <td>463.24</td>
      <td>2040.78</td>
      <td>1.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>365</td>
      <td>725</td>
      <td>3</td>
      <td>2218.15</td>
      <td>767.20</td>
      <td>2116.40</td>
      <td>1.34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>344</td>
      <td>935</td>
      <td>3</td>
      <td>2708.97</td>
      <td>856.19</td>
      <td>2570.65</td>
      <td>1.39</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2983542</th>
      <td>933</td>
      <td>458</td>
      <td>4</td>
      <td>2461.00</td>
      <td>467.36</td>
      <td>2418.48</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>2983543</th>
      <td>950</td>
      <td>497</td>
      <td>7</td>
      <td>2260.49</td>
      <td>609.04</td>
      <td>2185.16</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>2983544</th>
      <td>969</td>
      <td>307</td>
      <td>7</td>
      <td>1312.72</td>
      <td>305.35</td>
      <td>1280.95</td>
      <td>1.24</td>
    </tr>
    <tr>
      <th>2983545</th>
      <td>958</td>
      <td>417</td>
      <td>7</td>
      <td>1821.70</td>
      <td>460.59</td>
      <td>1763.67</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>2983546</th>
      <td>61</td>
      <td>186</td>
      <td>7</td>
      <td>1660.76</td>
      <td>570.09</td>
      <td>1591.59</td>
      <td>1.31</td>
    </tr>
  </tbody>
</table>
<p>2983547 rows × 7 columns</p>
</div>
<br><br>


# Parking density maps and Traffic activity rhythms
데이터 파일들은 아래 [Harvard Dataverse](https://dataverse.harvard.edu/){:target="_blank"} URL에 들어가 다운받을 수 있다.
```
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8HAJFE
```
<br>

## Unzip all zip files
배포되는 zip 파일들을 macosx 기반에서 압축된 듯 하다. 이 경우 __MACOSX 디렉토리(내부에는 _.DS_Store 파일들)가 Mac Finder(윈도우 OS의 탐색기 같은) 호환 용도로 함께 포함되어 있다. 본래 데이터만 활용해도 되니 압축해제 과정에서 삭제하도록 하자.

> **_NOTE:_** 나는 shutil.rmtree()로 __MACOSX 디렉토리를 없앨 건데, 이 함수는 os.rmdir() 와 달리 타겟 디렉토리 내부에 데이터가 있든 없든 무조건 지운다. 그만큼 사용할 때 더욱 신중을 기해야 한다. 더욱 안전한 방법을 원한다면, os.remove()로 먼저 내부 파일들을 삭제하여 empty directory를 만들어주고, os.rmdir()로 해당 빈 디렉토리를 삭제하는 방법도 있다.


```python
# 최초로 데이터셋 다운받을 때만 사용하는 셀임
BasePath = os.path.join(os.getcwd(), 'derived_dataset/dataverse_files')
for file in os.listdir(BasePath):
    if file.endswith('.zip'):
        dirname = file.split('.')[0]
        target_file_path = os.path.join(BasePath, file)
        target_dir_path = os.path.join(BasePath, dirname)
        if not os.path.exists(target_dir_path):
            os.makedirs(target_dir_path)
        with zipfile.ZipFile(target_file_path, 'r') as zip_obj:
            zip_obj.extractall(target_dir_path)
        
        # Remove a directory of __MACOSX
        # NOTE!! Be cautious when using shutil.rmtree().
        # shutil.rmtree() doesn't care if the directory is empty or not.
        macosx_dir = os.path.join(target_dir_path, '__MACOSX')
        if os.path.exists(macosx_dir):
            shutil.rmtree(macosx_dir)
```


```python
import fnmatch

def read_derived_dataset(BasePath:str, d_type:str, city:str, year:int, quarter:int, time_scope:str):
    """
        Dataset tree 에서 원하는 데이터 파일을 조금 더 간편하게 불러 오는 용도로 제작

    Parameters
    ----------
    BasePath : str
        Dataset Tree의 기본 경로
    d_type : str
        parking | driving
        parking: parking density.
        driving: traffic activity.
    city : str
        city name among 34 cities worldwide. 
        Don't worry about capitalization when writing city name.
        ['Amsterdam', 'Bangalore', 'Bogota', 'Boston', 'Brisbane', 'Brussels', 
        'Cairo', 'Cincinnati', 'Hyderabad', 'Johannesburg', 'Leeds', 'London', 
        'Los Angeles', 'Manchester', 'Melbourne', 'Miami', 'Mumbai', 'Nairobi', 
        'New Delhi', 'Orlando', 'Paris', 'Perth', 'Pittsburgh', 'San Francisco', 
        'Santiago', 'Sao Paulo', 'Seattle', 'Stockholm', 'Sydney', 'Taipei', 
        'Tampa Bay', 'Toronto', 'Washington DC', 'West Midlands UK']
    year : int
        year
    quarter : int
        quarter of a year, which must be matching one of (1, 2, 3, 4).
    time_scope : str
        all | weekends | weekdays
        각각 모든 날짜들 / 주말만 / 평일만 고려해 집계한 데이터에 해당.
    
    Returns
    -------
    pd.DataFrame
        if False returned, 아무튼 데이터 로드 실패
    """
    city = city.lower()
    city_dir = []
    for name in city.split(' '):
        if name in ['uk', 'dc']:
            city_dir.append(name.upper())
        else:
            city_dir.append(name[0].upper() + name[1:])
    city_dir = ' '.join(city_dir)
    
    false_msg = "[[ FALSE RETURNED !!! ]]"
    
    target_path = os.path.join(BasePath, city_dir, city_dir, 'data')
    if not os.path.exists(target_path):
        print(f"{false_msg:=^80}")
        print(target_path)
        print("Please, check again the parameters of 'BasePath' and 'city'.")
        print("The 'city' might not be currently supported.")
        print(''.ljust(80, "="), end='\n\n')
        return False

    else:
        type_dict = {'parking':'parkingdensities', 'driving':'trafficactivity'}
        scope_dict = {'all':'All', 'weekends':'OnlyWeekends', 'weekdays':'OnlyWeekdays'}
        for file in os.listdir(target_path):
            if fnmatch.fnmatch(file, f'*{type_dict[d_type]}*{year}-{quarter}-{scope_dict[time_scope]}*.csv'):
                print(f"LOADED FILE:: {file}")
                break
            
        else: # for loop가 break 되지 않는다면, else로 정상적으로 넘어 온다.
            print(f"{false_msg:=^80}")
            print("Please, check again the parameters of 'd_type', 'year', 'quarter', and 'time_scope'.")
            print("The set of parameters might not be currently supported.")
            print()
            print("NOTE!! 'd_type' must be one of ['parking', 'driving'].")
            print("NOTE!! 'time_scope' must be one of ['all', 'weekends', 'weekdays'].")
            print(''.ljust(80, "="), end='\n\n')
            return False
        
        df = pd.read_csv(os.path.join(target_path, file))
        return df
```
<br>

## About dataset

본 연구팀은 그들만의 모델을 적용하여 Uber Travel Time 데이터셋을 기반으로 새로운 두 종류의 데이터셋을 추출했다. 첫 번째로는 **Traffic Activity** 데이터셋이고, 두 번째론 **Parking Density Maps** 데이터셋이다. 그들의 모델에 대해 아직 정확히 이해하지 못한 단계라 확실치는 않지만, 지금까지 내가 이해한 바로는, 모델이 하는 기본적인 기능은 Uber Travel Time의 운행 기록을 바탕으로 특정 지역 및 시점에서 Driving Vehicles(운행 중인 차량들)과 Parking Vehicles(주차 중인 차량들)의 **수** (Counts)들을 추정해내는 것이다.
* * *
**Traffic Activity 데이터**는 Circadian rhythm of traffic activity 에 대한 데이터인데, 하루 전체 관찰 시점들 중 driving 차량 대수가 가장 많이 관찰된 시점을 기준으로, 나머지 시점에서의 차량 대수를 모두 Normalize 한다. 그렇게 0부터 1사이의 density 값이 기록되고, 이것이 traffic activity가 된다. 따로 공간적인 정보는 없기 때문에, temporal dataset (no spatial dataset)이라 볼 수 있다. (아래 **london_traffic 데이터프레임** 참고)
<br>

**Parking Density Maps 데이터**는 반면 spatio-temporal dataset 이라 볼 수 있다. 지역적 및 시간대 구분이 되어 있기 때문이다. 이 데이터에서는 현 관찰 시점에서 전체 Parking 차량 대수가 기준이 되는데, 예를 들어, 오후 1시부터 2시 사이에 런던 도시 안에 있는 모든 parking 차량 대수가 1000대로 파악됐다면, 이것이 기준이 되는 것이다. 그리고 런던 도시 내 모든 지역들의 parking 차량수를 이 기준으로 나눠서 density 값을 얻는다. 이것이 parking density maps 데이터이다.


```python
# 1. Circadian rhythm of traffic activity (temporal dataset)
# 2018년 4분기 내 평일 기준, 런던의 traffic activity는 평균적으로 오후 5시에 가장 많은 차량 대수를 보인다는 것을 확인할 수 있다.
BasePath = os.path.join(os.getcwd(), 'derived_dataset/dataverse_files')
london_traffic = read_derived_dataset(BasePath, d_type='driving', city='london', year=2018, quarter=4, time_scope='weekdays')
london_traffic
```

    LOADED FILE:: results_trafficactivity_london-lsoa-2018-4-OnlyWeekdays-HourlyAggregate.csv


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>t = 1h</th>
      <th>t = 2h</th>
      <th>t = 3h</th>
      <th>t = 4h</th>
      <th>t = 5h</th>
      <th>t = 6h</th>
      <th>t = 7h</th>
      <th>t = 8h</th>
      <th>t = 9h</th>
      <th>t = 10h</th>
      <th>...</th>
      <th>t = 15h</th>
      <th>t = 16h</th>
      <th>t = 17h</th>
      <th>t = 18h</th>
      <th>t = 19h</th>
      <th>t = 20h</th>
      <th>t = 21h</th>
      <th>t = 22h</th>
      <th>t = 23h</th>
      <th>t = 24h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.160637</td>
      <td>0.102701</td>
      <td>0.053747</td>
      <td>0.0</td>
      <td>0.19509</td>
      <td>0.572794</td>
      <td>0.874499</td>
      <td>0.942832</td>
      <td>0.792283</td>
      <td>0.7301</td>
      <td>...</td>
      <td>0.921995</td>
      <td>0.989038</td>
      <td>1.0</td>
      <td>0.885043</td>
      <td>0.692044</td>
      <td>0.512077</td>
      <td>0.43405</td>
      <td>0.432048</td>
      <td>0.386291</td>
      <td>0.223329</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 24 columns</p>
</div>
<br>


### Regional ID and GeoJson Files
Parking density maps의 지역ID는 Uber Movement Travel Time 데이터에서 등장하는 지역ID를 그대로 따른다. 그리고 Parking density 데이터 컬럼 순서(0부터~N까지)가 각각의 지역ID에 대응한다. 즉, 0번째 컬럼의 데이터 정보는 지역ID=0 에 해당하는 정보라는 의미다.
* * *
고유 지역ID를 지닌 도시별 GeoJson 파일은 Uber Movement 홈페이지에서 역시 다운받을 수 있다. 
> [Uber Movement 홈페이지](https://movement.uber.com/?lang=en-US){:target="_blank"} $ \rightarrow $ \[Products/Travel Times\] $ \rightarrow $ 'Download data' $ \rightarrow $ \[GEO BOUNDARIES\] $ \rightarrow $ '약관동의 및 .JSON 클릭 후 다운로드'

내려받은 GeoJson파일 내 **MOVEMENT_ID** 를 참조해서 사용하면 된다. 


```python
# 2. Parking Density Maps (spatio-temporal dataset)
london_parking = read_derived_dataset(BasePath, d_type='parking', city='london', year=2018, quarter=4, time_scope='weekdays')
london_parking
```

    LOADED FILE:: results_parkingdensities_london-lsoa-2018-4-OnlyWeekdays-HourlyAggregate.csv



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>t = 1h</th>
      <th>t = 2h</th>
      <th>t = 3h</th>
      <th>t = 4h</th>
      <th>t = 5h</th>
      <th>t = 6h</th>
      <th>t = 7h</th>
      <th>t = 8h</th>
      <th>t = 9h</th>
      <th>t = 10h</th>
      <th>...</th>
      <th>t = 15h</th>
      <th>t = 16h</th>
      <th>t = 17h</th>
      <th>t = 18h</th>
      <th>t = 19h</th>
      <th>t = 20h</th>
      <th>t = 21h</th>
      <th>t = 22h</th>
      <th>t = 23h</th>
      <th>t = 24h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000604</td>
      <td>0.000671</td>
      <td>0.000695</td>
      <td>0.000627</td>
      <td>0.000650</td>
      <td>0.000563</td>
      <td>0.000417</td>
      <td>0.000364</td>
      <td>0.000350</td>
      <td>0.000284</td>
      <td>...</td>
      <td>0.000536</td>
      <td>0.000586</td>
      <td>0.000655</td>
      <td>0.000628</td>
      <td>0.000656</td>
      <td>0.000662</td>
      <td>0.000554</td>
      <td>0.000552</td>
      <td>0.000619</td>
      <td>0.000649</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.001360</td>
      <td>0.001268</td>
      <td>0.001180</td>
      <td>0.001184</td>
      <td>0.001684</td>
      <td>0.002404</td>
      <td>0.002978</td>
      <td>0.002304</td>
      <td>0.002017</td>
      <td>0.001952</td>
      <td>...</td>
      <td>0.002284</td>
      <td>0.002211</td>
      <td>0.002128</td>
      <td>0.001860</td>
      <td>0.001603</td>
      <td>0.001620</td>
      <td>0.001639</td>
      <td>0.001581</td>
      <td>0.001523</td>
      <td>0.001458</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000549</td>
      <td>0.000418</td>
      <td>0.000297</td>
      <td>0.000273</td>
      <td>0.000491</td>
      <td>0.001504</td>
      <td>0.002121</td>
      <td>0.001568</td>
      <td>0.001148</td>
      <td>0.001172</td>
      <td>...</td>
      <td>0.000947</td>
      <td>0.000960</td>
      <td>0.000946</td>
      <td>0.000894</td>
      <td>0.000745</td>
      <td>0.000535</td>
      <td>0.000480</td>
      <td>0.000375</td>
      <td>0.000468</td>
      <td>0.000482</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.001396</td>
      <td>0.001282</td>
      <td>0.001209</td>
      <td>0.001179</td>
      <td>0.001438</td>
      <td>0.002173</td>
      <td>0.002609</td>
      <td>0.002144</td>
      <td>0.001882</td>
      <td>0.001870</td>
      <td>...</td>
      <td>0.002003</td>
      <td>0.001796</td>
      <td>0.001612</td>
      <td>0.001575</td>
      <td>0.001405</td>
      <td>0.001310</td>
      <td>0.001244</td>
      <td>0.001262</td>
      <td>0.001337</td>
      <td>0.001320</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.001492</td>
      <td>0.001407</td>
      <td>0.001347</td>
      <td>0.001394</td>
      <td>0.001474</td>
      <td>0.001572</td>
      <td>0.001705</td>
      <td>0.001577</td>
      <td>0.001660</td>
      <td>0.001907</td>
      <td>...</td>
      <td>0.001460</td>
      <td>0.001313</td>
      <td>0.001236</td>
      <td>0.001389</td>
      <td>0.001670</td>
      <td>0.001791</td>
      <td>0.001699</td>
      <td>0.001577</td>
      <td>0.001523</td>
      <td>0.001459</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>978</th>
      <td>0.001061</td>
      <td>0.001023</td>
      <td>0.000914</td>
      <td>0.000879</td>
      <td>0.001150</td>
      <td>0.001725</td>
      <td>0.002233</td>
      <td>0.002889</td>
      <td>0.003118</td>
      <td>0.003042</td>
      <td>...</td>
      <td>0.002073</td>
      <td>0.001916</td>
      <td>0.001621</td>
      <td>0.001719</td>
      <td>0.001709</td>
      <td>0.001670</td>
      <td>0.001521</td>
      <td>0.001403</td>
      <td>0.001257</td>
      <td>0.001154</td>
    </tr>
    <tr>
      <th>979</th>
      <td>0.001065</td>
      <td>0.000864</td>
      <td>0.000810</td>
      <td>0.000906</td>
      <td>0.001113</td>
      <td>0.001260</td>
      <td>0.001907</td>
      <td>0.002717</td>
      <td>0.003118</td>
      <td>0.003018</td>
      <td>...</td>
      <td>0.001836</td>
      <td>0.001439</td>
      <td>0.001415</td>
      <td>0.001703</td>
      <td>0.001842</td>
      <td>0.001556</td>
      <td>0.001432</td>
      <td>0.001320</td>
      <td>0.001208</td>
      <td>0.001108</td>
    </tr>
    <tr>
      <th>980</th>
      <td>0.000937</td>
      <td>0.000786</td>
      <td>0.000767</td>
      <td>0.000942</td>
      <td>0.001296</td>
      <td>0.001525</td>
      <td>0.001984</td>
      <td>0.002534</td>
      <td>0.002915</td>
      <td>0.003009</td>
      <td>...</td>
      <td>0.001626</td>
      <td>0.001326</td>
      <td>0.001267</td>
      <td>0.001367</td>
      <td>0.001568</td>
      <td>0.001437</td>
      <td>0.001238</td>
      <td>0.001033</td>
      <td>0.000977</td>
      <td>0.000878</td>
    </tr>
    <tr>
      <th>981</th>
      <td>0.001003</td>
      <td>0.000852</td>
      <td>0.000810</td>
      <td>0.000883</td>
      <td>0.001043</td>
      <td>0.001187</td>
      <td>0.001690</td>
      <td>0.002573</td>
      <td>0.002839</td>
      <td>0.002771</td>
      <td>...</td>
      <td>0.001669</td>
      <td>0.001498</td>
      <td>0.001418</td>
      <td>0.001585</td>
      <td>0.001746</td>
      <td>0.001521</td>
      <td>0.001397</td>
      <td>0.001211</td>
      <td>0.001182</td>
      <td>0.001064</td>
    </tr>
    <tr>
      <th>982</th>
      <td>0.000863</td>
      <td>0.000891</td>
      <td>0.000916</td>
      <td>0.000861</td>
      <td>0.000916</td>
      <td>0.000781</td>
      <td>0.000913</td>
      <td>0.001031</td>
      <td>0.001122</td>
      <td>0.000862</td>
      <td>...</td>
      <td>0.000594</td>
      <td>0.000624</td>
      <td>0.000584</td>
      <td>0.000550</td>
      <td>0.000486</td>
      <td>0.000453</td>
      <td>0.000573</td>
      <td>0.000724</td>
      <td>0.000821</td>
      <td>0.000890</td>
    </tr>
  </tbody>
</table>
<p>983 rows × 24 columns</p>
</div>
<br>


```python
# 3. GeoJson from Uber Movement
with open('cities_json/london_lsoa.json', 'r') as file:
    geojson = json.load(file)
    geojson = json.dumps(geojson)
    geojson_df = gpd.read_file(geojson)

geojson_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>msoa_code</th>
      <th>msoa_name</th>
      <th>la_code</th>
      <th>la_name</th>
      <th>geoeast</th>
      <th>geonorth</th>
      <th>popeast</th>
      <th>popnorth</th>
      <th>area_km2</th>
      <th>MOVEMENT_ID</th>
      <th>DISPLAY_NAME</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>E02000508</td>
      <td>Hillingdon 015</td>
      <td>00AS</td>
      <td>Hillingdon</td>
      <td>506163</td>
      <td>183536</td>
      <td>505978</td>
      <td>183811</td>
      <td>2.746600</td>
      <td>0</td>
      <td>Hillingdon, 00AS (0)</td>
      <td>MULTIPOLYGON (((-0.47794 51.55485, -0.47665 51...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>E02000716</td>
      <td>Newham 003</td>
      <td>00BB</td>
      <td>Newham</td>
      <td>541978</td>
      <td>186009</td>
      <td>541870</td>
      <td>185568</td>
      <td>1.565170</td>
      <td>1</td>
      <td>Newham, 00BB (1)</td>
      <td>MULTIPOLYGON (((0.05255 51.56171, 0.05310 51.5...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>E02000747</td>
      <td>Newham 034</td>
      <td>00BB</td>
      <td>Newham</td>
      <td>539578</td>
      <td>181317</td>
      <td>539891</td>
      <td>181438</td>
      <td>2.082410</td>
      <td>2</td>
      <td>Newham, 00BB (2)</td>
      <td>MULTIPOLYGON (((0.01001 51.52181, 0.01003 51.5...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>E02000748</td>
      <td>Newham 035</td>
      <td>00BB</td>
      <td>Newham</td>
      <td>542500</td>
      <td>181152</td>
      <td>542439</td>
      <td>181339</td>
      <td>1.331750</td>
      <td>3</td>
      <td>Newham, 00BB (3)</td>
      <td>MULTIPOLYGON (((0.05392 51.51611, 0.05174 51.5...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E02000749</td>
      <td>Newham 036</td>
      <td>00BB</td>
      <td>Newham</td>
      <td>541047</td>
      <td>181103</td>
      <td>540847</td>
      <td>181294</td>
      <td>1.419020</td>
      <td>4</td>
      <td>Newham, 00BB (4)</td>
      <td>MULTIPOLYGON (((0.03241 51.51704, 0.03179 51.5...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>978</th>
      <td>E02000974</td>
      <td>Westminster 015</td>
      <td>00BK</td>
      <td>Westminster</td>
      <td>527028</td>
      <td>181254</td>
      <td>527172</td>
      <td>181179</td>
      <td>0.689337</td>
      <td>978</td>
      <td>Westminster, 00BK (978)</td>
      <td>MULTIPOLYGON (((-0.17019 51.51994, -0.16019 51...</td>
    </tr>
    <tr>
      <th>979</th>
      <td>E02000975</td>
      <td>Westminster 016</td>
      <td>00BK</td>
      <td>Westminster</td>
      <td>526396</td>
      <td>181129</td>
      <td>526375</td>
      <td>181042</td>
      <td>0.484638</td>
      <td>979</td>
      <td>Westminster, 00BK (979)</td>
      <td>MULTIPOLYGON (((-0.17867 51.52008, -0.17898 51...</td>
    </tr>
    <tr>
      <th>980</th>
      <td>E02000980</td>
      <td>Westminster 021</td>
      <td>00BK</td>
      <td>Westminster</td>
      <td>529921</td>
      <td>178656</td>
      <td>529758</td>
      <td>178698</td>
      <td>0.539208</td>
      <td>980</td>
      <td>Westminster, 00BK (980)</td>
      <td>MULTIPOLYGON (((-0.12279 51.49453, -0.12305 51...</td>
    </tr>
    <tr>
      <th>981</th>
      <td>E02000981</td>
      <td>Westminster 022</td>
      <td>00BK</td>
      <td>Westminster</td>
      <td>529123</td>
      <td>178488</td>
      <td>529140</td>
      <td>178401</td>
      <td>0.363777</td>
      <td>981</td>
      <td>Westminster, 00BK (981)</td>
      <td>MULTIPOLYGON (((-0.14126 51.49455, -0.14080 51...</td>
    </tr>
    <tr>
      <th>982</th>
      <td>E02000983</td>
      <td>Westminster 024</td>
      <td>00BK</td>
      <td>Westminster</td>
      <td>529464</td>
      <td>178054</td>
      <td>529488</td>
      <td>178111</td>
      <td>0.559636</td>
      <td>982</td>
      <td>Westminster, 00BK (982)</td>
      <td>MULTIPOLYGON (((-0.12710 51.48764, -0.12957 51...</td>
    </tr>
  </tbody>
</table>
<p>983 rows × 12 columns</p>
</div>
<br>


## Parking Density Maps
본 글에서 traffic activity 데이터는 해당 데이터를 읽어본 것만으로 만족하고 넘어가도록 한다. (데이터 컬럼 한줄 있는거 보고 흥미/의욕 저하..)
대신 (spatio-temporal) Parking Density Maps 데이터를 중점적으로 뜯어보고 시각화해보고자 한다.
* * *
내가 할 일은 이러하다.
1. 우선, 특정 시점에서 density snapshot 그림을 그려본다.
2. density map의 24시간 하루 변화를 연속적으로 시각화해보자.

* * *
지역(구역)별 Density의 대수적 차이의 표현은 값의 크기에 따라 scatter size를 달리하여 표현하도록 하자. 아래 Rescaling equation을 density 컬럼에 적용하여 scatter size들을 추출한다.

<br>

## Rescaling data range
Parking density 값에 따라 Scatter size를 다르게 부여할 것이다. Parking density 값 자체가 작기 때문에 표현이 어렵기 때문이다. 주어진 데이터 분포를 기반으로, 아래 식에 따라 rescaling을 진행했다.
$$\Large x' = (s_{max} - s_{min})\times\left(\frac{x-x_{min}}{x_{max}-x_{min}}\right)^{1.5} + s_{min}$$

$ s_{min} $과 $ s_{max} $는 각각 내가 원하는 scattter size의 최소 및 최대 크기를 나타낸다. 따라서 임의로 조정 가능한 hyper-parameter들이다. 그리고 $ x_{min} $과 $ x_{max} $ 는 데이터 분포 상의 최소 및 최대값이다. 분모 term은 단순 min-max normalization으로서 [0, 1] range를 갖게 되고, 나머지 term들에 의해 최종적으로 $ x $값은 [$ s_{min} $, $ s_{max} $] range로 normalize 된다.

> **_NOTE:_** Min-max Normalization term 에 1.5 power를 준 이유는 작은 값의 density point 시각화는 억제하고, 어느 정도 큰 값 위주로만 강조하기 위함이다. power 가 커질 수록, density 가 큰 애들만 시각화에 강조된다. (아래 셀 참고)


```python
# Choosing the suitable power for visualization
x_test = np.arange(0, 1, 0.01)
fig, axs = plt.subplots(nrows=1, ncols=3, facecolor='w', figsize=(15, 3))
for ax, a in zip(axs.flatten(), [1.5, 3, 6]):
    ax.scatter(x_test, x_test**a, marker='*', facecolors='w', edgecolor='grey')
    ax.set_title(f"Power of {a}", fontsize=10)
    ax.set_ylabel("rescaled x")
    ax.set_xlabel("original x")
plt.show()
```

    
![png](/assets/img/post/parking_density/parking_maps_21_0.png)
<br><br>


```python
def rescaled_values(xs:pd.Series, x_min, x_max, s_min, s_max, power=1.5):
    """
        float 자릿수를 가지는 Density값을 가시적인 scatter size 로 rescaling 하기 위함.

    Parameters
    ----------
    xs : pd.Series
        Density series
    x_min : float
        Minimum Density correspodning to s_min
    x_max : float
        Maximum Density corresponding to s_max 
    s_min : int
        Minimum scatter size corresponding to minimum density
    s_max : int
        Maximum scatter size correspoding to maximum density
    power : float, optional
        parameter to control how making the small values to be much smaller, by default 1.5.

    Returns
    -------
    pd.Series
        Scatter size series as rescaled densities
    """
    x_min, x_max = min(xs), max(xs)
    cal_func = lambda x: ((s_max - s_min) * ((x - x_min) / (x_max - x_min)) ** power) + s_min
    new_xs = pd.Series([cal_func(x) for x in xs])
    return new_xs
```


```python
x_min = london_parking.min().min()
x_max = london_parking.max().max()

ms = rescaled_values(london_parking.iloc[:, 8], x_min, x_max, 0.5, 400)

fig, ax = plt.subplots(facecolor='w', figsize=(10, 10))
geojson_df.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=.4)
geojson_df.representative_point().plot(ax=ax, marker='o', color='blue', markersize=ms, alpha=.4)
cx.add_basemap(ax, crs=geojson_df.crs.to_string(), source=cx.providers.Stamen.Watercolor)
cx.add_basemap(ax, crs=geojson_df.crs.to_string(), source=cx.providers.Stamen.TonerLabels)
ax.axis('off')
ax.set_aspect('auto')
ax.set_title("LONDON, t = 9h", fontsize=20)
plt.show()
```

![png](/assets/img/post/parking_density/parking_maps_23_0.png)
<br><br>    


## Circadian Parking Density Maps
**London, Paris, LA, Melbourne, Mumbai, Sydney** 에 대해 모두 하루 중 Parking Density Maps을 추출하여 Graphics Interchange Format(GIF)로 만들어 보자.


```python
# Check if it's possible to load all datasets for each city.
BasePath = os.path.join(os.getcwd(), 'derived_dataset/dataverse_files')

cities = ['london', 'paris', 'los angeles', 'melbourne', 'mumbai', 'sydney']
parking_cities = []
geojson_cities = []
for city in cities:
    df = read_derived_dataset(BasePath, d_type='parking', city=city, year=2018, quarter=4, time_scope='weekdays')
    parking_cities.append(df)

    city = city.lower().replace(' ', '_')
    for fn in os.listdir('cities_json/'):
        if fnmatch.fnmatch(fn, f'{city}*.json'):
            with open(f'cities_json/{fn}', 'r') as file:
                gj = json.load(file)
                gj = json.dumps(gj)
                geojson_cities.append(gpd.read_file(gj))
            
            print(f"LOADED JSON:: {fn}")
            break
    else:
        print(f"Failed to find .json matching to {city}.")
    
    print()
```

    LOADED FILE:: results_parkingdensities_london-lsoa-2018-4-OnlyWeekdays-HourlyAggregate.csv
    LOADED JSON:: london_lsoa.json
    
    LOADED FILE:: results_parkingdensities_paris-iris-2018-4-OnlyWeekdays-HourlyAggregate.csv
    LOADED JSON:: paris_iris.json
    
    LOADED FILE:: results_parkingdensities_los_angeles-censustracts-2018-4-OnlyWeekdays-HourlyAggregate.csv
    LOADED JSON:: los_angeles_censustracts.json
    
    LOADED FILE:: results_parkingdensities_melbourne-tz-2018-4-OnlyWeekdays-HourlyAggregate.csv
    LOADED JSON:: melbourne_tz.json
    
    LOADED FILE:: results_parkingdensities_mumbai-hexclusters-2018-4-OnlyWeekdays-HourlyAggregate.csv
    LOADED JSON:: mumbai_hexclusters.json
    
    LOADED FILE:: results_parkingdensities_sydney-tz-2018-4-OnlyWeekdays-HourlyAggregate.csv
    LOADED JSON:: sydney_tz.json
    



```python
SavePath = os.path.join(os.getcwd(), 'img_output/')
for ii, city in enumerate(cities):
    city_dir_name = city.lower().replace(' ', '_')
    city_dir_path = os.path.join(SavePath, city_dir_name)
    if not os.path.exists(city_dir_path):
        os.mkdir(city_dir_path)
    
    target_city = parking_cities[ii]
    x_min = target_city.min().min()
    x_max = target_city.max().max()

    for jj, col in tqdm(enumerate(target_city.columns)):
        snap_parking = target_city.loc[:, col]
        markersizes = rescaled_values(snap_parking, x_min, x_max, 0.5, 400)

        fig, ax = plt.subplots(facecolor='w', figsize=(10, 10))
        geojson_cities[ii].representative_point().plot(ax=ax, marker='o', color='blue', markersize=markersizes, alpha=.5)
        cx.add_basemap(ax, crs=geojson_cities[2].crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik, alpha=.7)

        ax.axis('off')
        ax.set_aspect('auto') # turn-off of forcing to ignore my figsize
        ax.set_title(f'Weekdays in {city_dir_name.upper()}, {col}', fontsize=20)

        save_name = f'{jj}_{jj+1}h_ParkingDensity_{city_dir_name.upper()}_2018_4q_OnlyWeekdays.png'
        plt.savefig(os.path.join(city_dir_path, save_name), pad_inches=.15, bbox_inches='tight')
        plt.close()
        plt.clf()

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

24it [00:39,  1.65s/it]
24it [01:07,  2.82s/it]
24it [00:23,  1.01it/s]
24it [00:30,  1.27s/it]
24it [00:24,  1.03s/it]
24it [00:26,  1.10s/it]
```


```python
for city in cities:
    city = city.lower().replace(' ', '_')
    img_path = os.path.join(SavePath, city)
    img_set = []
    for img in sorted(os.listdir(img_path), key=lambda x: int(x.split('_')[0]), reverse=False):
        img_set.append(iio.imread(os.path.join(img_path, img)))
    
    gif_path = os.path.join(os.getcwd(), 'gif_output')
    if not os.path.exists(gif_path):
        os.mkdir(gif_path)
    gif_fname = f"{city.upper()}_ParkingDensity_2018-4q_OnlyWeekdays.gif"
    iio.imwrite(os.path.join(gif_path, gif_fname), img_set, duration=2.7, loop=0) # loop = 0 means infinite loop.
```

<table><tr>
<td> <img src="/assets/img/post/parking_density/LONDON_ParkingDensity_2018-4q_OnlyWeekdays.gif" alt="Drawing" style="width: 500px;"/> </td>
<td> <img src="/assets/img/post/parking_density/LOS_ANGELES_ParkingDensity_2018-4q_OnlyWeekdays.gif" alt="Drawing" style="width: 500px;"/> </td>
</tr></table>

<table><tr>
<td> <img src="/assets/img/post/parking_density/PARIS_ParkingDensity_2018-4q_OnlyWeekdays.gif" alt="Drawing" style="width: 500px;"/> </td>
<td> <img src="/assets/img/post/parking_density/MELBOURNE_ParkingDensity_2018-4q_OnlyWeekdays.gif" alt="Drawing" style="width: 500px;"/> </td>
</tr></table>

<table><tr>
<td> <img src="/assets/img/post/parking_density/MUMBAI_ParkingDensity_2018-4q_OnlyWeekdays.gif" alt="Drawing" style="width: 500px;"/> </td>
<td> <img src="/assets/img/post/parking_density/SYDNEY_ParkingDensity_2018-4q_OnlyWeekdays.gif" alt="Drawing" style="width: 500px;"/> </td>
</tr></table>
<br>

## Take-Home Messages and Conclusion
* Uber Movement에서 공개 배포 중인 Travel Time 데이터 정보를 바탕으로, Parking Density & Traffic Activity 값을 추산한 데이터를 살펴 보았다.
  * Uber Movement 팀은 Travel Time 말고도, 현재 (도로 단위별) Speed 데이터와 New Mobility Heatmap 데이터도 공개하고 있다. (!! 2023년 11월 기준, 데이터 비공개됨ㅜ)
* 본 연구팀에서 raw data로 활용한 Uber Travel Time 데이터는 어떤 형식과 내용을 지녔는지만 살짝 확인해봤다.
* 본 연구팀 모델의 주요 아웃풋이라 할 수 있는 Parking Density 데이터를 중점적으로 살펴 보았고, 해외 6개 주요 도시에 대해 Circadian rhythms of Parking Density 를 시각화해봤다. 

안타깝게도, 본 연구팀에서 추출한 도시별 데이터에서 우리 나라 '서울'같은 국내 도시는 찾을 수 없었다. 아무래도 Uber의 Travel Time 데이터를 raw data로 활용하는 이유와 국내에선 Uber 이용의 점유율이 높지 않아서 일 것이다. 사실 공개된 Uber Travel Time 데이터에도 국내 도시는 없다.
<br>

중요한 건, 연구팀에서 개발한 Parking Density & Traffic Activity를 <u>Travel Time 정보에 기반해</u> 추출해주는 모델은 굉장히 활용도가 높아 보인다. 어느 시간대/어느 지역에 주차 밀집도 및 차량 운행 수준이 높다는 정보는, 도시의 (교통, 상업 등 관련) 인프라 개발 단계에서 좋은 참고 자료가 될 것이기 때문이다. (물론 차량 내비게이션 데이터로는 Parking인지 Driving인지를 추론해내긴 쉽겠지만... 가용 데이터가 없다는 전제하에) 그래서 추후 여유가 된다면, 본 연구팀의 모델과 그 방법론을 더욱 자세히 답습하여, '서울'이나 국내 도시에도 적용해보고 싶다. 고맙게도 연구팀은 모델에 관한 소스 코드(**julia** 언어 사용) 역시 모두 공개하고 있다. 관심 있다면, 여기 [CODE OCEAN LAB](https://codeocean.com/capsule/2525498/tree/v1){:target="_blank"}에 들어가 열람해보자.

***fin***