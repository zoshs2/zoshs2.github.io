---
title: "SafeGraph Open Census Dataset"
date: 2023-06-26 11:48:49 +0900
categories: [Open-Data, EDA]
tags: [python, census, visualization, eda, safegraph, USA]
math: true
---

# 들어가며
오늘은 [SafeGraph Inc.](https://www.safegraph.com/free-data/){:target="_blank"}에서 재가공 및 공개 배포 중인 **Open Census Data**를 살펴본 내용이다. 본 데이터는 사실 US Census Bureau에서 이미 배포 중인 각종 demographic 데이터들을 하나로 종합해 정리한 데이터라고 보면 된다. 지역별 인구수, 소득 수준 같은 속성 뿐 아니라, 특정 시설수 같은 정보도 포함되어 있다.
* * *
[US Census Bureau](https://www.census.gov){:target="_blank"}(미국 인구조사국) 는 다양한 특성에 대한 다양한 인구통계적(demographic) 정보를 조사 및 수집하는 기관이다. 대표적인 '국내 인구수 조사(Census)'도 이런 여러 demographic data 중 하나이다. Census 조사는 보통 대규모의 인력과 비용, 그리고 시간이 요구되기 때문에, 미국같은 경우는 **10년**마다 모집단 조사(full survey)를 수행하고 결과를 발표한다. 이게 이른바, **Decennial Census**라고 부르는 공식 결과다. 그리고 미국 인구조사국은 이 외에도 지역마다의 성별수(Gender), 나이(Age), 소득(Income), 민족계통(Ethnicity; 라틴아메리카(히스패닉계) or 아시아계 or 아프리카계 or 유럽계) 등을 조사한다. 이 조사는 **American Community Survey; ACS**라는 프로젝트 이름으로 **매년** 샘플링 조사(sample survey)를 통해 집계하여 결과를 발표한다.
<br><br>

# SafeGraph - Open Census Data

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdown
from tqdm import tqdm
import tarfile
import geopandas as gpd
```
<br>

## Data Acquisition
현재 배포되고 있는 데이터들의 Google Drive **고유파일 ID**와 **파일이름**들을 정리해뒀다.

```python
# File ID and Name
IDnFN = [['1klKXB35iXyhfbgKTEXZXgdhZWJDwpbEi', 'safegraph_open_census_data_2020.tar.gz'], \
    ['1v2MTZG9MNW-ao8fSeO6r69AfHsXB2mwx', 'safegraph_open_census_data_2020_to_2029_geometry.tar.gz'], \
    ['1rMF7doWkgoKvAs4GPi5FpQNhOFd9V9b2', 'safegraph_open_census_data_2020_redistricting.tar.gz'], \
    ['1ab-dGzzDntCEE8wVBekAQpvsOZvxSbVV', 'safegraph_open_census_data_2019.tar.gz'], \
    ['1oUT_UBUCa6nRZ207taXHAeANmpHEwsGt', 'safegraph_open_census_data_2018.tar.gz'], \
    ['15TFKFONZquET0AvpFlsENSOP2dk2w39V', 'safegraph_open_census_data_2017.tar.gz'], \
    ['10InSSafTPUZ6tK-e8g6msYepCO9i2H3L', 'safegraph_open_census_data_2016.tar.gz'], \
    ['1QmKe7v7peaYAjDDh50hNP4b9s0B0JTWm', 'safegraph_open_census_data_2010_to_2019_geometry.tar.gz']]
```


```python
gdrive_base_path = 'https://drive.google.com/uc?id='
SavePath = '/open_census_data'
for file_id, file_name in tqdm(IDnFN):
    gdown.download(gdrive_base_path + file_id, os.path.join(SavePath, file_name), quiet=True)
```

    100%|██████████| 8/8 [04:37<00:00, 34.67s/it]



```python
# unzip 'safegraph_open_census_data_2020.tar.gz'
with tarfile.open(os.path.join(SavePath, IDnFN[0][1]), 'r:gz') as tr:
    tr.extractall(path=SavePath)

# unzip 'safegraph_open_census_data_2020_to_2029_geometry.tar.gz' to extract a geojson file of 'cbg_2020.geojson'
with tarfile.open(os.path.join(SavePath, IDnFN[1][1]), 'r:gz') as tr:
    tr.extractall(path=SavePath)
```
<br>

## US Census GeoJson
US Census Bureau의 집계 단위인 Census Block Group(cbg)의 polygon-styled and geometrical GeoJSON 파일이다. 용량이 커서(~1.9 GB) 불러오는데 꽤 시간이 소요된다. Polygon-style로 시각화 할 게 아니면, 각 census data 내의 '/metadata/cbg_geographic_data.csv'를 사용하자.

```python
BasePath = '/open_census_data'
FileContents = os.listdir(BasePath)

cbg_geo = gpd.read_file(os.path.join(BasePath, FileContents[1]))
cbg_geo
```

<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StateFIPS</th>
      <th>CountyFIPS</th>
      <th>TractCode</th>
      <th>BlockGroup</th>
      <th>CensusBlockGroup</th>
      <th>State</th>
      <th>County</th>
      <th>MTFCC</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01</td>
      <td>033</td>
      <td>020200</td>
      <td>1</td>
      <td>010330202001</td>
      <td>AL</td>
      <td>Colbert County</td>
      <td>G5030</td>
      <td>MULTIPOLYGON (((-87.70081 34.76189, -87.70081 ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01</td>
      <td>019</td>
      <td>956000</td>
      <td>1</td>
      <td>010199560001</td>
      <td>AL</td>
      <td>Cherokee County</td>
      <td>G5030</td>
      <td>MULTIPOLYGON (((-85.67917 34.15255, -85.67904 ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>01</td>
      <td>073</td>
      <td>004701</td>
      <td>2</td>
      <td>010730047012</td>
      <td>AL</td>
      <td>Jefferson County</td>
      <td>G5030</td>
      <td>MULTIPOLYGON (((-86.78478 33.51157, -86.78267 ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>01</td>
      <td>073</td>
      <td>004702</td>
      <td>1</td>
      <td>010730047021</td>
      <td>AL</td>
      <td>Jefferson County</td>
      <td>G5030</td>
      <td>MULTIPOLYGON (((-86.77400 33.51790, -86.77396 ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>01</td>
      <td>073</td>
      <td>004702</td>
      <td>2</td>
      <td>010730047022</td>
      <td>AL</td>
      <td>Jefferson County</td>
      <td>G5030</td>
      <td>MULTIPOLYGON (((-86.77621 33.50359, -86.77599 ...</td>
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
    </tr>
    <tr>
      <th>242330</th>
      <td>72</td>
      <td>127</td>
      <td>008300</td>
      <td>3</td>
      <td>721270083003</td>
      <td>PR</td>
      <td>San Juan Municipio</td>
      <td>G5030</td>
      <td>MULTIPOLYGON (((-66.09123 18.39897, -66.08954 ...</td>
    </tr>
    <tr>
      <th>242331</th>
      <td>72</td>
      <td>127</td>
      <td>010012</td>
      <td>3</td>
      <td>721270100123</td>
      <td>PR</td>
      <td>San Juan Municipio</td>
      <td>G5030</td>
      <td>MULTIPOLYGON (((-66.04081 18.36705, -66.04076 ...</td>
    </tr>
    <tr>
      <th>242332</th>
      <td>72</td>
      <td>127</td>
      <td>010022</td>
      <td>2</td>
      <td>721270100222</td>
      <td>PR</td>
      <td>San Juan Municipio</td>
      <td>G5030</td>
      <td>MULTIPOLYGON (((-66.05515 18.37903, -66.05482 ...</td>
    </tr>
    <tr>
      <th>242333</th>
      <td>72</td>
      <td>127</td>
      <td>010100</td>
      <td>1</td>
      <td>721270101001</td>
      <td>PR</td>
      <td>San Juan Municipio</td>
      <td>G5030</td>
      <td>MULTIPOLYGON (((-66.07215 18.34087, -66.07208 ...</td>
    </tr>
    <tr>
      <th>242334</th>
      <td>72</td>
      <td>127</td>
      <td>010100</td>
      <td>3</td>
      <td>721270101003</td>
      <td>PR</td>
      <td>San Juan Municipio</td>
      <td>G5030</td>
      <td>MULTIPOLYGON (((-66.08014 18.32918, -66.08002 ...</td>
    </tr>
  </tbody>
</table>
<p>242335 rows × 9 columns</p>
</div>
<br>


```python
fig, ax = plt.subplots(facecolor='w', figsize=(15, 15))
cbg_geo.plot(ax=ax, facecolor='None', edgecolor='black', linewidth=0.2)
ax.axis('off')
plt.show()
```

<br>    
![png](/assets/img/post/safegraph/OpenCensusData_10_0.png)
<br> 


* * *
이 글에선 미국 본토 내 "**48개 주 + District of Columbia; 즉 워싱턴 D.C**"만을 다룬다. \
한 가지 TMI,,, 미국 자체는 크게 3계층 구조를 지닌다고 한다.
1. United States: 50개 states + 워싱턴 D.C (미국 수도인 워싱턴 D.C는 어느 주에도 속하지 않음)
2. Continental United States: '하와이주'를 제외한 49개 states + 워싱턴 D.C
3. Conterminous/Contiguous United States: '하와이주'와 '알래스카주'를 제외한 48개 states + 워싱턴 D.C

즉, Conterminous(Contiguous) United States 만 다루겠다는 말이다. \
두 번째 TMI,,, '하와이주'와 '알래스카주'를 제외한 48개 states를 **Lower 48 states**라고 부르기도 한다고 한다.

```python
# 하와이주(HI), 알래스카주(AK) + 푸에르토리코(PR)까지 총 3개 제외
us_cbg_geo = cbg_geo[~cbg_geo['State'].isin(['AK', 'HI', 'PR'])].reset_index(drop=True)
```


```python
fig, ax = plt.subplots(facecolor='w', figsize=(15, 15))
us_cbg_geo.plot(ax=ax, facecolor='None', edgecolor='black', linewidth=0.2)
ax.axis('off')
plt.show()
```

<br>
![png](/assets/img/post/safegraph/OpenCensusData_12_0.png)
<br> 
<br>

## US Open Census Data

2023년 6월 기준, SafeGraph에서 재가공-배포 중인 demographic dataset은 다음과 같다.
+ **2016 5-year ACS** : 2012.01 ~ 2016.12까지의 매해 ACS의 결과를 평균 집계한 데이터
+ **2017 5-year ACS** : 2013.01 ~ 2017.12 ACS 평균 집계
+ **2018 5-year ACS** : 2014.01 ~ 2018.12 ACS 평균 집계
+ **2019 5-year ACS** : 2015.01 ~ 2019.12 ACS 평균 집계
+ **2010-2019 Census Block Group geometries** : 2010년 ~ 2019년 사이 데이터들의 집계 기준으로 활용한 GeoJSON
+ **(NEW) 2020 5-year ACS** : 2016.01 ~ 2020.12
+ **(NEW) 2020-2029 Census Block Group geometries** : 2020년 ~ 2029년 사이 데이터들의 집계 기준으로 활용할 GeoJSON
+ **(NEW) 2020 decennial redistricting data** : 2020년판 Decennial Survey (미국 인구총조사 발표 데이터; 인구수에 대한 데이터만 있음; ACS 아님)

참고로, 엄밀히 말하자면, 5년 묶음으로 취합 및 집계한 이 데이터들(Multiyear dataset)도 US Census Bureau 측에서 ACS 일환으로 수행한 자료이다. 자세한 내용이 궁금하다면 아래 미국 인구조사국 공식 홈페이지 내용을 참고하자.
* * *
https://www.census.gov/programs-surveys/acs/guidance/estimates.html<br>: 'When to Use 1-year or 5-year Estimates' by US Census Bureau \
https://www2.census.gov/programs-surveys/acs/tech_docs/accuracy/MultiyearACSAccuracyofData2019.pdf<br>: 'Detailed of multiyear(5-year) Dataset' by US Census Bureau
* * *

그러면 SafeGraph Inc. 측 홈페이지에 데이터 업로드의 취지와 목적이 궁금할 수 있는데, 그들은 아래와 같이 설명하고 있다.<br><br>
***"While the US Census Bureau offers free downloads of their data, it's often difficult and confusing to get bulk access to it at the granularity needed for advanced analysis.*** \
***(Therefore) We've pre-cleaned this data and packaged it into easy to use..."*** <br>
\- SafeGraph Inc. (https://www.safegraph.com/free-data/open-census-data). 
* * *

대충 정리하자면, 미국 인구 조사국에서 올려놓은 ACS 각종 자료들이 여기저기 흩어져있고 사용자들의 접근과 활용이 어려우니, 더욱 사용이 용이하게끔 우리가 잘 정리해서 재배포한다는 취지이다. 아무튼 그렇다. 아무튼 이 글에서 나는 **SafeGraph's <2020 5-year ACS>** 데이터(아래 Dataset Structure 참고)를 살펴보도록 한다.

```python
safegraph_open_census_data_2020
├── data
│   ├── cbg_b01.csv     # field 명으로 데이터들이 나눠져있다. (field: household income, median age, population etc...)
│   ├── cbg_b02.csv
│   ├── ...
│   ├── ...
│   └── cbg_c24.csv
└── metadata
    ├── cbg_field_descriptions.csv          # field 들에 대한 설명
    ├── cbg_fips_codes.csv                  # us fips code
    └── cbg_geographic_data.csv             # point-styled CBG geometry (longitude and latitude)
```

데이터에 포함된 모든 속성의 테이블 정의서는 여기 [미국 인구조사국 사이트](https://www.census.gov/programs-surveys/acs/technical-documentation/table-shells.2020.html#list-tab-79594641){:target="_blank"}에서 열람할 수 있다.


```python
BasePath = '/open_census_data/safegraph_open_census_data_2020'
SubDir = os.listdir(BasePath)
cbg_fd_desc = pd.read_csv(os.path.join(BasePath, SubDir[1], 'cbg_field_descriptions.csv'))
cbg_fd_desc.head()
```

<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>table_id</th>
      <th>table_number</th>
      <th>table_title</th>
      <th>table_topics</th>
      <th>table_universe</th>
      <th>field_level_1</th>
      <th>field_level_2</th>
      <th>field_level_3</th>
      <th>field_level_4</th>
      <th>field_level_5</th>
      <th>field_level_6</th>
      <th>field_level_7</th>
      <th>field_level_8</th>
      <th>field_level_9</th>
      <th>field_level_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B01001e1</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B01001e10</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>Male</td>
      <td>22 to 24 years</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B01001e11</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>Male</td>
      <td>25 to 29 years</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B01001e12</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>Male</td>
      <td>30 to 34 years</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B01001e13</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>Male</td>
      <td>35 to 39 years</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
<br>



```python
# field_level_1 에는 크게 'Estimate'과 'MarginOfError'에 해당하는 field가 있다. 
# 이 글에선 측정치(값) 자체만 보고자 하므로 'Estimate'으로만 관심 field 수를 제한하겠다.
cbg_fd_desc = cbg_fd_desc[cbg_fd_desc['field_level_1']=='Estimate'].reset_index(drop=True)
cbg_fd_desc
```

<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>table_id</th>
      <th>table_number</th>
      <th>table_title</th>
      <th>table_topics</th>
      <th>table_universe</th>
      <th>field_level_1</th>
      <th>field_level_2</th>
      <th>field_level_3</th>
      <th>field_level_4</th>
      <th>field_level_5</th>
      <th>field_level_6</th>
      <th>field_level_7</th>
      <th>field_level_8</th>
      <th>field_level_9</th>
      <th>field_level_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B01001e1</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B01001e10</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>Male</td>
      <td>22 to 24 years</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B01001e11</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>Male</td>
      <td>25 to 29 years</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B01001e12</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>Male</td>
      <td>30 to 34 years</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B01001e13</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>Male</td>
      <td>35 to 39 years</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>4077</th>
      <td>C24030e55</td>
      <td>C24030</td>
      <td>Sex By Industry For The Civilian Employed Popu...</td>
      <td>Age and Sex, Civilian Population, Industry</td>
      <td>Civilian employed population 16 years and over</td>
      <td>Estimate</td>
      <td>SEX BY INDUSTRY FOR THE CIVILIAN EMPLOYED POPU...</td>
      <td>Civilian employed population 16 years and over</td>
      <td>Total</td>
      <td>Female</td>
      <td>Public administration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4078</th>
      <td>C24030e6</td>
      <td>C24030</td>
      <td>Sex By Industry For The Civilian Employed Popu...</td>
      <td>Age and Sex, Civilian Population, Industry</td>
      <td>Civilian employed population 16 years and over</td>
      <td>Estimate</td>
      <td>SEX BY INDUSTRY FOR THE CIVILIAN EMPLOYED POPU...</td>
      <td>Civilian employed population 16 years and over</td>
      <td>Total</td>
      <td>Male</td>
      <td>Construction</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4079</th>
      <td>C24030e7</td>
      <td>C24030</td>
      <td>Sex By Industry For The Civilian Employed Popu...</td>
      <td>Age and Sex, Civilian Population, Industry</td>
      <td>Civilian employed population 16 years and over</td>
      <td>Estimate</td>
      <td>SEX BY INDUSTRY FOR THE CIVILIAN EMPLOYED POPU...</td>
      <td>Civilian employed population 16 years and over</td>
      <td>Total</td>
      <td>Male</td>
      <td>Manufacturing</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4080</th>
      <td>C24030e8</td>
      <td>C24030</td>
      <td>Sex By Industry For The Civilian Employed Popu...</td>
      <td>Age and Sex, Civilian Population, Industry</td>
      <td>Civilian employed population 16 years and over</td>
      <td>Estimate</td>
      <td>SEX BY INDUSTRY FOR THE CIVILIAN EMPLOYED POPU...</td>
      <td>Civilian employed population 16 years and over</td>
      <td>Total</td>
      <td>Male</td>
      <td>Wholesale trade</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4081</th>
      <td>C24030e9</td>
      <td>C24030</td>
      <td>Sex By Industry For The Civilian Employed Popu...</td>
      <td>Age and Sex, Civilian Population, Industry</td>
      <td>Civilian employed population 16 years and over</td>
      <td>Estimate</td>
      <td>SEX BY INDUSTRY FOR THE CIVILIAN EMPLOYED POPU...</td>
      <td>Civilian employed population 16 years and over</td>
      <td>Total</td>
      <td>Male</td>
      <td>Retail trade</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>4082 rows × 15 columns</p>
</div>
<br>


### Alternative for US and CBG Geometry
CBG Geometries GeoJSON 파일은 다루기 너무 무거워서, State-level의 다른 shapefiles(cb_2018_us_state_500k)을 찾아 사용하였다. 아래 URL 참조. \
https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html


```python
# Load Point-styled CBG geometry
cbg_geo_lonlat = pd.read_csv(os.path.join(BasePath, SubDir[1], 'cbg_geographic_data.csv'))
# cbg_geo_lonlat['census_block_group'] = cbg_geo_lonlat['census_block_group'].apply(lambda x: f"{x:012d}") # 자릿수맞추기: CBG 코드는 12글자
cbg_geo_lonlat.head()
```

<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>census_block_group</th>
      <th>amount_land</th>
      <th>amount_water</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10010201001</td>
      <td>4264299</td>
      <td>28435</td>
      <td>32.465832</td>
      <td>-86.489661</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10010201002</td>
      <td>5561005</td>
      <td>0</td>
      <td>32.485873</td>
      <td>-86.489672</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10010202001</td>
      <td>2058374</td>
      <td>0</td>
      <td>32.480082</td>
      <td>-86.474974</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10010202002</td>
      <td>1262444</td>
      <td>5669</td>
      <td>32.464435</td>
      <td>-86.469766</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10010203001</td>
      <td>3866513</td>
      <td>9054</td>
      <td>32.480175</td>
      <td>-86.460792</td>
    </tr>
  </tbody>
</table>
</div>
<br>


```python
us_states = gpd.read_file('/open_census_data/cb_2018_us_state_500k')
us_states = us_states[~us_states['STUSPS'].isin(['PR', 'AK', 'HI', 'AS', 'VI', 'GU', 'MP'])]

fig, ax = plt.subplots(facecolor='w', figsize=(15, 15))
us_states.plot(ax=ax, facecolor='None', edgecolor='black', linewidth=.5)
ax.axis('off')
plt.show()
```

<br>
![png](/assets/img/post/safegraph/OpenCensusData_18_0.png)
<br>    
<br>


### Gender Population for each CBG
* table_id(male) = B01001e2
* table_id(female) = B01001e26


```python
cbg_fd_desc[cbg_fd_desc['table_id'].isin(['B01001e2', 'B01001e26'])]
```

<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>table_id</th>
      <th>table_number</th>
      <th>table_title</th>
      <th>table_topics</th>
      <th>table_universe</th>
      <th>field_level_1</th>
      <th>field_level_2</th>
      <th>field_level_3</th>
      <th>field_level_4</th>
      <th>field_level_5</th>
      <th>field_level_6</th>
      <th>field_level_7</th>
      <th>field_level_8</th>
      <th>field_level_9</th>
      <th>field_level_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>B01001e2</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>B01001e26</td>
      <td>B01001</td>
      <td>Sex By Age</td>
      <td>Age and Sex</td>
      <td>Total population</td>
      <td>Estimate</td>
      <td>SEX BY AGE</td>
      <td>Total population</td>
      <td>Total</td>
      <td>Female</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
<br>



```python
# table_number 앞 세자리를 따서 /data에서 파일을 찾아 불러온다.
BasePath = '/open_census_data/safegraph_open_census_data_2020'
print(os.path.join(BasePath, SubDir[0]))
cbg_b01 = pd.read_csv(os.path.join(BasePath, SubDir[0], 'cbg_b01.csv'))
cbg_b01.head()
```

    /open_census_data/safegraph_open_census_data_2020/data



<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>census_block_group</th>
      <th>B01001e1</th>
      <th>B01001m1</th>
      <th>B01001e2</th>
      <th>B01001m2</th>
      <th>B01001e3</th>
      <th>B01001m3</th>
      <th>B01001e4</th>
      <th>B01001m4</th>
      <th>B01001e5</th>
      <th>...</th>
      <th>B01002He3</th>
      <th>B01002Hm3</th>
      <th>B01002Ie1</th>
      <th>B01002Im1</th>
      <th>B01002Ie2</th>
      <th>B01002Im2</th>
      <th>B01002Ie3</th>
      <th>B01002Im3</th>
      <th>B01003e1</th>
      <th>B01003m1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10010201001</td>
      <td>674</td>
      <td>192</td>
      <td>284</td>
      <td>88</td>
      <td>12</td>
      <td>15</td>
      <td>17</td>
      <td>23</td>
      <td>5</td>
      <td>...</td>
      <td>37.5</td>
      <td>27.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>674</td>
      <td>192</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10010201002</td>
      <td>1267</td>
      <td>401</td>
      <td>694</td>
      <td>244</td>
      <td>49</td>
      <td>65</td>
      <td>80</td>
      <td>88</td>
      <td>45</td>
      <td>...</td>
      <td>35.0</td>
      <td>5.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1267</td>
      <td>401</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10010202001</td>
      <td>706</td>
      <td>200</td>
      <td>354</td>
      <td>142</td>
      <td>50</td>
      <td>38</td>
      <td>72</td>
      <td>69</td>
      <td>15</td>
      <td>...</td>
      <td>28.8</td>
      <td>4.2</td>
      <td>65.8</td>
      <td>37.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>65.8</td>
      <td>37.6</td>
      <td>706</td>
      <td>200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10010202002</td>
      <td>1051</td>
      <td>229</td>
      <td>656</td>
      <td>175</td>
      <td>31</td>
      <td>26</td>
      <td>5</td>
      <td>9</td>
      <td>7</td>
      <td>...</td>
      <td>42.3</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1051</td>
      <td>229</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10010203001</td>
      <td>2912</td>
      <td>565</td>
      <td>1461</td>
      <td>289</td>
      <td>22</td>
      <td>23</td>
      <td>134</td>
      <td>69</td>
      <td>200</td>
      <td>...</td>
      <td>35.7</td>
      <td>2.8</td>
      <td>28.0</td>
      <td>22.1</td>
      <td>27.6</td>
      <td>0.5</td>
      <td>28.9</td>
      <td>58.7</td>
      <td>2912</td>
      <td>565</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 161 columns</p>
</div>
<br>



```python
cbg_b01 = pd.merge(cbg_b01, cbg_geo_lonlat, on='census_block_group')
cbg_b01 = gpd.GeoDataFrame(cbg_b01, geometry=gpd.points_from_xy(cbg_b01.longitude, cbg_b01.latitude))
cbg_b01 = cbg_b01.set_crs(epsg=4269) # The EPSG of 'cb_2018_us_state_500k' is 4269. But note that EPSG of CBG_geojson is 4326.

# gpd.sjoin(how='left/right/inner/'): ‘inner’: use intersection of keys from both dfs; retain only left_df geometry column
cbg_b01 = gpd.sjoin(cbg_b01, us_states[['NAME', 'geometry']], how='inner') # Spatial Join based on the Lower 48 states 
```


```python
def alpha_with_SquareMinMaxScaling(values, min_alpha, max_alpha):
    """
    min_alpha : int or float
        minimum alpha for color
    max_alpha : int or float
        maximum alpha for color

    Returns
    -------
    Sequential list
        the alpha list for each value, the order must be controlled carefully.
    """
    alphas = []
    min_val = np.min(values)
    max_val = np.max(values)
    for v in values:
        alp = min_alpha + (max_alpha - min_alpha) * ((v - min_val) / (max_val - min_val)) ** 2
        alphas.append(alp)

    return alphas
```


```python
cbg_gender_pop = cbg_b01.loc[:, ['census_block_group', 'longitude', 'latitude', 'B01001e2', 'B01001e26']].rename(columns={'B01001e2':'male', 'B01001e26':'female'})

# Only Male
male_alps = alpha_with_SquareMinMaxScaling(cbg_gender_pop['male'].values, 0.01, 0.85)
fig, ax = plt.subplots(facecolor='w', figsize=(15, 15))
us_states.plot(ax=ax, facecolor='None', edgecolor='black', linewidth=.5)
ax.scatter(cbg_gender_pop['longitude'], cbg_gender_pop['latitude'], s=15, c='blue', alpha=male_alps)
ax.axis('off')
plt.show()
```

<br>
![png](/assets/img/post/safegraph/OpenCensusData_24_0.png)
<br> 


```python
# Only Female
female_alps = alpha_with_SquareMinMaxScaling(cbg_gender_pop['female'].values, 0.01, 0.85)
fig, ax = plt.subplots(facecolor='w', figsize=(15, 15))
us_states.plot(ax=ax, facecolor='None', edgecolor='black', linewidth=.5)
ax.scatter(cbg_gender_pop['longitude'], cbg_gender_pop['latitude'], s=15, c='red', alpha=female_alps)
ax.axis('off')
plt.show()
```


<br>
![png](/assets/img/post/safegraph/OpenCensusData_25_0.png)
<br>
<br> 


### The Number of House by Housing Value for each CBG
부동산 가치가 높은 지역과 낮은 지역이 어디인지 살펴본다. 임의의 금액 기준을 두고, 해당하는 field 인덱스들을 찾아 분류했다.
* $500,000 이상 table_id: B25075e23, B25075e24, B25075e25, B25075e26, B25075e27
* $50,000 미만 table_id: B25075e2, B25075e3, B25075e4, B25075e5, B25075e6, B25075e7, B25075e8, B25075e9


```python
cntHouse_fd_desc = cbg_fd_desc[cbg_fd_desc['table_title']=='Value'].reset_index(drop=True)
cntHouse_fd_desc
```


<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>table_id</th>
      <th>table_number</th>
      <th>table_title</th>
      <th>table_topics</th>
      <th>table_universe</th>
      <th>field_level_1</th>
      <th>field_level_2</th>
      <th>field_level_3</th>
      <th>field_level_4</th>
      <th>field_level_5</th>
      <th>field_level_6</th>
      <th>field_level_7</th>
      <th>field_level_8</th>
      <th>field_level_9</th>
      <th>field_level_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B25075e1</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B25075e10</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$50 000 to $59 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B25075e11</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$60 000 to $69 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B25075e12</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$70 000 to $79 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B25075e13</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$80 000 to $89 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>B25075e14</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$90 000 to $99 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>B25075e15</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$100 000 to $124 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>B25075e16</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$125 000 to $149 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>B25075e17</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$150 000 to $174 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>B25075e18</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$175 000 to $199 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>B25075e19</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$200 000 to $249 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>B25075e2</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>Less than $10 000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>B25075e20</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$250 000 to $299 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>B25075e21</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$300 000 to $399 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>B25075e22</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$400 000 to $499 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>B25075e23</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$500 000 to $749 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>B25075e24</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$750 000 to $999 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>B25075e25</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$1 000 000 to $1 499 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>B25075e26</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$1 500 000 to $1 999 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>B25075e27</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$2 000 000 or more</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>B25075e3</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$10 000 to $14 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>B25075e4</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$15 000 to $19 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>B25075e5</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$20 000 to $24 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>B25075e6</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$25 000 to $29 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>B25075e7</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$30 000 to $34 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>B25075e8</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$35 000 to $39 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>B25075e9</td>
      <td>B25075</td>
      <td>Value</td>
      <td>Housing Value and Purchase Price, Owner Renter...</td>
      <td>Owner-occupied housing units</td>
      <td>Estimate</td>
      <td>VALUE</td>
      <td>Owner-occupied housing units</td>
      <td>Total</td>
      <td>$40 000 to $49 999</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
<br>



```python
# table_number 앞 세자리를 따서 /data에서 파일을 찾아 불러온다.
BasePath = '/open_census_data/safegraph_open_census_data_2020'
print(os.path.join(BasePath, SubDir[0]))
cbg_b25 = pd.read_csv(os.path.join(BasePath, SubDir[0], 'cbg_b25.csv'))
cbg_b25.head()
```

    /open_census_data/safegraph_open_census_data_2020/data


<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>census_block_group</th>
      <th>B25001e1</th>
      <th>B25001m1</th>
      <th>B25002e1</th>
      <th>B25002m1</th>
      <th>B25002e2</th>
      <th>B25002m2</th>
      <th>B25002e3</th>
      <th>B25002m3</th>
      <th>B25003e1</th>
      <th>...</th>
      <th>B25093e25</th>
      <th>B25093m25</th>
      <th>B25093e26</th>
      <th>B25093m26</th>
      <th>B25093e27</th>
      <th>B25093m27</th>
      <th>B25093e28</th>
      <th>B25093m28</th>
      <th>B25093e29</th>
      <th>B25093m29</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10010201001</td>
      <td>290</td>
      <td>77</td>
      <td>290</td>
      <td>77</td>
      <td>290</td>
      <td>77</td>
      <td>0</td>
      <td>12</td>
      <td>290</td>
      <td>...</td>
      <td>7</td>
      <td>8</td>
      <td>18</td>
      <td>31</td>
      <td>15</td>
      <td>23</td>
      <td>12</td>
      <td>21</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10010201002</td>
      <td>420</td>
      <td>116</td>
      <td>420</td>
      <td>116</td>
      <td>403</td>
      <td>113</td>
      <td>17</td>
      <td>27</td>
      <td>403</td>
      <td>...</td>
      <td>0</td>
      <td>12</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>12</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10010202001</td>
      <td>284</td>
      <td>57</td>
      <td>284</td>
      <td>57</td>
      <td>227</td>
      <td>53</td>
      <td>57</td>
      <td>49</td>
      <td>227</td>
      <td>...</td>
      <td>3</td>
      <td>5</td>
      <td>11</td>
      <td>17</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10010202002</td>
      <td>436</td>
      <td>83</td>
      <td>436</td>
      <td>83</td>
      <td>346</td>
      <td>86</td>
      <td>90</td>
      <td>52</td>
      <td>346</td>
      <td>...</td>
      <td>17</td>
      <td>17</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>12</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10010203001</td>
      <td>1147</td>
      <td>185</td>
      <td>1147</td>
      <td>185</td>
      <td>1034</td>
      <td>185</td>
      <td>113</td>
      <td>89</td>
      <td>1034</td>
      <td>...</td>
      <td>0</td>
      <td>12</td>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>8</td>
      <td>22</td>
      <td>26</td>
      <td>0</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1741 columns</p>
</div>
<br>



```python
cbg_b25 = pd.merge(cbg_b25, cbg_geo_lonlat, on='census_block_group')
cbg_b25 = gpd.GeoDataFrame(cbg_b25, geometry=gpd.points_from_xy(cbg_b25.longitude, cbg_b25.latitude))
cbg_b25 = cbg_b25.set_crs(epsg=4269)
cbg_b25 = gpd.sjoin(cbg_b25, us_states[['NAME', 'geometry']], how='inner').reset_index(drop=True) # Spatial Join based on the Lower 48 states 
```


```python
cbg_b25.head()
```


<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>census_block_group</th>
      <th>B25001e1</th>
      <th>B25001m1</th>
      <th>B25002e1</th>
      <th>B25002m1</th>
      <th>B25002e2</th>
      <th>B25002m2</th>
      <th>B25002e3</th>
      <th>B25002m3</th>
      <th>B25003e1</th>
      <th>...</th>
      <th>B25093m28</th>
      <th>B25093e29</th>
      <th>B25093m29</th>
      <th>amount_land</th>
      <th>amount_water</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>geometry</th>
      <th>index_right</th>
      <th>NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10010201001</td>
      <td>290</td>
      <td>77</td>
      <td>290</td>
      <td>77</td>
      <td>290</td>
      <td>77</td>
      <td>0</td>
      <td>12</td>
      <td>290</td>
      <td>...</td>
      <td>21</td>
      <td>0</td>
      <td>12</td>
      <td>4264299</td>
      <td>28435</td>
      <td>32.465832</td>
      <td>-86.489661</td>
      <td>POINT (-86.48966 32.46583)</td>
      <td>17</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10010201002</td>
      <td>420</td>
      <td>116</td>
      <td>420</td>
      <td>116</td>
      <td>403</td>
      <td>113</td>
      <td>17</td>
      <td>27</td>
      <td>403</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>12</td>
      <td>5561005</td>
      <td>0</td>
      <td>32.485873</td>
      <td>-86.489672</td>
      <td>POINT (-86.48967 32.48587)</td>
      <td>17</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10010202001</td>
      <td>284</td>
      <td>57</td>
      <td>284</td>
      <td>57</td>
      <td>227</td>
      <td>53</td>
      <td>57</td>
      <td>49</td>
      <td>227</td>
      <td>...</td>
      <td>12</td>
      <td>0</td>
      <td>12</td>
      <td>2058374</td>
      <td>0</td>
      <td>32.480082</td>
      <td>-86.474974</td>
      <td>POINT (-86.47497 32.48008)</td>
      <td>17</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10010202002</td>
      <td>436</td>
      <td>83</td>
      <td>436</td>
      <td>83</td>
      <td>346</td>
      <td>86</td>
      <td>90</td>
      <td>52</td>
      <td>346</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>12</td>
      <td>1262444</td>
      <td>5669</td>
      <td>32.464435</td>
      <td>-86.469766</td>
      <td>POINT (-86.46977 32.46444)</td>
      <td>17</td>
      <td>Alabama</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10010203001</td>
      <td>1147</td>
      <td>185</td>
      <td>1147</td>
      <td>185</td>
      <td>1034</td>
      <td>185</td>
      <td>113</td>
      <td>89</td>
      <td>1034</td>
      <td>...</td>
      <td>26</td>
      <td>0</td>
      <td>12</td>
      <td>3866513</td>
      <td>9054</td>
      <td>32.480175</td>
      <td>-86.460792</td>
      <td>POINT (-86.46079 32.48018)</td>
      <td>17</td>
      <td>Alabama</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 1748 columns</p>
</div>
<br>



```python
HighMortValue = 'B25075e23, B25075e24, B25075e25, B25075e26, B25075e27'.split(', ')
LowMortValue = 'B25075e2, B25075e3, B25075e4, B25075e5, B25075e6, B25075e7, B25075e8, B25075e9'.split(', ') 
```


```python
cbg_cntHouse_value = cbg_b25.loc[:, ['census_block_group', 'latitude', 'longitude']]
cbg_cntHouse_value['HighHouseCnt'] = cbg_b25.loc[:, HighMortValue].sum(axis=1)
cbg_cntHouse_value['LowHouseCnt'] = cbg_b25.loc[:, LowMortValue].sum(axis=1)
```


```python
# The number of House with high value (more than $500,000)
high_alps = alpha_with_SquareMinMaxScaling(cbg_cntHouse_value['HighHouseCnt'].values, 0, 0.85)
fig, ax = plt.subplots(facecolor='w', figsize=(15, 15))
us_states.plot(ax=ax, facecolor='None', edgecolor='black', linewidth=.5)
ax.scatter(cbg_cntHouse_value['longitude'], cbg_cntHouse_value['latitude'], s=15, c='blue', alpha=high_alps)
ax.axis('off')
plt.show()
```


<br>    
![png](/assets/img/post/safegraph/OpenCensusData_33_0.png)
<br> 



```python
# The number of House with low value (lower than $50,000)
low_alps = alpha_with_SquareMinMaxScaling(cbg_cntHouse_value['LowHouseCnt'].values, 0, 0.85)
fig, ax = plt.subplots(facecolor='w', figsize=(15, 15))
us_states.plot(ax=ax, facecolor='None', edgecolor='black', linewidth=.5)
ax.scatter(cbg_cntHouse_value['longitude'], cbg_cntHouse_value['latitude'], s=15, c='red', alpha=low_alps)
ax.axis('off')
plt.show()
```


<br>
![png](/assets/img/post/safegraph/OpenCensusData_34_0.png)
<br>
<br>



### Educational Attainment for only US citizens 18 years and over
교육 수준이 높고, 낮은 인구가 미국 어디에 몰려있는지 살펴본다.
* 초중고 중퇴 및 고졸 table_id: B29002e2, B29002e3, B29002e4
* 전문대 졸(Associate's degree), 일반대 졸(Bachelor's degree) 및 석사이상 table_id: B29002e6, B29002e7, B29002e8


```python
# table_number 앞 세자리를 따서 /data에서 파일을 찾아 불러온다.
BasePath = '/open_census_data/safegraph_open_census_data_2020'
print(os.path.join(BasePath, SubDir[0]))
cbg_b29 = pd.read_csv(os.path.join(BasePath, SubDir[0], 'cbg_b29.csv'))
cbg_b29.head()
```

    /open_census_data/safegraph_open_census_data_2020/data


<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>census_block_group</th>
      <th>B29001e1</th>
      <th>B29001m1</th>
      <th>B29001e2</th>
      <th>B29001m2</th>
      <th>B29001e3</th>
      <th>B29001m3</th>
      <th>B29001e4</th>
      <th>B29001m4</th>
      <th>B29001e5</th>
      <th>...</th>
      <th>B29002e8</th>
      <th>B29002m8</th>
      <th>B29003e1</th>
      <th>B29003m1</th>
      <th>B29003e2</th>
      <th>B29003m2</th>
      <th>B29003e3</th>
      <th>B29003m3</th>
      <th>B29004e1</th>
      <th>B29004m1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10010201001</td>
      <td>574</td>
      <td>161</td>
      <td>102</td>
      <td>81</td>
      <td>120</td>
      <td>79</td>
      <td>226</td>
      <td>77</td>
      <td>126</td>
      <td>...</td>
      <td>55</td>
      <td>35</td>
      <td>574</td>
      <td>161</td>
      <td>72</td>
      <td>63</td>
      <td>502</td>
      <td>164</td>
      <td>39167.0</td>
      <td>20140.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10010201002</td>
      <td>948</td>
      <td>256</td>
      <td>163</td>
      <td>80</td>
      <td>298</td>
      <td>130</td>
      <td>322</td>
      <td>134</td>
      <td>165</td>
      <td>...</td>
      <td>76</td>
      <td>34</td>
      <td>948</td>
      <td>256</td>
      <td>100</td>
      <td>70</td>
      <td>848</td>
      <td>245</td>
      <td>70699.0</td>
      <td>11633.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10010202001</td>
      <td>458</td>
      <td>121</td>
      <td>89</td>
      <td>52</td>
      <td>143</td>
      <td>67</td>
      <td>103</td>
      <td>40</td>
      <td>123</td>
      <td>...</td>
      <td>7</td>
      <td>9</td>
      <td>458</td>
      <td>121</td>
      <td>106</td>
      <td>66</td>
      <td>352</td>
      <td>108</td>
      <td>39750.0</td>
      <td>20003.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10010202002</td>
      <td>974</td>
      <td>211</td>
      <td>289</td>
      <td>143</td>
      <td>223</td>
      <td>83</td>
      <td>301</td>
      <td>73</td>
      <td>161</td>
      <td>...</td>
      <td>37</td>
      <td>27</td>
      <td>762</td>
      <td>201</td>
      <td>39</td>
      <td>37</td>
      <td>723</td>
      <td>204</td>
      <td>50221.0</td>
      <td>3210.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10010203001</td>
      <td>2045</td>
      <td>413</td>
      <td>317</td>
      <td>131</td>
      <td>730</td>
      <td>234</td>
      <td>680</td>
      <td>221</td>
      <td>318</td>
      <td>...</td>
      <td>172</td>
      <td>98</td>
      <td>2045</td>
      <td>413</td>
      <td>170</td>
      <td>93</td>
      <td>1875</td>
      <td>412</td>
      <td>66843.0</td>
      <td>10424.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>
<br>



```python
target_table = 'B29002e2, B29002e3, B29002e4, B29002e6, B29002e7, B29002e8'.split(', ')
target_table
```



    ['B29002e2', 'B29002e3', 'B29002e4', 'B29002e6', 'B29002e7', 'B29002e8']




```python
cbg_fd_desc[cbg_fd_desc['table_id'].isin(target_table)].reset_index(drop=True)
```


<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>table_id</th>
      <th>table_number</th>
      <th>table_title</th>
      <th>table_topics</th>
      <th>table_universe</th>
      <th>field_level_1</th>
      <th>field_level_2</th>
      <th>field_level_3</th>
      <th>field_level_4</th>
      <th>field_level_5</th>
      <th>field_level_6</th>
      <th>field_level_7</th>
      <th>field_level_8</th>
      <th>field_level_9</th>
      <th>field_level_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>B29002e2</td>
      <td>B29002</td>
      <td>Citizen, Voting-Age Population By Educational ...</td>
      <td>Age and Sex, Citizenship, Educational Attainment</td>
      <td>Citizens 18 years and over</td>
      <td>Estimate</td>
      <td>CITIZEN, VOTING-AGE POPULATION BY EDUCATIONAL ...</td>
      <td>Citizens 18 years and over</td>
      <td>Total</td>
      <td>Less than 9th grade</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B29002e3</td>
      <td>B29002</td>
      <td>Citizen, Voting-Age Population By Educational ...</td>
      <td>Age and Sex, Citizenship, Educational Attainment</td>
      <td>Citizens 18 years and over</td>
      <td>Estimate</td>
      <td>CITIZEN, VOTING-AGE POPULATION BY EDUCATIONAL ...</td>
      <td>Citizens 18 years and over</td>
      <td>Total</td>
      <td>9th to 12th grade no diploma</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B29002e4</td>
      <td>B29002</td>
      <td>Citizen, Voting-Age Population By Educational ...</td>
      <td>Age and Sex, Citizenship, Educational Attainment</td>
      <td>Citizens 18 years and over</td>
      <td>Estimate</td>
      <td>CITIZEN, VOTING-AGE POPULATION BY EDUCATIONAL ...</td>
      <td>Citizens 18 years and over</td>
      <td>Total</td>
      <td>High school graduate (includes equivalency)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>B29002e6</td>
      <td>B29002</td>
      <td>Citizen, Voting-Age Population By Educational ...</td>
      <td>Age and Sex, Citizenship, Educational Attainment</td>
      <td>Citizens 18 years and over</td>
      <td>Estimate</td>
      <td>CITIZEN, VOTING-AGE POPULATION BY EDUCATIONAL ...</td>
      <td>Citizens 18 years and over</td>
      <td>Total</td>
      <td>Associate's degree</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B29002e7</td>
      <td>B29002</td>
      <td>Citizen, Voting-Age Population By Educational ...</td>
      <td>Age and Sex, Citizenship, Educational Attainment</td>
      <td>Citizens 18 years and over</td>
      <td>Estimate</td>
      <td>CITIZEN, VOTING-AGE POPULATION BY EDUCATIONAL ...</td>
      <td>Citizens 18 years and over</td>
      <td>Total</td>
      <td>Bachelor's degree</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>B29002e8</td>
      <td>B29002</td>
      <td>Citizen, Voting-Age Population By Educational ...</td>
      <td>Age and Sex, Citizenship, Educational Attainment</td>
      <td>Citizens 18 years and over</td>
      <td>Estimate</td>
      <td>CITIZEN, VOTING-AGE POPULATION BY EDUCATIONAL ...</td>
      <td>Citizens 18 years and over</td>
      <td>Total</td>
      <td>Graduate or professional degree</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
<br>



```python
cbg_b29 = pd.merge(cbg_b29, cbg_geo_lonlat, on='census_block_group')
cbg_b29 = gpd.GeoDataFrame(cbg_b29, geometry=gpd.points_from_xy(cbg_b29.longitude, cbg_b29.latitude))
cbg_b29 = cbg_b29.set_crs(epsg=4269)
cbg_b29 = gpd.sjoin(cbg_b29, us_states[['NAME', 'geometry']], how='inner').reset_index(drop=True) # Spatial Join based on the Lower 48 states 
```


```python
HighEdu = 'B29002e6, B29002e7, B29002e8'.split(', ')
LowEdu = 'B29002e2, B29002e3, B29002e4'.split(', ')

cbg_cntPop_edu = cbg_b29.loc[:, ['census_block_group', 'latitude', 'longitude']]
cbg_cntPop_edu['HighEduPop'] = cbg_b29.loc[:, HighEdu].sum(axis=1)
cbg_cntPop_edu['LowEduPop'] = cbg_b29.loc[:, LowEdu].sum(axis=1)
```


```python
# The number of House with high value (more than $500,000)
high_alps = alpha_with_SquareMinMaxScaling(cbg_cntPop_edu['HighEduPop'].values, 0, 0.85)
low_alps = alpha_with_SquareMinMaxScaling(cbg_cntPop_edu['LowEduPop'].values, 0, 0.85)

fig, axs = plt.subplots(nrows=1, ncols=2, facecolor='w', figsize=(15, 15))
us_states.plot(ax=axs[0], facecolor='None', edgecolor='black', linewidth=.5)
us_states.plot(ax=axs[1], facecolor='None', edgecolor='black', linewidth=.5)

axs[0].scatter(cbg_cntPop_edu['longitude'], cbg_cntPop_edu['latitude'], s=15, c='blue', alpha=high_alps)
axs[1].scatter(cbg_cntPop_edu['longitude'], cbg_cntPop_edu['latitude'], s=15, c='red', alpha=low_alps)

axs[0].axis('off')
axs[1].axis('off')
fig.subplots_adjust(wspace=.1)
plt.show()
```


<br>
![png](/assets/img/post/safegraph/OpenCensusData_41_0.png)
<br>
<br>



## Take-Home Message and Discussion
- SafeGraph Inc.의 데이터 중 Open Census Data란 것을 살펴보았다.
- Census Block Group(CBG)라는 공간 스케일을 사용하고 있다.
- US Census Bureau가 매년 조사를 수행하는 American Community Survey(ACS)자료를 기반으로 한 데이터이다.
- 사용자로 하여금 ACS 자료 활용이 용이하게끔 하자는 것이 SafeGraph's Open Census Data의 제작 취지이다.
- 인구수 뿐 아니라 지역별 소득수준, 교육수준 등을 추정할 수 있는 다양한 정보들이 포함되어 있다.
- SafeGraph Inc.는 이 외에도 카드소비데이터 - 'Spend' 데이터, 전세계 매장정보 - 'Places' 데이터를 배포하고 있다. 하지만 매우 안타깝게도 해당 데이터 접근은 유료 구독형 서비스라 이 글에선 다루지 못하였다...

***fin***