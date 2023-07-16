---
title: "핀란드 헬싱키 생활인구 데이터 분석"
date: 2023-05-02 15:29:15 +0900
categories: [Open-Data, EDA]
tags: [python, population, visualization, eda, finland, helsinki]
math: true
---

# 들어가며
이전 [헬싱키 통행시간 데이터 분석](https://zoshs2.github.io/posts/HelsinkiTT/){:target="_blank"} 글을 정리하면서, 해당 연구팀이 추후 헬싱키 데이터에 대한 또 다른 논문과 데이터를 배포했다고 잠깐 언급한 바 있다. 오늘은 이건 또 어떤 데이터인지 한번 뜯어볼 계획이다. 이번 데이터셋 관련 상세한 내용은 [이 링크](https://www.nature.com/articles/s41597-021-01113-4){:target="_blank"}의 논문을 참고하면 된다.

본 연구팀은 **Elisa Oyj** 라는 핀란드 통신사로부터 **HSPA(3.5세대 통신망) 통화 기록 데이터**를 제공받아 가공하여 생활인구 데이터를 추출했다. 따라서, 본 생활인구는 핀란드 실거주민 뿐 아니라 외국 관광객, 지역외 주민 등을 포함한다.
- c.f.) HSPA(High-Speed Packet Access)는 글로벌 통신망 프로토콜(규격)이다. "업/다운로드 속도가 이 정도되면 HSPA(3.5세대)라 한다." 같은.
- e.g.) 통신망 프로토콜의 변천사: GSM(2G) >> UMTS(3G) >> HSPA(3.5G) >> HSPA+(3.5G보다는 빠른 수준) >> LTE(3.9G) >> LTE-Advanced(LTE+ 혹은 LTE-A; 4G) >> NR(New Radio; 5G)

# Dynamical Population in Helsinki, Finland

```python
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
```

## Data Acquisition
* 터미널에서 wget URL을 통해 데이터셋을 다운받을 수 있다.
```bash
>>> wget https://zenodo.org/record/4726996/files/Helsinki_dynpop_matrix.zip?download=1
>>> unzip <downloaded_file> -d <dir_name_after_unzip>
```
* 내려받은 데이터의 종류는 아래와 같다. 
  * _workdays.csv: Monday ~ Thursday 
  * _sat.csv: Saturdays only 
 
```python
dataset/
├── [2.5M]  HMA_Dynamic_population_24H_sat.csv
├── [2.5M]  HMA_Dynamic_population_24H_sun.csv
├── [2.5M]  HMA_Dynamic_population_24H_workdays.csv
├── [3.1K]  README.txt
└── [4.8M]  target_zones_grid250m_EPSG3067.geojson
```

```python
DataPath = '/home/ygkwon/helsinki/Helsinki_dynpop_matrix'
DataContents = [file for file in os.listdir(DataPath)]
print(DataContents)

    ['target_zones_grid250m_EPSG3067.geojson', 'HMA_Dynamic_population_24H_workdays.csv', 'HMA_Dynamic_population_24H_sat.csv', 'HMA_Dynamic_population_24H_sun.csv', 'README.txt', '.README.txt.swp']
```

## Geojson of Helsinki
"target_zones_grid250m_EPSG3067.geojson"
- Attributes
  - YKR_ID: 핀란드어 yhdyskuntarakenteen seurantajärjestelmä의 약자... (핀란드 정부에서 정의한 지질학적 그리드 ID)
  - geometry: geometrical POLYGON (EPSG: 3067 / Cartesian axis: East(metre) + North(metre) / Ellipsoid: GRS 1980)
- 13,231개의 독립적인 YKR_ID(grid) 가 존재

```python
geojson_df = gpd.read_file(os.path.join(DataPath, DataContents[0]))
geojson_df
```

<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YKR_ID</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5785640</td>
      <td>POLYGON ((382000.00014 6697750.00013, 381750.0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5785641</td>
      <td>POLYGON ((382250.00014 6697750.00013, 382000.0...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5785642</td>
      <td>POLYGON ((382500.00014 6697750.00013, 382250.0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5785643</td>
      <td>POLYGON ((382750.00014 6697750.00013, 382500.0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5787544</td>
      <td>POLYGON ((381250.00014 6697500.00013, 381000.0...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13226</th>
      <td>6016698</td>
      <td>POLYGON ((373000.00014 6665500.00013, 372750.0...</td>
    </tr>
    <tr>
      <th>13227</th>
      <td>6016699</td>
      <td>POLYGON ((373250.00014 6665500.00013, 373000.0...</td>
    </tr>
    <tr>
      <th>13228</th>
      <td>6018252</td>
      <td>POLYGON ((372500.00014 6665250.00013, 372250.0...</td>
    </tr>
    <tr>
      <th>13229</th>
      <td>6018253</td>
      <td>POLYGON ((372750.00014 6665250.00013, 372500.0...</td>
    </tr>
    <tr>
      <th>13230</th>
      <td>6018254</td>
      <td>POLYGON ((373000.00014 6665250.00013, 372750.0...</td>
    </tr>
  </tbody>
</table>
<p>13231 rows × 2 columns</p>
</div>
<br>


```python
fig, ax = plt.subplots(facecolor='w', figsize=(14, 8))
geojson_df.plot(ax=ax, color='w', edgecolor='black', linewidth=0.2)
ax.axis('off')
plt.show()
```

<br>
![png](/assets/img/post/helsinki_pop/helsink_dynpop_6_0.png)
<br> 


## Helsinki population
- Divided into Workdays(Mon~Thu) / Sat / Sun, seperately.
- 두달 반 가량의 study period 에 대해 aggregate 한 데이터.
- Attributes: [YKR_ID, H0, H1, ..., H23]
  - Hx: x시 ~ x+1시 사이의 Population ratio
- Population ratio는 특정 시점의 헬싱키 내 관측 인구 수를 100% 라 보았을 때, 각 YKR grid에서 100% 중 몇 %가 있는지를 나타낸 값이다.
- 논문에 따르면 Population ratio을 추산한 대략적인 과정은 아래와 같다.
    - (1) 두달 반 가량의 기간 동안, 각 Base Station(BS; 기지국)마다, 해당 기지국에 잡힌 HSPA calls(통화 기록)를 수집한다.
    - (2) Workdays / Sat / Sun 마다 'median' of HSPA calls를 구해서 BS에 할당한다. (이미 여기서 aggregate 됨)
    - (3) BS 위치에 대해 Voronoi Tessellation을 진행하여 각 BS의 영역권을 구한다.
    - (4) YKR 그리드를 바닥에 깐다.
    - (5) Voronoi 영역에 'grid가 차지하는 비율'로 BS가 가진 calls를 분배한다. (13,231개의 모든 YKR grid가 각각의 calls을 나눠 할당받게 된다)
    - (6) 여기까지가 그저그런 정확도를 지닌 일반적인 interpolation 과정이지만, 본 연구팀을 더욱 정밀한 가공을 위해 아래와 같은 추가적인 후처리를 진행한다.
    - (7) 여러 Meta 정보들을 여기저기서 가져와서, 한번 더 grid 내 calls를 재할당한다. (MFD interpolation; 자세히 알고싶으면 Järv et al. 2017 참조)
    - (8) 이렇게 시간마다, 그리고 YKR grid마다 calls수가 할당되어있다.
    - (9) 특정 시간 시점마다 grid의 calls를 '전체 calls 수 대비 grid 내 calls 수'란 비율로 전환한다. 
    - (10) 이게 H0, H1, ..., H23 에 들어있는 값이다. (= Population Ratio; 특정 시점에 대해 13,231개 pop을 전부 합하면 **약 100(%)**이 된다.)

```python
# 본 작업에서는 workdays 집계 데이터만 살펴보았다.
workday_pop = pd.read_csv(os.path.join(DataPath, DataContents[1]))
workday_pop
```

<br>
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YKR_ID</th>
      <th>H0</th>
      <th>H1</th>
      <th>H2</th>
      <th>H3</th>
      <th>H4</th>
      <th>H5</th>
      <th>H6</th>
      <th>H7</th>
      <th>H8</th>
      <th>...</th>
      <th>H14</th>
      <th>H15</th>
      <th>H16</th>
      <th>H17</th>
      <th>H18</th>
      <th>H19</th>
      <th>H20</th>
      <th>H21</th>
      <th>H22</th>
      <th>H23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5785640</td>
      <td>0.00083</td>
      <td>0.00078</td>
      <td>0.00085</td>
      <td>0.00082</td>
      <td>0.00075</td>
      <td>0.00102</td>
      <td>0.00126</td>
      <td>0.00149</td>
      <td>0.00124</td>
      <td>...</td>
      <td>0.00134</td>
      <td>0.00156</td>
      <td>0.00185</td>
      <td>0.00162</td>
      <td>0.00145</td>
      <td>0.00133</td>
      <td>0.00116</td>
      <td>0.00103</td>
      <td>0.00089</td>
      <td>0.00079</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5785641</td>
      <td>0.00185</td>
      <td>0.00174</td>
      <td>0.00182</td>
      <td>0.00177</td>
      <td>0.00170</td>
      <td>0.00219</td>
      <td>0.00250</td>
      <td>0.00254</td>
      <td>0.00207</td>
      <td>...</td>
      <td>0.00228</td>
      <td>0.00251</td>
      <td>0.00287</td>
      <td>0.00286</td>
      <td>0.00273</td>
      <td>0.00267</td>
      <td>0.00255</td>
      <td>0.00222</td>
      <td>0.00202</td>
      <td>0.00184</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5785642</td>
      <td>0.00518</td>
      <td>0.00481</td>
      <td>0.00489</td>
      <td>0.00477</td>
      <td>0.00466</td>
      <td>0.00593</td>
      <td>0.00638</td>
      <td>0.00580</td>
      <td>0.00479</td>
      <td>...</td>
      <td>0.00580</td>
      <td>0.00600</td>
      <td>0.00660</td>
      <td>0.00747</td>
      <td>0.00751</td>
      <td>0.00763</td>
      <td>0.00753</td>
      <td>0.00637</td>
      <td>0.00580</td>
      <td>0.00529</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5785643</td>
      <td>0.00561</td>
      <td>0.00524</td>
      <td>0.00531</td>
      <td>0.00520</td>
      <td>0.00512</td>
      <td>0.00642</td>
      <td>0.00687</td>
      <td>0.00614</td>
      <td>0.00499</td>
      <td>...</td>
      <td>0.00587</td>
      <td>0.00610</td>
      <td>0.00671</td>
      <td>0.00762</td>
      <td>0.00767</td>
      <td>0.00782</td>
      <td>0.00783</td>
      <td>0.00670</td>
      <td>0.00624</td>
      <td>0.00572</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5787544</td>
      <td>0.00088</td>
      <td>0.00086</td>
      <td>0.00093</td>
      <td>0.00092</td>
      <td>0.00089</td>
      <td>0.00109</td>
      <td>0.00128</td>
      <td>0.00127</td>
      <td>0.00086</td>
      <td>...</td>
      <td>0.00062</td>
      <td>0.00075</td>
      <td>0.00091</td>
      <td>0.00084</td>
      <td>0.00076</td>
      <td>0.00074</td>
      <td>0.00081</td>
      <td>0.00082</td>
      <td>0.00088</td>
      <td>0.00083</td>
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
      <th>13226</th>
      <td>6016698</td>
      <td>0.00047</td>
      <td>0.00045</td>
      <td>0.00053</td>
      <td>0.00053</td>
      <td>0.00051</td>
      <td>0.00046</td>
      <td>0.00040</td>
      <td>0.00029</td>
      <td>0.00021</td>
      <td>...</td>
      <td>0.00018</td>
      <td>0.00017</td>
      <td>0.00018</td>
      <td>0.00020</td>
      <td>0.00023</td>
      <td>0.00025</td>
      <td>0.00028</td>
      <td>0.00026</td>
      <td>0.00033</td>
      <td>0.00039</td>
    </tr>
    <tr>
      <th>13227</th>
      <td>6016699</td>
      <td>0.00018</td>
      <td>0.00017</td>
      <td>0.00020</td>
      <td>0.00020</td>
      <td>0.00020</td>
      <td>0.00017</td>
      <td>0.00015</td>
      <td>0.00010</td>
      <td>0.00007</td>
      <td>...</td>
      <td>0.00004</td>
      <td>0.00004</td>
      <td>0.00005</td>
      <td>0.00006</td>
      <td>0.00007</td>
      <td>0.00007</td>
      <td>0.00009</td>
      <td>0.00009</td>
      <td>0.00013</td>
      <td>0.00015</td>
    </tr>
    <tr>
      <th>13228</th>
      <td>6018252</td>
      <td>0.00021</td>
      <td>0.00020</td>
      <td>0.00023</td>
      <td>0.00023</td>
      <td>0.00022</td>
      <td>0.00020</td>
      <td>0.00018</td>
      <td>0.00013</td>
      <td>0.00009</td>
      <td>...</td>
      <td>0.00008</td>
      <td>0.00007</td>
      <td>0.00008</td>
      <td>0.00009</td>
      <td>0.00010</td>
      <td>0.00011</td>
      <td>0.00012</td>
      <td>0.00011</td>
      <td>0.00015</td>
      <td>0.00017</td>
    </tr>
    <tr>
      <th>13229</th>
      <td>6018253</td>
      <td>0.00018</td>
      <td>0.00018</td>
      <td>0.00021</td>
      <td>0.00021</td>
      <td>0.00020</td>
      <td>0.00018</td>
      <td>0.00016</td>
      <td>0.00011</td>
      <td>0.00008</td>
      <td>...</td>
      <td>0.00006</td>
      <td>0.00006</td>
      <td>0.00006</td>
      <td>0.00007</td>
      <td>0.00009</td>
      <td>0.00009</td>
      <td>0.00011</td>
      <td>0.00010</td>
      <td>0.00013</td>
      <td>0.00015</td>
    </tr>
    <tr>
      <th>13230</th>
      <td>6018254</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>...</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
<p>13231 rows × 25 columns</p>
</div>
<br>


### A Snaptime for workdays dataset, 08:00 ~ 08:59
- 08:00 ~ 08:59 = a column of 'H8'

```python
# 13,231개의 YKR grid에 대한 H8 시점의 population ratio 분포
snap_pop = workday_pop.loc[:, ['YKR_ID', 'H8']]
snap_pop = snap_pop.rename(columns={'H8':'pop'})
snap_merge = pd.merge(geojson_df, snap_pop, on='YKR_ID')

fig, ax = plt.subplots(facecolor='w', figsize=(7, 5))
ax.hist(snap_merge['pop'], bins=30, color='blue')
ax.set_ylabel("Count", fontsize=13)
ax.set_xlabel("Population Ratio", fontsize=13)
ax.set_yscale('log')
ax.set_title("Workdays, 08:00 ~ 08:59 (H8)", fontsize=15)
plt.show()
```

<br>
![png](/assets/img/post/helsinki_pop/helsink_dynpop_10_0.png)
<br> 


```python
# Spatial Distribution of population ratio
cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["navy", "darkolivegreen", "olive", "goldenrod", "gold", "yellow", "orange", "orangered", "red"])

manual_symbol = mpl.lines.Line2D([0], [0], label='non-existed', marker='s', markersize=10, markeredgecolor='black', markerfacecolor='black', linestyle='')
nonzero_pop_merge = snap_merge[snap_merge['pop']!=0].reset_index(drop=True)

fig, ax = plt.subplots(facecolor='w', figsize=(14, 8))
snap_merge.plot(ax=ax, edgecolor='grey', linewidth=.4, color='black')
nonzero_pop_merge.plot(column='pop', ax=ax, edgecolor='grey', linewidth=0.2, cmap=cmap)
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=nonzero_pop_merge['pop'].min(), vmax=nonzero_pop_merge['pop'].max()), cmap=cmap)
cbaxes = fig.add_axes([0.76, 0.2, 0.01, 0.4])
cbar = fig.colorbar(sm, cax=cbaxes)
cbar.ax.get_yaxis().labelpad = 17
cbar.ax.set_ylabel('Population Ratio', rotation=270, fontsize=15)

ax.legend(loc='upper left', handles=[manual_symbol], prop={'size': 15})

ax.set_title("Workdays, 08:00 ~ 08:59 (H8)", x=.55, y=1.03, fontsize=20)
ax.axis('off')
plt.show()
```

<br>
![png](/assets/img/post/helsinki_pop/helsink_dynpop_11_0.png)
<br> 


### 24h for workdays dataset
- Circadian change of spatial distribution of population ratio.

```python
# Default requirements
timeset = [f"H{time}"for time in range(24)]
cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["navy", "darkolivegreen", "olive", "goldenrod", "gold", "yellow", "orange", "orangered", "red"])
manual_symbol = mpl.lines.Line2D([0], [0], label='non-existed', marker='s', markersize=10, markeredgecolor='black', markerfacecolor='black', linestyle='')
merge_pop = pd.merge(geojson_df, workday_pop, on='YKR_ID')
total_cnt = merge_pop.shape[0]

# Plot for loop
fig, axs = plt.subplots(nrows=6, ncols=4, facecolor='w', figsize=(60, 72))
for time_col, ax in zip(timeset, axs.flatten()):
    snap_pop = merge_pop.loc[:, ['YKR_ID', 'geometry', time_col]]
    nonzero_snap_pop = snap_pop[snap_pop[time_col]!=0].reset_index(drop=True)
    if not nonzero_snap_pop.shape[0] == total_cnt:
        snap_pop.plot(ax=ax, edgecolor='grey', linewidth=.4, color='black')
        ax.legend(loc='upper left', handles=[manual_symbol], prop={'size': 15})

    nonzero_snap_pop.plot(column=time_col, ax=ax, edgecolor='grey', linewidth=.2, cmap=cmap)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=nonzero_snap_pop[time_col].min(), vmax=nonzero_snap_pop[time_col].max()), cmap=cmap)
    cbaxes = ax.inset_axes([0.55, 0.07, 0.5, 0.02])
    cbar = fig.colorbar(sm, cax=cbaxes, orientation='horizontal', ticks=None)
    time_unit = time_col.split('H')[1]
    ax.set_title(f"Workdays, {time_unit}:00 ~ {time_unit}:59 ({time_col})", x=.55, y=1.03, fontsize=14)
    ax.axis('off')

plt.subplots_adjust(wspace=.1, hspace=.3)
plt.show()
```

<br> 
![png](/assets/img/post/helsinki_pop/helsink_dynpop_13_0.png)
<br> 

## Take-Home Message and Discussion
- 핀란드의 수도 헬싱키의 생활인구'비율' 데이터를 살펴보았다.
- 두달 반간의 통화기록 raw dataset을 기반으로 시간 단위(1h resoultion)로 집계 및 가공한 데이터이다.
- Workdays(월~목) / Sat / Sun별로 집계 및 가공되어 공개 배포하고 있다.
- YKR Grid 라는 핀란드 정부에서 정의한 250m by 250m 크기의 grid를 사용하고 있다. 
- 헬싱키에 속하는 YKR Grid는 총 13,231개이다.

***fin***