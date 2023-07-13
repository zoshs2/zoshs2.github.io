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



<script type="text/javascript">
<!--

{
	var element = document.getElementById('NetworKit_script');
	if (element) {
		element.parentNode.removeChild(element);
	}
	element = document.createElement('script');
	element.type = 'text/javascript';
	element.innerHTML = 'function NetworKit_pageEmbed(id) { var i, j; var elements; elements = document.getElementById(id).getElementsByClassName("Plot"); for (i=0; i<elements.length; i++) { elements[i].id = id + "_Plot_" + i; var data = elements[i].getAttribute("data-image").split("|"); elements[i].removeAttribute("data-image"); var content = "<div class=\\"Image\\" id=\\"" + elements[i].id + "_Image\\" />"; elements[i].innerHTML = content; elements[i].setAttribute("data-image-index", 0); elements[i].setAttribute("data-image-length", data.length); for (j=0; j<data.length; j++) { elements[i].setAttribute("data-image-" + j, data[j]); } NetworKit_plotUpdate(elements[i]); elements[i].onclick = function (e) { NetworKit_overlayShow((e.target) ? e.target : e.srcElement); } } elements = document.getElementById(id).getElementsByClassName("HeatCell"); for (i=0; i<elements.length; i++) { var data = parseFloat(elements[i].getAttribute("data-heat")); var color = "#00FF00"; if (data <= 1 && data > 0) { color = "hsla(0, 100%, 75%, " + (data) + ")"; } else if (data <= 0 && data >= -1) { color = "hsla(240, 100%, 75%, " + (-data) + ")"; } elements[i].style.backgroundColor = color; } elements = document.getElementById(id).getElementsByClassName("Details"); for (i=0; i<elements.length; i++) { elements[i].setAttribute("data-title", "-"); NetworKit_toggleDetails(elements[i]); elements[i].onclick = function (e) { NetworKit_toggleDetails((e.target) ? e.target : e.srcElement); } } elements = document.getElementById(id).getElementsByClassName("MathValue"); for (i=elements.length-1; i>=0; i--) { value = elements[i].innerHTML.trim(); if (value === "nan") { elements[i].parentNode.innerHTML = "" } } elements = document.getElementById(id).getElementsByClassName("SubCategory"); for (i=elements.length-1; i>=0; i--) { value = elements[i].innerHTML.trim(); if (value === "") { elements[i].parentNode.removeChild(elements[i]) } } elements = document.getElementById(id).getElementsByClassName("Category"); for (i=elements.length-1; i>=0; i--) { value = elements[i].innerHTML.trim(); if (value === "") { elements[i].parentNode.removeChild(elements[i]) } } var isFirefox = false; try { isFirefox = typeof InstallTrigger !== "undefined"; } catch (e) {} if (!isFirefox) { alert("Currently the function\'s output is only fully supported by Firefox."); } } function NetworKit_plotUpdate(source) { var index = source.getAttribute("data-image-index"); var data = source.getAttribute("data-image-" + index); var image = document.getElementById(source.id + "_Image"); image.style.backgroundImage = "url(" + data + ")"; } function NetworKit_showElement(id, show) { var element = document.getElementById(id); element.style.display = (show) ? "block" : "none"; } function NetworKit_overlayShow(source) { NetworKit_overlayUpdate(source); NetworKit_showElement("NetworKit_Overlay", true); } function NetworKit_overlayUpdate(source) { document.getElementById("NetworKit_Overlay_Title").innerHTML = source.title; var index = source.getAttribute("data-image-index"); var data = source.getAttribute("data-image-" + index); var image = document.getElementById("NetworKit_Overlay_Image"); image.setAttribute("data-id", source.id); image.style.backgroundImage = "url(" + data + ")"; var link = document.getElementById("NetworKit_Overlay_Toolbar_Bottom_Save"); link.href = data; link.download = source.title + ".svg"; } function NetworKit_overlayImageShift(delta) { var image = document.getElementById("NetworKit_Overlay_Image"); var source = document.getElementById(image.getAttribute("data-id")); var index = parseInt(source.getAttribute("data-image-index")); var length = parseInt(source.getAttribute("data-image-length")); var index = (index+delta) % length; if (index < 0) { index = length + index; } source.setAttribute("data-image-index", index); NetworKit_overlayUpdate(source); } function NetworKit_toggleDetails(source) { var childs = source.children; var show = false; if (source.getAttribute("data-title") == "-") { source.setAttribute("data-title", "+"); show = false; } else { source.setAttribute("data-title", "-"); show = true; } for (i=0; i<childs.length; i++) { if (show) { childs[i].style.display = "block"; } else { childs[i].style.display = "none"; } } }';
	element.setAttribute('id', 'NetworKit_script');
	document.head.appendChild(element);
}


{
	var element = document.getElementById('NetworKit_style');
	if (element) {
		element.parentNode.removeChild(element);
	}
	element = document.createElement('style');
	element.type = 'text/css';
	element.innerHTML = '.NetworKit_Page { font-family: Arial, Helvetica, sans-serif; font-size: 14px; } .NetworKit_Page .Value:before { font-family: Arial, Helvetica, sans-serif; font-size: 1.05em; content: attr(data-title) ":"; margin-left: -2.5em; padding-right: 0.5em; } .NetworKit_Page .Details .Value:before { display: block; } .NetworKit_Page .Value { font-family: monospace; white-space: pre; padding-left: 2.5em; white-space: -moz-pre-wrap !important; white-space: -pre-wrap; white-space: -o-pre-wrap; white-space: pre-wrap; word-wrap: break-word; tab-size: 4; -moz-tab-size: 4; } .NetworKit_Page .Category { clear: both; padding-left: 1em; margin-bottom: 1.5em; } .NetworKit_Page .Category:before { content: attr(data-title); font-size: 1.75em; display: block; margin-left: -0.8em; margin-bottom: 0.5em; } .NetworKit_Page .SubCategory { margin-bottom: 1.5em; padding-left: 1em; } .NetworKit_Page .SubCategory:before { font-size: 1.6em; display: block; margin-left: -0.8em; margin-bottom: 0.5em; } .NetworKit_Page .SubCategory[data-title]:before { content: attr(data-title); } .NetworKit_Page .Block { display: block; } .NetworKit_Page .Block:after { content: "."; visibility: hidden; display: block; height: 0; clear: both; } .NetworKit_Page .Block .Thumbnail_Overview, .NetworKit_Page .Block .Thumbnail_ScatterPlot { width: 260px; float: left; } .NetworKit_Page .Block .Thumbnail_Overview img, .NetworKit_Page .Block .Thumbnail_ScatterPlot img { width: 260px; } .NetworKit_Page .Block .Thumbnail_Overview:before, .NetworKit_Page .Block .Thumbnail_ScatterPlot:before { display: block; text-align: center; font-weight: bold; } .NetworKit_Page .Block .Thumbnail_Overview:before { content: attr(data-title); } .NetworKit_Page .HeatCell { font-family: "Courier New", Courier, monospace; cursor: pointer; } .NetworKit_Page .HeatCell, .NetworKit_Page .HeatCellName { display: inline; padding: 0.1em; margin-right: 2px; background-color: #FFFFFF } .NetworKit_Page .HeatCellName { margin-left: 0.25em; } .NetworKit_Page .HeatCell:before { content: attr(data-heat); display: inline-block; color: #000000; width: 4em; text-align: center; } .NetworKit_Page .Measure { clear: both; } .NetworKit_Page .Measure .Details { cursor: pointer; } .NetworKit_Page .Measure .Details:before { content: "[" attr(data-title) "]"; display: block; } .NetworKit_Page .Measure .Details .Value { border-left: 1px dotted black; margin-left: 0.4em; padding-left: 3.5em; pointer-events: none; } .NetworKit_Page .Measure .Details .Spacer:before { content: "."; opacity: 0.0; pointer-events: none; } .NetworKit_Page .Measure .Plot { width: 440px; height: 440px; cursor: pointer; float: left; margin-left: -0.9em; margin-right: 20px; } .NetworKit_Page .Measure .Plot .Image { background-repeat: no-repeat; background-position: center center; background-size: contain; height: 100%; pointer-events: none; } .NetworKit_Page .Measure .Stat { width: 500px; float: left; } .NetworKit_Page .Measure .Stat .Group { padding-left: 1.25em; margin-bottom: 0.75em; } .NetworKit_Page .Measure .Stat .Group .Title { font-size: 1.1em; display: block; margin-bottom: 0.3em; margin-left: -0.75em; border-right-style: dotted; border-right-width: 1px; border-bottom-style: dotted; border-bottom-width: 1px; background-color: #D0D0D0; padding-left: 0.2em; } .NetworKit_Page .Measure .Stat .Group .List { -webkit-column-count: 3; -moz-column-count: 3; column-count: 3; } .NetworKit_Page .Measure .Stat .Group .List .Entry { position: relative; line-height: 1.75em; } .NetworKit_Page .Measure .Stat .Group .List .Entry[data-tooltip]:before { position: absolute; left: 0; top: -40px; background-color: #808080; color: #ffffff; height: 30px; line-height: 30px; border-radius: 5px; padding: 0 15px; content: attr(data-tooltip); white-space: nowrap; display: none; } .NetworKit_Page .Measure .Stat .Group .List .Entry[data-tooltip]:after { position: absolute; left: 15px; top: -10px; border-top: 7px solid #808080; border-left: 7px solid transparent; border-right: 7px solid transparent; content: ""; display: none; } .NetworKit_Page .Measure .Stat .Group .List .Entry[data-tooltip]:hover:after, .NetworKit_Page .Measure .Stat .Group .List .Entry[data-tooltip]:hover:before { display: block; } .NetworKit_Page .Measure .Stat .Group .List .Entry .MathValue { font-family: "Courier New", Courier, monospace; } .NetworKit_Page .Measure:after { content: "."; visibility: hidden; display: block; height: 0; clear: both; } .NetworKit_Page .PartitionPie { clear: both; } .NetworKit_Page .PartitionPie img { width: 600px; } #NetworKit_Overlay { left: 0px; top: 0px; display: none; position: absolute; width: 100%; height: 100%; background-color: rgba(0,0,0,0.6); z-index: 1000; } #NetworKit_Overlay_Title { position: absolute; color: white; transform: rotate(-90deg); width: 32em; height: 32em; padding-right: 0.5em; padding-top: 0.5em; text-align: right; font-size: 40px; } #NetworKit_Overlay .button { background: white; cursor: pointer; } #NetworKit_Overlay .button:before { size: 13px; display: inline-block; text-align: center; margin-top: 0.5em; margin-bottom: 0.5em; width: 1.5em; height: 1.5em; } #NetworKit_Overlay .icon-close:before { content: "X"; } #NetworKit_Overlay .icon-previous:before { content: "P"; } #NetworKit_Overlay .icon-next:before { content: "N"; } #NetworKit_Overlay .icon-save:before { content: "S"; } #NetworKit_Overlay_Toolbar_Top, #NetworKit_Overlay_Toolbar_Bottom { position: absolute; width: 40px; right: 13px; text-align: right; z-index: 1100; } #NetworKit_Overlay_Toolbar_Top { top: 0.5em; } #NetworKit_Overlay_Toolbar_Bottom { Bottom: 0.5em; } #NetworKit_Overlay_ImageContainer { position: absolute; top: 5%; left: 5%; height: 90%; width: 90%; background-repeat: no-repeat; background-position: center center; background-size: contain; } #NetworKit_Overlay_Image { height: 100%; width: 100%; background-repeat: no-repeat; background-position: center center; background-size: contain; }';
	element.setAttribute('id', 'NetworKit_style');
	document.head.appendChild(element);
}


{
	var element = document.getElementById('NetworKit_Overlay');
	if (element) {
		element.parentNode.removeChild(element);
	}
	element = document.createElement('div');
	element.innerHTML = '<div id="NetworKit_Overlay_Toolbar_Top"><div class="button icon-close" id="NetworKit_Overlay_Close" /></div><div id="NetworKit_Overlay_Title" /> <div id="NetworKit_Overlay_ImageContainer"> <div id="NetworKit_Overlay_Image" /> </div> <div id="NetworKit_Overlay_Toolbar_Bottom"> <div class="button icon-previous" onclick="NetworKit_overlayImageShift(-1)" /> <div class="button icon-next" onclick="NetworKit_overlayImageShift(1)" /> <a id="NetworKit_Overlay_Toolbar_Bottom_Save"><div class="button icon-save" /></a> </div>';
	element.setAttribute('id', 'NetworKit_Overlay');
	document.body.appendChild(element);
	document.getElementById('NetworKit_Overlay_Close').onclick = function (e) {
		document.getElementById('NetworKit_Overlay').style.display = 'none';
	}
}

-->
</script>



## Data Acquisition
-  **Figshare**: 그림, 데이터셋, 이미지 및 비디오 등을 포함한 연구 결과를 보존하고 공유하는 온라인 오픈 액세스 레포지토리
-  터미널 상에선, 아래 command line을 통해 데이터셋을 다운받을 수 있다.
```
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
```
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