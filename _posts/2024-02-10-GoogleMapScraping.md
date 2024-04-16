---
title: "Scrapping real-time traffic information from Google Maps"
date: 2024-02-10 17:33:50 +0900
categories: [etc, Scraping/Crawling]
tags: [Google Maps, Python, Flask, HTML, CSS, JS, JSFiddle, Scrape, Scraping, Traffic, Slurm, Chrome]
math: true
toc: false
---

# 구글맵 실시간 교통정보 크롤링

본 프로젝트는 [Google Map](https://www.google.com/maps){:target="_blank"}에서 **실시간 교통정보**를 데이터로써 수집하기 위해 시작한 프로젝트다. Google Map에 들어가보면, 다양한 레이어들이 존재하고, 각 레이어마다 각기 다른 정보를 제공한다. 예를 들어, 지형/자전거도로/스트리트뷰/대중교통인프라/교통정보 등이 레이어로 구분되어 있다. 

특히, 교통정보 레이어는 실시간으로 업데이트되며, Road segment마다 **Green/Yellow/Red/Darkred** 색으로 현재의 교통체증 심각도 수준을 표현한다. (Green:원활 ~ Darkred:심각한 혼잡). 나는 이러한 실시간 교통정보가 교통분석에 유용한 자료가 될 거라 생각되어, 본 실시간 정보들을 Scraping하는 프로그램을 제작했다. 

프로그램 소스 코드들은 [GitHub Repository](https://github.com/zoshs2/ScrapingGoogleMap){:target="_blank"}에 올려져 있다.

# Legal Concern <!-- omit in toc -->

구글맵의 정보를 본사의 동의없이 수집하여 활용해도 되는지에 대한 법적 책임은 물을 수 없다는 것이 내가 검색을 통해 (현재까진) 알고 있는 사실이다. 

![png](/assets/img/post/gmap_scraping/scraping_legal_question.png)*"Is scraping legal on Google Maps? Yes, absolutely." Source: https://www.lobstr.io/blog/is-scraping-google-maps-legal*

내가 이해한 바로는, 법적으로 문제삼을 순 없으나 구글 측에서 IP 제재 또는 Google 계정 및 Google Cloud API 정지 등의 **기업 차원의 제재는 가할 수 있다**는 것이다. 이 프로그램을 사용하려면, 이 점을 잘 유념하여 사용하시길 바란다.

## Google Cloud Platform 에서 API Key 발급 받기

먼저, 구글맵 정보를 처리하기 위해서 API 키를 얻어야 한다. 이 API키는 [Google Cloud Platform](https://cloud.google.com){:target="_blank"}에서 얻을 수 있다. 아래 사진처럼, 구글 클라우드 플랫폼(GCP) 홈페이지에 들어가면, 우측 상단에 **콘솔**이라는 버튼이 있다. 여기를 들어가준다.

<img src="/assets/img/post/gmap_scraping/GCP_Step1_Fig1.png" width="600px" height="400px">

이후, 아래 그림의 (1), (2) 순서로 클릭하여 새로운 프로젝트를 생성해준다.

<img src="/assets/img/post/gmap_scraping/GCP_Step1_Fig2.png" width="600px" height="300px">

프로젝트 이름을 자유롭게 작성하여 프로젝트 생성 후, 아래 우측 그림처럼, 만들어진 프로젝트를 선택해준다. 이후, 좌측최상단 더보기탭을 눌러 'API 및 서비스' - '라이브러리'에 들어가도록 한다.

<table><tr>
<td> <img src="/assets/img/post/gmap_scraping/GCP_Step1_Fig3.png" width="400px" height="300px" /> </td>
<td> <img src="/assets/img/post/gmap_scraping/GCP_Step1_Fig4.png" width="600px" height="300px" /> </td>
</tr></table>

라이브러리에서 "Map JavaScript API"를 눌러 '사용'을 클릭해 약관 동의 절차를 진행한다.

<img src="/assets/img/post/gmap_scraping/GCP_Step1_Fig5.png" width="600px" height="300px">

이 때, 결제 정보 입력 과정이 있는데, 이를 입력한다고 당장 무언가가 결제되는 것이 아니니 안심하고 입력해줘도 괜찮다. 
> 유료 서비스에 접근하거나 제한된 쿼리 수 이상을 호출하려고 할 시, 결제 청구 알람이 따로 온다고 하니 이 점만 잘 인지하고 있으면 된다.
{: .prompt-warning }

<img src="/assets/img/post/gmap_scraping/GCP_Step1_Fig6.png" width="300px" height="600px">

중간에 약관이나 키제한여부(나는 '나중에'를 눌러 스킵), 사용목적설문 등 부수적인 절차를 완료하고, 대충 여기까지 오면 API 발급은 끝이다. 발급된 API 키는 다시 좌측최상단 더보기탭에서 'API 및 서비스' - '사용자 인증 정보'에 들어가, 아래 그림처럼 '키표시'를 눌러 확인할 수 있다.

<img src="/assets/img/post/gmap_scraping/GCP_Step1_Fig7.png" width="800px" height="400px">

## API 키 환경변수 설정

API 키는 여러 군데에 노출시켜봤자 좋을게 없으므로, **.bash_profile**에서 시스템 환경변수로 설정시켜놓고 이를 사용하도록 하자.

<img src="/assets/img/post/gmap_scraping/Step2_Fig1.png" width="600px" height="300px">

## Crawling Procedure

여기까지 기초적인 준비는 모두 끝났다. 앞으로 map.html, app.py, CrawlingGmapTraffic.py 이름의 3가지 스크립트 파일을 소개할 건데, 이들을 서로 유기적으로 얽혀 구글맵 실시간 교통정보 이미지를 크롤링하게 된다. 

<img src="/assets/img/post/gmap_scraping/Workflow.png" width="1400px" height="700px">

실시간 교통정보 데이터 수집을 위해, 떠올린 개략적 계획은 다음과 같았다.
1. 특정 위치의 실시간 Google Traffic Layer를 웹(Web)에 띄우기.
2. 해당 웹을 직접 열어서, Web Screenshot(일명 화면캡쳐)을 찍어 PNG 파일로 저장시키기.
3. 5분 마다 1,2번 과정 반복. 

최초 CrawlingGmapTraffic.py 에서 '사용자가 원하는 입력값', **도시 이름 / 크롤링할 전용 포트번호 / 위도 & 경도 / 줌 스케일** 을 입력하면, 이 입력값들이 app.py 스크립트로 전달된다. 작업의 단순성을 높이고, 작업간 종속변수 에러를 최소화하기 위해, 사용자가 신경써야할 부분을 CrawlingGmapTraffic.py 에만 집중되도록 구성했다 (User Interface Region). 이후 CrawlingGmapTraffic.py 은 app.py를 동작시킨다. app.py 스크립트는 **python flask 라이브러리**를 사용하여, 내가 원하는 정보를 웹에 렌더링시키는 역할을 한다. 이 경우 '내가 원하는 정보'란 당연히 Google Map Traffic Layer고, 이는 map.html 파일에 기술되어 있고 이를 기반으로 웹에 정보가 표현된다. map.html은 HTML(웹 페이지 구조를 정의)/CSS(웹 페이지 스타일 정의)/JavaScript(웹 페이지 동작을 정의) 세 가지 언어를 통해 Google Maps 교통정보를 웹에 어떻게 표시할 것인지 정의한 내용이 담겨 있다.

입력값 전달과 스크립트 동작은 이런 식으로 진행되고, 이후 데이터를 수집하는 과정은 역순으로 진행된다. map.html (구글맵을 웹에 표현하는 방식을 정의)에서 나타낸 특정 위치의 실시간 교통정보를 app.py가 Web에 실제로 렌더링하고, CrawligGmapTraffic.py에서 이 Web을 **호출 및 캡쳐**하여 데이터화시키게 된다.

이제 스크립트를 하나하나 간단히 살펴보고, 실제로 동작까지 시켜 출력된 결과물이 어떤 지 확인해보자.

