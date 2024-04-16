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

