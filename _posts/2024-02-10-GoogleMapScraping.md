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

프로그램 소스 코드들과 **자세한 코드 설명**은 내 [GitHub Repository](https://github.com/zoshs2/ScrapingGoogleMap){:target="_blank"}에 올려두었다.

# Image Raw Dataset to Regularized 2D Dataset

위 스크래핑 프로그램으로 수집한 구글맵 교통정보 이미지를 어떤 식으로 전처리할 지에 대한 아이디어는, 23년 11월 Physical Review E 저널에 실린 [Ebrahimabadi, Sasan, et al. "Geometry of commutes in the universality of percolating traffic flows."](https://doi.org/10.1103/PhysRevE.108.054311){:target="_blank"} 논문을 참고했다. 

대충 요약하자면, 3차원(RGB) 텐서 행렬의 이미지 데이터를 2차원으로 투영시키고 Max Pooling으로 주변 픽셀 정보들을 압축해서 표현하는 전처리 과정이다.

본 글에서는 그 전처리에 대한 일부 결과들만 소개한다. 

![png](/assets/img/post/gmap_scraping/London_20240123_0800.png)*24년 1월 23일 (화) 영국 런던의 오전 8시 도심 아침 교통 상황*

![png](/assets/img/post/gmap_scraping/London_20240123_1330.png)*24년 1월 23일 (화) 영국 런던의 오후 1시 30분 도심 점심(낮) 교통 상황*

![png](/assets/img/post/gmap_scraping/London_20240123_1800.png)*24년 1월 23일 (화) 영국 런던의 오후 6시 도심 저녁 교통 상황*

![png](/assets/img/post/gmap_scraping/London_20240123_2300.png)*24년 1월 23일 (화) 영국 런던의 오후 11시 도심 야간 교통 상황*