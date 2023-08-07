---
title: "[NumPy] np.unique 기존 순서를 유지한 채로 반환받기"
date: 2021-03-18 17:34:29 +0900
categories: [Python, Library]
tags: [python, library, numpy, np.unique]
math: true
---

# 들어가며
결론부터 말하자면, numpy 말고 <u>**pandas의 unique 메소드**</u>를 쓰면 된다. 사실 이번 포스트의 내용은 다른 블로그 플랫폼에서 기록했던 ***과거의 글*** 을, 요즘 주로 다루고 있는, 여기 GitHub 블로그에 통합하면서 옮겨 적게 된 글이다. 
<br>

pandas.unique()라는 더욱 나은 대안을 지금은 알고 있긴 하지만, NumPy로도 원하는 결과를 얻는 방법이 있다는 것을 기록한다는 차원에서 이렇게 가져와 적는다.
<br><br>

# np.unique 메소드
어느 날, 내가 가진 데이터 array에서 중복되지 않는 값들(uniques)을 얻어내야 할 일이 있었다. 다만, 이 중복되지 않는 값들을 뽑아낼 때, <u>**기존의 순서를 유지**</u>해야할 필요가 있었다.
<br>

예를 들면,
```python
origin_arr = [3, 1, 3, 1, 5, 2, 4, 2]
```
인 상황에서, 중복되지 않는 값들을 뽑아낸다고 했을 때, **[3, 1, 5, 2, 4]** 를 추출해 내는 것처럼 말이다.
* * *
<br>

그리고 처음엔 그냥 막연하게 numpy.unique란 메소드를 활용하면 될 줄 알았는데, 결과는 추출 후 element간의 sorting이 이뤄진 채로 결과가 반환되었다. python built-in 함수인 **set**도 마찬가지.

```python
import numpy as np

>>> print(np.unique(origin_arr))

[1, 2, 3, 4, 5]     # Ascending sort된 반환 값들.
```

그렇다. np.unique 메소드는 유일한 값들을 찾아주면서, 동시에 값들을 sort 해준다. 그래서 그런지, 대상으로 하는 array의 크기가 굉장히 큰 경우엔 np.unique 속도 효율은 상당히 저하된다.
<br>

> **_NOTE:_** string elements를 지닌 array도 np.unique() 를 쓰면 정렬된 결과가 반환된다.
<br>
* * *
<br>

그래서, 어떻게 기존의 순서를 유지한 채 유일한 값들을 뽑아낼 지를 구글링을 좀 해봤다.
방법은 np.unique 메소드의 **return_index 옵션**을 활용하는 것이었다. 이 옵션은 명시하지 않을 경우, return_index=False가 default로 설정되어 있다. 이를 return_index=True 바꾸어 준다면, 다음과 같은 튜플(tuple) 형태의 결과가 반환된다.

```python
import numpy as np

origin_arr = [3, 1, 3, 1, 5, 2, 4, 2]
rslt = np.unique(origin_arr, return_index=True)

>>> print(rslt)

(array([1, 2, 3, 4, 5]), array([1, 5, 0, 6, 4]))
```

여기서 첫 번째 array는 (정렬이 이미 되어진 상태의) 유일한 값들을 나열한 행렬이고, 두 번째 array는 이 값들을 origin_arr에서 뽑아낸 위치(index)이다. 그리고 자세히 살펴보면, origin_arr에서 **해당 값이 처음 등장하는 위치**를 기준으로 index들이 기록되어 있다.
<br>

그럼 이제 이 index array 결과를 사용하면, 원하는 목적이었던 "**기존의 순서를 유지한 채로, unique 값들 뽑기**"를 할 수 있다.
* * *
<br>

먼저 이 index array를 오름차순으로 sort하여, 기존의 순서를 바꾸지 않고 유지하게끔 한다. 왜냐면 이 index들의 대수적 크기가 origin_arr에서의 본래 순서이기 때문이다. 그리고 이 (ascending) sorted index array를 origin_arr에다가 인덱싱해서 값들을 반환시키면 된다.

```python
idx_arr = rslt[1] # index array
sorted_idx_arr = sorted(idx_arr, reverse=False) # ascending sorted index array

>>> print(origin_arr[sorted_idx_arr]) # indexing to origin_arr

[3, 1, 5, 2, 4]
```

* * *
<br>

이렇게 np.unique() 메소드를 활용해서 (원치 않던) 정렬된 형태의 결과가 아닌, 기존 순서를 유지해서 얻는 방법을 알아 보았다. 순서를 굳이 유지하고 싶다면, 이 방법을 잘 기억해 사용해보자!