# SentenceBERT_with_Eldenring_Review
<br>
-> 스팀 게임 중 엘든링이라는 게임에 관한 리뷰를 분석해서 워드클라우드로 만들었음

## 목차
1. [문장 임베딩](#1-문장-임베딩)
2. [KMeans](#2-kmeans-클러스터링-from-scikit-learn)
3. [리뷰 크롤링](#3-리뷰데이터-크롤링)
4. [전처리](#4-preprocessing)
5. [리뷰 임베딩](#5-reviews-embedding)
6. [K=5 Clustering](#6-k5-클러스터링)
7. [워드클라우드](#7-워드클라우드-그리기)


---
## 1. 문장 임베딩

`SentenceTransformer` 라이브러리 사용
  * 말 그대로 문장을 임베딩 즉, 벡터화 시켜줌
  ```python
  # 예시코드
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('jhgan/ko-sroberta-multitask')

  sentences = ["안녕하세요?", "한국어 문장 임베딩을 위한 버트 모델입니다."]  # list형태로 저장
  embeddings = model.encode(sentences) 
  ```
  
  ## 2. `KMeans` 클러스터링 from `Scikit-Learn`
  
  Embedding Data들을 KMeans를 이용해서 군집화 하였음
  * Test Code
  ```python
  from sklearn.cluster import KMeans

  # Corpus with example sentences
  sentences = ['한 남자가 음식을 먹는다.',
            '한 남자가 빵 한 조각을 먹는다.',
            '그 여자가 아이를 돌본다.',
            '한 남자가 말을 탄다.',
            '한 여자가 바이올린을 연주한다.',
            '두 남자가 수레를 숲 속으로 밀었다.',
            '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
            '원숭이 한 마리가 드럼을 연주한다.',
            '치타 한 마리가 먹이 뒤에서 달리고 있다.',
            '한 남자가 파스타를 먹는다.',
            '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
            '치타가 들판을 가로 질러 먹이를 쫓는다.']

  embeddings = model.encode(sentences)

  # Then, we perform k-means clustering using sklearn:
  num_clusters = 5
  clustering_model = KMeans(n_clusters=num_clusters)
  clustering_model.fit(embeddings)
  cluster_assignment = clustering_model.labels_

  clustered_sentences = [[] for i in range(num_clusters)]
  for sentence_id, cluster_id in enumerate(cluster_assignment):
      clustered_sentences[cluster_id].append(sentences[sentence_id])

  for i, cluster in enumerate(clustered_sentences):
      print("Cluster ", i+1)
      print(cluster)
      print("")
  ```

## 3. 리뷰데이터 크롤링
* [Reference](https://github.com/arditoibryan/datasets/blob/main/220226_steam/get_reviews.ipynb)
* requests 내장 라이브러리 사용
* 총 2개의 함수 생성
  * get_reviews() 함수<br>
  => 원하는 url(해당 코드에서는 스팀게임리뷰 사이트) 에서 appid(게임코드)를 입력하면 해당되는 리뷰데이터를 얻을 수 있음
  * get_n_reviews() 함수<br>
  => get_reviews 함수를 총 n개의 페이지에서 크롤링<br>
  => 이후, 크롤링한 데이터를 dictionary형태로 저장
  
* `appid` : **1245620**
* `pandas`를 사용해서 dictionary를 DataFrame로 변경

## 4. Preprocessing
### 4-1. First Preprocessing
* 한글이 아닌 글자 제거
  * 정규표현식 사용
* 작은따옴표 제거
* 연속된 공백 제거
* 좌우 공백 제거
* `최대 글자는 255개로 제한`
<details open>
  <summary>Preprocessing Code</summary>
  
  ```
  df['review'] = df['review'] \
    .replace(r'[^가-힣 ]', ' ', regex=True) \
    .replace("'", '') \
    .replace(r'\s+', ' ', regex=True) \
    .str.strip() \
    .str[:255]
  ```
</details>

## 4-2. Second Preprocessing
* 공란인 리뷰 제거
```python
df = df[df['review'].str.strip().astype(bool)]
```
* 전처리가 완료된 데이터는 CSV파일로 저장

## 5. Reviews Embedding

* 위의 테스트 코드에서 사용한 Embedding모델(`SentenceTransformer`)사용
* 인코딩을 위해서 리뷰 칼럼을 리스트화
  ```python
  corpus = df['review'].values.tolist()
  ```
## 6. k=5 클러스터링
* 위의 테스트 코드와 마찬가지로 `KMeans`를 사용

<details open>
  <summary>Clustering Code</summary>
  
  ```python
  num_clusters = 5
  clustering_model = KMeans(n_clusters=num_clusters)
  clustering_model.fit(embeddings)
  cluster_assignment = clustering_model.labels_

  clustered_sentences = [[] for i in range(num_clusters)]
  for sentence_id, cluster_id in enumerate(cluster_assignment):
      clustered_sentences[cluster_id].append(corpus[sentence_id])

  for i, cluster in enumerate(clustered_sentences):
      print('Cluster %d (%d)' % (i+1, len(cluster)))
      print(cluster)
      print('')
  ```
</details>

## 7. 워드클라우드 그리기
### 형태소 분석 및 명사 추출
```bash
# Progress 상황을 볼 수 있음
pip install -q konlpy tqdm

# 한글폰트 다운로드
wget https://github.com/kairess/MBTI-wordcloud/raw/master/NanumSquareRoundR.ttf 
```
* 명사 추출
```python
from konlpy.tag import Komoran, Okt, Kkma, Hannanum
from tqdm import tqdm

extractor = Hannanum()

nouns = []

for review in tqdm(df['review'].values.tolist()):
    nouns.extend(extractor.nouns(review))

len(nouns)
```

### 워드클라우드 그리기
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(
    font_path='NanumSquareRoundR.ttf',
    width=2000,
    height=1000
).generate_from_frequencies(words)

plt.figure(figsize=(20, 10))
plt.imshow(wc)
plt.axis('off')
plt.show()
```
<img src = 'https://user-images.githubusercontent.com/103639510/216264475-78e9baf1-4db9-40a2-81f1-e154697e30c0.png'>
