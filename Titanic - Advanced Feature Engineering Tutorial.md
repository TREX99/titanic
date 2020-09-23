# Titanic - Advanced Feature Engineering Tutorial
원문 : https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial#2.-Feature-Engineering
## Public Score가 0.83732이라는데...사실 큰 관심은 Feature Engineering에 있음.

### 오늘은 관수원우님이 좋아하는 전처리에 대한 기초입니다.
### 자세한 내용은 원문을 참고하시면 됩니다만...여기 글만 읽어도 좋을 것 같습니다.
### Embarked와 Cabin 결측치 처리에 대한 원문을 읽을 때는 소오름이...쫘악....암튼 그랬음..
----------------------------------------------------------------------------------------------

## 1. Data Preprocessing (데이타 전처리) 개론
```
- Data Cleaning : remove noise, outliers, missing values
- Data Integration : combine multiple datasources
- Feature Construction / Extraction : derive features from raw data
- Feature Selection : choose most relevent features, remove redundant
- Normalization(Feature Scaling) : normalise contribution of each feature
- Sampling : subsample data to create trainning, cross validation, and testdatasets
※ 데이타 전처리는 설명하는 책과 경험자에 의해 더 많고 세분화한 분류가 있으나, 여기서는 교과내용을 중심으로 작성했습니다.
※ 실무에서는 위의 내용을 첫번째부터 하나씩 하나씩 딱딱 작업하기 보다는 경험과 그 날 기분(?)에 의해 생각나는대로 작업합니다.
   그래서 여러번 반복하기도 하지만...어찌되었든 위의 과정은 나도 모르는 사이 자연스럽게 다 하게 됩니다.
```

## 2. 서론 (원문 작성자 내용)
```
승객의 생존에 영향을 미친 비밀 요인을 찾으려고 노력했는데...아직 발견되기를 기다리는 다른 feature가 있다고 생각합니다.(me too ^^*)
이 커널에는 3 개의 주요 섹션이 있습니다. 
   - EDA (탐색적 데이터 분석) 
   - Feature Engineering & Model (기능 엔지니어링 및 모델)
   - Turned Random Forest Classifier
```

## 3. EDA (탐색적 데이터 분석)
```
다른 커널 유사하며 특별한 내용은 없습니다. 다만, 아주 꼼꼼하게 각 컬럼별 데이타를 몽땅 확인하고 있습니다. (하지만, 별거없음)
데이타를 확인할 때 df.head() 대신 df.sample()로 확인하는 것으로 보아 아마도 R을 사용하시는 분인 듯..R.sample() 무작위 
샘플 추출합니다.
```

## 4. Data Cleaning (Missing Value Process)
```
결측치 처리에 대한 의견입니다. 아주 좋은 의견이라 생각합니다.
  
- Age : Age의 누락된 값은 중간 연령으로 채워지지만 전체 데이터 세트의 중간 연령을 사용하는 것은 좋은 선택이 아닙니다. 
        Pclass 그룹의 중간 연령은 연령(0.408106) 및 생존(0.338481)과 높은 상관관계 때문에 최상의 선택입니다.
        또한 다른 기능 대신 "승객 등급별"로 연령을 그룹화하는 것이 더 논리적입니다.
        더 정확하게 하기 위해 누락된 연령값을 채우는 동안 두번째 수준의 groupby로 Sex 컬럼을 사용합니다. 
        Pclass와 Sex 그룹은 뚜렷한 중간 연령 값을 가지고 있습니다. 승객등급이 증가하면 남녀 모두의 중간 연령도 증가합니다.
        그러나 여성은 남성보다 나이 중앙값이 약간 낮은 경향이 있습니다.
        중간 연령은 Age 결측치를 채우는데 사용됩니다.
  ※ 위 내용이 사실인지 그리고 본인이 이해되는지 python 으로 프로그램하여 내용을 확인하면 좋을 것 같습니다.
  
- Embarked : Embarked는 범주형이며 결측값이 2개뿐입니다. 
             나이 각각 38세, 62세 / Cabin은 B28 / 같은 요금 / 둘다 생존 / 티켓번호는 둘 다 113572이며
             이름은 각각 "Icard, Miss. Amelie" 와 "Stone, Mrs. George Nelson (Martha Evelyn)" 입니다.
             "Mrs Martha Evelyn Stone"로 구글 검색하면 소름끼치는 결과가 나옵니다. 생존자에 대한 상세정보입니다.
             누락된 Embarked 는 결국 "S"로 채워집니다.
  ※ 검색하기 귀찮으면 : https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html 참조
  
- Fare : 운임값이 누락된 승객은 한명입니다. 요금이 가족규모(Parch 및 SibSp) 및 Pclass기능과 관련이 있다고 가정할 수 있습니다.
         3등석 티켓이 있고 가족이 없는 남성의 평균운임 값은 누락된 값을 채우기 위한 논리적선택입니다.
             
- Cabin : 681개의 결측치가 있는 Cabin은 무엇으로 채워야하나 ? 삭제해야 하나 ? 매우 까다롭다고 생각했던 항목입니다.
          저자의 노트북(커널)에서는 배의 전체 이미지와 Cabin의 위치를 상세하게 표시하고 생존에 대한 의견을 매우 논리적으로
          작성하였습니다. 여기다 작성하기에는 양이 많으니, 원문을 참고하시고 줄거리만 작성하면...
          타이타닉은 모두 15개의 데크를 가지고 있으며, 생존여부는 각 테크와 연결된 Cabin과 관계가 있어 논리적인 방법으로
          "Deck"라는 Feature를 만들고, 15개의 데크를 Cabin, Embarked, Survived를 연계하여 4개의 그룹으로 나누어 값을 입력한 후
          "Cabin" Feature는 삭제함.
          즉, "Cabin" 컬럼값은 많은 결측치를 추정하기 어렵고, 결측치가 없다 하더라도 생존에 영향을 주는 Feature로 사용하기에는
          무리가 있다고 판단하였습니다. 그래서 "배"라는 특수성을 감안하여 "Deck"라는 Feature를 생성하였답니다. (놀랬음..)
```
## 5. Correlations
```
별 내용 없음
```

## 6. Target Distribution in Features
```
Continuous Features : 연령 및 요금은 의사결정트리가 학습에 유용한 분할지점과 스파이크를 가지고 있다. (무슨 말인지 모르겠오. ㅜ.ㅜ)
Categorical Features : 최상의 범주형은 Pclass와 Sex 입니다.
Conclusion : 대부분의 feature는 연관되어 있다는데...그에 대한 설명은 무슨 말인지 하나도 모르겠음 .ㅋㅋㅋ
```

```
여기까지의 내용만으로도 이해하기 벅찰 것 같습니다. 
"100% 확률" 노트북(커널)을 못 봐서 그런지....몇 개의 노트북(커널)을 보니 정답은 없는 것 같습니다.
이것 저것 닥치는대로 보고 자기만의 정리를 하면 될 것 같습니다.


이후에 Feature Engineering 과 Feature Transformation, Model에 대한 설명으로 절반이 채워집니다.
Feature Engineering 과 Feature Transformation은 신규 Feature를 만들어 내는 논리적인 과정 설명이 있으며,
Model은 Random Forest를 사용합니다. Feature Importance 확인하고, ROC Curve 그려보는 것으로 마무리 되네요.

```




















