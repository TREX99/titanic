# 제목 : Titanic: A complete approach to Top 6% Rank
#### 원문 : https://www.kaggle.com/pedrodematos/titanic-a-complete-approach-to-top-6-rank

# Public Score가 0.78229라는데...정확한건 모르겠고..ㅎㅎㅎ

이 노트북의 주요 목표는 탐색적 데이터 분석에서 지도 및 비지도 학습기술을 데이터에 적용하는 모델링 문제에 대한 완전한 접근 방식을 제시하는 것입니다. 다만, 우리는 feature engineering 과 Pipeline에 대한 이해만 하도록 합니다.


## Section 1 - Data Exploration

### 1. pandas_profiling을 이용한 데이타 확인

```
import pandas_profiling
report = pandas_profiling.ProfileReport(df)
display(report)
```

### 2. AutoViz Class를 이용한 각종 feature 시각화
```
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
report_2 = AV.AutoViz("/kaggle/input/titanic/train.csv")
```

### 3. EDA결론
```
대부분의 커널에서 말하는 공통적인 내용은 집어치우고, 위에 나열된 두 자동 EDA 라이브러리 및 z-test, t-test 를 통해서 발견된 사항  
  - "연령"열의 값 중 거의 20%(177)가 누락되었습니다. 이러한 null을 분포의 평균으로 채우는 등 다양한 기술로 채워야 한다.
  - 연령 분포는 왜곡되어 있다. 평균 연령이 약 30세이고 표준 편차가 15에 가깝습니다. 가장 오래된 승객은 80 세입니다.
  - "Fare"(운임)은 많이 왜곡되어 있다. 평균값은 약 32이고 표준 편차는 50에 가깝습니다. 최소값은 0이고 최대 값은 512.3292입니다. 
  - 따라서 왜곡된 "Fare"를 사용하는 경우 SVM모델을 사용할 때 주의 깊게 다루어야 한다.
  - 연령과 생존은 관계가 있다.
  - 성별과 생존은 관계가 있다.
  - "Pclass"와 생존은 관계가 있다.
  - 운임과 생존은 관계가 있다.
  - 아울러 "Pclass"와 운임은 관계가 있다.
※ 시사점 : 왜곡여부의 확인은 "평균 연령이 약 30세이고 표준 편차가 15에 가깝습니다"와 같이 평균과 표준편차를 이용한다...
           도대체 평균과 표준편차는 무슨관계이길래..
```

### 4. Feature Engineering
```
참고사이트 https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial#2.-Feature-Engineering

# 나이를 이용한 신규 categorical variable 생성
df['AgeCat'] = ''
df['AgeCat'].loc[(df['Age'] < 18)] = 'young'
df['AgeCat'].loc[(df['Age'] >= 18) & (df['Age'] < 56)] = 'mature'
df['AgeCat'].loc[(df['Age'] >= 56)] = 'senior'


# 가족규모에 따른 신규 categorical variable 생성
df['FamilySize'] = ''
df['FamilySize'].loc[(df['SibSp'] <= 2)] = 'small'
df['FamilySize'].loc[(df['SibSp'] > 2) & (df['SibSp'] <= 5 )] = 'medium'
df['FamilySize'].loc[(df['SibSp'] > 5)] = 'large'


# 싱글 승객인지 여부에 대한 신규 categorical variable 생성
df['IsAlone'] = ''
df['IsAlone'].loc[((df['SibSp'] + df['Parch']) > 0)] = 'no'
df['IsAlone'].loc[((df['SibSp'] + df['Parch']) == 0)] = 'yes'


# 성별과 나이를 이용한 신규 categorical variable 생성
df['SexCat'] = ''
df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] <= 21)] = 'youngmale'
df['SexCat'].loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50)] = 'maturemale'
df['SexCat'].loc[(df['Sex'] == 'male') & (df['Age'] > 50)] = 'seniormale'
df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] <= 21)] = 'youngfemale'
df['SexCat'].loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50)] = 'maturefemale'
df['SexCat'].loc[(df['Sex'] == 'female') & (df['Age'] > 50)] = 'seniorfemale'


# 승객의 이름 앞 글자를 이용하여 신규 categorical variable 생성 (결혼여부도 알 수 있음..이건 좋은 아이디어라고 생각함)
df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df['Is_Married'] = 0
df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1
df['Title'] = df['Title'].replace(['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df['Title'] = df['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

# 같은 종류의 "Ticket" 갯수 variable 생성
df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')

# 학습에 불필요한 feature 삭제(drop)
df.drop(['PassengerId', 'Survived', 'Ticket', 'Name', 'Cabin'], axis=1, inplace=True)

# categorical 컬럼명과 numerical 컬러명을 분리 보관
categorical_df = df.select_dtypes(include=['object'])
numeric_df = df.select_dtypes(exclude=['object'])
categorical_columns = list(categorical_df.columns)
numeric_columns = list(numeric_df.columns)
```

### 5. 특이사항 (Balancing Data, 데이타 균형) : 이건 모르겠음...공부해야 할 것..
```
def balancingClassesRus(x_train, y_train):
    # Using RandomUnderSampler to balance our training data points
    rus = RandomUnderSampler(random_state=7)
    features_balanced, target_balanced = rus.fit_resample(x_train, y_train)
    print("Count for each class value after RandomUnderSampler:", collections.Counter(target_balanced))
    return features_balanced, target_balanced

def balancingClassesSmoteenn(x_train, y_train):
    # Using SMOTEEN to balance our training data points
    smn = SMOTEENN(random_state=7)
    features_balanced, target_balanced = smn.fit_resample(x_train, y_train)    
    print("Count for each class value after SMOTEEN:", collections.Counter(target_balanced))    
    return features_balanced, target_balanced

def balancingClassesSmote(x_train, y_train):
    # Using SMOTE to to balance our training data points
    sm = SMOTE(random_state=7)
    features_balanced, target_balanced = sm.fit_resample(x_train, y_train)
    print("Count for each class value after SMOTE:", collections.Counter(target_balanced))
    return features_balanced, target_balanced
```

### 6. Model training & Evaluation functions (모델훈련 및 평가함수)
```
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, make_scorer

위 모듈을 이용하여 아주 간편하게 평가점수를 구할 수 있음 (ADsP시험에 나오는 내용이며, 계산 공식을 이용하여 값을 구하는 문제가 출제됨)
```

### 7. Pipeline Construction
```
전체 파이프라인이 있을 때, 작은 크기의 파이프 라인으로 구성되었다고 생각하고, 각각의 파이프 라인이 모델링 프로세스 단계라고 가정함.
> Step 1: 숫자형 컬럼의 null 값을 채운다.
> Step 2: 정규화(표준화)하여 서로 다른 컬럼의 값이 동일한 척도에 있도록 한다. => (x - mean) / std  <-- 배운거
> Step 3: 범주형 컬럼의 null 값을 채운다.
> Step 4: OneHotEncode 로 변환한다.  <-- 앞으로 배울 것
> Step 5: 머신러닝 학습 및 평가

이러한 각 단계를 개별적으로 수행하는 대신 모든 단계를 통합하는 Pipeline 객체를 만든 다음에 객체를 학습 데이터에 맞출 수 있습니다.
즉, 편리하게 모델링하기 위한 방법인데, 이것은 단지 이론일뿐이고 전산처리 경험이 있는 실무자는 함수, 모듈, 라이브러리 등과 같은
하나의 단위 프로그램을 만드는 것으로 이해하면 됨. 실제로 아래 내용을 보면 defineBestModelPipeline() 함수 하나에서
전처리, 분류기 선택 등 한꺼번에 다 함. (덩치 큰 함수 하나로 만들기 좀 그러면 여러개의 함수로 쪼개서 순차적으로 호출해도 좋겠음)

왜 Pipeline 객체를 만들어야 하는지는 원문을 그대로 옮겨봅니다만...한번쯤 들었던 내용이니 그냥 생각없이 읽으면 될 듯..
1 - Production code gets much easier to implement
When deploying a Machine Learning model into production, the main goal is to use it on data it hasn't seen before. 
To do that, the new data needs to be transformed the same way training data was. Instead of having several different functions
for each one of the preprocessing tasks, you can use a single pipeline object to apply all of them sequentially. 
It means that, in 1 line of code, you can apply all needed transformations. 
Check an example of this in the "Predictions" section of this notebook.

2 - When combined with RandomSearchCV, it is possible to test several different pipeline options
You must have already asked yourself, when training your models: "for this type of data, what works best? 
Filling missing values with the average or the median of a column? Should I use MinMaxScaler or StandardScaler? 
Apply dimensionality reduction? Create more features using, for example, PolynomialFeatures?" 
Using Pipelines and hyperparameter search functions (like RandomSearchCV), you can search through entire 
sets of data pipelines, models and parameters automatically, saving up effort invested by you in the search 
for optimal feautre engineering methods and models/hyperparameters. (궁금했던 하이퍼파라메터 값도 여기서 찾아준다)

Suppose we have 4 different pipelines:

-> Pipeline 1: fill missing values from numeric features by imputing the mean of each column - apply MinMaxScaler
    - apply OneHotEncoder to categorical features - fits the data into a KNN Classifier with n_neighbors = 15.
-> Pipeline 2: fill missing values from numeric features by imputing the mean of each column - apply StandardScaler 
    - apply OneHotEncoder to categorical features - fits the data into a KNN Classifier with n_neighbors = 30.
-> Pipeline 3: fill missing values from numeric features by imputing the median of each column - apply MinMaxScaler 
    - apply OneHotEncoder to categorical features - fits the data into a Random Forest Classifier with n_estimators = 100.
-> Pipeline 4: fill missing values from numeric features by imputing the median of each column - apply StandardScaler
    - apply OneHotEncoder to categorical features - fits the data into a Random Forest Classifier with n_estimators = 150.
Initially, you might think that, to check which pipeline is better, all you need to do is to create all of them manually, 
fit your data, and then evaluate the results. But what if we want to increase the range of this search, 
let's say, to over hundreds of different pipelines? It would be really hard to do that manually. 
And that's where RandomSearchCV comes into play.

3 - No information leakage when Cross-Validating
This one is a bit trickier, specially for beginners. Basically, when cross-validating, 
data should be transformed inside each CV step, not before. When doing cross validation 
after transforming the training set (with a StandardScaler, for example), 
information from it is leaked to the validation set. This may lead to biased/unoptimal results.

The right way to do that is to normalize data inside cross-validation. 
That means, for each CV step, a scaler is fitted only on the training set. 
Then, this scaler transforms the validation set, and the model is evaluated. 
This way, no information from the training set is leaked to the validation set. 
When using pipelines inside RandomSearchCV (or GridSearchCV), this problem is taken care of.
```

### 8. Pipeline Construction 요약
```
이론적인 표현 : 원본에서 categorical, numeric columns별로 데이타 전처리와 분류기를 찾는 Pipeline을 생성하는 것
실무에서 표현 :  원본에서 categorical, numeric columns별로 데이타 전처리와 분류기 찾기 기능을 갖는 함수를 만들어라.

예를들어 아래와 같이 사용하는 defineBestModelPipeline() 함수를 만들어라
x_train, x_test, y_train, y_test, best_model_pipeline = defineBestModelPipeline(df, target, categorical_columns, numeric_columns)
```
 
 
# 요약
```
### 다른 노트북과 다른게 여기서는 pandas_profiling 과 AutoViz Class를 이용한 시각화 시도 : 이건 완전 편한 듯..
### 시각화에 근거한 추론으로 새로운 feature 생성 (AgeCat, FamilySize, IsAlone, SexCat, Is_Married 등)
### Pipeline 생성 (전처리 자동화, 최적의 분류기 선택, 운영환경 적용시 정확성 담보 등의 효과기대)
### 끝에 나오는 비지도학습에서는...내용은 알겠는데, 어떻게 머신러닝에 응용하는지 모르겠음 (so what ??!?!?!)
### ADsP 시험에 나오는 용어가 대박 많이 나옴. ㅋㅋ
```
