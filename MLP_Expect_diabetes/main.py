# MLP : Multi Layer Perceptron
# ML의 과정은 EDA (탐색적 데이터 분석, Exploratory Data Analysis) -> 데이터 전처리 (결측값, 표준화) -> 모델링!
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import confusion_matrix
import seaborn as sns


from sklearn.metrics import roc_curve

df = pd.read_csv('./diabetes.csv')

# ------------------------------------------------------- <데이터 전처리> -------------------------------------------------------

# (1) 결측값, 0 이 아닌데 0으로 되어있는값 -> 평균값으로
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

# (2) 표준화
df_scaled = preprocessing.scale(df)  # 스케일링 된 리스트 리턴
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)  # 다시 dataframe 객체로 변환
df_scaled['Outcome'] = df['Outcome']  # 결과 칼럼 추가 -> 결과는 normalize 안해줌
df = df_scaled

# (3) 데이터 셋 분할
# df.loc : 행리턴, df.loc[x,y] x 조건 (행 조건), y 조건 (열 조건) 에 해당하는 행 리턴
X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, 'Outcome']

# 1차 분할 (훈련 데이터셋 80%, 테스트 데이터셋 20%) > 자동으로 무작위로 분배해줌!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2차 분할 (훈련데이터셋 : 80%, 검증 데이터셋 : 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2)


# ------------------------------------------------------- <머신러닝 모델링(MLP)> -------------------------------------------------------
# keras 구성 : input > Model (layer 집합) -> Compile (Loss함수,  Optimizer 로 구성) -> Train (Loss를 Optimizer 로 최적화)


# ------------ keras 1. Model ------------
model = Sequential()

# Hidden Layer 1
# Sequential 에 Hiddenlayer 쌓음 > input layer 와 연결되는 첫번째 hidden layer 는 weight 을 잘 정해서 해줘야함 >> 이후 중간은 케라스가 알아서함
# weight 32 개 사용,  input [n X 8 ] , weight [8 X 32]  matmul 이후  activation 하면 [n x 32]
# weight 의 갯수를 정하는 거를 초매개변수 세팅이라고 함, 실제로는 실험을 반복해서 신중히 결정함.
model.add(Dense(32, activation='relu', input_dim=8))


# Hidden Layer 2
# 레이어를 더쌓으면 모델 복잡도를 올릴 수 있지만, Overfit 이 발생할 수도 있음 지금은 두개가 적당 (출력레이어 포함 총 3개 layer)
# weight 16개 사용, input [n X 32] , weight [32 X 16] , activation 이후 [n X 16]
model.add(Dense(16, activation='relu'))

# output layer
# output 은 [ n X 1 ] 로 나오게됨. 결과가 당뇨냐 아니냐 ( 0 or 1 인 binary classify 이므로 sigmoid 사용)
model.add(Dense(1, activation='sigmoid'))


# ------------ keras 2. Compile ------------
# Loss : 이진 문제의 경우 대개 binary_crossentropy 사용
# Optimizer : adam optimizer 사용, 케라스에서 잘 사용하며, 튜닝없이도 대부분의 데이터셋에 잘동작!
# metrics (평가지표) : accuracy 사용, 올바르게 분류된 샘플 비율 계산

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])


# ------------ keras 3. Train ------------
# 200 epoch 으로 훈련
model.fit(X_train, y_train, epochs=200)


# ------------------------------------------------------- <테스트 정확도 분석> -------------------------------------------------------
scores = model.evaluate(X_train, y_train)
print('Training Accuracy: %.2f%%\n' % (scores[1]*100))

scores = model.evaluate(X_test, y_test)
print('Test Accuracy: %.2f%%\n' % (scores[1]*100))


# ------------ Confusion Matrix 평가 ------------
y_test_pred = model.predict_classes(X_test)
#y_test_pred = np.argmax(model.predict(X_test), axis=-1)
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True, xticklabels=['No Diabetes', 'Diabetes'],
                 yticklabels=['No Diabetes', 'Diabetes'], cbar=False, cmap='Blues')
ax.set_xlabel('Prediction')
ax.set_ylabel('Actual')
plt.show()
plt.clf()

# ------------ ROC 곡선 ------------
# FPR : 위양성률 = 위양성 / 진음성 + 위양성 (x) > 실제가 False 인 데이터중 False 를 정확히 예측 못한 비율
# TPR : 진양성률 = 진양성 / 진양성 + 위음성 (y) > 실제가 True 인 데이터중 True 를 정확히 예측한 비율
# AUC(Area Under Curve) 로 평가한다 라고 함 (곡선 밑에 면적이 많을 수록 예측을 잘한것)
y_test_pred_probs = model.predict_proba(X_test)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
plt.plot(FPR, TPR)
plt.plot([0, 1], [0, 1], '--', color='black')
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
plt.clf()
