# 심층 전방향 신경망 (Deep Feedforward Network)
# 택시요금 추정 회귀모델

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import preprocess, feature_engineer

from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)


# 데이터 전처리
df = preprocess(df)
df = feature_engineer(df)

# 변수 스케일링
df_prescaled = df.copy()  # default : deep copy

df_scaled = df.drop(['fare_amount'], axis=1)  # 출력값 (amount) 는 스케일링 제외
df_scaled = scale(df_scaled)  # 리턴한 obj 가  df 가 아니므로 df 로 변환

cols = df.columns.tolist()
cols.remove('fare_amount')  # column 리스트 (df_scaled) 에 붙여줄 헤더

df_scaled = pd.DataFrame(df_scaled, columns=cols, index=df.index)
df_scaled = pd.concat([df_scaled, df['fare_amount']], axis=1)
df = df_scaled.copy()

# Data Sets
X = df.loc[:, df.columns != 'fare_amount']
y = df.loc[:, 'fare_amount']

# Data sets 분할 (20% 는 테스트데이터)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# -------------- [keras1] Modeling --------------

# 회귀손실함수
"""
택시요금예측의 경우 결과가 0 or 1 로 나오거나 A/B/C/D 와 같이 다중 클래스로 나오지 않는다.
택시요금예측의 경우 결과는 연속(Continuous)변수로 출력된다.

회귀모델은 연속변수 (비용, 시간, 높이 등) 의 값을 예측하며, 
분류모델은 클래스(0 or 1 , A/B/C/D) 를 예측한다.

분류모델의 경우, 평가를 위해 백분율 정확도 (epoch 에 따라 모델의 accuracy (출력을 몇개 맞췄냐),Confusion Matrix, ROC 곡선) 를 사용
이게 당연한게 이산적으로 구분이 되니까 몇개 맞췄는지 판단이 가능함.

허나 회귀모델의 경우....
RMSE (Root Mean Squared Error) 를 오차지표로 사용한다. 
예측값과 실제값 차의 제곱 을 루트 씌운 것인데.. 이게 평가지표로도 사용되고 실제 손실함수로도 사용된다!  > sungkim 형님 강의 하면 이거부터 가르치자늠

"""

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
# 입력레이어 : 128개 >> [input_dim , 128] 행렬이됨
model.add(Dense(64, activation='relu'))  # 0.5배
model.add(Dense(32, activation='relu'))  # 0.5배
model.add(Dense(8, activation='relu'))
# 마지막은 택시예측요금 인 Y 가 나오는데, activation 이 따로 없다! (왜냐면 실수가 나와야 하므로..)
model.add(Dense(1))

# 모델검증
# model.summary()


# -------------- [keras2] Compile, Fit --------------
# 이게 분류모델 할때는 metrices 이게 accuracy 였음. > terminal 에 mse 가 같이나온다는 뜻인듯
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(X_train, y_train, epochs=1)


# -------------- [keras3] Evaluate --------------

# 테스트 데이터셋에서 무작위로 1개의 샘플을 뽑아서 요금을 예측하는 함수!
def predict_random(df_prescaled, X_test, model):
    sample = X_test.sample(
        n=1, random_state=np.random.randint(low=0, high=10000))
    idx = sample.index[0]

    # 실제 요금
    actual_fare = df_prescaled.loc[idx, 'fare_amount']

    # 샘플 시간 추출
    day_names = ['Monday', 'Tuesday', 'Wednesday',
                 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # day of week 열이 값이 012345 로 되있음..;;
    day_of_week = day_names[df_prescaled.loc[idx, 'day_of_week']]
    hour = df_prescaled.loc[idx, 'hour']

    # 예측 요금
    predicted_fare = model.predict(sample)[0][0]

    rmse = np.sqrt(np.square(predicted_fare - actual_fare))

    print('Trip Details : {}, {}:00hrs'.format(day_of_week, hour))
    print('Actual fare: ${:0.2f}'.format(actual_fare))
    print('Predicted fare: ${:0.2f}'.format(predicted_fare))
    print('RMSE: ${:0.2f}'.format(rmse))


predict_random(df_prescaled, X_test, model)

train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print('Train RMSE : {:0.2f}'.format(train_rmse))
print('Test RMSE : {:0.2f}'.format(test_rmse))
