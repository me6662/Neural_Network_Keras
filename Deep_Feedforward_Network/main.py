# 심층 전방향 신경망 (Deep Feedforward Network)
# 택시요금 추정 회귀모델

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale

from utils import preprocess, feature_engineer

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

print(df.describe())