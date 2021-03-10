import numpy as np
import pandas as pd

df = pd.read_csv('./diabetes.csv')

# df[열헤더] : 해당 헤더의 열 나열
# df.loc[조건] : 조건을 만족하는 행 나열


# df 에서 0 인 값을 nan 으로 변경 > 판다스가 인식하도록함
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)


# 이제 nan 을 정상 값의 평균값으로 대체한다. 판다스의 fillna() 함수 사용
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
