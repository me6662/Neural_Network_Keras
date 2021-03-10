# 데이터 표준화
# 데이터 셋의 숫자변수를 평균이 0 분산이 1
# NN 의 역전파 알고리즘을 잘 적용하려면 표준화가 필요함.
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing

df = pd.read_csv('./diabetes.csv')

# print(df.describe())
"""
       Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin         BMI  DiabetesPedigreeFunction         Age     Outcome
count   768.000000  768.000000     768.000000     768.000000  768.000000  768.000000                768.000000  768.000000  768.000000
mean      3.845052  120.894531      69.105469      20.536458   79.799479   31.992578                  0.471876   33.240885    0.348958
std       3.369578   31.972618      19.355807      15.952218  115.244002    7.884160                  0.331329   11.760232    0.476951
min       0.000000    0.000000       0.000000       0.000000    0.000000    0.000000                  0.078000   21.000000    0.000000
25%       1.000000   99.000000      62.000000       0.000000    0.000000   27.300000                  0.243750   24.000000    0.000000
50%       3.000000  117.000000      72.000000      23.000000   30.500000   32.000000                  0.372500   29.000000    0.000000
75%       6.000000  140.250000      80.000000      32.000000  127.250000   36.600000                  0.626250   41.000000    1.000000
max      17.000000  199.000000     122.000000      99.000000  846.000000   67.100000                  2.420000   81.000000    1.000000

딱봐도 insulin 은 최대가 846 인데, DiabetesPedigreeFunction 은 최대가 2.42 임...
이러면 거의 insulin 의 영향만을 받게된다.
"""
df_scaled = preprocessing.scale(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)


# 여기서 outcome 값 까지 표준화 되버린 모습이므로 outcome 원래꺼 그대로 사용한다.
df_scaled['Outcome'] = df['Outcome']
df = df_scaled

print(df)
print(df.describe().loc[['mean', 'std', 'max'], ].round(2).abs())
