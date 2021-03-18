# 요일 및 시간별 승차 통계
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)

# pandas 의  dt 변수를 통해 손쉽게 년월일요일 시 분별
df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['day'] = df['pickup_datetime'].dt.day
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['hour'] = df['pickup_datetime'].dt.hour


# df['day_of_week'].plot.hist(bins= np.arange(8) - 0.5, ec='black', ylim=(60000,75000))
# plt.xlabel('Day of Week (0 : Mon, 6 : Sun)')
# plt.title('Day of Week Histogram')n
# plt.show()


df['hour'].plot.hist(bins= np.arange(24), ec='blue')
plt.xlabel('Hour')
plt.title('Pickup Hour Histogram')
plt.show()