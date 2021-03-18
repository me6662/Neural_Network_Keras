# 승차 및 하차 포인트 시각화
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)


# 뉴욕시 경도 범위
nyc_min_longitude = -74.05
nyc_max_longitude = -73.75

# 뉴욕시 위도 범위
nyc_min_latitude = 40.63
nyc_max_latitude = 40.85

# deep :  깊은 복사 vs 얕은복사
# 얕은복사 : 원본 데이터 주소 공유
# 깊은복사 : 원본데이터 기반 새로운 주소 파서 새로운 df
df2 = df.copy(deep=True)

# min ~ max 사이 경도만 필터링
for long in ['pickup_longitude', 'dropoff_longitude']:
    df2 = df2[(df2[long] > nyc_min_longitude) &
              (df2[long] < nyc_max_longitude)]

# min ~ max 사이 위도만 필터링
for long in ['pickup_latitude', 'dropoff_latitude']:
    df2 = df2[(df2[long] > nyc_min_latitude) & (df2[long] < nyc_max_latitude)]


# 랜드마크 좌표
landmarks = {
    'JFK Airport': (-73.78, 40.643),
    'Laguardia Airport': (-73.87, 40.77),
    'Midtown': (-73.98, 40.76),
    'Lower Manhattan': (-74.00, 40.72),
    'Upper Manhattan': (-73.94, 40.82),
    'Brooklyn': (-73.95, 40.66)
}


def plot_lat_long(df, landmarks, points='Pickup'):
    plt.figure(figsize=(12, 12))

    if points == 'Pickup':
        plt.plot(list(df.pickup_longitude), list(
            df.pickup_latitude), '.', markersize=1)
    else:
        plt.plot(list(df.dropoff_longitude), list(
            df.dropoff_latitude), '.', markersize=1)

    for landmark in landmarks:
        plt.plot(landmarks[landmark][0], landmarks[landmark]
                 [1], '*', markersize=15, alpha=1, color='r')
        plt.annotate(landmark, (landmarks[landmark][0] + 0.005,
                                landmarks[landmark][1] + 0.005), color='r', backgroundcolor='w')

    plt.title("{} Locations in NYC Illustrated".format(points))
    plt.grid(None)
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.show()


plot_lat_long(df2, landmarks, points='Pickup')
