# 머신 러닝 워크플로에는 데이터 처리단계가 너무많아 관리하기 어려운 경우가 많으므로, 평소에도 코드를 모듈화 해놓자.
import pandas as pd
import numpy as np

# 데이터 전처리 모듈


def preprocess(df):
    # DataFrame 내 결측값 제거
    def remove_missing_values(df):
        # 데이터 전처리
        # print(df.isnull().any()) #결측값 유무
        # print(df.isnull().sum()) # 결측값 수
        df.dropna()
        return df

    # 요금 이상치 제거

    # print(df.describe()) # 병신같은 데이터 있는지 확인
    # 요금이 -44달러가 나온게 있고, 500달러 나온게 있음 > 병신같다!
    # df['fare amount'].hist(bins = 500)
    # plt.xlabel('Fare')
    # plt.title('Histogram of Fares')
    # plt.show()
    # 히스토그램 분석 시, 500 달러 같은 이상치가 눈에 띄게 있진 않으므로 제거 해버리는게 좋다.
    # 0~100 달러 가지고만 분석!

    def remove_fare_amount_outliers(df, lower_bound, upper_bound):
        df = df[(df['fare_amount'] >= lower_bound) &
                (df['fare_amount'] <= upper_bound)]
        return df

    # 승객 최빈수 0 -> 1 대체
    # 그리고 자세히보면 승객수 min 이 0인 데이터가 있는데 이를 버리지 말고 1로 바꾸어 준다.

    def replace_passenger_count_outliers(df):
        # mode () : Return the highest frequency value in a Series.
        mode = df['passenger_count'].mode().values[0]
        df.loc[df['passenger_count'] == 0, 'passenger_count'] = mode
        return df

    # 경도 위도 또한 뉴욕시 내부로 제한하자.
    def remove_lat_long_outliers(df):
        # 뉴욕시 경도 범위
        nyc_min_longitude = -74.05
        nyc_max_longitude = -73.75

        # 뉴욕시 위도 범위
        nyc_min_latitude = 40.63
        nyc_max_latitude = 40.85

        # min ~ max 사이 경도만 필터링
        for long in ['pickup_longitude', 'dropoff_longitude']:
            df = df[(df[long] > nyc_min_longitude) &
                    (df[long] < nyc_max_longitude)]
        # min ~ max 사이 위도만 필터링
        for long in ['pickup_latitude', 'dropoff_latitude']:
            df = df[(df[long] > nyc_min_latitude) &
                    (df[long] < nyc_max_latitude)]
        return df

    df = remove_missing_values(df)
    df = remove_fare_amount_outliers(df, lower_bound=0, upper_bound=100)
    df = replace_passenger_count_outliers(df)
    df = remove_lat_long_outliers(df)
    return df


# 특징변수 모듈

# 특징변수 1 : 시간 관련 변수
def feature_engineer(df):
    # 연, 월, 일 ,요일 , 시간 칼럼 만들기(datetime 형식을 처리할 수 없음)
    def create_time_features(df):
        df['year'] = df['pickup_datetime'].dt.year
        df['month'] = df['pickup_datetime'].dt.month
        df['day'] = df['pickup_datetime'].dt.day
        df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['hour'] = df['pickup_datetime'].dt.hour
        df = df.drop(['pickup_datetime'], axis=1)
        return df

    # 특징변수  2 : 위도, 경도 좌표로 직선거리 계산

    def euc_distance(lat1, long1, lat2, long2):
        return (((lat1-lat2) ** 2 + (long1 - long2) ** 2) ** 0.5)

    def create_pickup_dropoff_dist_features(df):
        df['travel_distance'] = euc_distance(df['pickup_latitude'], df['pickup_longitude'],
                                             df['dropoff_latitude'], df['dropoff_longitude'])
        return df

    # 특징변수 3 : 공항과의 거리 (고정요금 붙음)

    # 위치 그래프 분석 코드
    # def euc_distance(lat1, long1, lat2, long2):
    #     return (((lat1-lat2) ** 2 + (long1 - long2) ** 2) ** 0.5)

    # df['distance'] = euc_distance(df['pickup_latitude'], df['pickup_longitude'],
    #                               df['dropoff_latitude'], df['dropoff_longitude'])

    # df.plot.scatter('fare_amount', 'distance')
    # plt.show()

    # 결론은 승차시, 하차시에 공항근처 라면 기존 거리 요금 + 통행요금 52달러 가 추가로 붙음 이를 반영해 줘야함.

    def create_airport_dist_features(df):
        airports = {'JFK_Airport': (-73.78, 40.643),
                    'Laguardia_Airport': (-73.87, 40.77),
                    'Newark_Airport': (-74.18, 40.69)}

        # 따라서 뉴욕시 주요공항과 승차 및 하차위치 사이의 거리를 새로운 특징변수로 추가할 필요가 있음.
        for airport in airports:
            df['pickup_dist_'+airport] = euc_distance(
                df['pickup_latitude'], df['pickup_longitude'], airports[airport][1], airports[airport][0])
            df['dropoff_dist_'+airport] = euc_distance(
                df['dropoff_latitude'], df['dropoff_longitude'], airports[airport][1], airports[airport][0])
        return df

    df = create_time_features(df)
    df = create_pickup_dropoff_dist_features(df)
    df = create_airport_dist_features(df)
    df = df.drop(['key'], axis=1)
    return df
