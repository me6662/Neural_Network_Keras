import pandas as pd


df = pd.read_csv('NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)
print(df.info())