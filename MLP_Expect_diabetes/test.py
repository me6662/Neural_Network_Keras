import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('./diabetes.csv')

print(df.loc[df['Insulin'] == 0].shape[0])
