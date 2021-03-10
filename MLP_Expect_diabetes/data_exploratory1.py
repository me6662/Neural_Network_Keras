import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('./diabetes.csv')

df.hist()
plt.tight_layout()
plt.show()
