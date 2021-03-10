import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


df = pd.read_csv('./diabetes.csv')

# 3 x 3 크기로 서브차트
plt.subplots(3, 3, figsize=(10, 10))
print(df.loc[df.Outcome == 0]['Glucose'])

# 각 특징의 밀도 차트 (df 의 column 별로 outcome 의 분포를 보여줌)
for idx, col in enumerate(df.columns[:-1]):
    ax = plt.subplot(3, 3, idx+1)
    ax.yaxis.set_ticklabels([])
    sns.distplot(df.loc[df.Outcome == 0][col], hist=False, axlabel=False, kde_kws={
                 'linestyle': '-', 'color': 'black', 'label': 'No Diabetes'})
    sns.distplot(df.loc[df.Outcome == 1][col], hist=False, axlabel=False, kde_kws={
                 'linestyle': '--', 'color': 'black', 'label': 'Diabetes'})
    ax.set_title(col)
# 차트 9개 중에 마지막 차트 는 숨김 (보여지는 것은 8개 이므로)
plt.subplot(3, 3, 9).set_visible(False)
plt.tight_layout()  # 화면 꽉 채우기
plt.show()
