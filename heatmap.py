import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##訓練データとテストデータ、提出用サンプルデータの読み込み
train = pd.read_table('train.tsv')
test = pd.read_table('test.tsv')
sample = pd.read_csv('sample_submit.csv', header=None)

#ヒートマップを作成し表示する
plt.figure(figsize=(11, 9))
corr_matrix = train.corr()
sns.heatmap(corr_matrix, annot=True,fmt='.2f',cmap='Blues',square=True)
corr_y = pd.DataFrame({"features":train.columns,"corr_y":corr_matrix["bot"]},index=None)
corr_y = corr_y.reset_index(drop=True)
plt.tight_layout()
plt.show()
