import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#データの読み込み&確認
train = pd.read_csv("/Users/e195767/VS code/SIGNATE/bot/train.tsv", sep="\t", index_col=0)
test = pd.read_csv("/Users/e195767/VS code/SIGNATE/bot/test.tsv", sep="\t", index_col=0)
submit = pd.read_csv("/Users/e195767/VS code/SIGNATE/bot/sample_submit.csv", index_col=0, header=None)

print(train.info())
print(train.describe())
res = train.corr()
print(res["bot"])

#可視化
hum_df = train[train["bot"] == 0]
bot_df = train[train["bot"] == 1]
train["bot"].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',shadow=True)

search_idx = "friends_count"

plt.hist(hum_df[search_idx], alpha=0.5, label="human")
plt.hist(bot_df[search_idx], alpha=0.5, label="bot")
plt.show()


#学習
features = train.drop("bot", axis=1)
target = train["bot"]

#print(features.shape)
#print(target.shape)

#X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, random_state=0)

#前処理
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss_features = ss.fit_transform(features)
ss_X_train, ss_X_test, ss_y_train, ss_y_test = train_test_split(ss_features, target, train_size=0.8, random_state=0)

#アンダーサンプリング
from imblearn.under_sampling import RandomUnderSampler
positive_count_train = ss_y_train.value_counts()[1]
strategy = {0:positive_count_train*2, 1:positive_count_train}

rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)
X_resampled, y_resampled = rus.fit_resample(ss_X_train, ss_y_train)
y_resampled.value_counts()

#EFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
clf = RandomForestClassifier(max_depth=2, random_state=0)
efs1 = EFS(clf, min_features=10, max_features=14)
efs1 = efs1.fit(X_resampled,  y_resampled)
print('Best accuracy score: %.2f' % efs1.best_score_)
print('Best subset:', efs1.best_feature_names_)

#GredSearchCV
from sklearn.model_selection import GridSearchCV
def params():
  ret = {
      "penalty":["none", "l1", "l2", "elasticnet"],
      "C":[10 ** i for i in range(-5, 6)], #10**-5 ~ 10**5
      "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
  }
  return ret

gscv = GridSearchCV(LogisticRegression(), params(), cv=4, verbose=2)
gscv.fit(X_resampled, y_resampled)

gscv_result = pd.DataFrame.from_dict(gscv.cv_results_)
gscv_result.sort_values("rank_test_score")

best = gscv.best_estimator_
pred = best.predict(ss_X_test)

score = best.score(ss_X_test, ss_y_test)
print(score)

"""
#ロジスティック回帰
LR = LogisticRegression()
LR.fit(ss_X_train, ss_y_train)

score = LR.score(ss_X_test, ss_y_test)
print(score)
"""

#予測
ss_test = ss.fit_transform(test)

pred = best.predict(test)
pred = pred.astype(np.int64)

submit[1] = pred
submit.to_csv("/Users/e195767/VS code/SIGNATE/bot/submit.csv", header=None)