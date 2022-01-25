import pandas as pd
import numpy as np
import sklearn.model_selection as cross_validation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve 
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.model_selection import GridSearchCV


train = pd.read_table('/Users/e195718/3year/secondSemester/DataMining/bot/train.tsv')
test = pd.read_table('/Users/e195718/3year/secondSemester/DataMining/bot/test.tsv')
sample = pd.read_csv('/Users/e195718/3year/secondSemester/DataMining/bot/sample_submit.csv', header=None)

trainX = train.drop('bot', axis=1)
trainX = trainX[['id', 'default_profile', 'default_profile_image', 'friends_count', 'followers_count', 'favourites_count', 'geo_enabled', 'listed_count', 'mean_mins_between_tweets', 'mean_tweet_length']]
y = train['bot']

testX = test.copy()

positive_count_train = y.value_counts()[1]
strategy = {0:positive_count_train*2, 1:positive_count_train}
rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)
X_resampled, y_resampled = rus.fit_resample(trainX, y)
y_resampled.value_counts()
#EFSで特徴量選択

"""
clf = RandomForestClassifier()
efs1 = EFS(clf, min_features=10, max_features=15)
efs1 = efs1.fit(trainX, y)
print('Best accuracy score: %.2f' % efs1.best_score_)
print('Best subset:', efs1.best_feature_names_)
"""

testX = testX[['id', 'default_profile', 'default_profile_image', 'friends_count', 'followers_count', 'favourites_count', 'geo_enabled', 'listed_count', 'mean_mins_between_tweets', 'mean_tweet_length']]

X_train,X_test,y_train,y_test = train_test_split(trainX,y,test_size=0.20,random_state=1)


parameters = {  
    'n_estimators': [600, 650, 700],     # 用意する決定木モデルの数
    'max_depth': [5, 6, 7],     # 決定木のノード深さの制限値
}


clf = RandomForestClassifier(max_depth=7, max_features=None, n_estimators=600)                                   


gridsearch = GridSearchCV(estimator = clf,        # モデル
                          param_grid = parameters,  # チューニングするハイパーパラメータ
                          cv = 5,
                          scoring = "f1"      # スコアリング
                         )   

gridsearch.fit(X_train, y_train)
print('Best params: {}'.format(gridsearch.best_params_)) 
print('Best Score: {}'.format(gridsearch.best_score_))



clf.fit(X_train, y_train)

# 学習データに対する精度
y_pred = clf.predict(X_test) #テスト用データの予測

trainaccuracy_random_forest = clf.score(X_train, y_train)
print('TrainAccuracy: {}'.format(trainaccuracy_random_forest))

accuracy = accuracy_score(y_test, y_pred)
print('TestAccuracy: {}'.format(accuracy))

pred = clf.predict(testX)
sample[1] = pred
sample.to_csv('undersumpling.csv', index=None, header=None)
