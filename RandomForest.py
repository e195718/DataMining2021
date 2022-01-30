import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.model_selection import GridSearchCV

#訓練データとテストデータ、提出用サンプルデータの読み込み
train = pd.read_table('train.tsv')
test = pd.read_table('test.tsv')
sample = pd.read_csv('sample_submit.csv', header=None)

#訓練データから目的変数を削除
trainX = train.drop('bot', axis=1)
#EFSを用いて選ばれた特徴量を選択
trainX = trainX[['default_profile', 'default_profile_image', 'friends_count', 'followers_count', 'favourites_count', 'geo_enabled', 'listed_count', 'mean_mins_between_tweets', 'mean_tweet_length']]
#目的変数
y = train['bot']

#テストデータの特徴量選択
testX = test[['default_profile', 'default_profile_image', 'friends_count', 'followers_count', 'favourites_count', 'geo_enabled', 'listed_count', 'mean_mins_between_tweets', 'mean_tweet_length']]


#EFSで特徴量選択
"""
clf = RandomForestClassifier()
efs1 = EFS(clf, min_features=10, max_features=15)
efs1 = efs1.fit(trainX, y)
print('Best accuracy score: %.2f' % efs1.best_score_)
print('Best subset:', efs1.best_feature_names_)
"""

#テストデータと検証用データを8:2で分割
X_train,X_test,y_train,y_test = train_test_split(trainX,y,test_size=0.20,random_state=1)

#アンダーサンプリング
positive_count_train = y_train.value_counts()[1]
strategy = {0:positive_count_train*2, 1:positive_count_train}
rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
y_resampled.value_counts()

#グリッドサーチで調整するパラメータ
"""
parameters = {  
    'n_estimators': [600, 650, 700],     # 用意する決定木モデルの数
    'max_depth': [5, 6, 7],     # 決定木のノード深さの制限値
}

gridsearch = GridSearchCV(estimator = clf,        # モデル
                          param_grid = parameters,  # チューニングするハイパーパラメータ
                          cv = 5,
                          scoring = "f1"      # スコアリング
                         )   

#グリッドサーチ実行し、ベストなパラメータの値を出力
gridsearch.fit(X_train, y_train)
print('Best params: {}'.format(gridsearch.best_params_)) 
print('Best Score: {}'.format(gridsearch.best_score_))
"""

#グリッドサーチによって求めたパラメータを引数に入力したモデル
clf = RandomForestClassifier(max_depth=7, max_features=None, n_estimators=600)                                   

#学習
clf.fit(X_resampled, y_resampled)

#テスト用データの予測
y_pred = clf.predict(X_test)

#訓練データに対する実行結果
trainaccuracy_random_forest = clf.score(X_resampled, y_resampled)
print('TrainAccuracy: {}'.format(trainaccuracy_random_forest))

#検証用データに対する実行結果
accuracy = accuracy_score(y_test, y_pred)
print('TestAccuracy: {}'.format(accuracy))

"""
#提出用データに対する予測と提出ファイル(.csv)作成
pred = clf.predict(testX)
sample[1] = pred
sample.to_csv('undersumpling.csv', index=None, header=None)
"""