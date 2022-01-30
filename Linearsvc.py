import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.model_selection import GridSearchCV
from sklearn import svm

##訓練データとテストデータ、提出用サンプルデータの読み込み
train = pd.read_table('train.tsv')
test = pd.read_table('test.tsv')
sample = pd.read_csv('sample_submit.csv', header=None)

#訓練データから目的変数を削除
trainX = train.drop('bot', axis=1)

#標準化
scaler = StandardScaler()
scaler.fit(trainX)
scaler.transform(trainX)

#EFSを用いて選ばれた特徴量を選択
trainX = trainX[['default_profile', 'default_profile_image', 'favourites_count', 'geo_enabled', 'listed_count', 'diversity', 'mean_tweet_length', 'mean_retweets', 'reply_rate']]

#目的変数
y = train['bot']

#テストデータのコピー
testX = test.copy()

#EFSで特徴量選択
"""
clf = svm.LinearSVC()
efs1 = EFS(clf, min_features=9, max_features=11)
efs1 = efs1.fit(trainX, y)
print('Best accuracy score: %.2f' % efs1.best_score_)
print('Best subset:', efs1.best_feature_names_)
"""

#テストデータの特徴量選択
testX = testX[['default_profile', 'default_profile_image', 'favourites_count', 'geo_enabled', 'listed_count', 'diversity', 'mean_tweet_length', 'mean_retweets', 'reply_rate']]

#テストデータと検証用データを8:2で分割
X_train,X_test,y_train,y_test = train_test_split(trainX,y,test_size=0.20,random_state=1)

#アンダーサンプリング
positive_count_train = y_train.value_counts()[1]
strategy = {0:positive_count_train*2, 1:positive_count_train}
rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
y_resampled.value_counts()

#グリッドサーチによって求めたパラメータを引数に入力したモデル
clf = svm.LinearSVC(penalty="l2", C=0.5, multi_class="ovr")                                   

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
sample.to_csv('LinearSVC.csv', index=None, header=None)
"""