import pandas as pd
import numpy as np
import sklearn.model_selection as cross_validation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

train = pd.read_table('train.tsv')
test = pd.read_table('test.tsv')
sample = pd.read_csv('sample_submit.csv', header=None)


plt.figure(figsize=(11, 9))
corr_matrix = train.corr()
sns.heatmap(corr_matrix, annot=True,fmt='.2f',cmap='Blues',square=True)
corr_y = pd.DataFrame({"features":train.columns,"corr_y":corr_matrix["bot"]},index=None)
corr_y = corr_y.reset_index(drop=True)
plt.tight_layout()
plt.show()
