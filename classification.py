import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import lightgbm as lgb
from google.colab import files

import lightgbm as lgb
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score
import time


uploaded = files.upload()
lgm = ('CompleteResponses.csv')
df = pd.read_csv(lgm)
X = df.drop(['brand'], axis=1)
y = df['brand']

time_gone_XGB=np.zeros(3)
accuracyXGB = np.zeros(3)
print('XGBoost Model accuracy score: ')
for i in range(2,5):
  time_start = time.time()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1*i, random_state = 0)
  clf = xgb.XGBClassifier()
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  accuracyXGB[i-2]=accuracy_score(y_pred, y_test)
  time_gone_XGB[i-2]=(time.time()-time_start)
  print('{0:0.1f}'.format(i*0.1), '{0:0.4f}'.format(accuracyXGB[i-2]), '{0:0.3f}'.format(time_gone_XGB[i-2]))

time_gone_light=np.zeros(3)
accuracy_light = np.zeros(3)
print('LightGBM Model accuracy score: ')
for i in range(2,5):
  time_start = time.time()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1*i)
  clf = lgb.LGBMClassifier()
  clf.fit(X_train, y_train)
  y_pred=clf.predict(X_test)
  accuracy_light[i-2]=accuracy_score(y_pred, y_test)
  time_gone_light[i-2]=(time.time()-time_start)
  print('{0:0.1f}'.format(i*0.1), '{0:0.4f}'.format(accuracy_light[i-2]), '{0:0.3f}'.format(time_gone_light[i-2]))

time_gone_logistic=np.zeros(9)
accuracy_logistic = np.zeros(9)

print('Logistic Regression Model accuracy score: ')
for i in range(2,5):
  time_start = time.time()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1*i, random_state = 0)
  logreg = LogisticRegression()
  logreg.fit(X_train, y_train)
  y_pred = logreg.predict(X_test)
  accuracy_logistic[i-2]=accuracy_score(y_pred, y_test)
  time_gone_logistic[i-2]=(time.time()-time_start)
  print('{0:0.1f}'.format(i*0.1), '{0:0.4f}'.format(accuracy_logistic[i-2]), '{0:0.3f}'.format(time_gone_logistic[i-2]))