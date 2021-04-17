import tqdm
import pandas as pd

import keras
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import interp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

num_classes = 3
epochs = 300
model_name = 'keras_cnn_trained_model_shallow.h5'

with open('./feature_engineering/data/x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open('./feature_engineering/data/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('./feature_engineering/data/x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('./feature_engineering/data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)


x_train_numpy = np.array(x_train)
x_test_numpy = np.array(x_test)
y_train_numpy = y_train['label'].values
y_test_numpy = y_test['label'].values

train_mask = (y_train_numpy != 3) & (y_train_numpy != 2)
test_mask = (y_test_numpy != 3) & (y_test_numpy != 2)

x_train_numpy = x_train_numpy[train_mask]
x_test_numpy = x_test_numpy[test_mask]
y_train_numpy = y_train_numpy[train_mask]
y_test_numpy = y_test_numpy[test_mask]

scaler = StandardScaler()
scaler.fit(x_train_numpy)
x_train_numpy = scaler.transform(x_train_numpy)
x_test_numpy = scaler.transform(x_test_numpy)


#####################
train_data = lgb.Dataset(x_train_numpy, label=y_train_numpy)
validation_data = lgb.Dataset(x_test_numpy, label=y_test_numpy)

params = {
    # 'num_leaves': 31,
    # 'min_data_in_leaf': 30,
    'objective': 'binary',
    'max_depth': -1,
    'learning_rate': 0.1,
    # 'min_sum_hessian_in_leaf': 6,
    'boosting': 'gbdt',
    # 'feature_fraction': 0.9,
    # 'bagging_freq': 1,
    # 'bagging_fraction': 0.8,
    # 'bagging_seed': 11,
    # 'lambda_l1': 0.1,
    # 'lambda_l2': 0.2,
    'verbosity': -1,
    'metric': 'binary',
    'random_state': 0,
}
clf = lgb.train(params, train_data, valid_sets=[train_data, validation_data],
                num_boost_round=100, early_stopping_rounds=5)

y_pred = clf.predict(x_test_numpy)
y_pred2 = (y_pred > 0.5).astype('int')
print(accuracy_score(y_test_numpy, y_pred2))
aaa = confusion_matrix(y_test_numpy, y_pred2)
bbb = (aaa.T / aaa.sum(axis=1)).T
print(pd.DataFrame(bbb))

#####################
clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=10, random_state=0, min_samples_leaf=1)
clf.fit(x_train_numpy, y_train_numpy)
y_pred = clf.predict_proba(x_test_numpy)[:, 1]
y_pred2 = (y_pred > 0.5).astype('int')
print(accuracy_score(y_test_numpy, y_pred2))
aaa = confusion_matrix(y_test_numpy, y_pred2)
bbb = (aaa.T / aaa.sum(axis=1)).T
print(pd.DataFrame(bbb))


plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = metrics.roc_curve(y_test_numpy, y_pred, pos_label=1)
plt.plot(fpr, tpr)
plt.grid()
plt.plot([0, 1], [0, 1])
plt.xlabel('FP')
plt.ylabel('TP')
plt.ylim([0, 1])
plt.xlim([0, 1])
auc = np.trapz(tpr, fpr)
print('AUC:', auc)
plt.title('AUC:' + str(auc))
plt.show()
