import tqdm
import pandas as pd

import keras
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler
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


index = top300
x_train_numpy = np.array(x_train[index])
x_test_numpy = np.array(x_test[index])
y_train_numpy = y_train['label'].values
y_test_numpy = y_test['label'].values

x_train_numpy = x_train_numpy[y_train_numpy != 3]
x_test_numpy = x_test_numpy[y_test_numpy != 3]
y_train_numpy = y_train_numpy[y_train_numpy != 3]
y_test_numpy = y_test_numpy[y_test_numpy != 3]

scaler = StandardScaler()
scaler.fit(x_train_numpy)
x_train_numpy = scaler.transform(x_train_numpy)
x_test_numpy = scaler.transform(x_test_numpy)

train_data = lgb.Dataset(x_train_numpy, label=y_train_numpy)
validation_data = lgb.Dataset(x_test_numpy, label=y_test_numpy)

params = {
    # 'num_leaves': 31,
    # 'min_data_in_leaf': 30,
    'objective': 'multiclass',
    'num_class': num_classes,
    'max_depth': 10,
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
    'metric': 'multi_logloss',
    # 'metric': 'multi_error',
    'random_state': 0,
}
clf = lgb.train(params, train_data, valid_sets=[train_data, validation_data],
                num_boost_round=100, early_stopping_rounds=5)

y_pred = clf.predict(x_test_numpy)
y_pred2 = y_pred.argmax(axis=1)
print(accuracy_score(y_test_numpy, y_pred2))
aaa = confusion_matrix(y_test_numpy, y_pred2)
bbb = (aaa.T / aaa.sum(axis=1)).T
print(pd.DataFrame(bbb))

