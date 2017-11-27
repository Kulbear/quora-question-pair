import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier

MODEL_PATH = 'model.mdl'

df_train = pd.read_csv('data/train.csv')

X_train = pd.read_csv('final_train_features.csv', index_col=False)
X_train = X_train.drop('Unnamed: 0', axis=1)
X_train.head()

y_train = df_train['is_duplicate'].values

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.1, random_state=4242)

# Up/down sampling
pos_train = X_train[y_train == 1]
neg_train = X_train[y_train == 0]
X_train = pd.concat((neg_train, pos_train.iloc[:int(0.8 * len(pos_train))],
                     neg_train))
y_train = np.array([0] * neg_train.shape[0] + [1] * pos_train.iloc[:int(
    0.8 * len(pos_train))].shape[0] + [0] * neg_train.shape[0])
print(np.mean(y_train))
del pos_train, neg_train

pos_valid = X_valid[y_valid == 1]
neg_valid = X_valid[y_valid == 0]
X_valid = pd.concat((neg_valid, pos_valid.iloc[:int(0.8 * len(pos_valid))],
                     neg_valid))
y_valid = np.array([0] * neg_valid.shape[0] + [1] * pos_valid.iloc[:int(
    0.8 * len(pos_valid))].shape[0] + [0] * neg_valid.shape[0])
print(np.mean(y_valid))
del pos_valid, neg_valid

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.025
params['max_depth'] = 7
params['subsample'] = 0.8
params['base_score'] = 0.2
params['n_estimators'] = 1000
# params['scale_pos_weight'] = 0.2

d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(
    params,
    d_train,
    2500,
    watchlist,
    early_stopping_rounds=75,
    verbose_eval=50)
print(log_loss(y_valid, bst.predict(d_valid)))
bst.save_model(MODEL_PATH)
