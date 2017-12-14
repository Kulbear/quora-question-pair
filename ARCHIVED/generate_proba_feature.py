import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm


def generate_models(model, paramter_grid, clfs, X_train, y_train):
    for params in list(paramter_grid):
        clf = model(**params)
        print('Training', clf.__class__.__name__, 'with', params)
        clf.fit(X_train, y_train)
        print(clf.score(X_train[:5000], y_train[:5000]), '\n')
        clfs.append(clf)


sgd_params = ParameterGrid({
    'loss': ['log'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'max_iter': [750, 1000],
    'tol': [1e-4, 1e-3],
    'random_state': [1, 2]
})
# knn_params = ParameterGrid({'n_neighbors': [4, 5, 6]})
rf_params = ParameterGrid({
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': [125, 250],
    'max_depth': [8, 9, 10],
    'random_state': [1, 2, 3]
})

dt_params = ParameterGrid({
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion': ['gini', 'entropy'],
    'max_depth': [8, 9, 10],
    'random_state': [1, 2, 3]
})

gnb_params = ParameterGrid({'priors ': [None]})

et_params = ParameterGrid({
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': [125, 250],
    'max_depth': [8, 9, 10],
    'random_state': [1, 2],
    'criterion': ['gini', 'entropy'],
})

adb_params = ParameterGrid({
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': [125, 250],
    'learning_rate': [1.],
    'random_state': [1, 2],
})

df_train = pd.read_csv('data/train.csv')
X_train = pd.read_csv('final_train_features.csv', index_col=False)
X_train = X_train.drop('Unnamed: 0', axis=1)
y_train = df_train['is_duplicate'].values

df_test = pd.read_csv('./data/test.csv')
X_test = pd.read_csv('./final_test_features.csv')
X_test = X_test.drop('Unnamed: 0', axis=1)

X_test = X_test.replace(np.nan, 0.)
X_test = X_test.replace(np.inf, 0.)
X_train = X_train.replace(np.nan, 0.)
X_train = X_train.replace(np.inf, 0.)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

idx = 0
clfs = []
generate_models(SGDClassifier, sgd_params, clfs, X_train_scaled, y_train)
for clf in clfs:
    idx += 1
    print('Running', clf.__class__.__name__)
    X_train_copy['clf_{}'.format(idx)] = clf.predict_proba(X_train)[:, 1]
    X_test_copy['clf_{}'.format(idx)] = clf.predict_proba(X_test)[:, 1]
    print(clf.__class__.__name__, ' is done!\n')

X_train_copy.to_csv('./train_with_prob.csv')
X_test_copy.to_csv('./test_with_prob.csv')

clfs = []
# generate_models(KNeighborsClassifier, knn_params, clfs)
generate_models(RandomForestClassifier, rf_params, clfs, X_train, y_train)
generate_models(DecisionTreeClassifier, dt_params, clfs, X_train, y_train)
for clf in clfs:
    idx += 1
    print('Running', clf.__class__.__name__)
    X_train_copy['clf_{}'.format(idx)] = clf.predict_proba(X_train)[:, 1]
    X_test_copy['clf_{}'.format(idx)] = clf.predict_proba(X_test)[:, 1]
    print(clf.__class__.__name__, ' is done!\n')

X_train_copy.to_csv('./train_with_prob.csv')
X_test_copy.to_csv('./test_with_prob.csv')

clfs = []
generate_models(GaussianNB, gnb_params, clfs, X_train, y_train)
generate_models(ExtraTreeClassifier, et_params, clfs, X_train, y_train)
generate_models(AdaBoostClassifier, adb_params, clfs, X_train, y_train)
for clf in clfs:
    idx += 1
    print('Running', clf.__class__.__name__)
    X_train_copy['clf_{}'.format(idx)] = clf.predict_proba(X_train)[:, 1]
    X_test_copy['clf_{}'.format(idx)] = clf.predict_proba(X_test)[:, 1]
    print(clf.__class__.__name__, ' is done!\n')

X_train_copy.to_csv('./train_with_prob.csv')
X_test_copy.to_csv('./test_with_prob.csv')
