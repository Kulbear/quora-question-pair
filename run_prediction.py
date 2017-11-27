import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, Booster

SUBMISSION_PATH = './submission.csv'

print('\nLoading :: original test set...')
df_test = pd.read_csv('./data/test.csv')

print('\nLoading :: pre-trained model...')
bst = xgb.Booster()
bst.load_model("./model.mdl")

print('\nLoading :: pre-processed test feature set...')
X_test = pd.read_csv('./final_test_features.csv')
X_test = X_test.drop('Unnamed: 0', axis=1)
print('Test feature size:', X_test.shape)

d_test = xgb.DMatrix(X_test)
print('\nPredicting :: yo~~~hooo~~~')
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
print('\nPrediction result:\n',sub.head(5))

print('Saving :: submission file...')
sub.to_csv(SUBMISSION_PATH, index=False)