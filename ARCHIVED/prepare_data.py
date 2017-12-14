import pandas as pd

from nltk.corpus import stopwords
from collections import Counter, defaultdict
from helpers import build_features, get_weight

print('Start data preparation')

print('\nLoading :: raw training data...')
df_train = pd.read_csv('./data/train.csv')
df_train = df_train.fillna(' ')

print('\nLoading :: raw test data...')
df_test = pd.read_csv('./data/test.csv')
ques = pd.concat([df_train[['question1', 'question2']], \
    df_test[['question1', 'question2']]], axis=0).reset_index(drop='index')

print('\nBuilding :: intersect features...')
q_dict = defaultdict(set)
for i in range(ques.shape[0]):
    q_dict[ques.question1[i]].add(ques.question2[i])
    q_dict[ques.question2[i]].add(ques.question1[i])


def q1_freq(row):
    return (len(q_dict[row['question1']]))


def q2_freq(row):
    return (len(q_dict[row['question2']]))


def q1_q2_intersect(row):
    return (len(
        set(q_dict[row['question1']]).intersection(
            set(q_dict[row['question2']]))))


df_train['q1_q2_intersect'] = df_train.apply(q1_q2_intersect, axis=1, raw=True)
df_train['q1_freq'] = df_train.apply(q1_freq, axis=1, raw=True)
df_train['q2_freq'] = df_train.apply(q2_freq, axis=1, raw=True)

df_test['q1_q2_intersect'] = df_test.apply(q1_q2_intersect, axis=1, raw=True)
df_test['q1_freq'] = df_test.apply(q1_freq, axis=1, raw=True)
df_test['q2_freq'] = df_test.apply(q2_freq, axis=1, raw=True)

test_leaky = df_test.loc[:, ['q1_q2_intersect', 'q1_freq', 'q2_freq']]
del df_test  # manual release memory

train_leaky = df_train.loc[:, ['q1_q2_intersect', 'q1_freq', 'q2_freq']]

stops = set(stopwords.words("english"))

df_train['question1'] = df_train['question1'].map(
    lambda x: str(x).lower().split())
df_train['question2'] = df_train['question2'].map(
    lambda x: str(x).lower().split())

train_qs = pd.Series(
    df_train['question1'].tolist() + df_train['question2'].tolist())

words = [x for y in train_qs for x in y]
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

print('\nLoading :: training leaky features...')
df_train_ = pd.read_csv('./data/train_features.csv', encoding="ISO-8859-1")
X_train_ab = df_train_.iloc[:, 2:-1]
X_train_ab = X_train_ab.drop('euclidean_distance', axis=1)
X_train_ab = X_train_ab.drop('jaccard_distance', axis=1)

print('Building :: other training features...')
X_train = build_features(df_train, stops, weights)
X_train = pd.concat((X_train, X_train_ab, train_leaky), axis=1)
y_train = df_train['is_duplicate'].values

try:
    print('Saving :: final train features...')
    X_train.to_csv('./final_train_features.csv')
except:
    print('WARNING :: Failed saving train features...')

print('\nLoading :: raw test data...')
df_test = pd.read_csv('./data/test.csv')
df_test = df_test.fillna(' ')

df_test['question1'] = df_test['question1'].map(
    lambda x: str(x).lower().split())
df_test['question2'] = df_test['question2'].map(
    lambda x: str(x).lower().split())

print('Loading :: test leaky features from files...')
df_test_ = pd.read_csv('./data/test_features.csv', encoding="ISO-8859-1")
x_test_ab = df_test_.iloc[:, 2:-1]
x_test_ab = x_test_ab.drop('euclidean_distance', axis=1)
x_test_ab = x_test_ab.drop('jaccard_distance', axis=1)

print('Building :: other test features...')
x_test = build_features(df_test, stops, weights)
x_test = pd.concat((x_test, x_test_ab, test_leaky), axis=1)

try:
    print('Saving :: final test features...')
    x_test.to_csv('./final_test_features.csv')
except:
    print('WARNING :: Failed saving test features...')
