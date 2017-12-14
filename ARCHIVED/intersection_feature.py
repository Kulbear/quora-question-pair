# Reference: https://www.kaggle.com/tour1st/magic-feature-v2-0-045-gain

import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def get_q_dict(questions):
    q_dict = defaultdict(set)

    for i in tqdm(range(questions.shape[0])):
        q_dict[questions.question1[i]].add(questions.question2[i])
        q_dict[questions.question2[i]].add(questions.question1[i])

    return q_dict


def q1_q2_intersect(row):
    global q_dict
    return len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']])))


if __name__ == '__main__':
    global q_dict
    train_data = pd.read_csv('train.csv', header=0)
    test_data = pd.read_csv('test.csv', header=0)

    all_questions = pd.concat([train_data[['question1', 'question2']], test_data[['question1', 'question2']]], axis=0).reset_index(drop='index')
    q_dict = get_q_dict(all_questions)

    train_data['q1_q2_intersect'] = train_data.apply(q1_q2_intersect, axis=1, raw=True)
    test_data['q1_q2_intersect'] = test_data.apply(q1_q2_intersect, axis=1, raw=True)

    train_data.to_csv('train.csv', index=False)
    test_data.to_csv('test.csv', index=False)
