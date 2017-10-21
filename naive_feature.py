import pandas as pd

pd.options.mode.chained_assignment = None


def generate_naive_features(data):
    df = data.copy()

    df['q1_char_length_with_space'] = df.question1.apply(lambda x: len(str(x)))  # with space
    df['q1_char_length_without_space'] = df.question1.apply(lambda x: len(str(x).replace(' ', '')))  # without space
    df['q1_word_length'] = df.question1.apply(lambda x: len(str(x).split(' ')))
    df['q1_question_mark'] = df.question1.apply(lambda x: str(x).count('?'))  # TODO: prob not good?

    df['q2_char_length_with_space'] = df.question2.apply(lambda x: len(str(x)))  # with space
    df['q2_char_length_without_space'] = df.question2.apply(lambda x: len(str(x).replace(' ', '')))  # without space
    df['q2_word_length'] = df.question2.apply(lambda x: len(str(x).split(' ')))
    df['q2_question_mark'] = df.question2.apply(lambda x: str(x).count('?'))  # TODO: prob not good?

    df['word_length_diff'] = abs(df.q2_word_length - df.q1_word_length)
    df['char_length_diff'] = abs(df.q2_char_length_without_space - df.q1_char_length_without_space)

    return df


if __name__ == '__main__':
    train_data = pd.read_csv('train.csv', header=0)
    test_data = pd.read_csv('test.csv', header=0)

    train_data = generate_naive_features(train_data)
    test_data = generate_naive_features(test_data)

    train_data.to_csv('train1.csv', index=False)
    test_data.to_csv('test1.csv', index=False)
