from fuzzywuzzy import fuzz
import pandas as pd

pd.options.mode.chained_assignment = None


def generate_naive_features(data):
    df = data.copy()
    
    df['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x.question1), str(x.question2)), axis=1)
    df['fuzz_WRatio'] = df.apply(lambda x: fuzz.WRatio(str(x.question1), str(x.question2)), axis=1)
    df['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x.question1), str(x.question2)), axis=1)
    df['fuzz_partial_token_set_ratio'] = df.apply(lambda x: fuzz.partial_token_set_ratio(str(x.question1), str(x.question2)), axis=1)
    df['fuzz_partial_token_sort_ratio'] = df.apply(lambda x: fuzz.partial_token_sort_ratio(str(x.question1), str(x.question2)), axis=1)
    df['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x.question1), str(x.question2)), axis=1)
    df['fuzz_token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(str(x.question1), str(x.question2)), axis=1)

    return df


if __name__ == '__main__':
    train_data = pd.read_csv('train.csv', header=0)
    test_data = pd.read_csv('test.csv', header=0)

    train_data = generate_naive_features(train_data)
    test_data = generate_naive_features(test_data)

    train_data.to_csv('train1.csv', index=False)
    test_data.to_csv('test1.csv', index=False)
