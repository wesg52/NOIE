import pandas as pd
from utils import get_arg_tokenizer, swap_tokens


def make(save_path, sample=-1):
    sents = pd.read_csv('data/neural_oie.sent', sep='\n', header=None)
    trips = pd.read_csv('data/neural_oie.triple', sep='\n', header=None)

    df = pd.concat([sents, trips], axis=1)
    df.columns = ['sents', 'trips']

    if sample > 0:
        df = df.sample(sample)

    df.loc[:, 'trips'] = df.loc[:, 'trips'].apply(lambda x: swap_tokens(x))

    tokenizer = get_arg_tokenizer()

    df.loc[:, 'sents'] = df.loc[:, 'sents'].apply(lambda t: ' '.join(tokenizer.tokenize(t)))
    df.loc[:, 'trips'] = df.loc[:, 'trips'].apply(lambda t: ' '.join(tokenizer.tokenize(t)))

    df.to_csv(save_path, index=False)

if __name__ == '__main__':
    save_path = 'data/dataset.csv'
    make(save_path)
