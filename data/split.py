import random
import pandas as pd

def get_splits(length, n_splits):
    splits = []
    ixs = [i for i in range(length)]
    random.shuffle(ixs)
    for i in range(n_splits):
        splits.append(ixs[i*length//(n_splits):min(length, (i+1)*length//(n_splits))])
    return splits

df = pd.read_csv('val_1mil.csv')

splits = get_splits(len(df), 9)

for ix, spt in enumerate(splits):
    subdf = df.iloc[spt, :]
    subdf.to_csv('full' + str(ix) + '.csv', index=False)