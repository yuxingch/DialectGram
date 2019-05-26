import pandas as pd
from collections import Counter
from scipy.io import loadmat

def load_mat_data(path ="./data/data.mat"):
    return loadmat(path)

def load_data(path):
    return pd.read_csv(path, sep="\t", header=None)

def build_vocab(docs, least_freq=3):
    vocab = Counter()
    for d in docs:
        vocab.update(Counter(d.split(" ")))
    vocab = {k for k, v in vocab.items() if v >= least_freq}
    return list(vocab)
