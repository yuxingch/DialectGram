import pandas as pd
from collections import Counter
from scipy.io import loadmat
import re
import numpy as np

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


def load_textfile(path = "./data/USA_tokenized.txt"):
    tweets_lst = []
    with open(path) as f:
        for line in f:
            values = line.split("\t")
            raw_text = values[-1].rstrip('\n')
            tweets_lst.append(clean_tweets(raw_text))
    return tweets_lst


def clean_tweets(text):
    text = re.sub(r'&[a-z]{3};', ' <sym> ', text)  # '&amp;'
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&lt;', '<', text) 
    text = re.sub(r'@\w+', ' <user> ', text)
    ENCODE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)  # emoji
    text = ENCODE_EMOJI.sub('', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()


def vec2dict(src_filename):
    data = {}
    with open(src_filename) as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data


def print_stats(pred, truth):
    assert len(pred) == len(truth)
    length = len(pred)
    acc = sum([i == j for i, j in zip(pred, truth)]) / length
    precision = sum([i == 1 and j == 1 for i, j in zip(pred, truth)]) / sum([i == 1 for i in pred])
    recall = sum([i == 0 and j == 0 for i, j in zip(pred, truth)]) / sum([i == 0 for i in pred])
    f1 = 2 * precision * recall / (precision + recall)
    print("Accuracy \t= %f\nPrecision \t= %f\nRecall   \t= %f\nF1 score \t= %f" %
          (acc, precision, recall, f1))

