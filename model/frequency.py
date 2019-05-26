import numpy as np
from tqdm import tqdm
from model.base import ModelBase
from typing import List
from collections import Counter

class FrequencyModel(ModelBase):

    def __init__(self, inverted_vocab, unk_ind = -1):
        self._inv_vocab = inverted_vocab
        self.unk_ind = unk_ind

    def fit(self, docs : List[np.ndarray]):
        term_raw_counts = np.zeros((len(self._inv_vocab), ))
        for i in tqdm(range(len(docs)), desc="Generating TF"):
            for k, v in Counter(docs[i]).items():
                if k != self.unk_ind:
                    term_raw_counts[k] += v
        self.term_prob = term_raw_counts / np.sum(term_raw_counts)

    def transform(self, word : str):
        return self.term_prob[self._inv_vocab[word]]