import numpy as np
from tqdm import tqdm
from model.base import ModelBase
from typing import List
from collections import Counter

class FrequencyModel(ModelBase):

    def __init__(self, inverted_vocab):
        self._inv_vocab = inverted_vocab

    def fit(self, docs : List[np.ndarray]):
        tf = np.zeros((len(docs), len(self._inv_vocab)))
        for i in tqdm(range(len(docs)), desc="Generating TF"):
            for k, v in Counter(docs[i]).items():
                tf[i, k - 1] = v
        term_raw_counts = np.sum(tf, axis=0)
        self.term_prob = term_raw_counts / np.sum(term_raw_counts)

    def transform(self, word : str):
        return self.term_prob[self._inv_vocab[word]]