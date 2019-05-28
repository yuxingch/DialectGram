import os
from collections import defaultdict
from copy import copy

import numpy as np
import torch
from torch.utils.data import Dataset, Dataloader
from tqdm import tqdm


class TwitterCorpus:

    def __init__(self, usa_path, uk_path):
        self.usa_path = usa_path
        self.uk_path = uk_path
        self.usa_corpus = self.load_textfile(path=self.usa_path)
        self.uk_corpus = self.load_textfile(path=self.uk_path)
        self.word2id_usa, self.id2word_usa = self.tokenize(self.usa_corpus, freq=4)
        self.word2id_uk, self.id2word_uk = self.tokenize(self.uk_corpus, freq=4)
        self.get_global_vocab()
        # TODO
    
    def load_textfile(self, path = "./data/USA_tokenized.txt"):
        tweets_lst = []
        with open(path) as f:
            for line in f:
                values = line.split("\t")
                tweets_lst.append(values[-1].rstrip('\n'))
        return tweets_lst

    def tokenize(self, docs, freq=1):
        "Tokenize each tweet and build up dictionary: wordID->embedding"
        wordFreq = defaultdict(int)
        word2id = defaultdict(int)
        id2word = defaultdict(str)
        idx = 0
        docs_size = len(docs)
        for tweet in tqdm(docs, total=docs_size):
            words = tweet.lstrip().rstrip().split(" ")
            for w in words:
                wordFreq[w] += 1
        word2id['<unk>'] = idx
        id2word[idx] = '<unk>'
        idx += 1
        for tweet in tqdm(docs, total=docs_size):
            words = tweet.lstrip().rstrip().split(" ")
            for w in words:
                if wordFreq[w] > freq:
                    if w not in word2id:
                        word2id[w] = idx
                        id2word[idx] = w
                        idx += 1
        final_vocab_size = len(word2id)
        assert final_vocab_size == idx
        assert len(word2id) == len(id2word)
        return word2id, id2word

    def get_global_vocab(self):
        self.word2id_global = copy(self.word2id_usa)
        self.id2word_global = copy(self.id2word_usa)
        idx = len(self.word2id_global)
        for w, _ in self.word2id_uk.items():
            if w not in self.word2id_global:
                self.word2id_global[w] = idx
                self.id2word_global[idx] = w
                self.idx += 1

    def sampleTokens(self):
        # TODO
        return


class Word2VecData(Dataset):

    def __init__(self, tweets, window_size):
        self.tweets = tweets
        self.window_size = window_size
    
    def __len__(self):
        return len(tweets)

    def __getitem__(self, idx):
        curr_tweet = self.tweets[idx]
        words = curr_tweet.split(" ")
        # TODO
        