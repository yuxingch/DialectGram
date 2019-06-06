import os
from collections import defaultdict
from copy import copy
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def in_vocab(w, vocab):
    if w in vocab:
        return w
    else:
        return '<unk>'


class TwitterCorpus:

    def __init__(self, usa_path, uk_path, freq):
        self.usa_path = usa_path
        self.uk_path = uk_path
        self.freq = freq
        self.usa_corpus = self.load_textfile(path=self.usa_path)
        self.uk_corpus = self.load_textfile(path=self.uk_path)
        self.get_all_tweets()
        self.word2id_usa, self.id2word_usa, self.wordFreq_usa = self.tokenize(self.usa_corpus, freq=self.freq)
        self.word2id_uk, self.id2word_uk, self.wordFreq_uk = self.tokenize(self.uk_corpus, freq=self.freq)
        self.uk_word_count = len(self.word2id_uk)
        self.usa_word_count = len(self.word2id_usa)
        self.get_global_vocab()
        self.global_word_count = len(self.word2id_global)
        self.generate_unigrams()
        self.global_vocab = [*self.word2id_global]
        self.build_regional_global_lookup()
        print('word count:', self.uk_word_count, self.usa_word_count)

    def build_regional_global_lookup(self):
        print('Building lookup table...')
        self.uk_global_lookup = []
        self.usa_global_lookup = []
        for uk_v in self.uk_vocab:
            self.uk_global_lookup.append(self.global_vocab.index(uk_v))
        for usa_v in self.usa_vocab:
            self.usa_global_lookup.append(self.global_vocab.index(usa_v))

    def load_textfile(self, path="./data/USA_tokenized.txt"):
        tweets_lst = []
        with open(path) as f:
            for line in f:
                values = line.split("\t")
                temp = values[-1].rstrip('\n')
                clean = re.sub(r"[,.;:@#?!&$”\"\-]+\ *", " ", temp)
                tweets_lst.append(clean)
        return tweets_lst

    def get_all_tweets(self):
        all_tweets_temp = [t.lstrip().rstrip().split(" ") for t in self.usa_corpus]
        self.usa_tweets = [t for t in all_tweets_temp if len(t) > 1]
        all_tweets_temp = [t.lstrip().rstrip().split(" ") for t in self.uk_corpus]
        self.uk_tweets = [t for t in all_tweets_temp if len(t) > 1]

    def tokenize(self, docs, freq):
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
                if w and wordFreq[w] > freq:
                    if w not in word2id:
                        word2id[w] = idx
                        id2word[idx] = w
                        idx += 1
        final_vocab_size = len(word2id)
        wordFreq = {key: val for key, val in wordFreq.items() if key and val > freq}
        wordFreq['<unk>'] = 1
        assert final_vocab_size == idx
        assert len(word2id) == len(id2word)
        assert len(wordFreq) == final_vocab_size
        return word2id, id2word, wordFreq

    def get_global_vocab(self):
        self.word2id_global = copy(self.word2id_usa)
        self.id2word_global = copy(self.id2word_usa)
        self.wordFreq_global = copy(self.wordFreq_usa)
        idx = len(self.word2id_global)
        for w, _ in self.word2id_uk.items():
            if w not in self.word2id_global:
                self.word2id_global[w] = idx
                self.id2word_global[idx] = w
                idx += 1
                self.wordFreq_global[w] = self.wordFreq_uk[w]
            else:
                self.wordFreq_global[w] += self.wordFreq_uk[w]

    def generate_unigrams(self):
        Z = 0.001
        # uk
        self.uk_unigram = []
        num_total_words_uk = sum([c for _, c in self.wordFreq_uk.items()])
        self.uk_vocab = [*self.word2id_uk]
        for v in self.uk_vocab:
            self.uk_unigram.extend([v] * int(((self.wordFreq_uk[v]/num_total_words_uk)**0.75)/Z))
        # usa
        self.usa_unigram = []
        num_total_words_usa = sum([c for _, c in self.wordFreq_usa.items()])
        self.usa_vocab = [*self.word2id_usa]
        for v in self.usa_vocab:
            self.usa_unigram.extend([v] * int(((self.wordFreq_usa[v]/num_total_words_usa)**0.75)/Z))

    def sample_negs(self, batch_size, num_negs, batch_target_input, region):
        negs_regional = []
        negs_global = []
        region_unigram = None
        region_word2id = None
        if region == 'UK':
            region_unigram = self.uk_unigram
            region_word2id = self.word2id_uk
        else:
            region_unigram = self.usa_unigram
            region_word2id = self.word2id_usa
        for i in range(batch_size):
            temp = random.sample(region_unigram, num_negs)
            while batch_target_input[i] in temp:
                temp = random.sample(region_unigram, num_negs)
            negs_regional.append([region_word2id[w] for w in temp])
            negs_global.append([self.word2id_global[w] for w in temp])
        assert batch_size == len(negs_regional)
        assert batch_size == len(negs_global)
        return negs_regional, negs_global

    def batch_sampler(self, batch_size, window_size=2):
        region_ID = random.randint(0, 1)
        if region_ID:
            region_tweets = self.usa_tweets
            geo_tag = 'US'
            vocab = self.usa_vocab
        else:
            region_tweets = self.uk_tweets
            geo_tag = 'UK'
            vocab = self.uk_vocab
        selected_list = []
        while len(selected_list) < batch_size:
            tweet_id = random.randint(0, len(region_tweets)-1)
            tokenized_tweet = region_tweets[tweet_id]
            center_word_id = random.randint(0, len(tokenized_tweet)-1)
            center_word = tokenized_tweet[center_word_id]
            center_word = in_vocab(center_word, vocab)
            context = tokenized_tweet[max(0, center_word_id-window_size):center_word_id]
            if center_word_id+1 < len(tokenized_tweet):
                context += tokenized_tweet[center_word_id+1:min(len(tokenized_tweet), center_word_id+window_size+1)]
            context = [in_vocab(w, vocab) for w in context]
            context = [w for w in context if w != center_word]
            remain = batch_size - len(selected_list)
            if remain > len(context):
                selected_list += [(center_word, context_word) for context_word in context]
            else:
                selected_list += [(center_word, context_word) for context_word in random.sample(context, remain)]
            selected_list = list(set(selected_list))
        assert len(selected_list) == batch_size
        batch_center_word = []
        batch_context_word = []
        for (center, context) in selected_list:
            batch_center_word.append(center)
            batch_context_word.append(context)
        return batch_center_word, batch_context_word, geo_tag
