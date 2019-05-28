import errno
import time
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataloader
from tqdm import tqdm

from model.dataloader import TwitterCorpus, Word2VecData
from model.base import ModelBase
from model.word2vec import SkipGram


MAX_EPOCH = 10
DEFAULT_EMB_DIM = 100

class GeodistModel(ModelBase):

    def __init__(self, usa_path, uk_path, batch_size, window_size, epochs=MAX_EPOCH,
                 emb_dim=DEFAULT_EMB_DIM, lr=0.01):
        self.usa_path = usa_path
        self.uk_path = uk_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.epochs = epochs
        self.emb_dim = emb_dim
        self.lr = lr
        self.corpus = TwitterCorpus(self.usa_path, self.uk_path)
        dataset = Word2VecData(self.corpus, window_size)
        self.data_loader = Dataloader(dataset, batch_size=self.batch_size, shuffle=False)
        self.load_skipgram()

    def load_skipgram(self):
        self.word2vec = SkipGram(len(self.corpus.word2id_usa), len(self.corpus.word2id_uk), len(self.corpus.word2id_global),
                                 self.emb_dim, self.corpus.word2id_uk, self.corpus.word2id_usa, self.corpus.word2id_global)

    def train(self):
        optimizer = optim.SparseAdam(self.word2vec.parameters(), lr=self.lr)
        epoch = 0
        while epoch < self.epochs+1:
            epoch += 1
            start_t = time.time()
            for i, batch in enumerate(tqdm(self.data_loader)):
                # TODO:
                if len(batch[0]) > 1:
                    optimizer.zero_grad()
                    loss = self.word2vec(batch)
                    loss.backward()
                    optimizer.step()
        return

    def fit(self, docs):
        return

    def transform(self, word):
        return


def mkdir_p(path):
    """Create a directory if not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    return