import errno
import time
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from tqdm import tqdm

from model.dataloader import TwitterCorpus
from model.base import ModelBase
from model.word2vec import SkipGram


MAX_STEPS = 10
DEFAULT_EMB_DIM = 100


def build_state_dict(config_net):
    """Build dictionary to store the state of our neural net"""
    return torch.load(config_net, map_location=lambda storage, loc: storage)['state_dict']


class GeodistModel(ModelBase):

    def __init__(self, usa_path, uk_path, batch_size, window_size, runs=MAX_STEPS,
                 emb_dim=DEFAULT_EMB_DIM, lr=0.01, model_dir='./outputs'):
        self.usa_path = usa_path
        self.uk_path = uk_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_steps = runs
        self.emb_dim = emb_dim
        self.lr = lr
        self.model_dir = model_dir
        mkdir_p(self.model_dir)
        self.corpus = TwitterCorpus(self.usa_path, self.uk_path)
        self.load_skipgram()

    def load_skipgram(self):
        self.word2vec = SkipGram(len(self.corpus.word2id_usa), len(self.corpus.word2id_uk), len(self.corpus.word2id_global),
                                 self.emb_dim, self.corpus.word2id_uk, self.corpus.word2id_usa, self.corpus.word2id_global)

    def train(self):
        optimizer = optim.SGD(self.word2vec.parameters(), lr=self.lr)
        lr = self.lr
        # optimizer = optim.SparseAdam(self.word2vec.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.corpus.uk_tweets+self.corpus.usa_tweets)
        step = 1
        avg_loss = 0
        while step < self.max_steps+1:
            start_t = time.time()
            batch_center_word, batch_context_word, geo_tag = self.corpus.batch_sampler(self.batch_size, self.window_size)
            negs_regional, negs_global = self.corpus.sample_negs(self.batch_size, 20, batch_center_word, geo_tag)
            
            if step % 1000 == 0:
                # update learning rate
                lr *= 0.8
                print(f'learning rate updated: {lr}')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            optimizer.zero_grad()
            loss = self.word2vec(batch_center_word, batch_context_word, geo_tag, negs_regional, negs_global)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            if step % 500 == 0:
                avg_loss /= 500
                print(f'Average loss at step {step} is {avg_loss:.4f}')
                avg_loss = 0
            if step % 2000 == 0:
                self.save_model(self.model_dir, step)
            step += 1
        torch.save({'step': step-1, 'state_dict': self.word2vec.state_dict()}, '%s/model_final.pth' % (self.model_dir))
        return

    def vector(self, word):
        return self.word2vec.predict(word)

    def save_model(self, model_dir, step):
        torch.save({'step': step, 'state_dict': self.word2vec.state_dict()}, '%s/model_step_%d.pth' % (model_dir, step))
        print('Save model at %s/model_step_%d.pth' % (model_dir, step))

    def load_model(self, model_path):
        self.word2vec.load_state_dict(build_state_dict(model_path))
        print(f'Successfully loaded checkpoint from {model_path}.')

    def save_vectors(self, path):
        mkdir_p(path)
        state_dict = self.word2vec.state_dict()
        embedding_global = state_dict['u_global_emb.weight'].tolist()
        embedding_uk = state_dict['u_uk_emb.weight'].tolist()
        embedding_usa = state_dict['u_usa_emb.weight'].tolist()
        curr_path = os.path.join(path, 'global_vector.txt')
        f = open(curr_path, 'w')
        for i in range(len(embedding_global)):
            term = self.corpus.id2word_global[i]
            emb = embedding_global[i]
            emb_lst = [str(e) for e in emb]
            emb_str = ' '.join(emb_lst)
            f.write(term + ' ' + emb_str + '\n')
        f.close()
        print(f'Finished writing global vectors to {curr_path}.')
        curr_path = os.path.join(path, 'uk_vector.txt')
        f = open(curr_path, 'w')
        for i in range(len(embedding_uk)):
            term = self.corpus.id2word_uk[i]
            emb = embedding_uk[i]
            emb_lst = [str(e) for e in emb]
            emb_str = ' '.join(emb_lst)
            f.write(term + ' ' + emb_str + '\n')
        f.close()
        print(f'Finished writing uk vectors to {curr_path}.')
        curr_path = os.path.join(path, 'usa_vector.txt')
        f = open(curr_path, 'w')
        for i in range(len(embedding_usa)):
            term = self.corpus.id2word_usa[i]
            emb = embedding_usa[i]
            emb_lst = [str(e) for e in emb]
            emb_str = ' '.join(emb_lst)
            f.write(term + ' ' + emb_str + '\n')
        f.close()
        print(f'Finished writing usa vectors to {curr_path}.')

    def fit(self, docs):
        return

    def transform(self, word):
        return self.vector(word)


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
