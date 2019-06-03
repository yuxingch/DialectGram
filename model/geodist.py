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
                 emb_dim=DEFAULT_EMB_DIM, lr=0.01, anneal_rate=0.9, model_dir='./outputs'):
        self.usa_path = usa_path
        self.uk_path = uk_path
        self.batch_size = batch_size
        self.window_size = window_size
        self.max_steps = runs
        self.emb_dim = emb_dim
        self.lr = lr
        self.anneal_rate = anneal_rate
        self.model_dir = model_dir
        self.vector_dir = os.path.join(self.model_dir, 'vectors')
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
        avg_loss_history = []
        while step < self.max_steps+1:
            start_t = time.time()
            batch_center_word, batch_context_word, geo_tag = self.corpus.batch_sampler(self.batch_size, self.window_size)
            negs_regional, negs_global = self.corpus.sample_negs(self.batch_size, 20, batch_center_word, geo_tag)
            
            if step % 1000 == 0:
                # update learning rate
                lr *= self.anneal_rate
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
                avg_loss_history.append(avg_loss)
                print(f'Average loss at step {step} is {avg_loss:.4f}')
                avg_loss = 0
            if step % 2000 == 0:
                self.neighbors('flat')
            if step % 4000 == 0:
                self.save_model(self.model_dir, step)
                self.save_vectors(self.vector_dir, step)
            step += 1
        torch.save({'step': step-1, 'state_dict': self.word2vec.state_dict()}, '%s/model_final.pth' % (self.model_dir))
        self.save_vectors(self.vector_dir, step-1, isFinal=True)
        return

    def vector(self, word, A=None):
        return self.word2vec.predict(word, A)

    def save_model(self, model_dir, step):
        torch.save({'step': step, 'state_dict': self.word2vec.state_dict()}, '%s/model_step_%d.pth' % (model_dir, step))
        print('Save model at %s/model_step_%d.pth' % (model_dir, step))

    def load_model(self, model_path):
        self.word2vec.load_state_dict(build_state_dict(model_path))
        print(f'Successfully loaded checkpoint from {model_path}.')

    def save_vectors(self, path, step, isFinal=False):
        mkdir_p(path)
        step_suffix = f'step_{step}'
        if isFinal:
            step_suffix = 'final'
        state_dict = self.word2vec.state_dict()
        embedding_global = state_dict['u_global_emb.weight'].tolist()
        embedding_uk = state_dict['u_uk_emb.weight'].tolist()
        embedding_usa = state_dict['u_usa_emb.weight'].tolist()
        curr_path = os.path.join(path, f'global_vector_{step_suffix}.txt')
        f = open(curr_path, 'w')
        for i in range(len(embedding_global)):
            term = self.corpus.id2word_global[i]
            emb = embedding_global[i]
            emb_lst = [str(e) for e in emb]
            emb_str = ' '.join(emb_lst)
            f.write(term + ' ' + emb_str + '\n')
        f.close()
        print(f'Finished writing global vectors to {curr_path}.')
        curr_path = os.path.join(path, f'uk_vector_{step_suffix}.txt')
        f = open(curr_path, 'w')
        for i in range(len(embedding_uk)):
            term = self.corpus.id2word_uk[i]
            emb = embedding_uk[i]
            emb_lst = [str(e) for e in emb]
            emb_str = ' '.join(emb_lst)
            f.write(term + ' ' + emb_str + '\n')
        f.close()
        print(f'Finished writing uk vectors to {curr_path}.')
        curr_path = os.path.join(path, f'usa_vector_{step_suffix}.txt')
        f = open(curr_path, 'w')
        for i in range(len(embedding_usa)):
            term = self.corpus.id2word_usa[i]
            emb = embedding_usa[i]
            emb_lst = [str(e) for e in emb]
            emb_str = ' '.join(emb_lst)
            f.write(term + ' ' + emb_str + '\n')
        f.close()
        print(f'Finished writing usa vectors to {curr_path}.')

    def nearest(self, word, id2word, word_emb, emb_matrix, region, lookup=None, k=10):
        emb_matrix_weight_curr = emb_matrix.weight
        if region is not 'global':
            emb_matrix_weight = emb_matrix_weight_curr + self.word2vec.u_global_emb.weight[lookup]
        else:
            emb_matrix_weight = emb_matrix_weight_curr
        sim = torch.matmul(emb_matrix_weight, word_emb)
        _, ind = (-sim).sort()
        ind = ind.tolist()
        nearest_k = ind[:k]
        output_sent = f"Nearest to '{word}', embedding type: {region}\n"
        for x in range(k):
            neighbor = id2word[nearest_k[x]]
            output_sent += f'{x}: {neighbor}\n'
        print(output_sent)

    def neighbors(self, word):
        print(f"====           Get nearest neighbors for word '{word}'         ====")
        if not word in self.word2vec.global_vocab:
            print(f"'{word}' is not in our vocabulary. Please try another word.")
            return
        A = ['global']
        if word in self.corpus.uk_vocab:
            A.append('uk')
        if word in self.corpus.usa_vocab:
            A.append('usa')
        global_emb, usa_emb, uk_emb = self.vector(word, A)
        self.nearest(word, self.corpus.id2word_global, global_emb, self.word2vec.u_global_emb, 'global')
        if usa_emb is not None:
            self.nearest(word, self.corpus.id2word_usa, global_emb+usa_emb, self.word2vec.u_usa_emb, 'usa', self.corpus.usa_global_lookup)
        if uk_emb is not None:
            self.nearest(word, self.corpus.id2word_uk, global_emb+uk_emb, self.word2vec.u_uk_emb, 'uk', self.corpus.uk_global_lookup)
        print(f"====    Finished finding nearest neighbors for word '{word}'   ====")

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
