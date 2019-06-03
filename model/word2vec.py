import torch
import torch.nn as nn
from torch.nn import init


class SkipGram(nn.Module):

    def __init__(self, us_vocab_size, uk_vocab_size, global_vocab_size, emb_dim,
                 word2id_uk, word2id_usa, word2id_global):
        super(SkipGram, self).__init__()
        self.us_vocab_size = us_vocab_size
        self.uk_vocab_size = uk_vocab_size
        self.global_vocab_size = global_vocab_size
        self.emb_dim = emb_dim
        self.word2id_uk = word2id_uk
        self.word2id_usa = word2id_usa
        self.word2id_global = word2id_global
        self.uk_vocab = [*word2id_uk]
        self.usa_vocab = [*word2id_usa]
        self.global_vocab = [*word2id_global]
        self.init_embs()
        self.define_module()

    def init_embs(self):
        self.u_global_emb = nn.Embedding(self.global_vocab_size, self.emb_dim, sparse=True)
        self.v_global_emb = nn.Embedding(self.global_vocab_size, self.emb_dim, sparse=True)
        self.u_usa_emb = nn.Embedding(self.us_vocab_size, self.emb_dim, sparse=True)
        self.v_usa_emb = nn.Embedding(self.us_vocab_size, self.emb_dim, sparse=True)
        self.u_uk_emb = nn.Embedding(self.uk_vocab_size, self.emb_dim, sparse=True)
        self.v_uk_emb = nn.Embedding(self.uk_vocab_size, self.emb_dim, sparse=True)

        # initrange = (2.0 / (self.global_vocab_size + self.emb_dim)) ** 0.5  # Xavier init
        # self.u_global_emb.weight.data.uniform_(-initrange, initrange)
        self.u_global_emb.weight.data.normal_(0.0, 0.02)
        self.v_global_emb.weight.data.fill_(0)
        # initrange = (2.0 / (self.us_vocab_size + self.emb_dim)) ** 0.5
        self.u_usa_emb.weight.data.fill_(0)
        self.v_usa_emb.weight.data.fill_(0)
        # initrange = (2.0 / (self.uk_vocab_size + self.emb_dim)) ** 0.5
        self.u_uk_emb.weight.data.fill_(0)
        self.v_uk_emb.weight.data.fill_(0)
        # initrange = (2.0 / (self.global_vocab_size + self.emb_dim) ** 0.5  # Xavier init
        # init.uniform_(self.u_global_emb.weight.data, -initrange, initrange)
        # init.uniform_(self.v_global_emb.weight.data, -0, 0)
        # init.uniform_(self.u_usa_emb.weight.data, -0, 0)
        # init.uniform_(self.v_usa_emb.weight.data, -0, 0)
        # init.uniform_(self.u_uk_emb.weight.data, -0, 0)
        # init.uniform_(self.v_uk_emb.weight.data, -0, 0)

    def define_module(self):
        self.logsigmoid = nn.LogSigmoid()

    def _tensor(self, v):
        return torch.tensor(v, dtype=torch.long)

    def forward(self, batch_center_word, batch_context_word, geo_tag,
                neg_regional_ids_batch, neg_global_ids_batch):
        neg_regional_ids_batch = self._tensor(neg_regional_ids_batch)
        neg_global_ids_batch = self._tensor(neg_global_ids_batch)
        if geo_tag == 'US':
            center_regional_id_batch = [self.word2id_usa[w] for w in batch_center_word]
            center_regional_id_batch = self._tensor(center_regional_id_batch)
            center_regional_emb_batch = self.u_usa_emb(center_regional_id_batch)
            context_regional_id_batch = [self.word2id_usa[w] for w in batch_context_word]
            context_regional_id_batch = self._tensor(context_regional_id_batch)
            context_regional_emb_batch = self.v_usa_emb(context_regional_id_batch)
            neg_regional_emb_batch = self.v_usa_emb(neg_regional_ids_batch)
        else:
            center_regional_id_batch = [self.word2id_uk[w] for w in batch_center_word]
            center_regional_id_batch = self._tensor(center_regional_id_batch)
            center_regional_emb_batch = self.u_uk_emb(center_regional_id_batch)
            context_regional_id_batch = [self.word2id_uk[w] for w in batch_context_word]
            context_regional_id_batch = self._tensor(context_regional_id_batch)
            context_regional_emb_batch = self.v_uk_emb(context_regional_id_batch)
            neg_regional_emb_batch = self.v_uk_emb(neg_regional_ids_batch)
        center_global_id_batch = [self.word2id_global[w] for w in batch_center_word]
        center_global_id_batch = self._tensor(center_global_id_batch)
        center_global_emb_batch = self.u_global_emb(center_global_id_batch)
        context_global_id_batch = [self.word2id_global[w] for w in batch_context_word]
        context_global_id_batch = self._tensor(context_global_id_batch)
        context_global_emb_batch = self.v_global_emb(context_global_id_batch)
        neg_global_emb_batch = self.v_global_emb(neg_global_ids_batch)

        center_combined_emb_batch = center_global_emb_batch + center_regional_emb_batch
        context_combined_emb_batch = context_global_emb_batch + context_regional_emb_batch
        neg_combined_emb_batch = neg_global_emb_batch + neg_regional_emb_batch

        positive_score = torch.sum(torch.mul(center_combined_emb_batch, context_combined_emb_batch), dim=1)
        # positive_score = self.logsigmoid(positive_score).squeeze()
        positive_score = torch.clamp(positive_score, max=10, min=-10)
        positive_score = self.logsigmoid(positive_score)

        neg_score = torch.bmm(neg_combined_emb_batch, center_combined_emb_batch.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = torch.sum(self.logsigmoid(-neg_score), dim=1)
        # neg_score = self.logsigmoid(-torch.sum(neg_score, dim=1))

        score = positive_score + neg_score
        return -torch.mean(score)

    def predict(self, word, A):
        global_emb = None
        uk_emb = None
        usa_emb = None
        # if the regions (A) are given:
        if A is not None:
            global_emb = self.u_global_emb(self._tensor(self.word2id_global[word]))
            if 'uk' in A:
                uk_emb = self.u_uk_emb(self._tensor(self.word2id_uk[word]))
            if 'usa' in A:
                usa_emb = self.u_usa_emb(self._tensor(self.word2id_usa[word]))
            return global_emb, usa_emb, uk_emb
        # if not given regions (A)
        if word not in self.global_vocab:
            global_emb = self.u_global_emb(self._tensor(self.word2id_global['<unk>']))
        else:
            global_emb = self.u_global_emb(self.word2id_global[word])
        if word not in self.uk_vocab:
            uk_emb = self.u_uk_emb(self._tensor(self.word2id_uk['<unk>']))
        else:
            uk_emb = self.u_uk_emb(self._tensor(self.word2id_uk[word]))
        if word not in self.usa_vocab:
            usa_emb = self.u_usa_emb(self._tensor(self.word2id_usa['<unk>']))
        else:
            usa_emb = self.u_usa_emb(self._tensor(self.word2id_usa[word]))
        return global_emb, usa_emb, uk_emb
