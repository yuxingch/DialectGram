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
        self.init_embs()
        self.define_module()

    def init_embs(self):
        self.u_global_emb = nn.Embedding(self.global_vocab_size, self.emb_dim, sparse=True)
        self.v_global_emb = nn.Embedding(self.global_vocab_size, self.emb_dim, sparse=True)
        self.u_usa_emb = nn.Embedding(self.us_vocab_size, self.emb_dim, sparse=True)
        self.v_usa_emb = nn.Embedding(self.us_vocab_size, self.emb_dim, sparse=True)
        self.u_uk_emb = nn.Embedding(self.uk_vocab_size, self.emb_dim, sparse=True)
        self.v_uk_emb = nn.Embedding(self.uk_vocab_size, self.emb_dim, sparse=True)

        initrange = (2.0 / (self.global_vocab_size + self.emb_dim) ** 0.5  # Xavier init
        init.uniform_(self.u_global_emb.weight.data, -initrange, initrange)
        init.uniform_(self.v_global_emb.weight.data, -0, 0)
        initrange = (2.0 / (self.us_vocab_size + self.emb_dim) ** 0.5  # Xavier init
        init.uniform_(self.u_usa_emb.weight.data, -initrange, initrange)
        init.uniform_(self.v_usa_emb.weight.data, -0, 0)
        initrange = (2.0 / (self.uk_vocab_size + self.emb_dim) ** 0.5  # Xavier init
        init.uniform_(self.u_uk_emb.weight.data, -initrange, initrange)
        init.uniform_(self.v_uk_emb.weight.data, -0, 0)

    def define_module(self):
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, batch):
        batch_center_word, batch_context_word, geo_tag, batch_neg = batch
        if geo_tag == 'US':
            center_regional_id_batch = [self.word2id_usa[w] for w in batch_center_word]
            center_regional_emb_batch = self.u_usa_emb(center_regional_id_batch)
            context_regional_id_batch = [self.word2id_usa[w] for w in batch_context_word]
            context_regional_emb_batch = self.v_usa_emb(context_regional_id_batch)
            neg_regional_ids_batch = [self.word2id_usa[w] for w in batch_neg]
            neg_regional_emb_batch = self.v_usa_emb(neg_regional_ids_batch)
        else:
            center_regional_id_batch = [self.word2id_uk[w] for w in batch_center_word]
            center_regional_emb_batch = self.u_uk_emb(center_regional_id_batch)
            context_regional_id_batch = [self.word2id_uk[w] for w in batch_context_word]
            context_regional_emb_batch = self.v_uk_emb(context_regional_id_batch)
            neg_regional_ids_batch = [self.word2id_uk[w] for w in batch_neg]
            neg_regional_emb_batch = self.v_uk_emb(neg_regional_ids_batch)
        center_global_id_batch = [self.word2id_global[w] for w in batch_center_word]
        center_global_emb_batch = self.u_global_emb(center_global_id_batch)
        context_global_id_batch = [self.word2id_global[w] for w in batch_context_word]
        context_global_emb_batch = self.v_global_emb(context_global_id_batch)
        neg_global_ids_batch = [self.word2id_global[w] for w in batch_neg]
        neg_global_emb_batch = self.v_global_emb(neg_global_ids_batch)

        center_combined_emb_batch = center_global_emb_batch + center_regional_emb_batch
        context_combined_emb_batch = context_global_emb_batch + context_regional_emb_batch
        neg_combined_emb_batch = neg_global_emb_batch + neg_regional_emb_batch

        positive_score = torch.sum(torch.mul(center_combined_emb_batch, context_combined_emb_batch), dim=1)
        # positive_score = self.logsigmoid(positive_score).squeeze()
        positive_score = self.logsigmoid(positive_score)

        neg_score = torch.bmm(neg_combined_emb_batch, center_combined_emb_batch.unsqueeze(2)).squeeze()
        neg_score = -torch.sum(self.logsigmoid(-neg_score), dim=1)

        score = positive_score + neg_score
        return torch.mean(score)
