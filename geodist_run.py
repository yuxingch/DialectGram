import argparse
import os
from collections import defaultdict

import numpy as np

import util
from model import GeodistModel


parser = argparse.ArgumentParser(
    description='Collecting parameters for training GEODIST ...')
parser.add_argument('--batch', dest='batch_size', default=128,
                    type=int, help='batch size, default: 128')
parser.add_argument('--window', dest='window_size', default=10,
                    type=int, help='window size of Skip-gram, default: 10')
parser.add_argument('--freq', dest='freq', default=20,
                    type=int, help='word frequency threshold, default: 20')
parser.add_argument('--step', dest='total_steps', default=80000,
                    type=int, help='total number of steps, default: 80000')
parser.add_argument('--dim', dest='dim', default=100,
                    type=int, help='dimension of word embeddings, default: 100')
parser.add_argument('--lr', dest='lr', default=0.05,
                    type=float, help='learning rate, default: 0.05')
parser.add_argument('--dir', dest='dir', default='./outputs',
                    type=str, help='output directory (storing word vectors), default: ./outputs/')
opt = parser.parse_args()
print(opt)

print(f'====             Initializing GEODIST model            ====')
model = GeodistModel('./data/USA_tokenized.txt', './data/UK_tokenized.txt',
                     opt.batch_size, opt.window_size, freq=opt.freq,
                     runs=opt.total_steps, emb_dim=opt.dim, lr=opt.lr,
                     model_dir=opt.dir)
print(f'==== Start training process, total {opt.total_steps} steps ====')
model.train()
print(f'====             Done!             ====')
