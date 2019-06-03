import os
from collections import defaultdict

import numpy as np

import util
from model import GeodistModel


DIM = 100
total_steps = 40000

print(f'====             Initializing GEODIST model            ====')
model = GeodistModel('./data/USA_tokenized.txt', './data/UK_tokenized.txt',
                     128, 5, runs=total_steps, emb_dim=DIM, lr=0.05)
print(f'==== Start training process, total {total_steps} steps ====')
model.train()
