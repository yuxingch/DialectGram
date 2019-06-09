import os
from collections import defaultdict

import numpy as np

import util
from model import GeodistModel


DIM = 100
total_steps = 80000

print(f'====             Initializing GEODIST model            ====')
model = GeodistModel('./data/USA_tokenized.txt', './data/UK_tokenized.txt',
                     128, 10, freq=20, runs=total_steps, emb_dim=DIM, lr=0.05,
                     model_dir='outputs')
print(f'==== Start training process, total {total_steps} steps ====')
model.train()
print(f'====             Done!             ====')
