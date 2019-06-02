import os
from collections import defaultdict

import numpy as np

import util
from model import GeodistModel


DIM = 100

model = GeodistModel('./data/USA_tokenized.txt', './data/UK_tokenized.txt', 128, 2, 30000, DIM, 0.025)
model.train()

model.save_vectors('./outputs/vectors/')

