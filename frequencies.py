import util
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import FrequencyModel

models = {}
for country in ["UK", "USA"]:
    print("Country", country)
    data = list(map(str, util.load_data("./data/%s_tokenized.txt" % country)[5].tolist()))
    vocab = util.build_vocab(data)
    inverted_vocab = {k:v for v,k in enumerate(vocab)}

    docs = []
    for d in tqdm(data, desc="Processing docs"):
        docs.append(np.array(list(map(lambda x : inverted_vocab[x] if x in inverted_vocab else -1, d.split(" ")))))

    model = FrequencyModel(inverted_vocab)
    model.fit(docs)
    models[country] = model

for c in models.keys():
    print("In %s, freq of 'fuck' = %f" % (c, models[c].transform("fuck")))