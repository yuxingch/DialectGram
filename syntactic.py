import sys
import util
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import FrequencyModel


# word we want to investigate, default: "sort"
w = "sort"
if len(sys.argv) == 2:
    w = sys.argv[1]

models = {}
for country in ["UK", "USA"]:
    print("Country", country)

    # Lazy: data is formed like str_POS
    # No need to build another model LOL
    data = list(map(str, util.load_data("./data/%s_tok_pos.txt" % country)[5].tolist()))
    vocab = util.build_vocab(data)
    inverted_vocab = {k:v for v,k in enumerate(vocab)}

    docs = []
    for d in tqdm(data, desc="Processing docs"):
        docs.append(np.array(list(map(lambda x : inverted_vocab[x] if x in inverted_vocab else -1, d.split(" ")))))

    model = FrequencyModel(inverted_vocab)
    model.fit(docs)
    models[country] = model

# Example
for c in models.keys():
    print("In %s, syntactic freq of '%s_N' = %f" % (c, w, models[c].transform("%s_N" % w)))
    print("In %s, syntactic freq of '%s_V' = %f" % (c, w, models[c].transform("%s_V" % w)))
    print("In %s, syntactic freq of '%s_A' = %f" % (c, w, models[c].transform("%s_A" % w)))
