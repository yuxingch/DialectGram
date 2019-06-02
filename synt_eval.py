import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import FrequencyModel
from sklearn.metrics import accuracy_score as acc
from scipy.stats import entropy as kl
from scipy.spatial.distance import jensenshannon

def pos_separate(pos_token):
    last_ind = pos_token.rfind("_")
    return pos_token[:last_ind], pos_token[last_ind + 1:]

def sub_keycounter(word, model):
    token_set = {}
    for pos_token, counts in model.tf.items():
        w, pos = pos_separate(pos_token)
        if w == word:
            token_set[pos_token] = counts
    return token_set

def categorical_counter(word, model):
    pos = ["NN", "VB", "JJ", "OTHER"]
    token_counter = sub_keycounter(word, model)
    ret = {}
    for p in pos:
        count = 1
        for k, v in token_counter.items():
            if p == "OTHER" and all(p_ not in k for p_ in pos):
                count += v
            elif p in pos_separate(k)[1]:
                count += v
        ret[p] = count / (sum(token_counter.values()) + 4)
    return ret

def score(w, models, measure=kl):
    uk_cat= categorical_counter(w, models["UK"])
    usa_cat = categorical_counter(w, models["USA"])
    return measure(list(uk_cat.values()), list(usa_cat.values()))

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

eval_train = pd.read_csv("./data/eval_train.csv")
eval_test = pd.read_csv("./data/eval_test.csv")

train_word = eval_train["word"].tolist()
train_label = eval_train["label"].tolist()
test_word = eval_test["word"].tolist()
test_label = eval_test["label"].tolist()

for measure in [kl, jensenshannon]:
    print(measure.__name__)
    pred = np.array([score(w, models) for w in tqdm(train_word, desc="Train Score")])

    linspace = np.linspace(min(pred), max(pred), 3000)
    accs = []
    for t in tqdm(linspace, desc="Threshold"):
        accs.append(acc(train_label, pred > t))
    plt.plot(linspace, accs)
    plt.xlabel("Threshold")
    plt.ylabel("Acc on training set")
    plt.show()

    t = linspace[np.argmax(accs)]
    print("Train Acc = %f" % acc(train_label, pred > t))
    print("Test Acc = %f" % acc(test_label, np.array([score(w, models) for w in tqdm(test_word, desc="Test Score")]) > t))
