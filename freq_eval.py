import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import FrequencyModel
from sklearn.metrics import accuracy_score as acc

def score(w, models, log_prob=True):
    raw = models["UK"].transform(w, log_prob=log_prob) / models["USA"].transform(w, log_prob=log_prob)
    return max(raw, 1 / raw) - 1

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

eval_train = pd.read_csv("./data/eval_train.csv")
eval_test = pd.read_csv("./data/eval_test.csv")

train_word = eval_train["word"].tolist()
train_label = eval_train["label"].tolist()
test_word = eval_test["word"].tolist()
test_label = eval_test["label"].tolist()
pred = np.array([score(w, models) for w in train_word])

linspace = np.linspace(0, 0.5, 1000)
accs = []
for t in linspace:
    accs.append(acc(train_label, pred > t))
plt.plot(linspace, accs)
plt.xlabel("Threshold")
plt.ylabel("Acc on training set")
plt.show()

t = linspace[np.argmax(accs)]
print("Train Acc = %f" % acc(train_label, pred > t))
print("Test Acc = %f" % acc(test_label, np.array([score(w, models) for w in test_word]) > t))