import util
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score as acc
from scipy.spatial.distance import cosine

random.seed(42)
COUNT = {'all': 0, 'missing': 0}


def score(w, us, uk):
    COUNT['all'] += 1
    if w not in us or w not in uk:
        COUNT['missing'] += 1
        return random.random()
    return 1 - cosine(us[w], uk[w])


def manhattan_distance(w, us, uk):
    COUNT['all'] += 1
    if w not in us or w not in uk:
        COUNT['missing'] += 1
        # print(w)
        return random.random()
    return sum(abs(a-b) for a, b in zip(us[w], uk[w]))

eval_train = pd.read_csv("./data/DialectSim_train.csv")
eval_test = pd.read_csv("./data/DialectSim_test.csv")

train_word = eval_train["word"].tolist()
train_label = eval_train["label"].tolist()
test_word = eval_test["word"].tolist()
test_label = eval_test["label"].tolist()

plain_us_dict = util.vec2dict("./outputs/vectors/usa_vector_step_24000.txt")
plain_uk_dict = util.vec2dict("./outputs/vectors/uk_vector_step_24000.txt")
plain_global_dict = util.vec2dict("./outputs/vectors/global_vector_step_24000.txt")
us_dict = {k: v + plain_global_dict[k] for k, v in plain_us_dict.items()}
uk_dict = {k: v + plain_global_dict[k] for k, v in plain_uk_dict.items()}

# TODO: Remove this
# us_dict = {k : np.random.rand(666) for k in train_word + test_word}
# uk_dict = {k : np.random.rand(666) for k in train_word + test_word}

pred = np.array([manhattan_distance(str(w), us_dict, uk_dict) for w in tqdm(train_word, desc="Geodist")])

linspace = np.linspace(min(pred), max(pred), 1000)
accs = []
for t in linspace:
    accs.append(acc(train_label, pred > t))
plt.plot(linspace, accs)
plt.xlabel("Threshold")
plt.ylabel("Acc on training set")
plt.show()

t = linspace[np.argmax(accs)]
print(t)
print("Training stats:")
util.print_stats(pred > t, train_label)
print("Testing stats:")
util.print_stats(np.array([manhattan_distance(str(w), us_dict, uk_dict) for w in test_word]) > t, test_label)
print(COUNT)
