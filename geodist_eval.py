import util
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score as acc
from scipy.spatial.distance import cosine

def score(w, us, uk):
    if w not in us or w not in uk:
        return random.random()
    return 1 - cosine(us[w], uk[w])

eval_train = pd.read_csv("./data/eval_train.csv")
eval_test = pd.read_csv("./data/eval_test.csv")

train_word = eval_train["word"].tolist()
train_label = eval_train["label"].tolist()
test_word = eval_test["word"].tolist()
test_label = eval_test["label"].tolist()

# TODO: Change the following code
plain_us_dict = util.vec2dict("./outputs_21/vectors/usa_vector_step_32000.txt")
plain_uk_dict = util.vec2dict("./outputs_21/vectors/uk_vector_step_32000.txt")
plain_global_dict = util.vec2dict("./outputs_21/vectors/global_vector_step_32000.txt")
us_dict = {k: v + plain_global_dict[k] for k, v in plain_us_dict.items()}
uk_dict = {k: v + plain_global_dict[k] for k, v in plain_uk_dict.items()}

# TODO: Remove this
# us_dict = {k : np.random.rand(666) for k in train_word + test_word}
# uk_dict = {k : np.random.rand(666) for k in train_word + test_word}

pred = np.array([score(str(w), us_dict, uk_dict) for w in tqdm(train_word, desc="Geodist")])

linspace = np.linspace(0, 1.0, 1000)
accs = []
for t in linspace:
    accs.append(acc(train_label, pred > t))
plt.plot(linspace, accs)
plt.xlabel("Threshold")
plt.ylabel("Acc on training set")
plt.show()

t = linspace[np.argmax(accs)]
print("Training stats:")
util.print_stats(pred > t, train_label)
print("Testing stats:")
util.print_stats(np.array([score(w, us_dict, uk_dict) for w in test_word]) > t, test_label)
