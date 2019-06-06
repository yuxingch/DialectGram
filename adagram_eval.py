from model import AdaGramModel
import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score as acc
from scipy.spatial.distance import cosine
from random import random
import pickle
import os

COUNT = {'all': 0, 'missing': 0}

def score(w, us, uk):
    w = str(w)
    COUNT['all'] += 1
    if (w not in us) or (w not in uk):
        COUNT['missing'] += 1
        return random()
    return 1 - cosine(us[w], uk[w])

df_val = pd.read_csv('data/eval_train_21.csv')
df_test = pd.read_csv('data/eval_test_21.csv')
print(df_val.head())
train_words = df_val['word'].tolist()[:]
train_labels = df_val['label'].tolist()[:]
test_words = df_test['word'].tolist()[:]
test_labels = df_test['label'].tolist()[:]

if os.path.isfile('usa_dict.pickle') and os.path.isfile('uk_dict.pickle'):
    with open('usa_dict.pickle', 'rb') as handle:
        us_dict = pickle.load(handle)
    with open('uk_dict.pickle', 'rb') as handle:
        uk_dict = pickle.load(handle)
else:
    with open("data/UK_tokenized.txt") as uk_f, open("data/USA_tokenized.txt") as usa_f:
        uk_tweets = list(map(lambda x: x.split('\t')[-1].strip(), uk_f))
        usa_tweets = list(map(lambda x: x.split('\t')[-1].strip(), usa_f))

    model_path = "pretrained/model.joblib"

    uk_adagram = AdaGramModel(model_path)
    uk_adagram.fit(uk_tweets)

    usa_adagram = AdaGramModel(model_path)
    usa_adagram.fit(usa_tweets)

    # print(uk_adagram.transform(train_words[0]))

    us_dict = {k : usa_adagram.transform(k) for k in tqdm(train_words + test_words, desc="US")}
    uk_dict = {k : uk_adagram.transform(k) for k in tqdm(train_words + test_words, desc="UK")}

    with open('usa_dict.pickle', 'wb') as handle:
        pickle.dump(us_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('uk_dict.pickle', 'wb') as handle:
        pickle.dump(uk_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(us_dict)
# input()
# print(uk_dict)
# input()
pred = np.array([score(w, us_dict, uk_dict) for w in tqdm(train_words, desc="AdaGram")])

linspace = np.linspace(0, 1.0, 1000)
accs = []
for t in linspace:
    accs.append(acc(train_labels, pred > t))
plt.plot(linspace, accs)
plt.xlabel("Threshold")
plt.ylabel("Acc on training set")
plt.show()

t = linspace[np.argmax(accs)]
print("Training stats:")
util.print_stats(pred > t, train_labels)
print("Testing stats:")
util.print_stats(np.array([score(w, us_dict, uk_dict) for w in test_words]) > t, test_labels)

print(COUNT)