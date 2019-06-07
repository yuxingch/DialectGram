from model import AdaGramModel
import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score, precision_score
from scipy.spatial.distance import cosine
from random import random
from math import *
import pickle
import os

COUNT = {'all': 0, 'missing': 0}

def cosine_distance(w, us, uk):
    w = str(w)
    COUNT['all'] += 1
    if (us[w].sum() == 0) or (uk[w].sum() == 0):
        COUNT['missing'] += 1
        return random()
    return 1 - cosine(us[w], uk[w])

def euclidean_distance(w, us, uk):
    COUNT['all'] += 1
    if (us[w].sum() == 0) or (uk[w].sum() == 0):
        COUNT['missing'] += 1
        return random()
    return sqrt(sum(pow(a-b,2) for a, b in zip(us[w], uk[w])))

def manhattan_distance(w, us, uk):
    COUNT['all'] += 1
    if (us[w].sum() == 0) or (uk[w].sum() == 0):
        COUNT['missing'] += 1
        return random()
    return sum(abs(a-b) for a,b in zip(us[w], uk[w]))

df_train = pd.read_csv('data/eval_train_21_balance.csv')
df_test = pd.read_csv('data/eval_test_21_balance.csv')
train_words = df_train['word'].tolist()[:]
train_labels = df_train['label'].tolist()[:]
test_words = df_test['word'].tolist()[:]
test_labels = df_test['label'].tolist()[:]

print(df_train.head())

usa_dict_file = 'usa_dict_balance.pickle'
uk_dict_file = 'uk_dict_balance.pickle'

if os.path.isfile(usa_dict_file) and os.path.isfile(uk_dict_file):
    with open(usa_dict_file, 'rb') as handle:
        us_dict = pickle.load(handle)
    with open(uk_dict_file, 'rb') as handle:
        uk_dict = pickle.load(handle)
else:
    with open("data/UK_tokenized.txt") as uk_f, open("data/USA_tokenized.txt") as usa_f:
        uk_tweets = list(map(lambda x: x.split('\t')[-1].strip(), uk_f))
        usa_tweets = list(map(lambda x: x.split('\t')[-1].strip(), usa_f))

    model_path = "pretrained/usa_uk_05_27_trial1.joblib"

    uk_adagram = AdaGramModel(model_path)
    uk_adagram.fit(uk_tweets)

    usa_adagram = AdaGramModel(model_path)
    usa_adagram.fit(usa_tweets)

    # print(uk_adagram.transform(train_words[0]))

    us_dict = {k : usa_adagram.greedy_transform(k) for k in tqdm(train_words + test_words, desc="US")}
    uk_dict = {k : uk_adagram.greedy_transform(k) for k in tqdm(train_words + test_words, desc="UK")}

    with open(usa_dict_file, 'wb') as handle:
        pickle.dump(us_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(uk_dict_file, 'wb') as handle:
        pickle.dump(uk_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(us_dict)
# input()
# print(uk_dict)
# input()
pred = np.array([manhattan_distance(w, us_dict, uk_dict) for w in tqdm(train_words, desc="AdaGram")])

linspace = np.linspace(0, 2.0, 1000)
metrics_scores = []
for t in linspace:
    metrics_scores.append(acc(train_labels, pred > t))
plt.plot(linspace, metrics_scores)
plt.xlabel("Threshold")
plt.ylabel("Acc on training set")
plt.show()

t = linspace[np.argmax(metrics_scores)]
print("Training stats:")
util.print_stats(pred > t, train_labels)
print("Testing stats:")
util.print_stats(np.array([manhattan_distance(w, us_dict, uk_dict) for w in test_words]) > t, test_labels)

print(COUNT)
