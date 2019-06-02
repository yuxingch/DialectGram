import pandas as pd
import numpy as np
import random
import re
import util

inverted_vocabs = {}
for c in ["UK", "USA"]:
    data = list(map(str, util.load_data("./data/%s_tokenized.txt" % c)[5].tolist()))
    vocab = util.build_vocab(data)
    inverted_vocabs[c] = {k: v for v, k in enumerate(vocab)}

joint_vocab = set(inverted_vocabs["UK"].keys()) & set(inverted_vocabs["USA"].keys())
word_list = pd.read_csv("./data/word_list.csv", encoding="gbk")

# Clean word list
words = word_list["Word"]
for i in range(len(words)):
    words[i] = re.sub("\(.+\)", "", words[i])
    words[i] = re.sub("\[.+\]", "", words[i])
    words[i] = re.sub("\r\n", ",", words[i])
    words[i] = re.sub(" ", "", words[i])
word_list["Word"] = words

# Match joint words
def contains(s, key):
    for k in key.split(","):
        if k in s:
            return k
    return None
filtered_set = set()
for w in word_list["Word"]:
    kw = contains(joint_vocab, w)
    if kw:
        filtered_set.add(kw)

pos_set = filtered_set
neg_set = set()
while len(pos_set) + len(neg_set) < 1000:
    sample = random.sample(joint_vocab, 1000 - len(pos_set) - len(neg_set))
    sample = set(filter(lambda x : x not in pos_set, sample))
    neg_set.update(sample)

word = list(pos_set) + list(neg_set)
label = [1] * len(pos_set) + [0] * len(neg_set)
from sklearn.model_selection import train_test_split
word_train, word_test, label_train, label_test = train_test_split(word, label)

train_data = pd.DataFrame.from_dict({
    "word" : word_train,
    "label" : label_train,
})
test_data = pd.DataFrame.from_dict({
    "word" : word_test,
    "label" : label_test,
})
train_data.to_csv("./data/eval_train.csv", index=False)
test_data.to_csv("./data/eval_test.csv", index=False)