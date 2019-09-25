from model import DialectGramModel
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

with open("data/UK_tokenized.txt") as uk_f, open("data/USA_tokenized.txt") as usa_f:
    uk_tweets_geos = list(map(lambda x: (x.split('\t')[-1].strip(), x.split('\t')[-4].strip()), uk_f)) # context, xy
    usa_tweets_geos = list(map(lambda x: (x.split('\t')[-1].strip(), x.split('\t')[-4].strip()), usa_f))

model_path = "pretrained/usa_uk_05_27_trial1.joblib"

uk_dialectgram = DialectGramModel(model_path)
uk_dialectgram.fit([tweet_geo[0] for tweet_geo in uk_tweets_geos])

usa_dialectgram = DialectGramModel(model_path)
usa_dialectgram.fit([tweet_geo[0] for tweet_geo in usa_tweets_geos])

select_words = ['buffalo', 'trainer', 'test', 'gas', 'subway', 'underground', 'flat']
for word in select_words:
    filename = './output/{}.csv'.format(word)
    show_in_uk_tweets = uk_dialectgram.index.get_sentences(word)
    show_in_usa_tweets = usa_dialectgram.index.get_sentences(word)
    df_lines = []
    print("Processing word [{}]".format(word))
    for idx in show_in_uk_tweets:
        context, xy = uk_tweets_geos[idx]
        proba = uk_dialectgram.predict_proba(word, context)
        line = [xy, context, proba, 'uk']
        df_lines.append(line)
    print("Add {} UK sentences".format(len(show_in_uk_tweets)))
    for idx in show_in_usa_tweets:
        context, xy = usa_tweets_geos[idx]
        proba = usa_dialectgram.predict_proba(word, context)
        line = [xy, context, proba, 'usa']
        df_lines.append(line)
    print("Add {} USA sentences".format(len(show_in_usa_tweets)))
    df = pd.DataFrame(df_lines, columns=['xy', 'text', 'prob', 'country'])
    df.to_csv(filename)
