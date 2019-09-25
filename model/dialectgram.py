import numpy as np
from tqdm import tqdm
from model.base import ModelBase
import adagram


class DialectGramModel(ModelBase):
    """Transforms a word to a weighted vector representation in a country

    """

    def __init__(self, pretrained):
        """Loads pretrained model

        :param pretrained: str, pretrained model name
        """

        self.tweets = None
        self.index = Index()
        self.vm = adagram.VectorModel.load(pretrained)
        self.dim = self.vm.dim

    def fit(self, docs):

        self.tweets = docs
        for i, tweet in enumerate(docs):
            self.index.add_sentence(tweet, i)
        print("Successfully loaded {} tweets!".format(len(docs)))

    def transform(self, word):
        """Transform a word to its vector representation in the current country (docs)

        :param word: str
        :return: np.array, word embeddings for a word
        """

        def _normalize_word_weight(sense_weights):
            """normalize weights s.t. sum = 1

            :param sense_weights: dict, raw weight sums
            :return: dict, weighted sense weights
            """
            weight_sum = sum([v for k, v in sense_weights.items()])
            return {k: v/weight_sum for k, v in sense_weights.items()}

        if word in self.vm.dictionary.word2id:
            sense_weights = {}
            sense_priors = {}
            for sense, prob in self.vm.word_sense_probs(word):
                sense_priors[sense] = prob
                sense_weights[sense] = 0
            for idx in self.index.get_sentences(word):
                context = self.tweets[idx]
                context = context.split()
                # disambiguate's context is a list of str
                prob_array = self.vm.disambiguate(word, context)[:len(sense_priors)]
                for sense in range(len(sense_weights)):
                    if sense in sense_weights:
                        sense_weights[sense] += prob_array[sense]
            sense_weights = _normalize_word_weight(sense_weights)
            # print(sense_weights)
            weighted_vector = np.zeros(self.dim)
            for sense, weight in sense_weights.items():
                weighted_vector += weight * self.vm.sense_vector(word, sense)
            return weighted_vector
        else:
            return np.zeros(self.dim)

    def greedy_transform(self, word):
        """Transform a word to a word embeddings in a greedy manner instead of adding weights

        :param word: str
        :return: np.array, word embedding
        """
        def _normalize_word_weight(sense_weights):
            """normalize weights s.t. sum = 1

            :param sense_weights: dict, raw weight sums
            :return: dict, weighted sense weights
            """
            weight_sum = sum([v for k, v in sense_weights.items()])
            return {k: v/weight_sum for k, v in sense_weights.items()}

        if word in self.vm.dictionary.word2id:
            sense_weights = {}
            sense_priors = {}
            for sense, prob in self.vm.word_sense_probs(word):
                sense_priors[sense] = prob
                sense_weights[sense] = 0
            for idx in self.index.get_sentences(word):
                context = self.tweets[idx]
                context = context.split()
                prob_array = self.vm.disambiguate(word, context)
                sense = np.argmax(prob_array)
                sense_weights[sense] += 1
            sense_weights = _normalize_word_weight(sense_weights)
            # print(sense_weights)
            weighted_vector = np.zeros(self.dim)
            # initialize with prior proba
            # for sense, weight in sense_priors.items():
            #     weighted_vector += weight * self.vm.sense_vector(word, sense)
            # add regional weighting
            for sense, weight in sense_weights.items():
                weighted_vector += weight * self.vm.sense_vector(word, sense)
            return weighted_vector
        else:
            return np.zeros(self.dim)


    def predict_proba(self, word, context):
        """

        :param word: str
        :param context: str, tweet text context where the word appears
        :return: np.array, proba distribution of a word in 30 senses
        """
        if word in self.vm.dictionary.word2id:
            return self.vm.disambiguate(word, context.split())
        return np.zeros(self.dim)


class Index:
    """
    This class stores word -> sentence ids.
    """

    def __init__(self):
        self.word2sentIDs = {}

    def add_sentence(self, sentence, sentence_idx):
        tokens = sentence.split()
        for token in tokens:
            if token not in self.word2sentIDs:
                self.word2sentIDs[token] = [sentence_idx]
            else:
                self.word2sentIDs[token].append(sentence_idx)

    def get_sentences(self, word):
        return self.word2sentIDs[word] if word in self.word2sentIDs else []
