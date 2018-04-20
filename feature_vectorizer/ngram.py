from collections import OrderedDict
from itertools import product
import numpy as np
from abstract_classes.feature_vectorizer import FeatureVectorizer
from constants import low_letters
from utils import max_columns
from sklearn.preprocessing import normalize


class NGram(FeatureVectorizer):

    def __init__(self, n, amt=None, lang='english', is_normalize=False):
        self.n = n
        self.lang = lang
        self.__init_ngrams__()
        self.amt = amt
        self.is_normalize = is_normalize

    def __init_ngrams__(self):
        letters = low_letters[self.lang]
        self.ngrams = list(map(''.join, product(letters, repeat=self.n)))
        self.gram_number = len(low_letters[self.lang]) ** self.n

    def doc2vec(self, texts):
        texts = list(texts)
        res_m = np.zeros((len(texts), self.gram_number), dtype=np.int32)
        counter = OrderedDict()

        for i, text in enumerate(texts):
            for gram in self.ngrams:
                counter[gram] = 0

            gram = text[:self.n]
            if gram in counter:
                counter[gram] += 1

            for ch in text[self.n:]:
                gram = gram[1:] + ch
                counter[gram] += 1

            res_m[i, :] = list(counter.values())

        res_m = max_columns(res_m, self.amt) if self.amt else res_m
        if self.is_normalize:
            return normalize(res_m)
        else:
            return res_m
