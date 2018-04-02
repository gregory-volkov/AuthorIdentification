from collections import OrderedDict
from itertools import product
from abstract_classes.feature_vectorizing_abstract import FeatureVectorizer
from constants import low_letters
import numpy as np
from my_utils import max_columns


class NGrams(FeatureVectorizer):

    def __init__(self, n, amt=None, lang='english'):
        self.n = n
        self.lang = lang
        self.__init_ngrams__()
        self.amt = amt

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
            counter[gram] += 1

            for ch in text[self.n:]:
                gram = gram[1:] + ch
                counter[gram] += 1

            res_m[i, :] = list(counter.values())

        return max_columns(res_m, self.amt) if self.amt else res_m
