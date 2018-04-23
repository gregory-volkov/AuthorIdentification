import numpy as np
from abstract_classes import FeatureVectorizer
from nltk import pos_tag
from constants import tag_ls
from itertools import product


class POSGrams(FeatureVectorizer):

    tag_ls = tag_ls
    langs = {
        'english': 'eng',
        'russian': 'rus'
    }

    def __init__(self, n=1, is_norm=False, amt=None, lang='english'):
        self.lang = self.langs[lang]
        self.n = n
        self.is_norm = is_norm
        self.amt = amt

    def doc2vec(self, texts):
        seq2id = {}
        res_m = []
        for text_id, text in enumerate(texts[:2]):
            print(text_id)
            all_tags = list(map(lambda x: x[1], pos_tag(text, lang=self.lang)))
            res_m.append([0 for _ in range(len(seq2id))])
            for i in range(len(text) - self.n + 1):
                tag_seq = tuple(all_tags[i:i + self.n])
                if tag_seq not in seq2id:
                    cur_id = len(seq2id)
                    seq2id[tag_seq] = cur_id
                    res_m[text_id].append(1)
                else:
                    cur_id = seq2id[tag_seq]
                    res_m[text_id][cur_id] += 1

        return np.array([np.array(row) for row in res_m])
