import numpy as np
from abstract_classes import FeatureVectorizer
from nltk import pos_tag
from itertools import product
from sklearn.preprocessing import normalize


class POSGrams(FeatureVectorizer):

    langs = {
        'english': 'eng',
        'russian': 'rus',
        'eng': 'eng',
        'rus': 'rus'
    }

    def __init__(self, n=1, is_norm=False, amt=None, lang='english'):
        self.lang = self.langs[lang]
        self.n = n
        self.is_norm = is_norm
        self.amt = amt

    def doc2vec(self, texts):
        tags = set()
        texts_tag = []

        for text in texts:
            cur_tags = [pair[1] for pair in pos_tag(text)]
            tags.update(set(cur_tags))
            texts_tag.append(cur_tags)

        seq2id = {v: k for k, v in enumerate(product(tags, repeat=self.n))}
        
        res_m = np.zeros((len(texts), len(seq2id)), dtype=np.int32)

        for text_id, tag_ls in enumerate(texts_tag):
            for i in range(len(tag_ls) - self.n + 1):
                tag_seq = tuple(tag_ls[i:i + self.n])
                tag_id = seq2id[tag_seq]
                res_m[text_id, tag_id] += 1

        if self.is_norm:
            res_m = res_m.astype(np.float)
            res_m = normalize(res_m, axis=1)

        return res_m
