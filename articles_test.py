from abstract_classes import Test
from sklearn.cluster import KMeans
from collections import defaultdict
from random import sample, randint
import os
from ml import PTHG, Identity
from utils import KMedoids
from scipy.spatial.distance import canberra
from feature_vectorizer import NGram, POSGrams
from preprocessing import OnlyLetters, PrepIdentity


def pthg_init(ngram_conf=(2, 100), ClusterAlgo=KMedoids, lang='english', dis=canberra, t=15, l=15):
    pthg = PTHG(
        OnlyLetters(lang=lang),
        NGram(ngram_conf[0], amt=ngram_conf[1], lang=lang)
    )

    pthg.set_dis(dis)
    pthg.set_cluster_algo(ClusterAlgo)
    pthg.T = t
    pthg.L = l
    return pthg


def bigram_init(n=2):
    bigram = Identity(OnlyLetters(), NGram(n=n, is_norm=True))
    bigram.set_cluster_algo(KMeans)
    return bigram


def pos_init(n=2):
    pos_gram = Identity(PrepIdentity(lang='english'), POSGrams(n=n, is_norm=True))
    pos_gram.set_cluster_algo(KMeans)
    return pos_gram

article_n = 10
names_n = 10
test_n = 10
path = 'data/articles/train/'

# dict: path -> filenames
path_dict = {}

names = os.listdir(path)
selected_names = sample(names, names_n)

name2texts = defaultdict(list)

models = {
    'Bi-grams': bigram_init(),
    'POS-grams': pos_init(),
    'PTHG': pthg_init(),
}

for name in selected_names:
    files = os.listdir(path + name)
    selected_files = sample(files, article_n)
    path_dict[path + name + '/'] = selected_files

res_ari = {}

for cluster_n in range(2, 15):
    print(cluster_n)
    res_ari[cluster_n] = {
        'Bi-grams': [],
        'POS-grams': [],
        'PTHG': []
    }
    for i in range(test_n):
        print(i)
        #cluster_n = randint(2, names_n)
        selected_names = sample(path_dict.keys(), cluster_n)
        cur_path_dict = {}
        for name in selected_names:
            cur_path_dict[name] = sample(path_dict[name], article_n)
        test = Test(cur_path_dict, str(i))

        for model_name, model in models.items():
            tested = test.test(model)
            res_ari[cluster_n][model_name].append(tested[2])
            print("{} model: {}".format(model_name, tested))

print(res_ari)
