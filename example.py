from scipy.spatial.distance import canberra
from feature_vectorizer import NGram
from ml import PTHG
from preprocessing import OnlyLetters
from tests import rus_tests, eng_tests
from utils import KMedoids

"""
Example of using PTHG class
"""
test_name = 'aus_king'


def pthg_init(ngram_conf, lang='english'):
    pthg = PTHG(OnlyLetters(lang=lang), NGram(ngram_conf[0], ngram_conf[1], lang=lang))
    pthg.L = 1000
    pthg.T = 15
    return pthg

test = next(t for t in rus_tests + eng_tests if t.test_name == test_name)

expected_doc_clustering = test.expected_clustering
print("expected clusters: {}".format(expected_doc_clustering))

pthg = pthg_init((2, 50), lang='english')
pthg.set_dis(canberra)
pthg.set_cluster_algo(KMedoids)
print("got: {}".format(
    pthg.cluster_docs(test.docs, k=2)
))
