from random import choice
from sklearn.metrics import adjusted_rand_score
from itertools import product
from abstract_classes.writing_style import WSVectorizer
import numpy as np
from collections import defaultdict


class PTHG(WSVectorizer):

    L = None
    T = None
    inv_T = None
    chunk2doc = {}
    Docs = []
    D = []
    n = None
    vectorized_chunks = None
    clustered_docs = None

    # Parameters, you need to set manually
    Dis = None
    ClusterAlgo = None

    # Parameters for training
    t_set = None
    l_set = None
    crand = None
    iter_number = None

    # Setting a mapping R^n x R^n -> R
    def set_dis(self, dis):
        self.Dis = dis

    # Setting a clustering algorithm with following properties:
    #   1. It's a class with constructor, accepting n_clusters argument
    #   2. Has a fit(X) method, which pass objects for clustering
    #   3. Has a field labels_ which consists of clusters numbers
    #      E.g. [1, 1, 2] means that objects with ids 0 and 1 are in the same cluster,
    #      but the 2nd object in other
    def set_cluster_algo(self, ClusterAlgo):
        self.ClusterAlgo = ClusterAlgo

    # Divide docs into k clusters
    def cluster_docs(self, docs, k):

        if type(docs) == dict:
            self.doc_names = docs.keys()
            doc_ls = docs.values()
        else:
            doc_ls = docs

        self.Docs = [''.join(self.preprocessing.mapping(text)) for text in doc_ls]
        self.chunk2doc = {}
        self.inv_T = 1 / self.T
        self.n = len(self.Docs)
        self.__chunk_dividing()
        self.__C()
        self.vectorized_chunks = self.feature_vectorizer.doc2vec(self.D)
        self.__dis_eval()
        self.__V()
        self.clustering(k=k)
        return self.clustered_docs

    # Evaluation of distances between all vectorized chunks
    def __dis_eval(self):
        chunk_number = len(self.vectorized_chunks)
        self.dis_cache = np.empty((chunk_number, chunk_number), dtype=np.float)
        # It is proposed, that Dis function is commutative
        for i in range(chunk_number):
            for j in range(i, chunk_number):
                self.dis_cache[i, j] = self.dis_cache[j, i] = \
                    self.Dis(self.vectorized_chunks[i], self.vectorized_chunks[j])

    # ZV distance, described in article: (2)
    def __ZV(self, chunk_num, last_precursor):
        chunk_id = self.C[chunk_num]
        last_precursor_id = self.C[last_precursor]
        acc = 0
        for i in range(last_precursor_id - self.T, last_precursor_id):
            acc += self.dis_cache[chunk_id, i]
        return self.inv_T * acc

    # DZV distance. described in article: (3)
    def __DZV(self, i, j):
        def A(a, b): return self.__ZV(a, b)
        return abs(A(i, i) + A(j, j) - A(i, j) - A(j, i))

    # Generation of list, which consists of chunk_id, "participated" in V matrix
    def __C(self):

        max_chunk_id = defaultdict(int)
        cnt = defaultdict(int)

        for chunk_id, doc_id in self.chunk2doc.items():
            cnt[doc_id] += 1
            if chunk_id > max_chunk_id[doc_id]:
                max_chunk_id[doc_id] = chunk_id

        chunks = [
            [chunk_id for chunk_id in range(max_chunk_id[doc_id] - cnt[doc_id] + self.T + 1, max_chunk_id[doc_id] + 1)]
            for doc_id in range(self.n)
                  ]

        self.C = sum(chunks, [])
        self.m = len(self.C)

    # Initialization of V matrix
    def __V(self):
        self.V = np.empty((self.m, self.m))
        for i in range(self.m):
            for j in range(i, self.m):
                self.V[i, j] = self.V[j, i] = self.__DZV(i, j)

    # Clustering of V matrix rows and finding clustering of original documents
    def clustering(self, k):
        clusters = self.ClusterAlgo(n_clusters=k).fit(self.V).labels_

        # Mapping doc_id -> cluster_id -> number_of_chunks
        doc_cnt = defaultdict(lambda: defaultdict(int))

        # Filling the doc_cnt dict: doc_id -> chunk_id -> number of chunks
        for chunk_num, cluster_id in enumerate(clusters):
            chunk_id = self.C[chunk_num]
            doc_id = self.chunk2doc[chunk_id]
            doc_cnt[doc_id][cluster_id] += 1

        # Filling res_dict: cluster_id -> set(doc_id)
        res_dict = defaultdict(set)
        for doc_id, cnt_dict in doc_cnt.items():
            max_occur_id = max(cnt_dict.items(), key=lambda x: x[1])[0]
            res_dict[max_occur_id].add(doc_id)

        # Finding clustered docs
        self.clustered_docs = [None for _ in range(self.n)]
        for cluster_id in res_dict:
            for doc_id in res_dict[cluster_id]:
                self.clustered_docs[doc_id] = cluster_id

        return self.clustered_docs

    # Setting parameters for training model
    def set_parameters(self, t_set, l_set, crand=0.5, iter_number=10):
        self.t_set = t_set
        self.l_set = l_set
        self.crand = crand
        self.iter_number = iter_number

    # Fit clustered documents for training
    def fit_clustered(self, docs):
        mean_dict = {}
        set_id_range = range(len(docs))

        for t, l in product(self.t_set, self.l_set):
            print(t, l)
            self.T = t
            self.L = l
            acc = 0
            for i in range(self.iter_number):
                print(i)
                set_id_1 = choice(set_id_range)
                set_id_2 = choice(set_id_range)
                expected_clusters = [set_id_1, set_id_2]
                doc1 = choice(docs[set_id_1])
                doc2 = choice(docs[set_id_2])
                self.cluster_docs([doc1, doc2], k=2)              
                acc += adjusted_rand_score(expected_clusters, self.clustered_docs)
            mean_dict[(t, l)] = acc / self.iter_number
        filtered_dict = dict(filter(lambda x: x[1] > self.crand, mean_dict.items()))
        res_l = min(filtered_dict, key=lambda x: x[1])[1]
        res_t = min(filter(lambda x: x[1] == res_l, filtered_dict), key=lambda x: x[0])[0]
        self.T, self.L = res_t, res_l
        print(mean_dict)
        print('chosen parameters: T = {}, L = {}'.format(self.T, self.L))

    # Dividing docs into chunks and saving result in self.D
    def __chunk_dividing(self):
        res_list = []
        chunk_size = self.L
        chunk_id = 0

        def add_chunk(doc, begin, end, doc_id):
            res_list.append(doc[begin: end])
            nonlocal chunk_id
            self.chunk2doc[chunk_id] = doc_id
            chunk_id += 1

        for doc_id, doc in enumerate(self.Docs):
            fst_end = len(doc) % chunk_size

            maybe_chunk = doc[:fst_end]
            if maybe_chunk:
                add_chunk(doc, 0, fst_end, doc_id)

            cur_p = fst_end
            while cur_p < len(doc):
                add_chunk(doc, cur_p, cur_p + chunk_size, doc_id)
                cur_p += chunk_size

        self.D = tuple(res_list)
