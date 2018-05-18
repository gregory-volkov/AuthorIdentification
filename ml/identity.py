from abstract_classes import WSVectorizer


class Identity(WSVectorizer):

    ClusterAlgo = None

    def set_cluster_algo(self, ClusterAlgo):
        self.ClusterAlgo = ClusterAlgo

    def fit_clustered(self, docs):
        pass

    def cluster_docs(self, docs, k):
        docs = [self.preprocessing.mapping(text) for text in docs]
        vectorized_texts = self.feature_vectorizer.doc2vec(docs)
        return self.ClusterAlgo(n_clusters=k).fit(vectorized_texts).labels_
