class WSVectorizer(object):

    preprocessing = None        # Instance of TextPreprocessing class
    feature_vectorizer = None   # Instance of FeatureVectorizer class
    doc_names = None            # List of doc names
    lang = None                 # Language of texts

    def __init__(self, prep, vectorizer, lang='english'):
        self.preprocessing = prep
        self.feature_vectorizer = vectorizer
        self.lang = lang

    # Get clustered docs for training the model
    def fit_clustered(self, docs):
        raise NotImplementedError

    # Get docs for clustering (docs is a dict doc_name -> text)
    def cluster_docs(self, docs, k):
        raise NotImplementedError
