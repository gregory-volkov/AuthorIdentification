class WSVectorizer(object):

    preprocessing = None        # instance of TextPreprocessing class
    feature_vectorizer = None   # instance of FeatureVectorizer class
    authors = None              # List of authors from training set

    def __init__(self, prep, vectorizer):
        self.preprocessing = prep
        self.feature_vectorizer = vectorizer

    # Get clustered docs for training the model
    def fit_clustered(self, docs):
        raise NotImplementedError

    # Get docs belonging to the author
    def fit_docs(self, docs):
        raise NotImplementedError

    def get_authors(self):
        return self.authors
