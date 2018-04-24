

class Test(object):

    # path_dict is a dict, where key - path to some directory, value - list of filenames
    def __init__(self, path_dict, test_name):
        self.test_name = test_name
        self.docs = []
        self.expected_clustering = []
        self.n_clusters = len(path_dict)
        self.path_dict = path_dict
        for i, dir_path in enumerate(path_dict):
            for filename in path_dict[dir_path]:
                self.expected_clustering.append(i)
                self.docs.append(
                    open(dir_path + filename).read()
                )

    def test(self, model):
        got = model.cluster_docs(self.docs, self.n_clusters)
        return self.expected_clustering, got
