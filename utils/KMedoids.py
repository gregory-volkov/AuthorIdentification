import random
from scipy.spatial.distance import euclidean
import numpy as np


class KMedoids(object):

    labels_ = None

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    def fit(self, X):
        k = self.n_clusters
        distances = np.empty((X.shape[0], X.shape[0]))
        for i, v in enumerate(X):
            for j, u in enumerate(X):
                distances[i, j] = euclidean(v, u)

        def compute_new_medoid(cluster, distances):
            mask = np.ones(distances.shape)
            mask[np.ix_(cluster, cluster)] = 0.
            cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
            costs = cluster_distances.sum(axis=1)
            return costs.argmin(axis=0, fill_value=10e9)

        def assign_points_to_clusters(medoids, distances):
            distances_to_medoids = distances[:, medoids]
            clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
            clusters[medoids] = medoids
            return clusters

        m = distances.shape[0]
        cur_medoids = np.array([-1] * k)

        while not len(np.unique(cur_medoids)) == k:
            cur_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
        old_medoids = np.array([-1] * k)
        new_medoids = np.array([-1] * k)

        while not ((old_medoids == cur_medoids).all()):
            clusters = assign_points_to_clusters(cur_medoids, distances)

            for cur_medoid in cur_medoids:
                cluster = np.where(clusters == cur_medoid)[0]
                new_medoids[cur_medoids == cur_medoid] = compute_new_medoid(cluster, distances)

            old_medoids[:] = cur_medoids[:]
            cur_medoids[:] = new_medoids[:]

        self.labels_ = clusters
        return self
