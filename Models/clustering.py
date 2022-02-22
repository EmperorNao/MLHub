import numpy as np
from Distances.distances import euclidian_distance


class KMeans:

    def __init__(self, k, max_iter=100, n_starts=10, dist=euclidian_distance):

        self.k = k
        self.max_iiter = max_iter
        self.n_starts = n_starts
        self.dist = dist

    def _centers(self, x, k):

        n_dims = x.shape[1]

        std = x.std(axis=0)
        max = x.max(axis=0)
        min = x.min(axis=0)
        diff = (max - min) / k

        return np.array([min + diff * np.random.randint(0, k) + np.random.normal(scale=std, size=(n_dims)) for i in range(k)])

    def fit_and_predict(self, x):

        if not self.n_starts % 2:
            self.n_starts += 1

        m = x.shape[0]

        n_labels = []
        n_centers = []

        for start in range(self.n_starts):

            iter = 0
            centers = self._centers(x, self.k)
            labels = np.zeros((m))
            while True and iter < self.max_iiter:

                dist = []
                for cls in range(0, self.k):
                    t = np.tile(centers[cls], (m, 1))
                    d = self.dist(x, t)
                    dist.append(d)

                dist = np.vstack(dist).T
                new_labels = np.argmin(dist, axis=1)
                iter += 1
                if not np.sum(new_labels == labels):
                    break

                labels = new_labels

                for cls in range(0, self.k):
                    cls_x = x[labels == cls]
                    centers[cls] = 0 if not cls_x.shape[0] else np.sum(cls_x, axis=0) / cls_x.shape[0]



            # idx = []
            # for cls in range(0, self.k):
            #     cls_idx = labels == cls
            #     idx.append(cls_idx)
            #
            # sort_idx = np.argsort(centers, axis = 0)
            #
            # for idxs in sort_idx:
            #     pass

            return labels

            n_labels.append(labels)


        stacked = np.vstack(n_labels).T
        return np.median(stacked, axis=1)

