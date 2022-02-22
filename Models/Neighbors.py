import numpy as np
from Distances.distances import euclidian_distance


class KNNClassifier:

    def __init__(self, k = 1, dist = euclidian_distance):

        self.k = k
        self.dist = dist

        self.answ = None
        self.obj = None
        self.classes = None

    def fit(self, x, y):

        self.obj = x
        self.answ = y
        self.classes = np.unique(self.answ)

    def predict(self, x):

        m = x.shape[0]
        n = self.obj.shape[0]

        pred = np.zeros(m, dtype='int32')
        for i, obj in enumerate(x):
            idx = np.argsort(np.array([self.dist(obj, example) for example in self.obj]))[:min(self.k, n)]

            labels = np.squeeze(self.answ[idx], -1)
            pred[i] = np.bincount(labels).argmax()

        return np.array(pred)
