import numpy as np
from Distances.distances import euclidian_distance


class KNN:

    def __init__(self, k = 1, dist = euclidian_distance):

        self.k = k
        self.dist = dist

        self.answ = None
        self.obj = None
        self.classes = None

    def fit(self, x, y):

        self.answ = y
        self.obj = x
        self.classes = np.unique(self.answ)


    def predict(self, x):

        m = x.shape[0]
        n = self.obj.shape[0]
        w = np.array([np.argsort(np.array([self.dist(x[j], self.obj[i]) for i in range(n)]))[:min(self.k, n)] for j in range(m)])

        classes = np.zeros((m, self.classes.shape[0]))
        for i, cls in enumerate(self.classes):
            sq = np.squeeze(self.answ, -1)
            idx = np.array([sq for i in range(m)])

            prod = idx * w
            sum = np.sum(prod, axis = 1)
            classes[:, i] = sum

        return np.expand_dims(self.classes[np.argmax(classes, axis = 1)], -1)
