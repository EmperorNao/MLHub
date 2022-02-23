import numpy as np
from Distances.distances import euclidian_distance


class KNNClassifier:

    def __init__(self, k = 1, dist = euclidian_distance, weighted=None, q=0.9, parzen=False, kernel='gauss', fix_h=True, h=None):

        self.k = k
        self.dist = dist

        self.answ = None
        self.obj = None
        self.classes = None
        self.weighted = weighted

        self.parzen = parzen

        if kernel == "gauss":
            self.kernel = lambda x: np.exp(-2 * x * x)

        self.h = h
        self.fix_h = fix_h
        self.q = q

    def fit(self, x, y):

        self.obj = x
        self.answ = y
        self.classes = np.unique(self.answ)

    def make_weights(self, size):

        weights = None
        if not self.weighted:
            weights = np.ones(size)
        elif self.weighted == 'linear':
            weights = np.array([(self.k + 1 - i) / self.k for i in range(1, self.k + 1)])
        elif self.weighted == 'exponential':

            weights = np.zeros(size)
            p = self.q
            for i in range(size):
                weights[i] = p
                p *= self.q

        else:
            raise KeyError("Wrong 'weighted' argument")

        return weights

    def _non_parzen_predict(self, x):

        m = x.shape[0]
        n = self.obj.shape[0]

        weights = self.make_weights(min(self.k, n))
        pred = np.zeros(m, dtype='int32')
        for i, obj in enumerate(x):
            idx = np.argsort(np.array([self.dist(obj, example) for example in self.obj]))[:min(self.k, n)]

            labels = self.answ[idx]

            answ = np.zeros(np.unique(self.answ).shape[0])
            for w, cls in zip(weights, labels):
                answ[cls] += w

            pred[i] = np.argmax(answ)

        return np.array(pred)

    def _parzen_predict(self, x):

        m = x.shape[0]
        n = self.obj.shape[0]

        pred = np.zeros(m, dtype='int32')
        for i, obj in enumerate(x):
            d = np.array([self.dist(obj, example) for example in self.obj])
            idx = np.argsort(d)[:min(self.k, n)]

            labels = self.answ[idx]

            answ = np.zeros(np.unique(self.answ).shape[0])

            if self.fix_h:
                weights = np.array([self.kernel(dist/self.h) for dist in d[idx]])

            else:
                last = d[idx][min(self.k, n) - 1]
                weights = np.array([self.kernel(dist / last) for dist in d[idx]])

            for w, cls in zip(weights, labels):
                answ[cls] += w

            pred[i] = np.argmax(answ)

        return np.array(pred)

    def predict(self, x):

       if self.parzen:
           return self._parzen_predict(x)
       else:
           return self._non_parzen_predict(x)
