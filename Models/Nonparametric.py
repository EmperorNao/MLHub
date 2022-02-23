import numpy as np
from Distances.distances import euclidian_distance, manhattan_distance


def gausian(x):
    return np.exp(-2 * x * x)



class KernelRegression:

    def __init__(self, kernel='gaussian', dist='euclidean', h=None):

        if kernel == 'gaussian':
            self.kernel = gausian

        self.h = h

        if dist == 'euclidean':
            self.dist = euclidian_distance
        elif dist == 'manhattan':
            self.dist = manhattan_distance

    def fit(self, x, y):

        self.obj = x
        self.answ = y

    def predict(self, x):

        pred = np.ones(x.shape[0])
        for i, obj in enumerate(x):

            dist = [self.dist(obj[0], obj_[0]) for obj_ in self.obj]
            K = np.array([self.kernel(d / self.h) for d in dist])
            pred[i] = (K @ self.answ) / np.sum(K)

        return pred