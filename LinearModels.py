import numpy as np
from exceptions import DimensionsException
from Optimizators import sgd


class LinearRegression:
    # L(w, X, y) = (X*w - y) ^ 2
    # x.shape = (l, n)
    # w.shape = (n, 1)
    # y.shape = (l, 1)
    # dL/dw = 2 * X.T * (X*w - y)
    # dL/dw.shape = (l, 1)

    def __init__(self, weights: np.ndarray = None, L1=False, L2=False, L2_coefficient: int = 0,
                 analytic_solution=True):

        self.weights = weights
        self.L2 = L2
        if not self.L2:
            self.L2_coefficient = 0
        else:
            self.L2_coefficient = L2_coefficient if L2_coefficient != 0 else 0.05

        self.analytic_solution = analytic_solution

    def train(self, x: np.ndarray, y: np.ndarray):

        if x.shape[0] != y.shape[0]:
            raise DimensionsException("X and y has different number of objects")

        n_objects = x.shape[0]
        n_features = x.shape[1] + 1
        ones = -1 * np.ones((n_objects, 1))

        x_padded = np.hstack([x, ones])

        l_i = np.eye(n_features, n_features)

        if self.analytic_solution:
            w = np.dot(np.linalg.inv(np.dot(x_padded.T, x_padded) + self.L2_coefficient * l_i), np.dot(x_padded.T, y))
        else:
            w, q = sgd(x, y, 0.01, self.loss(), self.grad_loss, batch_size=32)
        self.weights = w

    def predict(self, x: np.ndarray) -> np.ndarray:

        n_objects = x.shape[0]
        ones = -1 * np.ones((n_objects, 1))
        x_padded = np.hstack([x, ones])

        if x_padded.shape[1] != self.weights.shape[0]:
            raise DimensionsException("Wrong number of features")

        return np.dot(x_padded, self.weights)
