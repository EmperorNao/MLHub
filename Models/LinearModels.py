import numpy as np
from exceptions import DimensionsException


class LinearRegression:
    # L(w, X, y) = (X*w - y) ^ 2
    # x.shape = (l, n)
    # w.shape = (n, 1)
    # y.shape = (l, 1)
    # dL/dw = 2 * X.T * (X*w - y)
    # dL/dw.shape = (l, 1)

    def __init__(self, weights: np.ndarray = None, optimizer=None, L2_coefficient: float = 0,
                 analytic_solution=True):

        self.weights = weights
        self.optimizer = optimizer
        self.L2_coefficient = L2_coefficient
        self.analytic_solution = analytic_solution

    def fit(self, x: np.ndarray, y: np.ndarray):

        if x.shape[0] != y.shape[0]:
            raise DimensionsException("X and y has different number of objects")

        n_objects = x.shape[0]
        n_features = x.shape[1] + 1
        ones = 1 * np.ones((n_objects, 1))

        x_padded = np.hstack([x, ones])

        l_i = np.eye(n_features, n_features)

        if self.analytic_solution or not self.optimizer:
            w = np.dot(np.linalg.inv(np.dot(x_padded.T, x_padded) +  self.L2_coefficient * l_i), np.dot(x_padded.T, y))
        else:
            w, q = self.optimizer.fit(x_padded, y, self.loss, self.grad_loss)
        self.weights = w

    def loss(self, w: np.ndarray, x: np.ndarray, y: np.ndarray):
        # x.shape = m x n
        # y.shape = m x 1
        # w.shape = n x 1

        return np.square(np.dot(x, w) - y)

    def grad_loss(self, w: np.ndarray, x: np.ndarray, y: np.ndarray):
        # x.shape = m x n
        # y.shape = m x 1
        # w.shape = n x 1
        # out shape = ((m x n) * (n x 1) - (m x 1)) = (n x m) * (m x 1) = (n x 1)

        return 2 * np.dot(x.T, (np.dot(x, w) - y))

    def predict(self, x: np.ndarray) -> np.ndarray:

        n_objects = x.shape[0]
        ones = 1 * np.ones((n_objects, 1))
        x_padded = np.hstack([x, ones])

        if x_padded.shape[1] != self.weights.shape[0]:
            raise DimensionsException("Wrong number of features")

        return np.dot(x_padded, self.weights)


class BinaryClassifier:

    def __init__(self, optimizer, weights: np.ndarray = None, L2_coefficient: float = 0, logging=False):

        self.optimizer = optimizer
        self.weights = weights
        self.L2_coefficient = L2_coefficient if L2_coefficient != 0 else 0.05
        self.logging=logging

    def fit(self, x: np.ndarray, y: np.ndarray):

        if x.shape[0] != y.shape[0]:
            raise DimensionsException("X and y has different number of objects")

        n_objects = x.shape[0]
        n_features = x.shape[1] + 1
        ones = 1 * np.ones((n_objects, 1))

        x_padded = np.hstack([x, ones])

        y_transformed = []
        for el_y in y:
            if el_y == 1:
                y_transformed.append(1)
            else:
                y_transformed.append(0)

        w, q, history = self.optimizer.fit(x_padded, np.expand_dims(y, -1), self.loss, self.grad_loss, get_hist=True)
        self.weights = w

        return history

    def sigmoid(self, x, w):
        # x.shape = m x n
        # y.shape = m x 1
        # w.shape = n x 1
        # out shape = m x 1

        rev = np.exp(-1 * np.dot(x, w))
        return 1 / (np.ones([x.shape[0], w.shape[1]]) + rev)

    def loss(self, w: np.ndarray, x: np.ndarray, y: np.ndarray):
        # x.shape = m x n
        # y.shape = m x 1
        # w.shape = n x 1

        sigm_res = self.sigmoid(x, w)

        return np.mean(-1 * y * np.log(sigm_res) + \
        -1 * (np.ones_like(y) - y) * np.log(np.ones([x.shape[0], w.shape[1]]) - sigm_res))

    def grad_loss(self, w: np.ndarray, x: np.ndarray, y: np.ndarray):
        # x.shape = m x n
        # y.shape = m x 1
        # w.shape = n x 1
        # out shape = ((m x n) * (n x 1) - (m x 1)) = (n x m) * (m x 1) = (n x 1)

        return np.dot(x.T, (self.sigmoid(x, w) - y))

    def predict(self, x: np.ndarray) -> np.ndarray:

        n_objects = x.shape[0]
        n_features = x.shape[1] + 1
        ones = 1 * np.ones((n_objects, 1))

        x_padded = np.hstack([x, ones])

        pred = self.sigmoid(x_padded, self.weights)
        if self.logging:
            print(self.weights)
            print(pred)
        return (pred >= 0.5).astype(dtype="int").squeeze(-1)


class LogisticRegression:

    def __init__(self, optimizer, weights: np.ndarray = None, L2_coefficient: float = 0, logging=False):

        self.optimizer = optimizer
        self.weights = weights
        self.L2_coefficient = L2_coefficient if L2_coefficient != 0 else 0.05
        self.logging=logging

    def fit(self, x: np.ndarray, y: np.ndarray):

        if x.shape[0] != y.shape[0]:
            raise DimensionsException("X and y has different number of objects")

        n_objects = x.shape[0]
        n_features = x.shape[1] + 1
        self.classes = np.unique(y)

        n_classes = self.classes.shape[0]

        ones = 1 * np.ones((n_objects, 1))

        self.weights = np.random.normal(scale=1/(2 * np.pi * x.shape[0]), size=(n_features, n_classes))

        x_padded = np.hstack([x, ones])

        y_transformed = []
        for el_y in y:
            z = np.zeros(n_classes)
            z[np.where(self.classes == el_y)] = 1
            y_transformed.append(z)

        w, q, history = self.optimizer.fit(x_padded, np.array(y_transformed), self.loss, self.grad_loss, w=self.weights, get_hist=True)

        self.weights = w
        return history

    def softmax(self, v):
        # v = m * k
        # outp = m * k

        exp = np.exp(v)
        denom = np.expand_dims(np.sum(exp, axis=1), -1)
        ret = exp / denom
        return ret

    def softmax_grad(self, v):
        # v = m x k
        # outp = m x k x k

        sm = self.softmax(v)
        m = v.shape[0]
        c = v.shape[1]
        return np.array(np.array([np.array([sm[k][i] * (1 - sm[i]) if i == j else -1 * sm[k][i] * sm[k][j] for i in range(c)]) for j in range(c)]) for k in range(m))

    def loss(self, w: np.ndarray, x: np.ndarray, y: np.ndarray):
        # x.shape = m x n
        # y.shape = m x k
        # w.shape = k x n
        # outp = k x 1

        # sm(k x n * n x m = k x m)

        sm = self.softmax(np.dot(x, w))
        return -1 * np.mean(np.sum(y * np.log(sm), axis=1), axis=0)

    def grad_loss(self, w: np.ndarray, x: np.ndarray, y: np.ndarray):
        # x.shape = m x n
        # y.shape = m x k
        # w.shape = k x n
        # outp =

        # sm(k x n * n x m = k x m)
        sm = self.softmax(np.dot(x, w))
        prod = sm * y
        diff = prod - y
        return np.dot(x.T, diff)

    def predict(self, x: np.ndarray) -> np.ndarray:

        n_objects = x.shape[0]
        n_features = x.shape[1] + 1
        ones = 1 * np.ones((n_objects, 1))

        x_padded = np.hstack([x, ones])

        prob = self.softmax(self.softmax(np.dot(x_padded, self.weights)))
        pred = np.argmax(prob, axis=1)
        if self.logging:
            print(prob)
            print(pred)

        return pred