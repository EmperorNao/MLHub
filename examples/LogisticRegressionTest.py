import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from Models.LinearModels import BinaryClassifier
from datasets import train_test_split
from Metrics.metrics import accuracy

from Optimizers.Optimizators import SGDOptimizer
# будем бенчмаркать с sklearn

# пообучаем на данных с разным шумом и посмотрим метрики:


def log_test():

    for scale in [3, 5, 9]:

        optim = SGDOptimizer(lr=1e-4, max_iter=10000, lam=0.80)
        simple = BinaryClassifier()
        sklearn = linear_model.LogisticRegression()

        N = 200
        N2 = 25

        n_pos = int(N // 2 + np.random.randint(-N2, N2))
        n_neg = int(N // 2 + np.random.randint(-N2, N2))

        pos_x = 10
        pos_y = 10

        neg_x = -10
        neg_y = 10

        pos_pairs = np.array([np.array(
            [pos_x + np.random.normal(scale=scale), pos_y + np.random.normal(scale=scale)])
            for i in range(0, n_pos)])

        pos_answers = np.array([1] * n_pos)

        neg_pairs = np.array([np.asarray(
            [neg_x + np.random.normal(scale=scale), neg_y + np.random.normal(scale=scale)])
            for i in range(0, n_neg)])
        neg_answers = np.array([0] * n_neg)

        x = np.vstack([pos_pairs, neg_pairs])
        y = np.hstack([pos_answers, neg_answers])

        idx = np.random.permutation([i for i in range(x.shape[0])])

        x = x[idx]
        y = y[idx]

        x_pair, y_pair = train_test_split(x, y, ratio=0.75)

        x_train, x_test = x_pair
        y_train, y_test = y_pair

        simple.fit(x_train, y_train, optim)
        sklearn.fit(x_train, y_train)

        y_pred_simple = simple.predict(x_test)
        y_pred_sklearn = sklearn.predict(x_test)

        print(f"Scale = {scale}")
        print(f"ACC on SLKEARN = {accuracy(y_test, y_pred_sklearn)}")
        print(f"ACC on our LR = {accuracy(y_test, y_pred_simple)}")

        fig = plt.figure(figsize = (15, 10))

        idx_one = y_test == 1
        idx_zero = y_test == 0

        plt.title("Two classes")

        plt.scatter(x_test[idx_one, 0], x_test[idx_one, 1], color="red", s=50)

        plt.scatter(x_test[idx_zero, 0], x_test[idx_zero, 1], color="blue", s=50)

        b = simple.weights[2]
        w1 = simple.weights[0]
        w2 = simple.weights[1]
        x = [i for i in range(-15, 15)]
        y = [-i * w1 / w2 - b / w2 for i in x]
        plt.plot(x, y, color="black")
        plt.ylim([-25, 25])

        plt.show()


if __name__ == "__main__":
    binlog_test()
