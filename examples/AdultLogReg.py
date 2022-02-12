import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from Models.LinearModels import LogisticRegression
from datasets import train_test_split
from Metrics.metrics import accuracy

from Optimizers.Optimizators import SGDOptimizer
from datasets import get_dataset


def adult():

    x, y = get_dataset("adult")

    optim = SGDOptimizer(lr=1e-4, max_iter=1000, lam=0.80)
    simple = LogisticRegression()
    sklearn = linear_model.LogisticRegression(max_iter=100)

    x_pair, y_pair = train_test_split(x, y, ratio=0.75)

    x_train, x_test = x_pair
    y_train, y_test = y_pair

    history = simple.fit(x_train, y_train, optim)
    sklearn.fit(x_train, y_train)

    y_pred_simple = simple.predict(x_test)
    y_pred_sklearn = sklearn.predict(x_test)

    print(f"ACC on SLKEARN = {accuracy(y_test, y_pred_sklearn)}")
    print(f"ACC on our LR = {accuracy(y_test, y_pred_simple)}")

    fig = plt.figure(figsize=(15, 10))

    plt.title("Loss")
    plt.plot([i for i in range(len(history))], history)
    plt.show()


if __name__ == "__main__":
    adult()



