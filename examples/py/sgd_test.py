import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from Models.LinearModels import LogisticRegression
from datasets import train_test_split
from Metrics.metrics import accuracy, precision, recall

from Optimizers.Optimizators import SGDOptimizer
from datasets import get_dataset


def titanic():

    np.random.seed(41)
    x, y = get_dataset("titanic")

    optim = SGDOptimizer(lr=1e-4, max_iter=10000, lam=0.85, batch_size=128)
    simple = LogisticRegression()
    sklearn = linear_model.LogisticRegression(max_iter=100)

    x_pair, y_pair = train_test_split(x, y, ratio=0.75)

    x_train, x_test = x_pair
    y_train, y_test = y_pair

    y_test = np.squeeze(y_test, -1)

    history = simple.fit(x_train, y_train, optim)
    sklearn.fit(x_train, np.squeeze(y_train, -1))

    y_pred_simple = simple.predict(x_test)
    y_pred_sklearn = sklearn.predict(x_test)

    print(f"ACC on SLKEARN = {accuracy(y_test, y_pred_sklearn)}")
    print(f"ACC on our LR = {accuracy(y_test, y_pred_simple)}")

    print(f"precision on SLKEARN = {precision(y_test, y_pred_sklearn)}")
    print(f"precision on our LR = {precision(y_test, y_pred_simple)}")

    print(f"recall on SLKEARN = {recall(y_test, y_pred_sklearn)}")
    print(f"recall on our LR = {recall(y_test, y_pred_simple)}")

    pos_index = np.where(y_test == 1)
    neg_index = np.where(y_test == 0)
    print(f"SKLEARN:\n"
          f"TP = {np.sum(y_test[pos_index] == y_pred_sklearn[pos_index])}\n"
          f"FP = {np.sum(y_test[pos_index] != y_pred_sklearn[pos_index])}\n"
          f"FN = {np.sum(y_test[neg_index] != y_pred_sklearn[neg_index])}\n"
          f"TN = {np.sum(y_test[neg_index] == y_pred_sklearn[neg_index])}")

    print(f"LR:\n"
          f"TP = {np.sum(y_test[pos_index] == y_pred_simple[pos_index])}\n"
          f"FP = {np.sum(y_test[pos_index] != y_pred_simple[pos_index])}\n"
          f"FN = {np.sum(y_test[neg_index] != y_pred_simple[neg_index])}\n"
          f"TN = {np.sum(y_test[neg_index] == y_pred_simple[neg_index])}")

    fig = plt.figure(figsize=(15, 10))

    plt.title("Loss")
    plt.plot([i for i in range(len(history))], history)
    plt.show()


if __name__ == "__main__":
    titanic()



