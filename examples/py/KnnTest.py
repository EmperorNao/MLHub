from Pipelines.simple import pipeline, get_pipeline_res
import numpy as np
from sklearn import linear_model
from Models.LinearModels import LinearRegression
from datasets import get_dataset
from datasets import train_test_split
from Optimizers.Optimizators import SGDOptimizer
from Metrics.metrics import accuracy, precision, recall
from sklearn.metrics import accuracy_score
from Models.Neighbors import KNNClassifier
from sklearn import neighbors
from sklearn import model_selection
import matplotlib.pyplot as plt


def get_color(l):

    if l == 0:
        return "red"
    elif l == 1:
        return "blue"


def knn_test_basic():

    np.random.seed(41)
    x, y = get_dataset("random-binary_clouds")

    knn = KNNClassifier(k=20)
    knn_linear = KNNClassifier(k=20, weighted='linear')
    knn_expo = KNNClassifier(k=20, weighted='exponential', q=0.9)

    knn.fit(x, y)
    knn_linear.fit(x, y)
    knn_expo.fit(x, y)

    plt.title("knn")
    plt.scatter(x[:, 0], x[:, 1], c = [get_color(label) for label in knn.predict(x)])
    plt.show()

    plt.title("knn_linear")
    plt.scatter(x[:, 0], x[:, 1], c = [get_color(label) for label in knn_linear.predict(x)])
    plt.show()

    plt.title("knn_expo")
    plt.scatter(x[:, 0], x[:, 1], c = [get_color(label) for label in knn_expo.predict(x)])
    plt.show()

    plt.title("original")
    plt.scatter(x[:, 0], x[:, 1], c = [get_color(label) for label in y])
    plt.show()


def knn_test_prodv():

    np.random.seed(41)
    x, y = get_dataset("titanic")

    knn = KNNClassifier(k=15)
    knn_linear = KNNClassifier(k=15, weighted='linear')
    knn_expo = KNNClassifier(k=15, weighted='exponential', q=0.9)
    sk_knn = neighbors.KNeighborsClassifier(n_neighbors=15)

    knn005 = KNNClassifier(k=15, parzen=True, h=0.05)
    knn02 = KNNClassifier(k=15, parzen=True, h=0.2)
    knn01 = KNNClassifier(k=15, parzen=True, h=0.1)
    knn05 = KNNClassifier(k=15, parzen=True, h=0.5)
    knn1 = KNNClassifier(k=15, parzen=True, h=1)

    knn_non_fixed = KNNClassifier(k=15, parzen=True, fix_h=False)


    pipeline({'knn': knn, 'sk_knn': sk_knn, 'linear': knn_linear, 'expo': knn_expo,
              'knn005': knn005, 'knn02': knn02, 'knn01': knn01, 'knn05': knn05, 'knn1': knn1, 'knn_non_fixed':knn_non_fixed
              },
             {'acc': accuracy, 'precision': precision, 'recall': recall},
             x, y)


def knn_param_test(log = True):

    np.random.seed(41)
    x, y = get_dataset("adult")

    idx = np.random.randint(0, x.shape[0], size = 500)
    x = x[idx]
    y = y[idx]

    hist = dict()
    k_value = [k for k in range(1, min(x.shape[0], 50))]
    for modelname in ['knn', 'sklearn']:
        hist[modelname + ": " + "train"] = []
        hist[modelname + ": " + "test"] = []

    for k in k_value:
        if log:
            print(f"k = {k}")
        knn = KNNClassifier(k=k)
        sk_knn = neighbors.KNeighborsClassifier(n_neighbors=k)

        models = {'knn': knn, 'sklearn': sk_knn}
        x_pair, y_pair = train_test_split(x, y, ratio=0.75)

        x_train, x_test = x_pair
        y_train, y_test = y_pair

        for model in models.values():
            model.fit(x_train, y_train)

        for modelname, model in models.items():

            y_pred_train = model.predict(x_train)
            y_pred_test = model.predict(x_test)

            t = accuracy(np.squeeze(y_train, -1), y_pred_train)
            hist[modelname + ": " + "train"].append(t)

            if log:
                print(f"{modelname} : train acc is {t}")

            t = accuracy(np.squeeze(y_test, -1), y_pred_test)
            hist[modelname + ": " + "test"].append(t)
            if log:
                print(f"{modelname} : test acc is {t}")

    for name, history in hist.items():

        plt.plot(k_value, history, label=name)

    plt.legend()
    plt.show()


if __name__ == "__main__":

    knn_test_prodv()