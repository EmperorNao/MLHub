from Models.clustering import KMeans
from datasets import get_dataset
from Pipelines.clustering_pipeline import get_clustering_res
from Metrics.metrics import accuracy
import matplotlib.pyplot as plt


def get_color(l):

    if l == 0:
        return "red"
    elif l == 1:
        return "blue"
    elif l == 2:
        return "yellow"


def plot_clustering(x, y, title):

    fig = plt.figure(figsize=(x.shape[1] * 5, x.shape[1] * 5))

    plt.title(title)
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):

            plt.subplot(x.shape[1], x.shape[1], 1 + i * x.shape[1] + j)
            plt.scatter(x[:, i], x[:, j], c = [get_color(lab) for lab in y])
    plt.show()


def clustering_test():

    cluster = KMeans(k=3)

    x, y = get_dataset("iris")

    outp = get_clustering_res({"kmeans": cluster}, x)

    plot_clustering(x, y, "original")
    plot_clustering(x, outp[0], "clustering")

if __name__ == "__main__":

    clustering_test()
