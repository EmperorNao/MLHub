from Pipelines.simple import pipeline
import numpy as np
from sklearn import linear_model
from Models.LinearModels import LinearRegression
from datasets import get_dataset
from Optimizers.Optimizators import SGDOptimizer
from Metrics.metrics import accuracy, precision, recall
from Models.Neighbors import KNN
from sklearn import neighbors


def knn_test():

    np.random.seed(41)
    x, y = get_dataset("random-binary_clouds")

    knn = KNN(k=20)
    sk_knn = neighbors.KNeighborsClassifier()
    pipeline({'knn': knn, 'sk_knn': sk_knn}, {'acc': accuracy, 'precision': precision, 'recall': recall}, x, y)



if __name__ == "__main__":

    knn_test()