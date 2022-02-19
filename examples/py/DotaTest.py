from Pipelines.simple import pipeline
import numpy as np
from sklearn import linear_model
from Models.LinearModels import LogisticRegression
from datasets import get_dataset
from Optimizers.Optimizators import SGDOptimizer
from Metrics.metrics import accuracy, precision, recall


def dota_test():

    np.random.seed(41)
    x, y = get_dataset("dota")

    optim = SGDOptimizer(lr=1, max_iter=10000, lam=0.85, batch_size=1)
    simple = LogisticRegression(optimizer=optim)
    sklearn = linear_model.LogisticRegression()
    pipeline({'simple': simple, 'sklearn': sklearn}, {'acc': accuracy, 'precision': precision, 'recall': recall}, x, y)



if __name__ == "__main__":

    dota_test()