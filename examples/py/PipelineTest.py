from Pipelines.simple import pipeline
import numpy as np
from sklearn import linear_model
from Models.LinearModels import LinearRegression
from datasets import get_dataset
from Optimizers.Optimizators import SGDOptimizer
from Metrics.metrics import rmse, mse


def pipeline_test():

    np.random.seed(41)
    x, y = get_dataset("random-linear-dots")

    optim = SGDOptimizer(lr=1, max_iter=10000, lam=0.85, batch_size=1)
    simple = LinearRegression(optimizer=optim)
    sklearn = linear_model.LinearRegression()
    pipeline({'simple': simple, 'sklearn': sklearn}, {'rmse': rmse, 'mse': mse}, x, y)



if __name__ == "__main__":

    pipeline_test()