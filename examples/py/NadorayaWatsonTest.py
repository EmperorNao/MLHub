from Pipelines.simple import pipeline
import numpy as np
from sklearn import linear_model
from Models.LinearModels import LinearRegression
from datasets import get_dataset
from Optimizers.Optimizators import SGDOptimizer
from Metrics.metrics import rmse, mse
from Models.Nonparametric import KernelRegression


def kernel_regression_test():

    np.random.seed(41)
    x, y = get_dataset("auto-mpg")

    sklearn = linear_model.LinearRegression()

    kernel005 = KernelRegression(h=0.05)
    kernel01 = KernelRegression(h=0.1)
    kernel02 = KernelRegression(h=0.2)

    kernel005_m = KernelRegression(h=0.05, dist='manhattan')
    kernel01_m = KernelRegression(h=0.1, dist='manhattan')
    kernel02_m = KernelRegression(h=0.2, dist='manhattan')

    y = np.squeeze(y, -1)
    pipeline({'kernel005': kernel005, 'kernel01': kernel01, 'kernel02': kernel02,
              'kernel005_m': kernel005_m, 'kernel01': kernel01_m, 'kernel02_m': kernel02_m,
              'sklearn': sklearn}, {'rmse': rmse, 'mse': mse}, x, y)



if __name__ == "__main__":

    kernel_regression_test()