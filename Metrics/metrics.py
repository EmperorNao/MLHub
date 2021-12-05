import numpy as np


def rmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    return np.sum(np.square(y_real - y_pred)) / y_real.shape[0]


def mse(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    return np.sum(np.abs(y_real - y_pred)) / y_real.shape[0]


def accuracy(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    return np.mean(y_real == y_pred)