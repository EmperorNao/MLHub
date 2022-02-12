import numpy as np
from exceptions import DimensionsException

def rmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    return np.sum(np.square(y_real - y_pred)) / y_real.shape[0]


def mse(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    return np.sum(np.abs(y_real - y_pred)) / y_real.shape[0]


def accuracy(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    return np.mean(y_real == y_pred)


def precision(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    pos_idx = np.argwhere(y_real == 1)
    return np.sum(y_real[pos_idx] == y_pred[pos_idx]) / (np.sum(y_real[pos_idx] == y_pred[pos_idx]) + np.sum(y_real[pos_idx] != y_pred[pos_idx]))


def recall(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    pos_neg = np.argwhere(y_real == 0)
    return np.sum(y_real[pos_neg] == y_pred[pos_neg]) / (np.sum(y_real[pos_neg] == y_pred[pos_neg]) + np.sum(y_real[pos_neg] != y_pred[pos_neg]))