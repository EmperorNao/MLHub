import numpy as np


def is_valid_input(y_real, y_pred):
    assert len(y_real.shape) == 1
    assert len(y_pred.shape) == 1


def rmse(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    is_valid_input(y_real, y_pred)
    return np.sum(np.square(y_real - y_pred)) / y_real.shape[0]


def mse(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    is_valid_input(y_real, y_pred)
    return np.sum(np.abs(y_real - y_pred)) / y_real.shape[0]


def accuracy(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    is_valid_input(y_real, y_pred)
    return np.mean(y_real == y_pred)


def precision(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    is_valid_input(y_real, y_pred)
    pos_idx = np.argwhere(y_real == 1)
    return np.sum(y_real[pos_idx] == y_pred[pos_idx]) / (np.sum(y_real[pos_idx] == y_pred[pos_idx]) + np.sum(y_real[pos_idx] != y_pred[pos_idx]))


def recall(y_real: np.ndarray, y_pred: np.ndarray) -> float:

    is_valid_input(y_real, y_pred)
    pos_neg = np.argwhere(y_real == 0)
    return np.sum(y_real[pos_neg] == y_pred[pos_neg]) / (np.sum(y_real[pos_neg] == y_pred[pos_neg]) + np.sum(y_real[pos_neg] != y_pred[pos_neg]))