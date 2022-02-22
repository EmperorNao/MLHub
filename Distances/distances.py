import numpy as np


def euclidian_distance(x_1, x_2):

    return np.sum(np.square(x_1 - x_2), axis=-1) ** 0.5
