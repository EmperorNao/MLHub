import pandas as pd
import numpy as np
from exceptions import DimensionsException
from math import ceil


def get_dataset(name) -> (np.ndarray, np.ndarray):

    if name == "forestfires":
        df = pd.read_csv("./Datasets/forestfires/forestfires.csv")
        train_col = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
        test_col = ["area"]

    elif name == "auto-mpg":
        columns = ["mpg", "cylinders", "displacement", "horsepower", "weight",
                   "acceleration", "model year", "origin", "car name"]
        train_col = ["displacement", "horsepower", "weight", "acceleration"]
        test_col = ["mpg"]

        df = pd.read_csv("./Datasets/auto_mpg/auto-mpg.csv")
        """
        # old read
        df = pd.DataFrame(columns=columns)
        with open("./Datasets/auto_mpg/auto-mpg.data") as file:
            for line in file:

                split = line.split()
                split = split[:8] + [" ".join(split[8:])]
                df = df.append({columns[i]: split[i] for i in range(0, len(split))}, ignore_index=True)
        df["horsepower"][df["horsepower"] == "?"] = df["horsepower"][df["horsepower"] != "?"].astype(
                np.float32).mean()
        df.to_csv("./Datasets/auto_mpg/auto-mpg.csv", index=False)
        """

    return df[train_col].to_numpy(dtype=np.float32), df[test_col].to_numpy(dtype=np.float32)


def train_test_split(x: np.ndarray,
                     y: np.ndarray,
                     ratio: float = 0) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):

    if x.shape[0] != y.shape[0]:
        raise DimensionsException("X and y has different number of objects")

    if 1 < ratio < 0:
        raise ValueError(f"Ratio need to be in [0, 1], provided {ratio}")

    idx_full = range(0, x.shape[0])
    size = ceil(x.shape[0] * ratio)
    idx_train = list(np.random.choice(idx_full, size))
    idx_test = list(set(idx_full).difference(idx_train))

    return (x[idx_train, :], x[idx_test, :]), (y[idx_train, :], y[idx_test, :])
