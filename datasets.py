import pandas as pd
import numpy as np
from exceptions import DimensionsException
from math import ceil
from sklearn.preprocessing import LabelEncoder


def one_hot_encoding(x: pd.DataFrame) -> np.ndarray:

    unique = x.unique()

    ohe = np.zeros([len(x), len(unique)])
    for i, kv in enumerate(x.iteritems()):
        index, v = kv
        ohe[i][np.where(unique == v)] = 1

    return ohe


def scale(x: np.ndarray) -> np.ndarray:

    z = ((x - x.mean()) / x.std())
    return z


def adult(df):

    numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex','capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '>50K']
    target = ['>50K']

    df.columns = columns
    un = df[target].iloc[:,0].unique()
    y = (df[target] == un[0]).to_numpy(dtype="int")
    x = []

    processed = []
    #one-hot-encoding categorical
    for col in categorical:
        x.append(one_hot_encoding(df[col]))

    x += [np.expand_dims(scale(df[col].to_numpy()), -1) for col in numerical]
    for col in numerical:
        x.append(np.expand_dims(scale(df[col].to_numpy()), -1))

    for ohe in processed:
        x.append(scale(ohe))

    return np.hstack(x), y


def titanic(df):

    #df = df.dropna()
    df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    numerical = ['Age', 'Fare', ]
    categorical = ['Pclass', 'Sex', 'SibSp', 'Embarked']
    # columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    #            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    #            '>50K']
    target = ['Survived']

    #df.columns = columns
    un = df[target].iloc[:, 0].unique()
    y = (df[target] == un[0]).to_numpy(dtype="int")
    x = []

    processed = []
    # one-hot-encoding categorical
    for col in categorical:
        x.append(one_hot_encoding(df[col]))

    x += [np.expand_dims(scale(df[col].to_numpy()), -1) for col in numerical]
    for col in numerical:
        x.append(np.expand_dims(scale(df[col].to_numpy()), -1))

    for ohe in processed:
        x.append(scale(ohe))

    return np.hstack(x), y


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

    elif name == "iris":
        columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]
        train_col = ["sepal length", "sepal width", "petal length", "petal width"]
        test_col = ["class"]

        df = pd.read_csv("./Datasets/iris/iris.data", names=columns)

        df.loc[df[test_col] == "Iris-setosa"][test_col] = 0
        df.loc[df[test_col] == "Iris-versicolour"][test_col] = 1
        df.loc[df[test_col] == "Iris-viriginica"][test_col] = 2

    elif name == "adult":

        df = pd.read_csv(r"P:\D\Programming\MLHub\Datasets\adult\adult.data", header=None)

        return adult(df)

    elif name == "titanic":

        df = pd.read_csv(r"P:\D\Programming\MLHub\Datasets\titanic\train.csv")
        return titanic(df)

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

    return (x[idx_train, :], x[idx_test, :]), (y[idx_train], y[idx_test])
