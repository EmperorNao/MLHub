import pandas as pd
import numpy as np
from exceptions import DimensionsException
from math import ceil
from sklearn.preprocessing import LabelEncoder
from os.path import join


REL_PATH = '..\\..\\Datasets\\'


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
    columns = ['age', 'wofkclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex','capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '>50K']
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

    df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    numerical = ['Age', 'Fare', ]
    categorical = ['Pclass', 'Sex', 'SibSp', 'Embarked']
    target = ['Survived']

    #df.columns = columns
    un = df[target].iloc[:, 0].unique()
    y = (df[target] == un[0]).to_numpy(dtype="int")
    x = []

    processed = []
    # one-hot-encoding categorical
    for col in categorical:
        x.append(scale(one_hot_encoding(df[col])))

    x += [np.expand_dims(scale(df[col].to_numpy()), -1) for col in numerical]

    return np.hstack(x), y


def dota(df):

    target = ['RadiantWon']
    numerical = ['AverWRAllyHeroes','AverWRAllyGamers','AverWREnemyHeroes','AverWREnemyGamers']

    un = df[target].iloc[:, 0].unique()
    y = (df[target] == un[0]).to_numpy(dtype="int")

    x = []
    for col in numerical:
        x.append(np.expand_dims(scale(df[col].to_numpy()), -1))

    return np.hstack(x), y


def iris():

    path = join(REL_PATH, 'iris\\iris.data')
    columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]
    train_col = ["sepal length", "sepal width", "petal length", "petal width"]
    test_col = "class"

    df = pd.read_csv(path, names=columns)
    df.loc[df[test_col] == "Iris-setosa", test_col] = 0
    df.loc[df[test_col] == "Iris-versicolor", test_col] = 1
    df.loc[df[test_col] == "Iris-virginica", test_col] = 2

    return df[train_col].to_numpy(dtype=np.float32), df[test_col].to_numpy(dtype=np.float32)


def get_dataset(name, scale_categorical=True, scale_numerical=True) -> (np.ndarray, np.ndarray):

    df = None
    train_col = []
    test_col = []
    if name == "forestfires":
        df = pd.read_csv("./Datasets/forestfires/forestfires.csv")
        train_col = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
        test_col = ["area"]

    elif name == "auto-mpg":
        columns = ["mpg", "cylinders", "displacement", "horsepower", "weight",
                   "acceleration", "model year", "origin", "car name"]
        train_col = ["displacement", "horsepower", "weight", "acceleration"]
        test_col = ["mpg"]

        df = pd.read_csv(r"P:\D\Programming\MLHub\Datasets\auto_mpg\auto-mpg.csv")
        return scale(df[train_col].to_numpy(dtype=np.float32)), df[test_col].to_numpy(dtype=np.float32)

    elif name == "iris":

        return iris(rel_path)

    elif name == "adult":

        df = pd.read_csv(r".\Datasets\adult\adult.data", header=None)

        return adult(df)

    elif name == "titanic":

        df = pd.read_csv(r"P:\D\Programming\MLHub\Datasets\titanic\train.csv")
        return titanic(df)

    elif name == "random-linear-dots":

        N = 1000
        # x, y = get_dataset("auto-mpg")
        pairs = np.array([np.array([i, i * 5 + 3 + np.random.normal(scale=1)]) for i in range(0, N)])
        x = np.expand_dims(pairs[:, 0], -1)
        y = pairs[:, 1]
        return x, y

    elif name == "random-binary_clouds" or name == "binary":

        N = 200
        N2 = 25

        n_pos = int(N // 2 + np.random.randint(-N2, N2))
        n_neg = int(N // 2 + np.random.randint(-N2, N2))

        pos_x = 10
        pos_y = 10

        neg_x = -10
        neg_y = 10

        pos_pairs = np.array([np.array(
            [pos_x + np.random.normal(scale=1), pos_y + np.random.normal(scale=1)])
            for i in range(0, n_pos)])

        pos_answers = np.array([1] * n_pos)

        neg_pairs = np.array([np.asarray(
            [neg_x + np.random.normal(scale=1), neg_y + np.random.normal(scale=1)])
            for i in range(0, n_neg)])
        neg_answers = np.array([0] * n_neg)

        x = np.vstack([pos_pairs, neg_pairs])
        y = np.expand_dims(np.hstack([pos_answers, neg_answers]), -1)
        return x, y

    elif name == "binary_high_scale":

        N = 1000
        N2 = 300

        n_pos = int(N // 2 + np.random.randint(-N2, N2))
        n_neg = int(N // 2 + np.random.randint(-N2, N2))

        pos_x = 1
        pos_y = 3

        neg_x = -1
        neg_y = -3

        pos_pairs = np.array([np.array(
            [pos_x + np.random.normal(scale=1.2), pos_y + np.random.normal(scale=25)])
            for i in range(0, n_pos)])

        pos_answers = np.array([1] * n_pos)

        neg_pairs = np.array([np.asarray(
            [neg_x + np.random.normal(scale=1.2), neg_y + np.random.normal(scale=25)])
            for i in range(0, n_neg)])
        neg_answers = np.array([0] * n_neg)

        x = np.vstack([pos_pairs, neg_pairs])
        y = np.expand_dims(np.hstack([pos_answers, neg_answers]), -1)
        return x, y

    elif name == "dota":

        df = pd.read_csv(r"P:\D\Programming\MLHub\Datasets\dota\datafile.txt")
        return dota(df)


    return df[train_col].to_numpy(dtype=np.float32), df[test_col].to_numpy(dtype=np.float32)


def train_test_split(x: np.ndarray,
                     y: np.ndarray,
                     ratio: float = 0) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):

    if x.shape[0] != y.shape[0]:
        raise DimensionsException("X and y has different number of objects")

    if 1 < ratio < 0:
        raise ValueError(f"Ratio need to be in [0, 1], provided {ratio}")

    idx = np.random.permutation(x.shape[0])

    x = x[idx]
    y = y[idx]

    idx_full = range(0, x.shape[0])
    size = ceil(x.shape[0] * ratio)
    idx_train = list(np.random.choice(idx_full, size))
    idx_test = list(set(idx_full).difference(idx_train))

    return (x[idx_train, :], x[idx_test, :]), (y[idx_train], y[idx_test])
