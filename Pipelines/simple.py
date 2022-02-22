import numpy as np
from datasets import train_test_split


def pipeline(models: dict, metrics: dict, x, y):

    x_pair, y_pair = train_test_split(x, y, ratio=0.75)

    x_train, x_test = x_pair
    y_train, y_test = y_pair

    for model in models.values():
        model.fit(x, y)

    for modelname, model in models.items():
        y_pred = model.predict(x_test)

        print(f"Model {modelname}")
        for metricname, metric in metrics.items():
            print(f"{metricname} = {metric(np.squeeze(y_test, -1), y_pred)}")

        print()


def get_pipeline_res(models: dict, x, y):

    x_pair, y_pair = train_test_split(x, y, ratio=0.75)

    x_train, x_test = x_pair
    y_train, y_test = y_pair

    for model in models.values():
        model.fit(x, y)

    out = []
    for modelname, model in models.items():
        y_pred = model.predict(x_test)
        out.append(y_pred)

    return out
