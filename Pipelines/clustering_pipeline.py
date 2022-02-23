import numpy as np


def test_clustering(models: dict, metrics: dict, x, y):

    out = [model.fit_and_predict(x) for model in models.values()]

    for modelname, outp in zip(list(models.keys()), out):

        print(f"Model {modelname}")
        for metricname, metric in metrics.items():
            print(f"{metricname} = {metric(y, outp)}")


def get_clustering_res(models: dict, x):

    out = [model.fit_and_predict(x) for model in models.values()]

    return out