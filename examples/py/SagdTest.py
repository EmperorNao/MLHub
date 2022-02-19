import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from Models.LinearModels import LinearRegression
from datasets import train_test_split
from Metrics.metrics import rmse, mse

from Optimizers.Optimizators import SAGDOptimizer
# будем бенчмаркать с sklearn

# пообучаем на данных с разным шумом и посмотрим метрики:


def sagd_test():

    for scale in [0, 1.41, 4, 9, 25, 36, 100]:

        # max iter обеспечивает гораздо более лучшую сходимость
        # для улучшения качества пробуйте увеличить max_iter

        optim = SAGDOptimizer(lr=1e-6, max_iter=10000)

        simple = LinearRegression()
        sklearn = linear_model.LinearRegression()
        l2_reg = LinearRegression(L2_coefficient=0.05)
        sgd_lr = LinearRegression(analytic_solution=False, optimizer=optim)

        N = 100
        #x, y = get_dataset("auto-mpg")
        pairs = np.array([np.array([i, i * 5 + 2 + np.random.normal(scale=scale)]) for i in range(0, N)])
        x = np.expand_dims(pairs[:, 0], -1)
        y = np.expand_dims(pairs[:, 1], -1)

        x_pair, y_pair = train_test_split(x, y, ratio=0.75)

        x_train, x_test = x_pair
        y_train, y_test = y_pair

        simple.fit(x_train, y_train)
        sklearn.fit(x_train, y_train)
        sgd_lr.fit(x_train, y_train)
        l2_reg.fit(x_train, y_train)

        y_pred_simple = simple.predict(x_test)
        y_pred_sklearn = sklearn.predict(x_test)
        y_pred_sgd = sgd_lr.predict(x_test)
        y_pred_l2_reg = sgd_lr.predict(x_test)

        print(f"Scale = {scale}")
        print(f"RMSE on SLKEARN = {rmse(y_test, y_pred_sklearn)}")
        print(f"RMSE on our LR = {rmse(y_test, y_pred_simple)}")
        print(f"RMSE on LR + L2_reg = {rmse(y_test, y_pred_l2_reg)}")
        print(f"RMSE on SGD = {rmse(y_test, y_pred_sgd)}")

        print(f"MSE on SKLEARN = {mse(y_test, y_pred_sklearn)}")
        print(f"MSE on our LR = {mse(y_test, y_pred_simple)}")
        print(f"MSE on LR + L2_reg = {mse(y_test, y_pred_l2_reg)}")
        print(f"MSE on SGD = {mse(y_test, y_pred_sgd)}")
        print()

        n_obj = y_test.shape[0]

        fig = plt.figure(figsize = (15, 10))

        plt.subplot(2, 2, 1)
        plt.title("Our LR")
        plt.plot(x_test, y_pred_simple, color="red", label="simple")
        plt.scatter(x_test, y_test, color="black", s=50)

        plt.subplot(2, 2, 2)
        plt.title("SKLEARN")
        plt.plot(x_test, y_pred_sklearn, color="blue", label="sklearn")
        plt.scatter(x_test, y_test, color="black", s=50)

        plt.subplot(2, 2, 3)
        plt.title("LR + L2 regularization")
        plt.plot(x_test, y_pred_l2_reg, color="green", label="reg")
        plt.scatter(x_test, y_test, color="black", s=50)

        plt.subplot(2, 2, 4)
        plt.title("SGD")
        plt.plot(x_test, y_pred_sgd, color="yellow", label="sgd")
        plt.scatter(x_test, y_test, color="black", s=50)

        plt.show()


if __name__ == "__main__":
    sagd_test()