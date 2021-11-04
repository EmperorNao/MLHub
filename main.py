from LinearModels import LinearRegression
from datasets import get_dataset, train_test_split
from metrics import rmse, mse
from Optimizators import sgd
import numpy as np

# с моим SGD явно что-то не так :)
# а может быть всё правильно)
print("SGD test")


def f(w, x, y):
    """
    n_obj = x.shape[0]
    res = np.zeros(n_obj)
    for i in range(n_obj):
        res[i] = (x[i][0] * w[0] - y[i][0]) ** 2
    """
    """
    xw = np.expand_dims(np.dot(x, w), -1)
    diff = xw - y
    res = np.square(diff)
    """
    return np.square(np.dot(x, w) - y)


def grad_f(w, x, y):
    xw = (np.dot(x, w))
    diff = xw - y
    product = np.dot(x.T, diff)
    # w.shape = nx1
    # x.shape = lxn
    # xw.shape = lx1
    # diff.shape = lx1
    # product.shape= (nxl) * (lx1) = (nx1)
    return 2 * product

N = 1000

# исправил, т.к мы не сможем нормально предсказывать линейной регрессий y = x * 2
# теперь это y = 5 * x (без b т.к нет w_0)
pairs = np.array([np.array([i, i * 5 + np.random.normal(0, 0.01)]) for i in range(0, N)])

x = np.expand_dims(pairs[:, 0], -1)
y = np.expand_dims(pairs[:, 1], -1)
w = np.ndarray([1])
w[0] = 5
w, q = sgd(x, y, 0.001, f, grad_f, w=w)
print(w, q)




# print("Linear test")
#
# simple = LinearRegression()
# regular = LinearRegression(L2=True, L2_coefficient=0.5)
#
# x, y = get_dataset("auto-mpg")
# x_pair, y_pair = train_test_split(x, y, ratio=0.75)
#
# x_train, x_test = x_pair
# y_train, y_test = y_pair
#
#
# simple.train(x_train, y_train)
# regular.train(x_train, y_train)
#
# y_pred_simple = simple.predict(x_test)
# y_pred_regular = simple.predict(x_test)
#
#
# n_obj = y_test.shape[0]
# for i in range(n_obj):
#     print(f"real = {y_test[i]}, predicted w\o l2 = {y_pred_simple[i]}, predicted w l2 = {y_pred_regular[i]}")
#
# print(rmse(y_test, y_pred_regular))
# print(rmse(y_test, y_pred_simple))
#
# print(mse(y_test, y_pred_regular))
# print(mse(y_test, y_pred_simple))
#
# print(f"weights = {simple.weights}")
# print(f"weights lw = {regular.weights}")
