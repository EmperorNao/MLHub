from Models.LinearModels import LinearRegression
from datasets import train_test_split
from Metrics.metrics import rmse, mse
import numpy as np

import matplotlib.pyplot as plt

#from sklearn.linear_model import LinearRegression

# с моим SGD явно что-то не так :)
# а может быть всё правильно)
print("SGD test")
N = 1000

# исправил, т.к мы не сможем нормально предсказывать линейной регрессий y = x ^ 2
# теперь это y = 5 * x (без b т.к нет w_0)
#pairs = np.array([np.array([i, i * 5 + 1]) for i in range(0, N)])
#x = np.expand_dims(pairs[:, 0], -1)

print("Linear test")

simple = LinearRegression(analytic_solution=False)
#regular = LinearRegression(L2=True, L2_coefficient=0.5)

#x, y = get_dataset("auto-mpg")
pairs = np.array([np.array([i, i * 5 + 2]) for i in range(0, N)])
x = np.expand_dims(pairs[:, 0], -1)
y = np.expand_dims(pairs[:, 1], -1)

x_pair, y_pair = train_test_split(x, y, ratio=0.75)

x_train, x_test = x_pair
y_train, y_test = y_pair


simple.train(x_train, y_train)
#regular.train(x_train, y_train)

y_pred_simple = simple.predict(x_test)
#y_pred_regular = simple.predict(x_test)

# print(rmse(y_test, y_pred_regular))
print(rmse(y_test, y_pred_simple))
#
# print(mse(y_test, y_pred_regular))
print(mse(y_test, y_pred_simple))
#
print(f"weights = {simple.weights}")

n_obj = y_test.shape[0]

fig = plt.figure()

#for i in range(n_obj):
plt.plot(x_test, y_pred_simple, color="red", label="simple")
plt.plot(x_test, [simple.weights[0] * x[0] + simple.weights[-1] for x in x_test], color="green")
plt.show()


# cls = SignClassifier()
#
# x, y = get_dataset("iris")
# x_pair, y_pair = train_test_split(x, y, ratio=0.75)
#
# x_train, x_test = x_pair
# y_train, y_test = y_pair
#
#
# cls.train(x_train, y_train)
#
# y_pred_cls = cls.predict(x_test)
# print(f"accuracy {accuracy(y_test, y_pred_cls)}")


# print(f"weights = {simple.weights}")
# ind = np.argmax(simple.weights[:-1])
#
# n_obj = y_test.shape[0]
#
# fig = plt.figure()
#
# #for i in range(n_obj):
# plt.plot(x_test[:,ind], y_pred_simple, color="red", label="simple")
# #plt.scatter(range(0, n_obj), y_pred_regular, color="blue", label="penalty")
# plt.plot(x_test[:,ind], y_test, color="green", label="real")
# #plt.plot(x_test, [simple.weights[ind] * x[ind] + simple.weights[-1] for x in x_test], color="yellow")
# plt.show()
#
#
# print(rmse(y_test, y_pred_regular))
# print(rmse(y_test, y_pred_simple))
#
# print(mse(y_test, y_pred_regular))
# print(mse(y_test, y_pred_simple))
#
# print(f"weights = {simple.weights}")
# print(f"weights lw = {regular.weights}")
