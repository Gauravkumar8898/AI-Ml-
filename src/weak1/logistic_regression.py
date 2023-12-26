# ---------------------------------Assignment 2------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.constant import titanic_train_x, titanic_train_y, titanic_test_x, titanic_test_y

X_train = pd.read_csv(titanic_train_x)
Y_train = pd.read_csv(titanic_train_y)

X_test = pd.read_csv(titanic_test_x)
Y_test = pd.read_csv(titanic_test_y)

X_train = X_train.drop("Id", axis=1)
Y_train = Y_train.drop("Id", axis=1)
X_test = X_test.drop("Id", axis=1)
Y_test = Y_test.drop("Id", axis=1)
print(X_train)
X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_test = Y_test.values

X_train = X_train.T
Y_train = Y_train.reshape(1, X_train.shape[1])

X_test = X_test.T
Y_test = Y_test.reshape(1, X_test.shape[1])

print("Shape of X_train : ", X_train.shape)
print("Shape of Y_train : ", Y_train.shape)
print("Shape of X_test : ", X_test.shape)
print("Shape of Y_test : ", Y_test.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_model(x, y, learning_rates, iteration):
    m = X_train.shape[1]
    n = X_train.shape[0]
    wc = np.zeros((n, 1))
    bc = 0
    cost_list = []
    for i in range(iteration):
        z = np.dot(wc.T, x) + bc
        a = sigmoid(z)

        # using  cost function
        cost = -(1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

        # using  Gradient Descent
        dw = (1 / m) * np.dot(a - y, x.T)
        db = (1 / m) * np.sum(a - y)
        # Changing in weight and basis
        wc = wc - learning_rates * dw.T
        bc = bc - learning_rates * db
        # Keeping track of our cost function value
        cost_list.append(cost)
        if i % (iteration / 10) == 0:
            print("cost after ", i, "iteration is : ", cost)

    return wc, bc, cost_list


def accuracy(x, y, wu, bu):
    # print(X,W)
    z = np.dot(wu.T, x) + bu
    a = sigmoid(z)
    a = a > 0.5
    a = np.array(a, dtype='int64')
    acc = (1 - np.sum(np.absolute(a - y)) / y.shape[1]) * 100
    print("Accuracy of the logistic model is : ", round(acc, 2), "%")


iterations = 100000
learning_rate = 0.00145
w, b, cost_lists = logistic_model(X_train, Y_train, learning_rate, iterations)
# print(W,B)

plt.plot(np.arange(iterations), cost_lists)
plt.xlabel('Iteration')
plt.ylabel('cost')


def predict_logistic_regression(x, wu, bu):
    # Calculate the raw predicted value
    z = np.dot(wu.T, x) + bu
    probability = sigmoid(z)

    # print(probability)
    predictions = 1 if probability >= 0.5 else 0

    return predictions


accuracy(X_test, Y_test, w, b)
test = np.array([3, 0, 41.0, 0, 0, 7.85, 1])

# print(W)
prediction = predict_logistic_regression(test, w, b)

print("Prediction of logistic regression is :", prediction)
plt.show()
