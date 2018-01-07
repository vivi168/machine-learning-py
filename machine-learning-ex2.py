import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sop
import sklearn.preprocessing as skp


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y):
    # minimize x0 is a 1D vector, here we need a column vector
    theta = theta.reshape(-1, 1)

    m = len(y)
    h = sigmoid(X.dot(theta))
    J = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))
    grad = (1 / m) * (X.T.dot(h - y))

    return J, grad


def cost_function_reg(theta, X, y, l):
    theta = theta.reshape(-1, 1)
    theta_reg = np.copy(theta)
    theta_reg[0] = 0

    m = len(y)
    h = sigmoid(X.dot(theta))
    J_reg = l / (2 * m) * (theta_reg.T.dot(theta_reg))

    J = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + J_reg
    grad = (1 / m) * (X.T.dot(h - y)) + l / m * theta_reg

    return J, grad


def run():
    print("### Logistic Regression ###")
    data = pd.read_csv(
        'data/ex2data1.txt',
        header=None,
        names=['Exam 1', 'Exam 2', 'Admitted'])
    data.insert(0, 'intercept', 1)

    X = data.values[:, :-1]
    y = data.values[:, -1:]

    initial_theta = np.zeros([len(X.T), 1])

    res = sop.minimize(
        cost_function, initial_theta, args=(X, y), method='TNC', jac=True)
    theta = res.x.reshape(-1, 1)

    p = np.round(sigmoid(X.dot(theta)))
    acc = np.mean(p == y) * 100

    print("prediction accuracy: {:.2f}%".format(acc))


def run_reg():
    print("### Regularized logistic regression ###")
    data = pd.read_csv(
        'data/ex2data2.txt',
        header=None,
        names=['Microchip Test 1', 'Microchip Test 2', 'Accepted'])

    X = data.values[:, :-1]
    y = data.values[:, -1:]

    poly = skp.PolynomialFeatures(6)
    X = poly.fit_transform(X)

    l = 1
    initial_theta = np.zeros([len(X.T), 1])

    res = sop.minimize(
        cost_function_reg,
        initial_theta,
        args=(X, y, l),
        method='TNC',
        jac=True)
    theta = res.x.reshape(-1, 1)

    p = np.round(sigmoid(X.dot(theta)))
    acc = np.mean(p == y) * 100

    print("prediction accuracy: {:.2f}%".format(acc))


if __name__ == '__main__':
    run()
    run_reg()
