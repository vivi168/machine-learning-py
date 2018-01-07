import numpy as np
import pandas as pd


def compute_cost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    J = (h - y).T.dot(h - y) / (2 * m)

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros([num_iters, 1])

    for i in range(num_iters):
        h = X.dot(theta)
        grad = X.T.dot(h - y)

        theta = theta - (alpha / m) * grad
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def feature_normalize(X):
    mu = np.zeros([1, len(X.T)])
    sigma = np.zeros([1, len(X.T)])

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_norm = (X - mu) / sigma
    X_norm = np.append(np.ones([len(X_norm), 1]), X_norm, axis=1)
    return X_norm, mu, sigma


def run():
    print("### Linear regression with one variable ###")
    data = pd.read_csv(
        'data/ex1data1.txt', header=None, names=['Population', 'Profit'])
    data.insert(0, 'intercept', 1)

    X = data.values[:, :-1]
    y = data.values[:, -1:]

    initial_theta = np.zeros([len(X.T), 1])
    iterations = 1500
    alpha = 0.01

    theta, J_history = gradient_descent(X, y, initial_theta, alpha, iterations)

    pred = np.array([[1, 3.5]]).dot(theta) * 10000
    pred2 = np.array([[1, 7]]).dot(theta) * 10000

    print("Prediction for population of 35,000: {}".format(pred))
    print("Prediction for population of 70,000: {}".format(pred2))


def run_multi():
    print("### Linear regression with multiple variables ###")
    data = pd.read_csv(
        'data/ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])

    X = data.values[:, :-1]
    y = data.values[:, -1:]

    X_norm, mu, sigma = feature_normalize(X)
    initial_theta = np.zeros([len(X_norm.T), 1])

    alpha = 0.15
    num_iters = 400

    theta, J_history = gradient_descent(X_norm, y, initial_theta, alpha,
                                        num_iters)

    x_t = np.array([[1650, 3]])
    x_t_norm = (x_t - mu) / sigma
    x_t = np.append(np.ones([len(x_t), 1]), x_t_norm, axis=1)
    pred = x_t.dot(theta)

    print("Prediction for 1650sq ft house with 3 bedrooms: {}".format(pred))


if __name__ == '__main__':
    run()
    run_multi()
