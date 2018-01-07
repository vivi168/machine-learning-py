import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("### Linear regression with one variable ###")
data = pd.read_csv(
    'data/ex1data1.txt', header=None, names=['Population', 'Profit'])
data.insert(0, 'bias', 1)

train_X = data.values[:, :-1]
train_Y = data.values[:, -1:]
m, n = train_X.shape

initial_theta = tf.zeros([n, 1], tf.float32)
theta = tf.Variable(initial_theta)
iterations = 1500
alpha = 0.01

X = tf.placeholder(tf.float32, [None, n])
Y = tf.placeholder(tf.float32, [None, 1])

h = tf.matmul(X, theta)
cost_function = tf.matmul(tf.transpose(h - Y), h - Y) / (2 * m)
gradient = tf.train.GradientDescentOptimizer(alpha).minimize(cost_function)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    J_history = np.zeros([iterations, 1])

    print("Optimizing...")
    for i in range(iterations):
        sess.run(gradient, feed_dict={X: train_X, Y: train_Y})
        J_history[i] = sess.run(
            cost_function, feed_dict={
                X: train_X,
                Y: train_Y
            })

    t = sess.run(theta)

    pred = tf.matmul(tf.constant([[1., 3.5]]), theta) * 10000
    pred2 = tf.matmul(tf.constant([[1., 7]]), theta) * 10000

    print("Prediction for population of 35,000: {}".format(sess.run(pred)))
    print("Prediction for population of 70,000: {}".format(sess.run(pred2)))
