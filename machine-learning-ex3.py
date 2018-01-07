import numpy as np
import scipy.optimize as sop
import scipy.io as sio

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def lr_cost_function(theta, X, y, l):
  theta = theta.reshape(-1,1)
  theta_reg = np.copy(theta)
  theta_reg[0] = 0

  m = len(y)
  h = sigmoid(X.dot(theta))
  J = (1/m) * (-y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h))) + l/(2*m) * (theta_reg.T.dot(theta_reg))
  grad = (1/m) * (X.T.dot(h-y)) + l/m * theta_reg

  return J, grad

def one_vs_all(X, y, num_labels, l):
  m,n = X.shape
  X = np.append(np.ones([m,1]), X, axis=1)
  all_theta = np.zeros([num_labels, n+1])

  print("Optimizing one-vs-all...")
  for c in range(num_labels):
    initial_theta = np.zeros(n+1)
    res = sop.minimize(lr_cost_function, initial_theta, args=(X,(y==c).astype(int), l), method='TNC', jac=True)
    all_theta[c,:] = res.x
    print(c)

  return all_theta

def predict_one_vs_all(all_theta, X):
  m = len(X)
  num_labels = len(all_theta)
  X = np.append(np.ones([m,1]), X, axis=1)
  pred = np.argmax(X.dot(all_theta.T), axis=1)

  return pred

def predict(Theta1, Theta2, X):
  m = len(X)
  num_labels = len(Theta2)

  p = np.zeros([m, 1])
  X = np.append(np.ones([m,1]), X, axis=1)

  z1 = X.dot(Theta1.T)
  a2 = sigmoid(z1)
  a2 = np.append(np.ones([m,1]), a2, axis=1)

  z2 = a2.dot(Theta2.T)
  a3 = sigmoid(z2)

  pred = np.argmax(a3, axis=1)

  # in matlab, index starts at 1
  pred += 1

  return pred

def run():
  print('### One-vs-all logistic regression ###')

  data = sio.loadmat('data/ex3data1.mat')
  X, y = np.copy(data['X']), np.copy(data['y'])
  # in the original dataset, 0 was labeled as 10
  y[y==10] = 0

  l = 0.1
  num_labels = len(np.unique(y))
  all_theta = one_vs_all(X, y, num_labels, 0.1)

  pred = predict_one_vs_all(all_theta, X)
  acc = np.mean(pred == y.T) * 100

  print("prediction accuracy: {:.2f}%".format(acc))

  m,n = X.shape
  rp = np.random.permutation(m)
  for i in range(5):
    # we need a row vector
    rand_x = X[rp[i], :].reshape(1,-1)
    pred = predict_one_vs_all(all_theta, rand_x)
    print('prediction: {} digit: {}'.format(pred, pred % 10))

def run_nn():
  print('### Neural Network ###')

  data = sio.loadmat('data/ex3data1.mat')
  X, y = np.copy(data['X']), np.copy(data['y'])

  weights = sio.loadmat('data/ex3weights.mat')
  Theta1, Theta2 = np.copy(weights['Theta1']), np.copy(weights['Theta2'])

  pred = predict(Theta1, Theta2, X)
  acc = np.mean(pred == y.T) * 100

  print("prediction accuracy: {:.2f}%".format(acc))

  m,n = X.shape
  rp = np.random.permutation(m)
  for i in range(5):
      # we need a row vector
      rand_x = X[rp[i], :].reshape(1,-1)
      pred = predict(Theta1, Theta2, rand_x)
      print('prediction: {} digt: {}'.format(pred, pred % 10))

if __name__ == '__main__':
  run()
  run_nn()