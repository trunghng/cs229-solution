import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def lwlr(X_train, y_train, x, tau):
    theta = np.zeros(X_train.shape[1])
    max_iters = 20
    LAMBDA = 0.0001

    # compute weights
    w = [np.exp(-(np.linalg.norm(x - x_i)**2) / (2 * tau * tau)) for x_i in X_train]

    # Newton's method iterations
    for _ in range(max_iters):
        z = [w[i] * (y_train[i] - sigmoid(theta.T.dot(X_train[i]))) for i in range(X_train.shape[0])]
        grad = X_train.T.dot(z) - LAMBDA * theta
        D_diagonal = [-w[i] * sigmoid(theta.T.dot(X_train[i])) * (1 - sigmoid(theta.T.dot(X_train[i])))
                      for i in range(X_train.shape[0])]
        D = np.diag(D_diagonal)
        H = X_train.T.dot(D).dot(X_train) - LAMBDA * np.identity(X_train.shape[1])
        theta = theta - np.linalg.inv(H).dot(grad)

    print('Theta after fitting for tau={}: {}'.format(tau, theta))
    return theta


def predict(theta, x):
    if sigmoid(theta.T.dot(x)) > 0.5:
        return 1
    return 0
