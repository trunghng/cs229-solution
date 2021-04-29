import numpy as np
import load_data


def l1ls(X, y, LAMBDA):
    theta = np.zeros((X.shape[1], 1))

    while True:
        theta_old = theta.copy()

        for i in range(theta.shape[0]):
            theta[i] = 0
            theta_i = [0, 0]
            theta_i[0] = -(X[:, i].T.dot(X).dot(theta) + X[:, i].T.dot(y) - LAMBDA) * 1.0 / (X[:, i].T.dot(X[:, i]))
            theta_i[0] = max(theta_i[0], 0)
            theta_i[1] = (-X[:, i].T.dot(X).dot(theta) + X[:, i].T.dot(y) + LAMBDA) * 1.0 / (X[:, i].T.dot(X[:, i]))
            theta_i[1] = min(theta_i[1], 0)

            theta[i] = theta_i[0]
            obj = [0, 0]
            obj[0] = 1 / 2 * np.linalg.norm(X.dot(theta) - y)**2 + LAMBDA * np.linalg.norm(theta, ord=1)
            theta[i] = theta_i[1]
            obj[1] = 1 / 2 * np.linalg.norm(X.dot(theta) - y)**2 + LAMBDA * np.linalg.norm(theta, ord=1)
            theta[i] = theta_i[np.argmin(obj)]

        if np.linalg.norm(theta_old - theta) < 1e-5:
            break

    return theta


X, y, theta = load_data.read_file('./x.dat', './y.dat', './theta.dat')
X = np.array(X, dtype=np.float64)
y = np.array(y, dtype=np.float64)
theta_true = np.array(theta, dtype=np.float64)

LAMBDAS = [0.0001, 0.001, 0.01, 0.1, 1, 10]
for l in LAMBDAS:
    theta = l1ls(X, y, l)
    print(f'For lambda = {l}, theta = {theta}')
