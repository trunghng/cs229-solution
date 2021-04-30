import load_data as load
import lwlr
import numpy as np
import matplotlib.pyplot as plt


def plot(X, y, tau, res):
    # choose a querry point
    x = X[3]
    theta = lwlr.lwlr(X, y, x, tau)
    grid = np.linspace(-1, 1, res)

    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(wspace=0.25, hspace=0.25)

    ax1 = fig.add_subplot(1, 2, 1)
    for i in grid:
        for j in grid:
            label = lwlr.predict(theta, np.array([1, i, j]))
            color = 'r' if label == 0 else 'b'
            ax1.scatter(i, j, c=color)
    ax1.scatter(x[1].item(), x[2].item(), marker='x')
    ax1.set_title('Output data')

    labels = np.array(y.astype(int))
    colors = ['r' if label == 0 else 'b' for label in labels]
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(np.array(X[:, 1]), np.array(X[:, 2]), c=colors)
    ax2.set_title('Training data')

    plt.show()


if __name__ == '__main__':
    X_train, y_train = load.load_data('x.dat', 'y.dat')
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    taus = [0.01, 0.05, 0.1, 0.5, 1, 5]
    res = 20
    for tau in taus:
        plot(X_train, y_train, tau, res)
