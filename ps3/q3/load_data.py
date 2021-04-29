import numpy as np


def read_file(feature_file, label_file, theta_file):
    X, y, theta = [], [], []
    with open(feature_file, 'r') as f:
        read_data = f.readline()
        while len(read_data) > 0:
            X_i = read_data.strip().split()
            X.append(X_i)
            read_data = f.readline()

    with open(label_file, 'r') as f:
        read_data = f.readline()
        while len(read_data) > 0:
            y_i = read_data.strip()
            y.append(y_i)
            read_data = f.readline()

    with open(theta_file, 'r') as f:
        read_data = f.readline()
        while len(read_data) > 0:
            theta_i = read_data.strip()
            theta.append(theta_i)
            read_data = f.readline()

    return X, y, theta
