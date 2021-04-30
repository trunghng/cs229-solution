import numpy as np


def load_data(features_file, label_file):
    X, y = [], []
    with open(features_file, 'r') as f:
        read_data = f.readline()
        while len(read_data) > 0:
            data = read_data.strip().split()
            x1, x2 = data[0], data[1]
            X.append([1, x1, x2])
            read_data = f.readline()

    with open(label_file, 'r') as f:
        read_data = f.readline()
        while len(read_data) > 0:
            data = read_data.strip()
            y.append(data)
            read_data = f.readline()

    return X, y
