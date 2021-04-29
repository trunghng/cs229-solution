import numpy as np
import random
import draw_clusters as draw


def read_file(feature_file):
    X = []
    with open(feature_file, 'r') as f:
        read_data = f.readline()
        while len(read_data) > 0:
            X_i = read_data.strip().split()
            X.append(X_i)
            read_data = f.readline()

    return X


def kmeans(X, k):
    random.seed(1)
    centroids = np.array([X[random.randrange(0, X.shape[0])] for _ in range(k)])  # k x n
    clusters = np.zeros(X.shape[0],)  # m x 1

    while True:
        centroids_old = centroids.copy()

        # Assign each training example to the closest cluster centroid
        for i in range(X.shape[0]):
            similarities = [similarity_cal(X[i], centroid) for centroid in centroids]
            clusters[i] = np.argmin(similarities)

        draw.draw_clusters(X, clusters, centroids)

        # Update centroids
        for i in range(centroids.shape[0]):
            cluster_members = X[np.where(clusters == i)]
            centroids[i] = np.average(cluster_members, axis=0)

        if np.linalg.norm(centroids_old - centroids) < 1e-10:
            break


def similarity_cal(member, centroid):
    # Using Euclidean distance as our similarity
    return np.linalg.norm(member - centroid)**2


X = read_file('./X.dat')
X = np.array(X, dtype=np.float64)
k = 3
kmeans(X, k)
