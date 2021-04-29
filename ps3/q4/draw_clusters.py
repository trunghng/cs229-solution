import matplotlib.pyplot as plt


def draw_clusters(X, clusters, centroids):
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    fig, ax = plt.subplots()

    colors = [color_list[i] for i in clusters.astype(int)]
    ax.scatter(X[:, 0], X[:, 1], c=colors, edgecolors=None)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='k')
    ax.grid(True)

    plt.show()
