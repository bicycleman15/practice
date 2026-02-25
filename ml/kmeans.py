# practice kmeans clustering

import numpy as np

# np.random.seed(41)

cluster_1 = np.random.randn(100, 2) + np.array([2, 2])
cluster_2 = np.random.randn(100, 2) + np.array([-2, -2])
cluster_3 = np.random.randn(100, 2) + np.array([2, -2])

X = np.vstack([cluster_1, cluster_2, cluster_3])
np.random.shuffle(X)


def one_step_kmeans(X, centroids):
    # X: [N, D]
    # centroids: [K, D]
    # take old centroids
    # return new ones

    N, D = X.shape
    C, D = centroids.shape
    
    X = X.reshape(N, 1, D)
    centroids = centroids.reshape(1, C, D)

    dist = ((X - centroids)**2).sum(axis=-1) # [N, C]

    which_cluster_x = np.argmin(dist, axis=-1) # [N]

    new_centroids = list()

    for i in range(C):
        mask = which_cluster_x == i
        x_in_this_cluster = X[mask] # [N', D]
        c = x_in_this_cluster.mean(axis=0) # [D]
        c = c.reshape(1, -1)
        new_centroids.append(c)

    return np.concat(new_centroids, axis=0)


K = 3
# initial_centroids = X[np.random.choice(X.shape[0], K, replace=False)]
indices = [0, 1, 2]
initial_centroids = X[indices, :]
print(initial_centroids)

centroids = initial_centroids
for step in range(1):
    centroids = one_step_kmeans(X, centroids)

print("Final centroids:")
print(centroids)
print("\nTrue centers: [2,2], [-2,-2], [2,-2]")