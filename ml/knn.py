# practice KNN classifier

import numpy as np

np.random.seed(42)

cluster_1 = np.random.randn(100, 2) + np.array([2, 2])
cluster_2 = np.random.randn(100, 2) + np.array([-2, -2])
cluster_3 = np.random.randn(100, 2) + np.array([2, -2])

X = np.vstack([cluster_1, cluster_2, cluster_3])
y = np.array([0] * 100 + [1] * 100 + [2] * 100)

shuffle_idx = np.random.permutation(X.shape[0])
X = X[shuffle_idx]
y = y[shuffle_idx]

X_train, y_train = X[:240], y[:240]
X_test, y_test = X[240:], y[240:]


def knn_predict(X_train, y_train, X_test, k):
    # X_train: [N_train, D]
    # y_train: [N_train]
    # X_test:  [N_test, D]
    # k: number of neighbors
    # return: y_pred [N_test]

    N, D = X_train.shape
    N_test, D = X_test.shape

    X_test = X_test.reshape(N_test, 1, D)
    X = X_train.reshape(1, N, D)
    y = y_train.reshape(N, 1)

    dist = -((X_test - X)**2).sum(axis=-1) # [N_test, N]

    top_k_train_indices = np.argsort(dist, axis=-1)[:, -k:] # [N_test, K]

    pred_y_k = y_train[top_k_train_indices] # [N_test, K]

    # choose majority label y for every row
    pred_y = np.array([np.bincount(x).argmax() for x in pred_y_k]) # [N_test,]

    return pred_y



def test_knn():
    k = 5
    y_pred = knn_predict(X_train, y_train, X_test, k)

    accuracy = (y_pred == y_test).mean()
    print(f"k={k}, Accuracy: {accuracy:.2%}")
    print(f"Predictions: {y_pred[:10]}")
    print(f"Actual:      {y_test[:10]}")


test_knn()

