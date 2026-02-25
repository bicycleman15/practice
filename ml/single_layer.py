import time

import numpy as np
from tqdm import tqdm

np.random.seed(42)

in_dim = 10
hidden = 5
out_dim = 2

batch_size = 4

weight1 = np.random.randn(in_dim, hidden)
weight2 = np.random.randn(hidden, out_dim)

x = np.random.randn(batch_size, in_dim)
y = np.random.randn(batch_size, out_dim)

def grad_relu(x):
    return (x > 0).astype(float)

def take_step(x, y, weight1, weight2, lr=0.01, wd=0.1):

    h = x @ weight1 # [B, H]
    x1 = np.maximum(0, h) # take relu

    pred = x1 @ weight2 # [B, V]


    loss = ((pred - y)**2).mean()

    dweight2 = 1/batch_size * x1.T @ (pred - y)

    dx1 = 1/batch_size * (pred - y) @ weight2.T

    dh = grad_relu(h) * dx1

    dweight1 = x.T @ dh

    # weight = weight * (1 - wd) - grad * lr # decoupled wd
    # weight = weight * (1 - wd) - grad * lr # decoupled wd

    weight1 = weight1 - lr * dweight1
    weight2 = weight2 - lr * dweight2

    return loss, weight1, weight2

# bar = tqdm(range(1000))
# for _ in bar:
#     loss, weight1, weight2 = take_step(x, y, weight1, weight2)
#     bar.set_postfix(loss=f"{loss:.4f}")
#     time.sleep(0.01)
#     # print(weight)

def test_nonlinearity():
    np.random.seed(0)

    # XOR-like problem: linear model can't solve this
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # train WITH relu
    w1 = np.random.randn(2, 8) * 0.5
    w2 = np.random.randn(8, 1) * 0.5
    for _ in range(5000):
        _, w1, w2 = take_step(X, y, w1, w2, lr=0.05)
    h = X @ w1
    x1 = np.maximum(0, h)
    pred_relu = x1 @ w2
    loss_relu = ((pred_relu - y)**2).mean()

    # train WITHOUT relu (linear)
    np.random.seed(0)
    w_lin = np.random.randn(2, 1) * 0.5
    for _ in range(5000):
        pred_lin = X @ w_lin
        grad = 1/4 * X.T @ (pred_lin - y)
        w_lin = w_lin - 0.05 * grad
    pred_lin = X @ w_lin
    loss_lin = ((pred_lin - y)**2).mean()

    print(f"With ReLU    -> loss: {loss_relu:.4f}, preds: {pred_relu.flatten().round(2)}")
    print(f"Without ReLU -> loss: {loss_lin:.4f}, preds: {pred_lin.flatten().round(2)}")
    print(f"Target:                                preds: {y.flatten()}")
    print(f"\nReLU beats linear: {loss_relu < loss_lin}")


test_nonlinearity()
