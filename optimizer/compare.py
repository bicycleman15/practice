"""Compare SGD vs SGD with momentum on 2D ravine and MNIST."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


# --- Rosenbrock ---
def rosenbrock(params, a=1.0, b=100.0):
    """f(x,y) = (a-x)^2 + b*(y-x^2)^2, minimum at (1,1)."""
    x, y = params[0], params[1]
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def optimize_rosenbrock(start, lr, momentum, steps, use_rmsprop=False):
    params = torch.tensor(start, dtype=torch.float32, requires_grad=True)
    if use_rmsprop:
        optimizer = torch.optim.RMSprop([params], lr=lr)
    else:
        optimizer = torch.optim.SGD([params], lr=lr, momentum=momentum)
    
    trajectory = [start]
    losses = []
    
    for _ in range(steps):
        optimizer.zero_grad()
        loss = rosenbrock(params)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        trajectory.append((params[0].item(), params[1].item()))
    
    return np.array(trajectory), losses


# --- MNIST ---
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_mnist(lr, momentum, epochs=5, batch_size=64, use_rmsprop=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)
    
    model = SmallNet().to(device)
    if use_rmsprop:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    train_losses = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
        test_accs.append(100 * correct / len(test_data))
        print(f"  Epoch {epoch+1}: loss={train_losses[-1]:.4f}, acc={test_accs[-1]:.2f}%")
    
    return train_losses, test_accs


# --- Sparse Embeddings (where adaptive LR shines) ---
def train_sparse_embeddings(lr, momentum, steps=2000, use_rmsprop=False, use_adam=False):
    """
    Learn embeddings with sparse, imbalanced feature access.
    Some features are accessed frequently, others rarely.
    Adaptive methods shine here because they give larger updates to rare features.
    """
    vocab_size = 1000
    embed_dim = 32
    num_classes = 5
    
    # Create skewed distribution - some words very common, most rare (Zipf-like)
    probs = 1.0 / np.arange(1, vocab_size + 1)
    probs = probs / probs.sum()
    
    # Generate training data
    np.random.seed(42)
    n_samples = 5000
    words = np.random.choice(vocab_size, size=n_samples, p=probs)
    labels = torch.tensor(words % num_classes)  # simple label rule
    words = torch.tensor(words)
    
    # Model: embedding -> linear -> output
    embedding = nn.Embedding(vocab_size, embed_dim)
    classifier = nn.Linear(embed_dim, num_classes)
    params = list(embedding.parameters()) + list(classifier.parameters())
    
    if use_adam:
        optimizer = torch.optim.Adam(params, lr=lr)
    elif use_rmsprop:
        optimizer = torch.optim.RMSprop(params, lr=lr)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    
    losses = []
    batch_size = 64
    
    for step in range(steps):
        idx = np.random.randint(0, n_samples, batch_size)
        x, y = words[idx], labels[idx]
        
        optimizer.zero_grad()
        out = classifier(embedding(x))
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            losses.append(loss.item())
    
    # Final accuracy
    with torch.no_grad():
        out = classifier(embedding(words))
        acc = (out.argmax(1) == labels).float().mean().item() * 100
    
    return losses, acc


# --- Linear Regression with different feature scales ---
def train_scaled_regression(lr, momentum, steps=2000, use_rmsprop=False, use_adam=False, normalize=False, deep=False):
    """
    Linear regression where features have wildly different scales.
    x1 ~ [0, 1], x2 ~ [0, 1000], x3 ~ [0, 0.001]
    SGD struggles because one LR can't fit all scales.
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Features with different scales (not too extreme)
    x1 = np.random.randn(n_samples, 1)          # scale ~1
    x2 = np.random.randn(n_samples, 1) * 100    # scale ~100
    x3 = np.random.randn(n_samples, 1) * 0.01   # scale ~0.01
    X = np.hstack([x1, x2, x3]).astype(np.float32)
    
    # True weights
    true_w = np.array([[2.0], [0.03], [500.0]])  # adjusted so contributions are similar
    y = (X @ true_w + np.random.randn(n_samples, 1) * 0.5).astype(np.float32)
    
    # Normalize features to zero mean, unit variance
    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    X = torch.tensor(X)
    y = torch.tensor(y)
    
    # Residual block: x -> Linear -> ReLU -> (+x)
    class ResBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
        
        def forward(self, x):
            return x + F.relu(self.linear(x))
    
    # Model: deep or shallow
    if deep == 'residual':
        model = nn.Sequential(
            nn.Linear(3, 64),      # project to hidden dim
            ResBlock(64),          # x -> Linear -> ReLU -> (+x)
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            nn.Linear(64, 1)       # project to output
        )
    elif deep:
        model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    else:
        model = nn.Linear(3, 1)
    
    if use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif use_rmsprop:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    losses = []
    batch_size = 64
    
    for step in range(steps):
        idx = np.random.randint(0, n_samples, batch_size)
        xb, yb = X[idx], y[idx]
        
        optimizer.zero_grad()
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            losses.append(loss.item())
    
    # Final loss on all data
    with torch.no_grad():
        final_loss = F.mse_loss(model(X), y).item()
    
    return losses, final_loss


def smooth(losses, window=20):
    """Moving average smoothing."""
    smoothed = []
    for i in range(len(losses)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(losses[start:i+1]))
    return smoothed


def plot_scaled_regression(results, configs, title='Linear Regression with Multi-Scale Features'):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for losses, (_, _, _, color, label) in zip(results, configs):
        ax.semilogy(smooth(losses), color=color, linewidth=2, label=label)
    
    ax.set(xlabel='Step (x10)', ylabel='Loss (log)', title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scaled_regression_comparison.png', dpi=150)
    plt.show()


def plot_sparse(results, configs):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for losses, (_, _, _, color, label) in zip(results, configs):
        ax.plot(smooth(losses), color=color, linewidth=2, label=label)
    
    ax.set(xlabel='Step (x10)', ylabel='Loss', title='Sparse Embeddings (Zipf distribution)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sparse_comparison.png', dpi=150)
    plt.show()


def plot_rosenbrock(results, configs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x, y = np.linspace(-2, 2, 200), np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100*(Y - X**2)**2
    
    ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.5)
    for (traj, _), (_, _, color, label) in zip(results, configs):
        ax1.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=2.5, label=label)
        ax1.scatter(traj[::50, 0], traj[::50, 1], color=color, marker='o', s=40, edgecolors='black', zorder=5)
    ax1.scatter(1, 1, color='lime', marker='*', s=200, edgecolors='black', label='Optimum')
    ax1.set(xlabel='x', ylabel='y', title='Rosenbrock Trajectories', xlim=(-2, 2), ylim=(-1, 3))
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    for (_, losses), (_, _, color, label) in zip(results, configs):
        ax2.semilogy(losses, color=color, linewidth=2, label=label)
    ax2.set(xlabel='Iteration', ylabel='Loss', title='Rosenbrock Convergence')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rosenbrock_comparison.png', dpi=150)
    plt.show()


def plot_mnist(mnist_results, configs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for (losses, accs), (_, _, color, label) in zip(mnist_results, configs):
        epochs = range(1, len(losses) + 1)
        ax1.plot(epochs, losses, color=color, linewidth=2, marker='o', label=label)
        ax2.plot(epochs, accs, color=color, linewidth=2, marker='o', label=label)
    
    ax1.set(xlabel='Epoch', ylabel='Training Loss', title='MNIST Training Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set(xlabel='Epoch', ylabel='Test Accuracy (%)', title='MNIST Test Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mnist_comparison.png', dpi=150)
    plt.show()


def main():
    configs = [
        (0.0, False, False, 'red', 'SGD (no momentum)'),
        (0.9, False, False, 'blue', 'SGD (momentum=0.9)'),
        (0.0, True, False, 'orange', 'RMSprop'),
        (0.0, False, True, 'purple', 'Adam'),
    ]
    
    # Deep network regression (normalized input)
    print("=== Deep Network (no residuals) ===")
    deep_results = []
    for momentum, use_rmsprop, use_adam, color, label in configs:
        lr = 0.001 if (use_rmsprop or use_adam) else 0.01
        losses, final = train_scaled_regression(lr, momentum, steps=3000, use_rmsprop=use_rmsprop, use_adam=use_adam, normalize=True, deep=True)
        print(f"{label}: final_loss={final:.4f}")
        deep_results.append(losses)
    
    plot_scaled_regression(deep_results, configs, title='Deep Network (no residuals)')
    
    # Deep network with residuals (no LayerNorm)
    print("\n=== Deep Network with Residuals (no LayerNorm) ===")
    deep_ln_results = []
    for momentum, use_rmsprop, use_adam, color, label in configs:
        lr = 0.001 if (use_rmsprop or use_adam) else 0.01
        losses, final = train_scaled_regression(lr, momentum, steps=3000, use_rmsprop=use_rmsprop, use_adam=use_adam, normalize=True, deep='residual')
        print(f"{label}: final_loss={final:.4f}")
        deep_ln_results.append(losses)
    
    plot_scaled_regression(deep_ln_results, configs, title='Deep Network (with residuals, no LayerNorm)')
    
    # Sparse embeddings - where adaptive LR really matters
    print("\n=== Sparse Embeddings ===")
    sparse_results = []
    for momentum, use_rmsprop, use_adam, color, label in configs:
        lr = 0.01 if (use_rmsprop or use_adam) else 0.1
        losses, acc = train_sparse_embeddings(lr, momentum, steps=2000, use_rmsprop=use_rmsprop, use_adam=use_adam)
        print(f"{label}: final_acc={acc:.2f}%")
        sparse_results.append(losses)
    
    plot_sparse(sparse_results, configs)
    
    # Skip MNIST for now - sparse task is more interesting
    # Uncomment below to also run MNIST
    """
    print("\n=== MNIST ===")
    mnist_results = []
    for momentum, use_rmsprop, use_adam, color, label in configs:
        print(f"\n{label}:")
        lr = 0.001 if (use_rmsprop or use_adam) else 0.01
        losses, accs = train_mnist(lr=lr, momentum=momentum, epochs=5, use_rmsprop=use_rmsprop)
        mnist_results.append((losses, accs))
    plot_mnist(mnist_results, configs)
    """


if __name__ == "__main__":
    main()
