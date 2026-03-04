import torch
import math
import matplotlib.pyplot as plt

torch.manual_seed(42)


def make_data(M: int, D: int):
    """
    Part 1: Generate data with Chebyshev polynomial features.
    
    Args:
        M: Number of samples
        D: Number of features (Chebyshev polynomials T_0 to T_{D-1})
    
    Returns:
        x: Input vector of shape (M,)
        y: Target vector of shape (M,)
        X: Feature matrix of shape (M, D)
    """
    # (i) Sample x uniformly from [-1, 1]
    x = 2 * torch.rand(M) - 1  # Uniform in [-1, 1]
    
    # (ii) Construct feature matrix X using Chebyshev polynomials
    # T_0(x) = 1, T_1(x) = x, T_i(x) = 2x * T_{i-1}(x) - T_{i-2}(x)
    X = torch.zeros(M, D)
    

    X[:, 0] = 1  # T_0(x) = 1
    X[:, 1] = x  # T_1(x) = x
    
    for i in range(2, D):
        X[:, i] = 2 * x * X[:, i - 1] - X[:, i - 2]
    
    # (iii) Generate targets: y = 2x + sin(8πx)
    y = 2 * x + torch.sin(8 * math.pi * x)
    
    return x, y, X


def stochastic_solution(X: torch.Tensor, y: torch.Tensor, N: int):
    """
    Part 2: Compute minimum-norm least-squares solution on a random subset.
    
    Args:
        X: Feature matrix of shape (M, D)
        y: Target vector of shape (M,)
        N: Number of samples to use (N < M)
    
    Returns:
        w: Weight vector of shape (D,)
    """
    M = X.shape[0]
    
    # Uniformly sample N rows without replacement
    indices = torch.randperm(M)[:N]
    X_sub = X[indices]
    y_sub = y[indices]
    
    # Compute minimum-norm least-squares solution using pseudoinverse
    w = torch.linalg.pinv(X_sub) @ y_sub
    
    return w


def chebyshev_features(x: torch.Tensor, D: int):
    """
    Compute Chebyshev polynomial features for arbitrary x values.
    
    Args:
        x: Input tensor of shape (N,)
        D: Number of Chebyshev polynomials
    
    Returns:
        X: Feature matrix of shape (N, D)
    """
    N = x.shape[0]
    X = torch.zeros(N, D)
    
    X[:, 0] = 1  # T_0(x) = 1
    X[:, 1] = x  # T_1(x) = x
    
    for i in range(2, D):
        X[:, i] = 2 * x * X[:, i - 1] - X[:, i - 2]
    
    return X


def part3_plots():
    """
    Part 3: Generate plots for D ∈ {5, 15, 30}.
    """
    M = 30
    N = 15
    T = 3
    D_values = [5, 15, 30]
    
    # Dense x for plotting smooth curves
    x_plot = torch.linspace(-1, 1, 200)
    y_true = 2 * x_plot + torch.sin(8 * math.pi * x_plot)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, D in enumerate(D_values):
        ax = axes[idx]
        
        # (i) Generate data
        x, y, X = make_data(M, D)
        
        # Plot data points
        ax.scatter(x.numpy(), y.numpy(), c='black', s=20, label='Data $(x, y)$', zorder=5)
        
        # Plot ground truth
        ax.plot(x_plot.numpy(), y_true.numpy(), 'k--', linewidth=2, label='Ground truth')
        
        # (ii) Run stochastic_solution T=3 times
        X_plot = chebyshev_features(x_plot, D)
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        
        for t in range(T):
            w = stochastic_solution(X, y, N)
            f_t = X_plot @ w
            ax.plot(x_plot.numpy(), f_t.numpy(), color=colors[t], 
                    linewidth=1.5, label=f'Trial {t+1}')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'D = {D} (N={N}, M={M})')
        ax.set_ylim(-3, 3)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('p6_part3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved p6_part3.png")


if __name__ == "__main__":
    part3_plots()
