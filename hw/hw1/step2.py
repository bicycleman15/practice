import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Ground truth weight (fixed for all experiments)
w1_true = torch.randn(1).item()


def generate_data_and_estimate(D: int, c: float, N: int = 50):
    """
    Step 2: Generate data and compute least-squares estimates.
    
    Args:
        D: Dimension of feature vector x
        c: Off-diagonal entries of covariance matrix (correlation between features)
        N: Number of samples (default 50)
    
    Returns:
        w1_all_hat: First coefficient from least-squares using all D features
        w1_hat: Coefficient from least-squares using only the first feature
    """
    
    # (1) Sample X
    # c would cancel out :)
    sigma = (1 - c) * torch.eye(D) + c * torch.ones(D, D)
    
    mean = torch.zeros(D)
    mvn = torch.distributions.MultivariateNormal(mean, sigma)
    X = mvn.sample((N,))  # Shape: (N, D)
    
    # (2) Compute y = w1_true * x_1 + ε_y where ε_y ~ N(0, 1)
    epsilon_y = torch.randn(N)
    y = w1_true * X[:, 0] + epsilon_y  # y only depends on first feature
    
    # (3) Compute least-squares solutions
    
    # w_all_hat: Using all D features, solve X @ w = y
    # Least squares solution: w = (X^T X)^{-1} X^T y
    w_all_hat = torch.linalg.inv(X.T @ X) @ X.T @ y
    w1_all_hat = w_all_hat[0].item()  # Take first coefficient
    
    # w1_hat: Using only the first feature
    # Solve X[:, 0] * w1 = y
    x1 = X[:, 0:1]  # Shape: (N, 1)
    w1_hat = torch.linalg.inv(x1.T @ x1) @ x1.T @ y
    w1_hat = w1_hat[0].item()
    
    return w1_all_hat, w1_hat


def run_trials(D: int, c: float, T: int = 100):
    """
    Step 3: Run T trials of Step 2 and compute statistics.
    
    Args:
        D: Dimension of feature vector x
        c: Off-diagonal entries of covariance matrix
        T: Number of trials (default 100)
    
    Returns:
        mean_w1_all: Mean of w1_all_hat across T trials
        std_w1_all: Std of w1_all_hat across T trials
        mean_w1: Mean of w1_hat across T trials
        std_w1: Std of w1_hat across T trials
    """
    w1_all_estimates = []
    w1_estimates = []
    
    for _ in range(T):
        w1_all_hat, w1_hat = generate_data_and_estimate(D, c)
        w1_all_estimates.append(w1_all_hat)
        w1_estimates.append(w1_hat)
    
    w1_all_tensor = torch.tensor(w1_all_estimates)
    w1_tensor = torch.tensor(w1_estimates)
    
    return (
        w1_all_tensor.mean().item(),
        w1_all_tensor.std().item(),
        w1_tensor.mean().item(),
        w1_tensor.std().item(),
    )


def step4_plots():
    """
    Step 4: Generate a single figure with 10 subplots (means and stds for each D).
    """
    D_values = [2, 4, 8, 16, 32]
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    
    for i, D in enumerate(D_values):
        means_all = []
        stds_all = []
        means_1 = []
        stds_1 = []
        
        print(f"Running D={D}...")
        for c in c_values:
            mean_all, std_all, mean_1, std_1 = run_trials(D, c, T=100)
            means_all.append(mean_all)
            stds_all.append(std_all)
            means_1.append(mean_1)
            stds_1.append(std_1)
        
        # Plot means (left column)
        ax_mean = axes[i, 0]
        ax_mean.plot(c_values, means_all, 'o-', label=r'$\hat{w}_1^{all}$')
        ax_mean.plot(c_values, means_1, 's-', label=r'$\hat{w}_1$')
        ax_mean.axhline(y=w1_true, color='k', linestyle='--', label=r'$w_1^{true}$')
        ax_mean.set_xlabel('c')
        ax_mean.set_ylabel('Mean')
        ax_mean.set_title(f'Mean (D={D})')
        ax_mean.set_ylim(0, 0.7)
        ax_mean.legend(fontsize=8)
        ax_mean.grid(True, alpha=0.3)
        
        # Plot standard deviations (right column)
        ax_std = axes[i, 1]
        ax_std.plot(c_values, stds_all, 'o-', label=r'$\hat{w}_1^{all}$')
        ax_std.plot(c_values, stds_1, 's-', label=r'$\hat{w}_1$')
        ax_std.set_xlabel('c')
        ax_std.set_ylabel('Std')
        ax_std.set_title(f'Std (D={D})')
        ax_std.set_ylim(0, 0.8)
        ax_std.legend(fontsize=8)
        ax_std.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('step4_all_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Done! Saved step4_all_plots.png")


if __name__ == "__main__":
    print(f"Ground truth w1_true = {w1_true:.4f}")
    print()
    step4_plots()
