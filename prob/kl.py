# Optimize p_θ towards bimodal target using Forward vs Reverse KL
# Demonstrates mode-covering (forward KL) vs mode-seeking (reverse KL) behavior
# Now with both unimodal and expressive (mixture) parameterizations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal, MixtureSameFamily, Categorical


class UnimodalGaussian(nn.Module):
    """Learnable unimodal Gaussian p_θ(x) with mean and covariance parameters."""
    
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        self.n_components = 1
        # Learnable parameters
        self.mean = nn.Parameter(torch.zeros(dim))
        # Parameterize covariance as L @ L.T for positive definiteness
        self.L_raw = nn.Parameter(torch.eye(dim))
    
    @property
    def covariance(self):
        L = torch.tril(self.L_raw)
        return L @ L.T + 1e-4 * torch.eye(self.dim)
    
    def get_distribution(self):
        return MultivariateNormal(self.mean, self.covariance)
    
    def log_prob(self, x):
        return self.get_distribution().log_prob(x)
    
    def sample(self, n_samples):
        return self.get_distribution().rsample((n_samples,))


class MixtureOfGaussians(nn.Module):
    """Learnable Mixture of Gaussians p_θ(x) - expressive enough to fit multimodal targets."""
    
    def __init__(self, dim=2, n_components=2):
        super().__init__()
        self.dim = dim
        self.n_components = n_components
        
        # Learnable mixture weights (logits, will be softmaxed)
        self.mix_logits = nn.Parameter(torch.zeros(n_components))
        
        # Learnable means for each component
        self.means = nn.Parameter(torch.randn(n_components, dim) * 0.5)
        
        # Learnable Cholesky factors for each component's covariance
        self.L_raw = nn.Parameter(torch.stack([torch.eye(dim) for _ in range(n_components)]))
    
    @property
    def mix_weights(self):
        return F.softmax(self.mix_logits, dim=0)
    
    def get_covariances(self):
        """Get covariance matrices from Cholesky factors."""
        covs = []
        for k in range(self.n_components):
            L = torch.tril(self.L_raw[k])
            cov = L @ L.T + 1e-4 * torch.eye(self.dim)
            covs.append(cov)
        return torch.stack(covs)
    
    def get_distribution(self):
        """Return the mixture distribution."""
        mix = Categorical(self.mix_weights)
        comp = MultivariateNormal(self.means, self.get_covariances())
        return MixtureSameFamily(mix, comp)
    
    def log_prob(self, x):
        return self.get_distribution().log_prob(x)
    
    def sample(self, n_samples):
        """Sample using reparameterization trick (approximate via Gumbel-softmax or component sampling)."""
        dist = self.get_distribution()
        # MixtureSameFamily doesn't support rsample, so we use a workaround:
        # Sample component indices, then sample from those components
        with torch.no_grad():
            component_indices = Categorical(self.mix_weights).sample((n_samples,))
        
        # Sample from all components and select based on indices
        samples = []
        for k in range(self.n_components):
            L = torch.tril(self.L_raw[k])
            cov = L @ L.T + 1e-4 * torch.eye(self.dim)
            comp_dist = MultivariateNormal(self.means[k], cov)
            samples.append(comp_dist.rsample((n_samples,)))
        
        samples = torch.stack(samples, dim=1)  # (n_samples, n_components, dim)
        # Select samples based on component indices
        selected = samples[torch.arange(n_samples), component_indices]
        return selected


def create_bimodal_target(mode_distance=3.0):
    """Create bimodal Gaussian mixture target distribution."""
    # Two modes separated along x-axis
    means = torch.tensor([
        [-mode_distance/2, -mode_distance/2],
        [mode_distance/2, mode_distance/2]
    ])
    # Covariance for each component (slightly elongated)
    covs = torch.stack([
        torch.tensor([[0.5, 0.3], [0.3, 0.5]]),
        torch.tensor([[0.5, 0.3], [0.3, 0.5]])
    ])
    
    mix = Categorical(torch.ones(2))
    comp = MultivariateNormal(means, covs)
    return MixtureSameFamily(mix, comp)


def forward_kl_loss(p_theta, target, n_samples=1000):
    """
    Forward KL: D_KL(t || p_θ) = E_t[log t(x) - log p_θ(x)]
    
    Minimizing this is equivalent to maximum likelihood on samples from target.
    Results in MODE-COVERING behavior (p_θ tries to cover all of t).
    """
    # Sample from target distribution
    with torch.no_grad():
        x_target = target.sample((n_samples,))
    
    # We minimize -E_t[log p_θ(x)] (the entropy of t is constant w.r.t θ)
    log_prob_p = p_theta.log_prob(x_target)
    return -log_prob_p.mean()


def reverse_kl_loss_reparam(p_theta, target, n_samples=1000):
    """
    Reverse KL: D_KL(p_θ || t) = E_{p_θ}[log p_θ(x) - log t(x)]
    
    Uses reparameterization trick for gradient estimation.
    Results in MODE-SEEKING behavior (p_θ locks onto one mode of t).
    """
    # Sample from p_θ using reparameterization trick (rsample)
    x_p = p_theta.sample(n_samples)
    
    # KL = E_{p_θ}[log p_θ(x)] - E_{p_θ}[log t(x)]
    log_prob_p = p_theta.log_prob(x_p)
    log_prob_t = target.log_prob(x_p)
    
    return (log_prob_p - log_prob_t).mean()


def reverse_kl_loss_score(p_theta, target, n_samples=1000, baseline=None, 
                          analytic_entropy=True):
    """
    Reverse KL: D_KL(p_θ || t) = E_{p_θ}[log p_θ(x)] - E_{p_θ}[log t(x)]
                               = -H(p_θ) - E_{p_θ}[log t(x)]
    
    Uses score function gradient (REINFORCE) for gradient estimation.
    
    ∇_θ E_{p_θ}[f(x)] = E_{p_θ}[f(x) · ∇_θ log p_θ(x)]
    
    Args:
        analytic_entropy: If True, use analytic entropy for Gaussian.
                         If False, estimate entropy gradient via score function too.
    """
    dist = p_theta.get_distribution()
    
    # Sample WITHOUT gradients flowing through samples
    with torch.no_grad():
        x_p = dist.sample((n_samples,))
    
    # log p_θ(x) with gradients enabled
    log_prob_p = dist.log_prob(x_p)
    
    # Term 1: Negative entropy -H(p_θ) = E_{p_θ}[log p_θ(x)]
    if analytic_entropy:
        # For Gaussian, entropy is analytic: H = 0.5 * log((2πe)^d |Σ|)
        neg_entropy = -dist.entropy()
    else:
        # Score function gradient for entropy term:
        # ∇_θ E_{p_θ}[log p_θ(x)] = E_{p_θ}[log p_θ(x) · ∇_θ log p_θ(x)]
        # 
        # In PyTorch: use (log_prob.detach() * log_prob) so that:
        # - log_prob.detach() acts as the "reward" (no gradient)
        # - log_prob provides ∇_θ log p_θ(x) via backprop
        #
        # We can use a baseline b to reduce variance:
        # E[(log p - b) · ∇log p] = E[log p · ∇log p] - b·E[∇log p]
        #                         = E[log p · ∇log p]  (since E[∇log p] = 0)
        entropy_baseline = log_prob_p.detach().mean()
        advantage_entropy = log_prob_p.detach() - entropy_baseline
        neg_entropy = (advantage_entropy * log_prob_p).mean()
    
    # Term 2: -E_{p_θ}[log t(x)] using REINFORCE
    with torch.no_grad():
        log_prob_t = target.log_prob(x_p)
        
        # Reward: we want to maximize E[log t(x)], so reward = log t(x)
        reward = log_prob_t
        
        # Optional baseline for variance reduction
        if baseline is None:
            baseline = reward.mean()
        advantage = reward - baseline
    
    # REINFORCE loss: -E[(reward - baseline) · log p_θ(x)]
    # Negative because we want to MAXIMIZE E[log t(x)]
    cross_entropy_loss = -(advantage * log_prob_p).mean()
    
    # Total reverse KL loss
    return neg_entropy + cross_entropy_loss


def reverse_kl_loss(p_theta, target, n_samples=1000, method='reparam'):
    """Wrapper to select gradient estimation method.
    
    Methods:
        'reparam': Reparameterization trick (lowest variance, requires differentiable sampling)
        'score': Score function with analytic entropy (medium variance)
        'score_full': Pure score function for both terms (highest variance, most general)
    """
    if method == 'reparam':
        return reverse_kl_loss_reparam(p_theta, target, n_samples)
    elif method == 'score':
        return reverse_kl_loss_score(p_theta, target, n_samples, analytic_entropy=True)
    elif method == 'score_full':
        return reverse_kl_loss_score(p_theta, target, n_samples, analytic_entropy=False)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'reparam', 'score', 'score_full'")


def optimize_kl(kl_type='forward', n_iters=500, lr=0.05, mode_distance=3.0, 
                method='reparam', seed=42, init_mean=None, model_type='unimodal',
                n_components=2, n_samples=500):
    """Optimize p_θ using specified KL divergence.
    
    Args:
        kl_type: 'forward' for D(t||p_θ) or 'reverse' for D(p_θ||t)
        n_iters: number of optimization iterations
        lr: learning rate
        mode_distance: separation between modes of target
        method: 'reparam' (reparameterization) or 'score' (REINFORCE) for reverse KL
        seed: random seed for reproducibility
        init_mean: initial mean for p_θ (default: [0.0, 2.0]) - only for unimodal
        model_type: 'unimodal' for single Gaussian, 'mixture' for MoG
        n_components: number of components for mixture model
        n_samples: number of samples for gradient estimation
    """
    
    torch.manual_seed(seed)
    
    # Create target and model
    target = create_bimodal_target(mode_distance)
    
    if model_type == 'unimodal':
        p_theta = UnimodalGaussian(dim=2)
        # Initialize mean
        if init_mean is None:
            init_mean = [0.0, 2.0]
        p_theta.mean.data = torch.tensor(init_mean, dtype=torch.float32)
        optimizer = torch.optim.Adam(p_theta.parameters(), lr=lr)
    else:  # mixture
        p_theta = MixtureOfGaussians(dim=2, n_components=n_components)
        # Initialize means spread out
        if init_mean is None:
            # Spread initial means randomly
            p_theta.means.data = torch.randn(n_components, 2) * 2
        else:
            p_theta.means.data = torch.tensor(init_mean, dtype=torch.float32)
        
        # Use separate learning rates for mixture:
        # - Normal LR for means and covariances (good gradients via reparameterization)
        # - Lower LR for mixing weights (biased gradients due to discrete sampling)
        optimizer = torch.optim.Adam([
            {'params': [p_theta.means, p_theta.L_raw], 'lr': lr},
            {'params': [p_theta.mix_logits], 'lr': lr * 0.02}  # Much lower for weights
        ])
    
    history = {'loss': [], 'mean': [], 'cov': []}
    
    for i in range(n_iters):
        optimizer.zero_grad()
        
        if kl_type == 'forward':
            loss = forward_kl_loss(p_theta, target, n_samples=n_samples)
        else:
            loss = reverse_kl_loss(p_theta, target, n_samples=n_samples, method=method)
        
        loss.backward()
        optimizer.step()
        
        history['loss'].append(loss.item())
        
        # Track means (handle both unimodal and mixture)
        if model_type == 'unimodal':
            history['mean'].append(p_theta.mean.detach().clone().numpy())
            history['cov'].append(p_theta.covariance.detach().clone().numpy())
            mean_str = f"[{p_theta.mean[0].item():.2f}, {p_theta.mean[1].item():.2f}]"
        else:
            history['mean'].append(p_theta.means.detach().clone().numpy())
            history['cov'].append(p_theta.get_covariances().detach().clone().numpy())
            means = p_theta.means.detach()
            weights = p_theta.mix_weights.detach()
            mean_str = f"w={weights.numpy().round(2)}, μ1=[{means[0,0]:.1f},{means[0,1]:.1f}], μ2=[{means[1,0]:.1f},{means[1,1]:.1f}]"
        
        if (i + 1) % 100 == 0:
            print(f"Iter {i+1}: Loss = {loss.item():.4f}, {mean_str}")
    
    return p_theta, target, history


def plot_results(p_theta, target, history, kl_type, ax=None):
    """Plot contours of target and optimized p_θ."""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create grid for contour plot
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x, y)
    pos = torch.tensor(np.stack([X, Y], axis=-1), dtype=torch.float32)
    
    # Evaluate target density
    with torch.no_grad():
        Z_target = target.log_prob(pos.reshape(-1, 2)).reshape(X.shape).exp().numpy()
        Z_p = p_theta.log_prob(pos.reshape(-1, 2)).reshape(X.shape).exp().numpy()
    
    # Plot contours
    ax.contour(X, Y, Z_target, levels=6, colors='blue', linewidths=1.5, alpha=0.8)
    ax.contour(X, Y, Z_p, levels=6, colors='red', linewidths=1.5, alpha=0.8)
    
    # Plot mean trajectory
    means = np.array(history['mean'])
    ax.plot(means[:, 0], means[:, 1], 'g.-', alpha=0.5, markersize=2, linewidth=1)
    ax.plot(means[-1, 0], means[-1, 1], 'g*', markersize=15, label='Final mean')
    ax.plot(means[0, 0], means[0, 1], 'ko', markersize=8, label='Initial mean')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    title = "Forward KL: D(t||p_θ) - Mode Covering" if kl_type == 'forward' else "Reverse KL: D(p_θ||t) - Mode Seeking"
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    return ax


def main():
    """Run optimization with both forward and reverse KL."""
    
    print("=" * 60)
    print("Optimizing unimodal Gaussian towards bimodal target")
    print("=" * 60)
    
    # With mode_distance=4.0, modes are at (-2, -2) and (2, 2)
    # Use different inits to show mode-seeking behavior
    init_near_mode1 = [-2.0, -1.5]  # Start near mode 1
    init_near_mode2 = [2.0, 1.5]    # Start near mode 2
    init_between = [0.0, 3.0]       # Between modes (above)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    mode_dist = 4.0  # Larger separation for clearer mode-seeking
    
    # Forward KL - Mode Covering (init doesn't matter much)
    print("\n--- Forward KL (Mode Covering) ---")
    p_forward, target, hist_forward = optimize_kl(
        'forward', n_iters=500, lr=0.05, init_mean=init_between, mode_distance=mode_dist
    )
    plot_results(p_forward, target, hist_forward, 'forward', axes[0, 0])
    axes[0, 0].set_title("Forward KL: D(t||p_θ)\nMode Covering")
    
    # Reverse KL - init near mode 1 -> locks onto mode 1
    print("\n--- Reverse KL (Init near Mode 1) ---")
    p_reverse1, target, hist_reverse1 = optimize_kl(
        'reverse', n_iters=500, lr=0.05, method='reparam', seed=42,
        init_mean=init_near_mode1, mode_distance=mode_dist
    )
    plot_results(p_reverse1, target, hist_reverse1, 'reverse', axes[0, 1])
    axes[0, 1].set_title("Reverse KL: D(p_θ||t)\nInit near Mode 1")
    
    # Reverse KL - init near mode 2 -> locks onto mode 2
    print("\n--- Reverse KL (Init near Mode 2) ---")
    p_reverse2, target, hist_reverse2 = optimize_kl(
        'reverse', n_iters=500, lr=0.05, method='reparam', seed=42,
        init_mean=init_near_mode2, mode_distance=mode_dist
    )
    plot_results(p_reverse2, target, hist_reverse2, 'reverse', axes[1, 0])
    axes[1, 0].set_title("Reverse KL: D(p_θ||t)\nInit near Mode 2")
    
    # Reverse KL - init between modes (can go either way)
    print("\n--- Reverse KL (Init between modes) ---")
    p_reverse_between, target, hist_reverse_between = optimize_kl(
        'reverse', n_iters=500, lr=0.05, method='reparam', seed=42,
        init_mean=init_between, mode_distance=mode_dist
    )
    plot_results(p_reverse_between, target, hist_reverse_between, 'reverse', axes[1, 1])
    axes[1, 1].set_title("Reverse KL: D(p_θ||t)\nInit between modes")
    
    plt.suptitle("Forward KL (mode covering) vs Reverse KL (mode seeking)\n"
                 "Blue = target t, Red = learned p_θ, Green = optimization trajectory", 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('prob/kl_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Second figure: Compare gradient estimators with same init
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    print("\n--- Gradient Estimator Comparison (same init) ---")
    
    p_reparam, _, hist_reparam = optimize_kl(
        'reverse', n_iters=500, lr=0.05, method='reparam', seed=42,
        init_mean=init_near_mode1, mode_distance=mode_dist
    )
    plot_results(p_reparam, target, hist_reparam, 'reverse', axes[0])
    axes[0].set_title("Reparameterization")
    
    p_score, _, hist_score = optimize_kl(
        'reverse', n_iters=500, lr=0.05, method='score', seed=42,
        init_mean=init_near_mode1, mode_distance=mode_dist
    )
    plot_results(p_score, target, hist_score, 'reverse', axes[1])
    axes[1].set_title("Score + Analytic Entropy")
    
    p_score_full, _, hist_score_full = optimize_kl(
        'reverse', n_iters=500, lr=0.05, method='score_full', seed=42,
        init_mean=init_near_mode1, mode_distance=mode_dist
    )
    plot_results(p_score_full, target, hist_score_full, 'reverse', axes[2])
    axes[2].set_title("Full Score Function")
    
    plt.suptitle("Reverse KL: Gradient Estimator Comparison\n(Same init near Mode 1)", 
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('prob/kl_gradient_methods.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(hist_forward['loss'], label='Forward KL')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Forward KL Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(hist_reparam['loss'], label='Reparameterization', alpha=0.8)
    axes[1].plot(hist_score['loss'], label='Score + Analytic H', alpha=0.8)
    axes[1].plot(hist_score_full['loss'], label='Full Score Function', alpha=0.8)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reverse KL: Gradient Estimator Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prob/kl_losses.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print final results
    print("\n" + "=" * 60)
    print("Final Results:")
    print("=" * 60)
    print(f"\nForward KL final mean: [{p_forward.mean[0].item():.3f}, {p_forward.mean[1].item():.3f}]")
    print("  -> Covers both modes with large covariance")
    print(f"\nReverse KL (init near mode 1): [{p_reverse1.mean[0].item():.3f}, {p_reverse1.mean[1].item():.3f}]")
    print(f"Reverse KL (init near mode 2): [{p_reverse2.mean[0].item():.3f}, {p_reverse2.mean[1].item():.3f}]")
    print("  -> Locks onto nearest mode (mode-seeking)")
    
    print("\n" + "=" * 60)
    print("Key Insight:")
    print("=" * 60)
    print("""
    Forward KL D(t||p_θ): "Where t has mass, p_θ must have mass"
      -> p_θ spreads to COVER all modes (mode-covering)
      
    Reverse KL D(p_θ||t): "Where p_θ has mass, t must have mass"  
      -> p_θ concentrates on ONE mode to avoid penalty (mode-seeking)
      -> Which mode depends on initialization!
    """)
    
    # ==========================================
    # EXPRESSIVE MODEL: Mixture of Gaussians
    # ==========================================
    print("\n" + "=" * 60)
    print("EXPRESSIVE MODEL: Mixture of Gaussians")
    print("=" * 60)
    print("Now p_θ can fit both modes -> KL can go to ~0!")
    print("(Using lower LR for mixing weights to prevent drift)")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    n_samp = 2000  # More samples for better gradient estimates
    
    # Forward KL with mixture
    print("\n--- Forward KL with Mixture of Gaussians ---")
    p_mix_forward, target, hist_mix_forward = optimize_kl(
        'forward', n_iters=1000, lr=0.05, mode_distance=mode_dist,
        model_type='mixture', n_components=2, seed=42, n_samples=n_samp
    )
    plot_results(p_mix_forward, target, hist_mix_forward, 'forward', axes[0])
    axes[0].set_title("Forward KL: D(t||p_θ)\nMixture of Gaussians")
    
    # Reverse KL with mixture
    print("\n--- Reverse KL with Mixture of Gaussians ---")
    p_mix_reverse, target, hist_mix_reverse = optimize_kl(
        'reverse', n_iters=1000, lr=0.05, mode_distance=mode_dist,
        model_type='mixture', n_components=2, seed=42, method='reparam', n_samples=n_samp
    )
    plot_results(p_mix_reverse, target, hist_mix_reverse, 'reverse', axes[1])
    axes[1].set_title("Reverse KL: D(p_θ||t)\nMixture of Gaussians")
    
    # Reverse KL with unimodal for comparison
    print("\n--- Reverse KL with Unimodal (for comparison) ---")
    p_unimodal, target, hist_unimodal = optimize_kl(
        'reverse', n_iters=1000, lr=0.05, mode_distance=mode_dist,
        model_type='unimodal', seed=42, init_mean=init_near_mode1, n_samples=n_samp
    )
    plot_results(p_unimodal, target, hist_unimodal, 'reverse', axes[2])
    axes[2].set_title("Reverse KL: D(p_θ||t)\nUnimodal (can't fit both)")
    
    plt.suptitle("Expressive p_θ (Mixture) can fit both modes\n"
                 "Blue = target t, Red = learned p_θ", fontsize=12)
    plt.tight_layout()
    plt.savefig('prob/kl_expressive.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Loss comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_mix_forward['loss'], label='Forward KL (Mixture)', alpha=0.8)
    ax.plot(hist_mix_reverse['loss'], label='Reverse KL (Mixture)', alpha=0.8)
    ax.plot(hist_unimodal['loss'], label='Reverse KL (Unimodal)', alpha=0.8)
    ax.axhline(y=np.log(2), color='gray', linestyle='--', label=f'log(2) ≈ {np.log(2):.3f}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Expressive Model: Loss can go below log(2)!')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('prob/kl_expressive_loss.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("Expressive Model Results:")
    print("=" * 60)
    print(f"\nForward KL (Mixture) final loss: {hist_mix_forward['loss'][-1]:.4f}")
    print(f"Reverse KL (Mixture) final loss: {hist_mix_reverse['loss'][-1]:.4f}")
    print(f"Reverse KL (Unimodal) final loss: {hist_unimodal['loss'][-1]:.4f}")
    print(f"\nlog(2) = {np.log(2):.4f} (theoretical minimum for unimodal)")
    print("\n-> Mixture model can achieve loss < log(2) because it can fit BOTH modes!")


if __name__ == "__main__":
    main()
